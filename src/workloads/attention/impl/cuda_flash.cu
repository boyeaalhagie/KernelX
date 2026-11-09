#include "workloads/attention/impl/cuda_flash.hpp"
#include "utils/cuda_utils.hpp"
#include <cfloat>

// ─────────────────────────────────────────────────────────────────────────────
// FlashAttention-1 kernel — single-pass tiled attention, no N×N scratch.
//
// Tile geometry:
//   kBc = 32   K/V tile width (kBc == warpSize: one key per lane per tile)
//   kD  = 64   head dimension (compile-time constant)
//
// Thread/block mapping:
//   Block: (32,)   Grid: (N,)   — one warp per query row i
//
// Lane ownership of output dimensions:
//   Lane tid owns output dims [tid] and [tid+32]  (kD/32 = 2 dims per lane).
//   o_reg[2] holds the running unnormalised output for those 2 dimensions.
//   This keeps the accumulator entirely in registers (no spill) and makes the
//   V-accumulation loop O(1) per lane per j, vs O(kD) in the naïve layout.
//
// Shared memory per block (kBc=32, kD=64):
//   s_Q[kD]      = 256 B   (Q[i] loaded once, shared across all tiles)
//   s_K[kBc][kD] = 8  KB   (K tile, reloaded per tile)
//   s_V[kBc][kD] = 8  KB   (V tile, reloaded per tile)
//   s_S[kBc]     = 128 B   (per-tile attention scores, one per lane)
//   Total ≈ 16.5 KB (well within 48 KB L1 on Ampere/Hopper)
//
// Per-tile algorithm:
//   1. Load K/V tile collaboratively (lane tid loads row tid of each tile).
//   2. Lane tid computes full dot product: S[tid] = dot(s_Q, s_K[tid]) * scale
//      — 64 sequential FMAs per lane; this is unavoidable, O(N·D) total work.
//   3. Warp-reduce for tile max → tile_m; compute m_new = max(m, tile_m).
//   4. Rescale o_reg[0..1] by exp(m - m_new)  (2 ops, not 64).
//   5. For each j in [0, tile_width):
//        e_j = exp(S[j] - m_new);  tile_l += e_j
//        o_reg[0] += e_j * s_V[j][tid];      ← only 2 FMAs per j per lane
//        o_reg[1] += e_j * s_V[j][tid + 32];
//   6. Update l, m.
//
// Final: o_row[tid] = o_reg[0] / l;   o_row[tid+32] = o_reg[1] / l
// ─────────────────────────────────────────────────────────────────────────────

namespace kernelx {

static constexpr int kBc = 32;   // K/V tile size; must equal warpSize
static constexpr int kD  = 64;   // head dimension (compile-time)

__global__ void flash_attn_kernel(
        const float* __restrict__ Q,   // [N, D]
        const float* __restrict__ K,   // [N, D]
        const float* __restrict__ V,   // [N, D]
        float*       __restrict__ O,   // [N, D]
        int N, float scale)
{
    const int i   = blockIdx.x;   // query row
    const int tid = threadIdx.x;  // lane index [0, 32)

    // Shared memory layout: Q row | K tile | V tile | score tile
    __shared__ float s_Q[kD];
    __shared__ float s_K[kBc][kD];
    __shared__ float s_V[kBc][kD];
    __shared__ float s_S[kBc];

    // Load Q[i] collaboratively — kD=64, 2 elements per lane (64/32)
    // Each lane loads its 2 columns: s_Q[tid], s_Q[tid+32]
    s_Q[tid]      = Q[i * kD + tid];
    s_Q[tid + 32] = Q[i * kD + tid + 32];
    __syncwarp();

    // Online state.  o_reg[0] accumulates dim tid; o_reg[1] accumulates dim tid+32.
    float m       = -FLT_MAX;
    float l       = 0.0f;
    float o_reg[2] = {};   // 2 elements per lane, kD/32 = 2 dims owned by each lane

    const int tc = (N + kBc - 1) / kBc;   // number of K/V tiles

    for (int tile = 0; tile < tc; ++tile) {
        const int tile_start = tile * kBc;
        const int tile_width = min(kBc, N - tile_start);  // handle tail

        // ── Load K tile: each lane loads one K row ───────────────────────
        if (tid < tile_width) {
            const float* k_row = K + (tile_start + tid) * kD;
            for (int d = 0; d < kD; ++d)
                s_K[tid][d] = k_row[d];
        }

        // ── Load V tile: each lane loads one V row ───────────────────────
        if (tid < tile_width) {
            const float* v_row = V + (tile_start + tid) * kD;
            for (int d = 0; d < kD; ++d)
                s_V[tid][d] = v_row[d];
        }
        __syncwarp();

        // ── Compute attention scores S[j] = dot(Q[i], K[j]) * scale ─────
        // Lane j computes score for key j within this tile.
        float s_j = 0.0f;
        if (tid < tile_width) {
            for (int d = 0; d < kD; ++d)
                s_j += s_Q[d] * s_K[tid][d];
            s_j *= scale;
        }
        s_S[tid] = (tid < tile_width) ? s_j : -FLT_MAX;
        __syncwarp();

        // ── Tile max via warp reduction ───────────────────────────────────
        float tile_m = s_S[tid];
        for (int offset = 16; offset >= 1; offset >>= 1)
            tile_m = fmaxf(tile_m, __shfl_down_sync(0xFFFFFFFF, tile_m, offset));
        tile_m = __shfl_sync(0xFFFFFFFF, tile_m, 0);

        // ── Rescale running output with new max ───────────────────────────
        const float m_new   = fmaxf(m, tile_m);
        const float rescale = expf(m - m_new);
        o_reg[0] *= rescale;   // dim tid
        o_reg[1] *= rescale;   // dim tid+32

        // ── Accumulate V contribution for this tile ───────────────────────
        // Each lane owns 2 output dimensions: [tid] and [tid+32].
        // The j-loop is sequential (O(kBc) iterations) but each iteration is
        // just 2 FMAs per lane — O(kBc·kD/32) total work per lane per tile.
        // Collectively across all 32 lanes this covers all kD output dims.
        float tile_l = 0.0f;
        for (int j = 0; j < tile_width; ++j) {
            const float e_j = expf(s_S[j] - m_new);
            tile_l  += e_j;
            o_reg[0] += e_j * s_V[j][tid];       // dimension tid
            o_reg[1] += e_j * s_V[j][tid + 32];  // dimension tid+32
        }

        l = l * rescale + tile_l;
        m = m_new;
        __syncwarp();
    }

    // ── Normalise and write output ────────────────────────────────────────
    const float inv_l = 1.0f / l;
    float* o_row = O + static_cast<ptrdiff_t>(i) * kD;
    o_row[tid]      = o_reg[0] * inv_l;   // dim tid     (lane ownership)
    o_row[tid + 32] = o_reg[1] * inv_l;   // dim tid+32
}

void AttentionCudaFlash::setup(const WorkloadConfig& cfg) {
    init_host_data(cfg);
    compute_reference();
    alloc_device();
    upload_inputs();
}

void AttentionCudaFlash::run() {
    // Block: one warp (32 threads). Grid: one block per query row.
    const dim3 block(kBc);
    const dim3 grid(M_);
    flash_attn_kernel<<<grid, block>>>(d_Q_, d_K_, d_V_, d_O_, M_, scale());
    CUDA_CHECK(cudaGetLastError());
    download_output();
}

bool AttentionCudaFlash::verify() { return AttentionWorkload::verify(); }

void AttentionCudaFlash::teardown() {
    free_device();
    h_Q_.clear(); h_K_.clear(); h_V_.clear();
    h_O_ref_.clear(); h_O_out_.clear();
}

} // namespace kernelx
