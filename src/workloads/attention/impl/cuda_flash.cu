#include "workloads/attention/impl/cuda_flash.hpp"
#include "utils/cuda_utils.hpp"
#include <cfloat>

// ─────────────────────────────────────────────────────────────────────────────
// FlashAttention-1 kernel — single-pass tiled attention, no N×N scratch.
//
// Tile geometry:
//   kBc = 32   K/V tile width (one column per lane; one full tile per warp)
//   kD  = 64   head dimension (compile-time constant for register arrays)
//
// Thread/block mapping:
//   Block: (32,)       — one warp per block
//   Grid:  (N,)        — one block (warp) per query row i
//
// Shared memory per block (kBc=32, kD=64):
//   s_K[kBc][kD] = 32×64×4 = 8  KB
//   s_V[kBc][kD] = 32×64×4 = 8  KB
//   s_S[kBc]     = 32×4    = 128 B
//   s_Q[kD]      = 64×4    = 256 B
//   Total ≈ 16.5 KB   (well within 48 KB L1 on Ampere/Hopper)
//
// Online state (registers, per query row):
//   m          = -FLT_MAX   (running max)
//   l          =  0.0f      (running sum = partition function)
//   o_reg[kD]  =  0.0f      (running output accumulator, unnormalised)
//
// Tile loop (tc = tile count = ceil(N/kBc)):
//   1. Collaboratively load K[tc*kBc:(tc+1)*kBc, :] → s_K  (tid as lane)
//   2. Collaboratively load V[tc*kBc:(tc+1)*kBc, :] → s_V
//   3. __syncwarp()
//   4. Compute S[j] = dot(Q[i], s_K[j]) * scale,  j in [0, tile_width)
//      (each lane j computes one score; tid==j for exact alignment)
//   5. __syncwarp() — ensure s_S broadcast-ready
//   6. tile_m = max(S[0..tile_width))  via warp shfl reduction
//   7. m_new = max(m, tile_m)
//   8. Rescale o_reg: o_reg[d] *= exp(m - m_new)
//   9. For each j in tile: o_reg[d] += exp(S[j] - m_new) * s_V[j][d]
//  10. tile_l = sum(exp(S[j] - m_new))  via warp shfl reduction
//  11. l = l * exp(m - m_new) + tile_l
//  12. m = m_new
//
// Final: O[i,d] = o_reg[d] / l
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

    // Online state
    float m = -FLT_MAX;
    float l = 0.0f;
    float o_reg[kD] = {};   // zero-initialised; kD is compile-time constant

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
        const float m_new    = fmaxf(m, tile_m);
        const float rescale  = expf(m - m_new);
        for (int d = 0; d < kD; ++d)
            o_reg[d] *= rescale;

        // ── Accumulate: for each key j, add exp(S[j]-m_new) * V[j] ──────
        // All lanes cooperate: lane j "owns" row j of s_V; we iterate over j
        // using the full warp to cover all kD output dimensions.
        // Each lane broadcasts its exp(S[j]-m_new) and V[j][d] contribution.
        float tile_l = 0.0f;
        for (int j = 0; j < tile_width; ++j) {
            const float e_j = expf(s_S[j] - m_new);
            tile_l += e_j;
            for (int d = 0; d < kD; ++d)
                o_reg[d] += e_j * s_V[j][d];
        }

        l = l * rescale + tile_l;
        m = m_new;
        __syncwarp();
    }

    // ── Normalise and write output ────────────────────────────────────────
    const float inv_l = 1.0f / l;
    float* o_row = O + static_cast<ptrdiff_t>(i) * kD;
    // Each lane writes kD/32 = 2 output elements
    for (int d = tid; d < kD; d += 32)
        o_row[d] = o_reg[d] * inv_l;
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
