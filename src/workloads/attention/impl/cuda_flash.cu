#include "workloads/attention/impl/cuda_flash.hpp"
#include "utils/cuda_utils.hpp"
#include <cfloat>

// ─────────────────────────────────────────────────────────────────────────────
// FlashAttention-1 kernel — single-pass tiled attention, no N×N scratch.
//
// Tile geometry:
//   kBc = 32   K/V tile width (kBc == warpSize: lane tid owns key/value row tid)
//   kD  = 64   head dimension (compile-time constant)
//
// Thread/block mapping:
//   Block: (32,)   Grid: (N,)   — one warp per query row i
//
// Lane ownership:
//   Each lane tid owns 2 output dimensions: [tid] and [tid+32].
//   o_reg[2] holds the running unnormalised output for those 2 dims.
//   Accumulation is 2 FMAs per j per lane (not kD=64), eliminating register spill.
//
// Shared memory per block (kBc=32, kD=64):
//   s_Q[kD]            = 256  B  (Q[i] loaded once, reused every tile)
//   s_K[kBc][kD + 1]   = 8.25 KB (K tile; +1 pad eliminates bank conflicts)
//   s_V[kBc][kD + 1]   = 8.25 KB (V tile; +1 pad eliminates bank conflicts)
//   s_S[kBc]           = 128  B  (per-tile attention scores)
//   Total ≈ 17 KB (well within 48 KB L1 on Ampere/Hopper)
//
// Bank conflict analysis (s_K[kBc][kD+1], kD+1=65):
//   s_K[k][d] is at word offset k*65+d.  Bank = (k*65+d)%32 = (k+d)%32.
//   For 32 lanes accessing s_K[tid][d] simultaneously (d fixed):
//     bank(tid) = (tid+d)%32 — all 32 banks hit exactly once.  No conflict.
//
// Coalesced global memory loading (tile loop):
//   Loading kBc*kD = 2048 elements with 32 lanes:
//     for idx = tid, tid+32, ...:   row = idx/kD,  col = idx%kD
//     addr = K[(tile_start + row)*kD + col]
//   At each idx: 32 lanes access consecutive addresses → one 128-byte transaction.
//   Contrast with naive row-by-row: lane tid reads K[tile_start+tid][d] for all d
//   simultaneously → stride-kD = 256 B → 32 independent cache-line fetches per d.
//
// Per-tile algorithm:
//   1. Coalesced load K/V tile → s_K, s_V (each: 64 transactions of 128 B).
//   2. Lane tid computes full dot product: S[tid] = dot(s_Q, s_K[tid]) * scale.
//      — 64 sequential FMAs per lane; shared memory bank-conflict-free via +1 pad.
//   3. Warp-reduce for tile max → tile_m; m_new = max(m, tile_m).
//   4. Rescale o_reg[0..1] by exp(m − m_new)  (2 ops, not kD=64).
//   5. For j in [0, tile_width):
//        e_j = exp(S[j] − m_new);  tile_l += e_j
//        o_reg[0] += e_j * s_V[j][tid];      ← 2 FMAs per lane per j
//        o_reg[1] += e_j * s_V[j][tid + 32];
//   6. l = l*exp(m−m_new) + tile_l;  m = m_new.
//
// Final: o_row[tid] = o_reg[0] / l;   o_row[tid+32] = o_reg[1] / l
// ─────────────────────────────────────────────────────────────────────────────

namespace kernelx {

static constexpr int kBc  = 32;        // K/V tile size; must equal warpSize
static constexpr int kD   = 64;        // head dimension (compile-time)
static constexpr int kDp1 = kD + 1;    // padded row stride to avoid bank conflicts

__global__ void flash_attn_kernel(
        const float* __restrict__ Q,   // [N, kD]
        const float* __restrict__ K,   // [N, kD]
        const float* __restrict__ V,   // [N, kD]
        float*       __restrict__ O,   // [N, kD]
        int N, float scale)
{
    const int i   = blockIdx.x;   // query row
    const int tid = threadIdx.x;  // lane index [0, 32)

    // ── Shared memory ────────────────────────────────────────────────────────
    __shared__ float s_Q[kD];
    __shared__ float s_K[kBc][kDp1];   // +1 pad per row to eliminate bank conflicts
    __shared__ float s_V[kBc][kDp1];
    __shared__ float s_S[kBc];

    // ── Load Q[i] — 64 consecutive floats, 2 loads per lane, coalesced ──────
    s_Q[tid]      = Q[i * kD + tid];
    s_Q[tid + 32] = Q[i * kD + tid + 32];
    __syncwarp();

    // ── Online state ─────────────────────────────────────────────────────────
    float m        = -FLT_MAX;
    float l        = 0.0f;
    float o_reg[2] = {};   // lane tid owns dims [tid, tid+32]

    const int tc = (N + kBc - 1) / kBc;

    for (int tile = 0; tile < tc; ++tile) {
        const int tile_start = tile * kBc;
        const int tile_width = min(kBc, N - tile_start);

        // ── Coalesced load of K tile into s_K ────────────────────────────
        // idx ranges over kBc*kD=2048 elements in strides of 32.
        // row=idx/kD, col=idx%kD — consecutive lanes, consecutive addresses.
        for (int idx = tid; idx < kBc * kD; idx += 32) {
            const int row = idx / kD;
            const int col = idx % kD;
            const int src_row = tile_start + row;
            s_K[row][col] = (src_row < N) ? K[src_row * kD + col] : 0.0f;
        }

        // ── Coalesced load of V tile into s_V ────────────────────────────
        for (int idx = tid; idx < kBc * kD; idx += 32) {
            const int row = idx / kD;
            const int col = idx % kD;
            const int src_row = tile_start + row;
            s_V[row][col] = (src_row < N) ? V[src_row * kD + col] : 0.0f;
        }
        __syncwarp();

        // ── Score: S[tid] = dot(s_Q, s_K[tid]) * scale ───────────────────
        // s_K[tid][d]: bank = (tid+d)%32 — no conflict (see header).
        float s_j = 0.0f;
        if (tid < tile_width) {
            for (int d = 0; d < kD; ++d)
                s_j += s_Q[d] * s_K[tid][d];
            s_j *= scale;
        }
        s_S[tid] = (tid < tile_width) ? s_j : -FLT_MAX;
        __syncwarp();

        // ── Tile max via warp butterfly reduction ─────────────────────────
        float tile_m = s_S[tid];
        for (int offset = 16; offset >= 1; offset >>= 1)
            tile_m = fmaxf(tile_m, __shfl_down_sync(0xFFFFFFFF, tile_m, offset));
        tile_m = __shfl_sync(0xFFFFFFFF, tile_m, 0);

        // ── Rescale running output with new max ───────────────────────────
        const float m_new   = fmaxf(m, tile_m);
        const float rescale = expf(m - m_new);
        o_reg[0] *= rescale;   // dim tid
        o_reg[1] *= rescale;   // dim tid+32

        // ── Accumulate V contribution ─────────────────────────────────────
        // s_V[j][tid]:    bank = (j+tid)%32   — no conflict.
        // s_V[j][tid+32]: bank = (j+tid)%32   — same bank but sequential instruction.
        float tile_l = 0.0f;
        for (int j = 0; j < tile_width; ++j) {
            const float e_j = expf(s_S[j] - m_new);
            tile_l  += e_j;
            o_reg[0] += e_j * s_V[j][tid];
            o_reg[1] += e_j * s_V[j][tid + 32];
        }

        l = l * rescale + tile_l;
        m = m_new;
        __syncwarp();
    }

    // ── Normalise and write output ────────────────────────────────────────────
    const float inv_l = 1.0f / l;
    float* o_row = O + static_cast<ptrdiff_t>(i) * kD;
    o_row[tid]      = o_reg[0] * inv_l;
    o_row[tid + 32] = o_reg[1] * inv_l;
}

void AttentionCudaFlash::setup(const WorkloadConfig& cfg) {
    init_host_data(cfg);
    compute_reference();
    alloc_device();
    upload_inputs();
}

void AttentionCudaFlash::run() {
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
