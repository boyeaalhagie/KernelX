#include "workloads/attention/impl/cuda_flash.hpp"
#include "utils/cuda_utils.hpp"
#include <cfloat>

// ─────────────────────────────────────────────────────────────────────────────
// FlashAttention-1 kernel — single-pass tiled attention, no N×N scratch.
//
// Block structure: (kBc=32, kBr=32) = 1024 threads / 32 warps.
//   Warp wid (threadIdx.y) processes query row (block_start + wid).
//   Lane lid (threadIdx.x) owns key column lid within each tile.
// Grid: (ceil(N/kBr),) blocks.
//
// Memory traffic analysis (compare at N=4096, D=64):
//   This kernel (kBr=kBc=32):
//     K reads = (N/kBr)·(N/kBc)·kBc·D·4 = N²·D·4/kBr = 128 MB  (D=64, kBr=32)
//     V reads = 128 MB.  Q reads = O writes = 1 MB each.  Total ≈ 258 MB.
//   cuda_naive (3-kernel, materialises S):
//     write S + 2× read S + write P = 4·N²·4 = 256 MB at N=4096.  Total ≈ 260 MB.
//   → Roughly equal traffic at D=64, kBr=32 (break-even: D/kBr = 2).
//   → Flash's advantage: (a) no O(N²) VRAM allocation — naive OOMs at N≈16K on
//     80GB H100; (b) one kernel launch vs three; (c) Q tile stays warm in L2.
//
// Shared memory per block:
//   s_Q[kBr][kD]      = 8    KB  (Q tile, loaded once)
//   s_K[kBc][kDp1]    = 8.125 KB (K tile, +1 pad — bank conflict avoidance)
//   s_V[kBc][kDp1]    = 8.125 KB (V tile, +1 pad)
//   s_S[kBr][kBc]     = 4    KB  (scores for all (query, key) pairs in tile)
//   Total ≈ 28.25 KB (fits in 48 KB L1 on Ampere/Hopper).
//
// Bank conflict analysis:
//   s_K[kBc][kDp1]: s_K[k][d] at word k·(kD+1)+d.  Bank = (k+d)%32 for 32 lanes
//     simultaneously accessing s_K[lid][d] → all 32 banks distinct, zero conflict.
//   s_V[kBc][kDp1]: s_V[j][lid] for 32 lanes → bank = (j+lid)%32, no conflict.
//   s_S[kBr][kBc]: s_S[wid][lid] write → bank = lid%32, no conflict within warp.
//     s_S[wid][j] accumulation read → broadcast (all lanes same address), no conflict.
//   s_Q[kBr][kD]: s_Q[wid][d] read → broadcast within warp, no conflict.
//
// Coalesced global loads:
//   Tile loads index by flat idx = tid_flat, tid_flat+1024, ...;
//   row = idx/kD, col = idx%kD → 32 consecutive lanes, consecutive addresses.
//
// Per-tile algorithm:
//   1. Load K and V tiles collaboratively (all 1024 threads, coalesced).
//   2. __syncthreads()
//   3. Lane lid computes s_S[wid][lid] = dot(s_Q[wid], s_K[lid]) * scale (64 FMAs).
//   4. __syncwarp()
//   5. Warp-reduce tile max → tile_m; m_new = max(m, tile_m).
//   6. Rescale o_reg[0..1] by exp(m − m_new)  (2 FMAs per lane).
//   7. For j in [0, tile_width): accumulate exp(s_S[wid][j] − m_new)·s_V[j][lid/lid+32].
//   8. Update l, m.  __syncthreads() before next tile.
//
// Final: o_row[qi][lid] = o_reg[0]/l, o_row[qi][lid+32] = o_reg[1]/l.
// ─────────────────────────────────────────────────────────────────────────────

namespace kernelx {

static constexpr int kBr  = 32;        // query rows per block (== warps per block)
static constexpr int kBc  = 32;        // K/V tile width (== warpSize)
static constexpr int kD   = 64;        // head dimension
static constexpr int kDp1 = kD + 1;    // padded row stride to eliminate bank conflicts

// Block: (kBc, kBr) = (32, 32) = 1024 threads
// Grid:  (ceil(N/kBr),)
__global__ void flash_attn_kernel(
        const float* __restrict__ Q,   // [N, kD]
        const float* __restrict__ K,   // [N, kD]
        const float* __restrict__ V,   // [N, kD]
        float*       __restrict__ O,   // [N, kD]
        int N, float scale)
{
    const int lid         = threadIdx.x;         // lane  [0, kBc)
    const int wid         = threadIdx.y;         // warp  [0, kBr)
    const int block_start = blockIdx.x * kBr;   // first query row in this block
    const int qi          = block_start + wid;   // global query row for this warp

    __shared__ float s_Q[kBr][kD];
    __shared__ float s_K[kBc][kDp1];   // +1 pad per row: bank = (row+col)%32
    __shared__ float s_V[kBc][kDp1];
    __shared__ float s_S[kBr][kBc];

    // ── Load Q tile (kBr×kD = 2048 elements, 1024 threads → 2 loads each) ──
    // Coalesced: flat_idx = wid*kBc+lid; consecutive threads → consecutive addrs.
    const int tid_flat = wid * kBc + lid;
    for (int idx = tid_flat; idx < kBr * kD; idx += kBr * kBc) {
        const int row = idx / kD;
        const int col = idx % kD;
        const int src = block_start + row;
        s_Q[row][col] = (src < N) ? Q[src * kD + col] : 0.0f;
    }
    __syncthreads();

    // ── Online state (registers, per warp) ────────────────────────────────────
    float m        = -FLT_MAX;
    float l        = 0.0f;
    float o_reg[2] = {};   // lane lid owns output dims [lid] and [lid+32]

    const int tc = (N + kBc - 1) / kBc;

    for (int tile = 0; tile < tc; ++tile) {
        const int tile_start = tile * kBc;
        const int tile_width = min(kBc, N - tile_start);

        // ── Coalesced load K and V tiles ──────────────────────────────────
        // kBc×kD = 2048 elements, 1024 threads → 2 elements per thread.
        // row = idx/kD, col = idx%kD: consecutive flat_idx → consecutive addrs.
        for (int idx = tid_flat; idx < kBc * kD; idx += kBr * kBc) {
            const int row = idx / kD;
            const int col = idx % kD;
            const int src = tile_start + row;
            s_K[row][col] = (src < N) ? K[src * kD + col] : 0.0f;
            s_V[row][col] = (src < N) ? V[src * kD + col] : 0.0f;
        }
        __syncthreads();

        // ── Score: s_S[wid][lid] = dot(s_Q[wid], s_K[lid]) * scale ──────
        // All 32 warps execute simultaneously; each computes one row of s_S.
        // s_K[lid][d]: bank = (lid+d)%32 (padded layout) — no conflict.
        // s_Q[wid][d]: broadcast within warp — no conflict.
        float s_j = 0.0f;
        if (lid < tile_width) {
            for (int d = 0; d < kD; ++d)
                s_j += s_Q[wid][d] * s_K[lid][d];
            s_j *= scale;
        }
        s_S[wid][lid] = (lid < tile_width) ? s_j : -FLT_MAX;
        __syncwarp();

        // ── Tile max via warp butterfly ───────────────────────────────────
        float tile_m = s_S[wid][lid];
        for (int offset = 16; offset >= 1; offset >>= 1)
            tile_m = fmaxf(tile_m, __shfl_down_sync(0xFFFFFFFF, tile_m, offset));
        tile_m = __shfl_sync(0xFFFFFFFF, tile_m, 0);

        // ── Rescale running output with new max ───────────────────────────
        const float m_new   = fmaxf(m, tile_m);
        const float rescale = expf(m - m_new);
        o_reg[0] *= rescale;   // dim lid
        o_reg[1] *= rescale;   // dim lid+32

        // ── Accumulate V contribution ─────────────────────────────────────
        // s_S[wid][j]: broadcast per j iteration — no conflict.
        // s_V[j][lid]:    bank = (j+lid)%32   — no conflict.
        // s_V[j][lid+32]: bank = (j+lid)%32   — same, sequential instr, OK.
        float tile_l = 0.0f;
        for (int j = 0; j < tile_width; ++j) {
            const float e_j = expf(s_S[wid][j] - m_new);
            tile_l  += e_j;
            o_reg[0] += e_j * s_V[j][lid];
            o_reg[1] += e_j * s_V[j][lid + 32];
        }

        l = l * rescale + tile_l;
        m = m_new;
        __syncthreads();   // ensure all warps finish before next tile overwrites s_K/s_V
    }

    // ── Normalise and write output ─────────────────────────────────────────────
    if (qi < N) {
        const float inv_l = 1.0f / l;
        float* o_row = O + static_cast<ptrdiff_t>(qi) * kD;
        o_row[lid]      = o_reg[0] * inv_l;
        o_row[lid + 32] = o_reg[1] * inv_l;
    }
}

void AttentionCudaFlash::setup(const WorkloadConfig& cfg) {
    init_host_data(cfg);
    compute_reference();
    alloc_device();
    upload_inputs();
}

void AttentionCudaFlash::run() {
    // Block: (kBc, kBr) = (32, 32) = 1024 threads (32 warps, one per query row).
    // Grid:  (ceil(N/kBr),) blocks.
    const dim3 block(kBc, kBr);
    const dim3 grid((M_ + kBr - 1) / kBr);
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
