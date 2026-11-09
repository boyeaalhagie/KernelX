#include "workloads/reduction/impl/cuda_warp.hpp"
#include "utils/cuda_utils.hpp"

// ─────────────────────────────────────────────────────────────────────────────
// Row-wise sum reduction — warp-shuffle binary tree.
//
// Replaces intra-warp __syncthreads() with __shfl_down_sync(), reducing barrier
// count from 8 (naive, 256-thread tree) to 1 (one inter-warp sync).
//
// Three phases:
//  1. Register accumulation  — same coalesced stride loop as naive.
//  2. Intra-warp butterfly   — 5 __shfl_down_sync steps, no barrier.
//  3. Cross-warp reduction   — 1 __syncthreads() + warp-0 butterfly.
//
// kBlockDim = 256 → kWarps = 8 → s_partial[8] = 32 bytes shared mem per block.
//
// Grid/block: grid=(M,), block=(kBlockDim=256) — one CTA per row.
// ─────────────────────────────────────────────────────────────────────────────

namespace kernelx {

static constexpr int kBlockDim = 256;
static constexpr int kWarpSize = 32;
static constexpr int kWarps    = kBlockDim / kWarpSize;   // 8

// Reduce 32 values within one warp using butterfly shuffle.
// Returns the full warp sum in every lane.
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = kWarpSize / 2; offset >= 1; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;   // meaningful only in lane 0 after the loop
}

__global__ void reduce_warp_kernel(
        const float* __restrict__ X,   // [M × N]
        float*       __restrict__ Y,   // [M]
        int N)
{
    const int tid     = threadIdx.x;
    const int lane_id = tid % kWarpSize;    // [0, 32)
    const int warp_id = tid / kWarpSize;    // [0, kWarps)
    const int row     = blockIdx.x;

    const float* x_row = X + static_cast<ptrdiff_t>(row) * N;

    // s_partial holds one partial sum per warp (after phase 2).
    __shared__ float s_partial[kWarps];

    // ── Phase 1: thread-private accumulation ─────────────────────────────────
    // Coalesced strided load — identical access pattern to cuda_naive.
    // Consecutive threads read consecutive addresses (no stride gap in cache).
    float acc = 0.0f;
    for (int j = tid; j < N; j += kBlockDim)
        acc += x_row[j];

    // ── Phase 2: intra-warp butterfly (no __syncthreads) ─────────────────────
    acc = warp_reduce_sum(acc);
    // After this, lane 0 of each warp holds the warp's partial sum.

    // ── Phase 3: inter-warp reduction ────────────────────────────────────────
    if (lane_id == 0)
        s_partial[warp_id] = acc;
    __syncthreads();   // exactly one barrier — make all partial sums visible

    // Warp 0 reduces the kWarps partial sums with another butterfly.
    if (warp_id == 0) {
        // Load: thread tid (= lane) reads s_partial[tid] if tid < kWarps, else 0
        acc = (lane_id < kWarps) ? s_partial[lane_id] : 0.0f;
        acc = warp_reduce_sum(acc);
    }

    // ── Phase 4: write result ─────────────────────────────────────────────────
    if (tid == 0)
        Y[row] = acc;
}

void ReductionCudaWarp::setup(const WorkloadConfig& cfg) {
    init_host_data(cfg);
    compute_reference();
    alloc_device();
    upload_inputs();
}

void ReductionCudaWarp::run() {
    const dim3 grid(M_);
    const dim3 block(kBlockDim);
    reduce_warp_kernel<<<grid, block>>>(d_X_, d_Y_, N_);
    CUDA_CHECK(cudaGetLastError());
    download_output();
}

bool ReductionCudaWarp::verify() { return ReductionWorkload::verify(); }

void ReductionCudaWarp::teardown() {
    free_device();
    h_X_.clear();
    h_Y_ref_.clear();
    h_Y_out_.clear();
}

} // namespace kernelx
