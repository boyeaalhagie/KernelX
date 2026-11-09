#include "workloads/reduction/impl/cuda_naive.hpp"
#include "utils/cuda_utils.hpp"

// ─────────────────────────────────────────────────────────────────────────────
// Row-wise sum reduction — shared-memory binary tree.
//
// Design follows NVIDIA "Optimizing Parallel Reduction" §4:
//   • No idle threads on first step: each thread accumulates its share of the
//     row in a register before entering the tree phase.
//   • Sequential addressing in the tree (not interleaved) — avoids shared
//     memory bank conflicts in the first tree step.
//   • One __syncthreads() per tree level (kBlockDim=256 → 8 barriers).
//
// Grid/block: grid=(M,), block=(kBlockDim=256)
//   One CTA per row; blockIdx.x selects the row.
//
// Shared memory: s_sum[kBlockDim] = 1 KB per block.
// ─────────────────────────────────────────────────────────────────────────────

namespace kernelx {

static constexpr int kBlockDim = 256;

__global__ void reduce_naive_kernel(
        const float* __restrict__ X,   // [M × N]
        float*       __restrict__ Y,   // [M]
        int N)
{
    const int tid = threadIdx.x;
    const int row = blockIdx.x;

    const float* x_row = X + static_cast<ptrdiff_t>(row) * N;

    __shared__ float s_sum[kBlockDim];

    // ── Phase 1: thread-private accumulation ─────────────────────────────────
    // Each thread strides across its share of the row and accumulates in a
    // register — zero shared-memory traffic until the tree phase.
    float acc = 0.0f;
    for (int j = tid; j < N; j += kBlockDim)
        acc += x_row[j];
    s_sum[tid] = acc;
    __syncthreads();

    // ── Phase 2: binary tree in shared memory ─────────────────────────────────
    // Sequential addressing: active thread tid reads s_sum[tid] and
    // s_sum[tid + s], which are always in different banks → no bank conflict.
    for (int s = kBlockDim / 2; s > 0; s >>= 1) {
        if (tid < s)
            s_sum[tid] += s_sum[tid + s];
        __syncthreads();
    }

    // ── Phase 3: write result ─────────────────────────────────────────────────
    if (tid == 0)
        Y[row] = s_sum[0];
}

void ReductionCudaNaive::setup(const WorkloadConfig& cfg) {
    init_host_data(cfg);
    compute_reference();
    alloc_device();
    upload_inputs();
}

void ReductionCudaNaive::run() {
    const dim3 grid(M_);
    const dim3 block(kBlockDim);
    reduce_naive_kernel<<<grid, block>>>(d_X_, d_Y_, N_);
    CUDA_CHECK(cudaGetLastError());
    download_output();
}

bool ReductionCudaNaive::verify() { return ReductionWorkload::verify(); }

void ReductionCudaNaive::teardown() {
    free_device();
    h_X_.clear();
    h_Y_ref_.clear();
    h_Y_out_.clear();
}

} // namespace kernelx
