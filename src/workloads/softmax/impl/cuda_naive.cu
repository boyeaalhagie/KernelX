#include "workloads/softmax/impl/cuda_naive.hpp"
#include "utils/cuda_utils.hpp"

#include <cfloat>

namespace kernelx {

// softmax_naive_kernel
//
// One thread handles one complete row of the input matrix.
//
// Three passes over the row's N elements:
//   Pass 1 — scan for max (numerical stability)
//   Pass 2 — compute exp(x - max), accumulate sum, store intermediate
//   Pass 3 — normalise by dividing each element by the sum
//
// Memory access pattern:
//   Each pass reads/writes X/Y[row * N + 0 .. N-1].
//   Within a warp, 32 consecutive threads handle 32 consecutive rows.
//   Passes 1-2 load X: threads read X[row*N+j] with consecutive rows ×
//   consecutive j → not coalesced across the warp for the same j.
//   This is an inherent limitation of the one-thread-per-row strategy:
//   coalescing would require multiple threads to collaborate on a row.
//   The cuda_online kernel addresses this with warp-level cooperation.
//
// --use_fast_math makes expf map to __expf (hardware approximation, ~2 ULP).

__global__ void softmax_naive_kernel(
    const float* __restrict__ X,
          float* __restrict__ Y,
    int M, int N)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;

    const float* x_row = X + static_cast<size_t>(row) * N;
    float*       y_row = Y + static_cast<size_t>(row) * N;

    // Pass 1: find row maximum for numerical stability
    float max_val = -FLT_MAX;
    for (int j = 0; j < N; ++j)
        max_val = fmaxf(max_val, x_row[j]);

    // Pass 2: compute exp(x - max) and accumulate sum
    float sum = 0.0f;
    for (int j = 0; j < N; ++j) {
        const float e = expf(x_row[j] - max_val);
        y_row[j] = e;
        sum += e;
    }

    // Pass 3: normalise — multiply by reciprocal (one division, N multiplies)
    const float inv_sum = 1.0f / sum;
    for (int j = 0; j < N; ++j)
        y_row[j] *= inv_sum;
}

// ─────────────────────────────────────────────────────────────────────────────

void SoftmaxCudaNaive::setup(const WorkloadConfig& cfg) {
    init_host_data(cfg);
    compute_reference();
    alloc_device();
    upload_inputs();
    CUDA_CHECK(cudaMemset(d_Y_, 0, static_cast<size_t>(M_) * N_ * sizeof(float)));
}

void SoftmaxCudaNaive::run() {
    const dim3 block(kBlockDim);
    const dim3 grid((M_ + kBlockDim - 1) / kBlockDim);
    softmax_naive_kernel<<<grid, block>>>(d_X_, d_Y_, M_, N_);
    // No cudaDeviceSynchronize — BenchmarkRunner uses CUDA events
}

bool SoftmaxCudaNaive::verify() {
    CUDA_CHECK(cudaDeviceSynchronize());
    download_output();
    return SoftmaxWorkload::verify();
}

void SoftmaxCudaNaive::teardown() {
    free_device();
    h_X_.clear();
    h_Y_ref_.clear();
    h_Y_out_.clear();
}

} // namespace kernelx
