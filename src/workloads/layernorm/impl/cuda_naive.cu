#include "workloads/layernorm/impl/cuda_naive.hpp"
#include "utils/cuda_utils.hpp"

namespace kernelx {

// layernorm_naive_kernel
//
// One thread per row.  Three passes over X[row][0..N-1].
//
// Pass 1: mean
//   sum = Σ_j X[row][j];  μ = sum / N
//
// Pass 2: variance (population, not sample)
//   var = Σ_j (X[row][j] − μ)² / N
//   inv_std = rsqrtf(var + ε)
//
// Pass 3: normalise + affine transform
//   Y[row][j] = γ[j] · (X[row][j] − μ) · inv_std + β[j]
//
// Memory access notes:
//   Within a warp, 32 threads each handle a different row.  For a given
//   column j, they load X[row0][j], X[row1][j], ..., X[row31][j] — these
//   are stride-N apart, so consecutive threads do NOT load consecutive
//   addresses.  This is the coalescing failure that motivates cuda_welford.

__global__ void layernorm_naive_kernel(
    const float* __restrict__ X,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
          float* __restrict__ Y,
    int M, int N, float epsilon)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;

    const float* x_row = X + static_cast<size_t>(row) * N;
    float*       y_row = Y + static_cast<size_t>(row) * N;

    // Pass 1: mean
    float sum = 0.0f;
    for (int j = 0; j < N; ++j) sum += x_row[j];
    const float mean = sum / static_cast<float>(N);

    // Pass 2: variance
    float var = 0.0f;
    for (int j = 0; j < N; ++j) {
        const float d = x_row[j] - mean;
        var += d * d;
    }
    const float inv_std = rsqrtf(var / static_cast<float>(N) + epsilon);

    // Pass 3: normalise + scale + bias
    for (int j = 0; j < N; ++j)
        y_row[j] = gamma[j] * (x_row[j] - mean) * inv_std + beta[j];
}

// ─────────────────────────────────────────────────────────────────────────────

void LayerNormCudaNaive::setup(const WorkloadConfig& cfg) {
    init_host_data(cfg);
    compute_reference();
    alloc_device();
    upload_inputs();
    CUDA_CHECK(cudaMemset(d_Y_, 0, static_cast<size_t>(M_) * N_ * sizeof(float)));
}

void LayerNormCudaNaive::run() {
    const dim3 block(kBlockDim);
    const dim3 grid((M_ + kBlockDim - 1) / kBlockDim);
    layernorm_naive_kernel<<<grid, block>>>(d_X_, d_gamma_, d_beta_, d_Y_, M_, N_, kEpsilon);
}

bool LayerNormCudaNaive::verify() {
    CUDA_CHECK(cudaDeviceSynchronize());
    download_output();
    return LayerNormWorkload::verify();
}

void LayerNormCudaNaive::teardown() {
    free_device();
    h_X_.clear(); h_gamma_.clear(); h_beta_.clear();
    h_Y_ref_.clear(); h_Y_out_.clear();
}

} // namespace kernelx
