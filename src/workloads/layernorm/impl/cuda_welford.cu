#include "workloads/layernorm/impl/cuda_welford.hpp"
#include "utils/cuda_utils.hpp"

namespace kernelx {

// ── Welford accumulator merge ─────────────────────────────────────────────────
//
// Combines two partial (count, mean, M2) accumulators into one, using the
// parallel Welford formula (Chan et al., 1979).  Called during the warp-level
// tree reduction to merge results from adjacent lanes.
//
// In-place: (na, ma, m2a) is updated to hold the merged result.

__device__ __forceinline__
void merge_welford(int& na, float& ma, float& m2a,
                   int  nb, float  mb, float  m2b)
{
    const int   n     = na + nb;
    if (n == 0) return;
    const float delta = mb - ma;
    // New mean: weighted average
    ma  = (static_cast<float>(na) * ma + static_cast<float>(nb) * mb)
          / static_cast<float>(n);
    // New M2: sum of local M2 plus the cross-term
    m2a = m2a + m2b + delta * delta
          * (static_cast<float>(na) * static_cast<float>(nb)
             / static_cast<float>(n));
    na  = n;
}

// ── Warp-level Welford reduction ─────────────────────────────────────────────
//
// Reduces 32 partial (count, mean, M2) accumulators across the warp using a
// butterfly pattern.  After the reduction, lane 0 holds the globally merged
// accumulator; mean and variance are then broadcast to all lanes.
//
// count is an int but broadcast as float to avoid a separate __shfl for ints.

__device__ __forceinline__
void warp_reduce_welford(int& count, float& mean, float& m2)
{
    constexpr unsigned FULL_MASK = 0xFFFFFFFFu;

#pragma unroll
    for (int offset = 16; offset >= 1; offset >>= 1) {
        const int   nb  = __shfl_down_sync(FULL_MASK, count, offset);
        const float mb  = __shfl_down_sync(FULL_MASK, mean,  offset);
        const float m2b = __shfl_down_sync(FULL_MASK, m2,    offset);
        merge_welford(count, mean, m2, nb, mb, m2b);
    }
    // Broadcast global mean and total M2 from lane 0 to all lanes
    mean  = __shfl_sync(FULL_MASK, mean,  0);
    m2    = __shfl_sync(FULL_MASK, m2,    0);
    count = __shfl_sync(FULL_MASK, count, 0);
}

// ── Kernel ────────────────────────────────────────────────────────────────────
//
// layernorm_welford_kernel
//
// Block layout: threadIdx.x = lane (0..31), threadIdx.y = warp in block.
// One warp processes one row: row = blockIdx.x * kWarpsPerBlock + threadIdx.y.
//
// Pass 1 — Welford scan:
//   Lane l sweeps elements j = l, l+32, l+64, ...
//   Each lane maintains a local (count, mean, M2) using the sequential Welford
//   update.  After the scan, warp_reduce_welford merges all 32 partial
//   accumulators to give the global mean and total M2 for the row.
//
//   variance = M2 / N    (population variance, consistent with PyTorch's default)
//   inv_std  = rsqrtf(variance + ε)
//
// Pass 2 — normalise + affine transform:
//   Y[row][j] = γ[j] · (X[row][j] − mean) · inv_std + β[j]
//
// Memory access in Pass 2:
//   Lane l writes Y[row][l], Y[row][l+32], ...  — stride-32 within a row.
//   Within a warp: threads access Y[row][0..31] simultaneously → coalesced.

__global__ void layernorm_welford_kernel(
    const float* __restrict__ X,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
          float* __restrict__ Y,
    int M, int N, float epsilon)
{
    const int lane    = threadIdx.x;
    const int warp_id = threadIdx.y;
    const int row     = blockIdx.x * blockDim.y + warp_id;

    if (row >= M) return;

    const float* __restrict__ x_row = X + static_cast<size_t>(row) * N;
    float*                    y_row = Y + static_cast<size_t>(row) * N;

    // ── Pass 1: Welford online mean+variance ──────────────────────────────────
    int   count = 0;
    float mean  = 0.0f;
    float m2    = 0.0f;   // sum of squared deviations from running mean

    for (int j = lane; j < N; j += 32) {
        count++;
        const float x     = x_row[j];
        const float delta = x - mean;
        mean += delta / static_cast<float>(count);
        const float delta2 = x - mean;   // uses updated mean
        m2   += delta * delta2;
    }

    // Merge all 32 local accumulators → global (mean, m2) broadcast to all lanes
    warp_reduce_welford(count, mean, m2);

    // Population variance and reciprocal standard deviation
    const float inv_std = rsqrtf(m2 / static_cast<float>(N) + epsilon);

    // ── Pass 2: normalise + scale + bias ──────────────────────────────────────
    for (int j = lane; j < N; j += 32)
        y_row[j] = gamma[j] * (x_row[j] - mean) * inv_std + beta[j];
}

// ─────────────────────────────────────────────────────────────────────────────

void LayerNormCudaWelford::setup(const WorkloadConfig& cfg) {
    init_host_data(cfg);
    compute_reference();
    alloc_device();
    upload_inputs();
    CUDA_CHECK(cudaMemset(d_Y_, 0, static_cast<size_t>(M_) * N_ * sizeof(float)));
}

void LayerNormCudaWelford::run() {
    const dim3 block(32, kWarpsPerBlock);
    const dim3 grid((M_ + kWarpsPerBlock - 1) / kWarpsPerBlock);
    layernorm_welford_kernel<<<grid, block>>>(
        d_X_, d_gamma_, d_beta_, d_Y_, M_, N_, kEpsilon);
}

bool LayerNormCudaWelford::verify() {
    CUDA_CHECK(cudaDeviceSynchronize());
    download_output();
    return LayerNormWorkload::verify();
}

void LayerNormCudaWelford::teardown() {
    free_device();
    h_X_.clear(); h_gamma_.clear(); h_beta_.clear();
    h_Y_ref_.clear(); h_Y_out_.clear();
}

} // namespace kernelx
