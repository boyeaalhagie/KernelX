#include "workloads/softmax/impl/cuda_online.hpp"
#include "utils/cuda_utils.hpp"

#include <cfloat>

namespace kernelx {

// ── Warp-level reduction helpers ──────────────────────────────────────────────
//
// merge_md: combine two (max, sum-of-exp) pairs produced by the online update.
//
// If thread A has accumulated (m_a, d_a) over its elements, and thread B has
// (m_b, d_b), the merged result over all their elements is:
//   m = max(m_a, m_b)
//   d = d_a * exp(m_a - m) + d_b * exp(m_b - m)
//
// This identity is the core of the Milakov & Gimelshein (2018) algorithm.

__device__ __forceinline__
void merge_md(float& m, float& d, float m_other, float d_other) {
    const float m_new = fmaxf(m, m_other);
    d = d * expf(m - m_new) + d_other * expf(m_other - m_new);
    m = m_new;
}

// warp_reduce_md: tree reduction over all 32 lanes to produce the global
// (max, sum-of-exp) in lane 0, then broadcast to all lanes.

__device__ __forceinline__
void warp_reduce_md(float& m, float& d) {
    constexpr unsigned FULL_MASK = 0xFFFFFFFFu;

    // Butterfly reduction: each step halves the active set
#pragma unroll
    for (int offset = 16; offset >= 1; offset >>= 1) {
        const float m_peer = __shfl_down_sync(FULL_MASK, m, offset);
        const float d_peer = __shfl_down_sync(FULL_MASK, d, offset);
        merge_md(m, d, m_peer, d_peer);
    }

    // Broadcast global (m, d) from lane 0 to all lanes
    m = __shfl_sync(FULL_MASK, m, 0);
    d = __shfl_sync(FULL_MASK, d, 0);
}

// ── Kernel ────────────────────────────────────────────────────────────────────

// softmax_online_kernel
//
// Block layout: (32 threads × kWarpsPerBlock warps)
//   threadIdx.x = lane  (0..31) within the warp
//   threadIdx.y = warp index within the block
//
// Each warp processes one row (row = blockIdx.x * kWarpsPerBlock + threadIdx.y).
//
// Pass 1 — online scan:
//   Lane l sweeps elements j = l, l+32, l+64, ..., accumulating local (m, d)
//   via the online update.  After the scan, warp_reduce_md merges all 32
//   local (m, d) pairs into the global (m_global, d_global).
//
// Pass 2 — normalise:
//   Each lane writes: Y[j] = expf(X[j] − m_global) / d_global.
//   d_global is already the correct denominator because warp_reduce_md
//   produces the sum of exp(x − m_global) over all N elements.

__global__ void softmax_online_kernel(
    const float* __restrict__ X,
          float* __restrict__ Y,
    int M, int N)
{
    const int lane    = threadIdx.x;                               // 0..31
    const int warp_id = threadIdx.y;                               // 0..kWarpsPerBlock-1
    const int row     = blockIdx.x * blockDim.y + warp_id;

    if (row >= M) return;

    const float* __restrict__ x_row = X + static_cast<size_t>(row) * N;
    float*                    y_row = Y + static_cast<size_t>(row) * N;

    // ── Pass 1: online max+sum reduction ─────────────────────────────────────
    float m = -FLT_MAX;   // running maximum
    float d = 0.0f;       // running sum: Σ exp(x − m) relative to current m

    for (int j = lane; j < N; j += 32) {
        const float x  = x_row[j];
        const float m_new = fmaxf(m, x);
        // Online update: adjust existing sum for new max, add new element
        d = d * expf(m - m_new) + expf(x - m_new);
        m = m_new;
    }

    // Merge all 32 local (m, d) → global (m_global, d_global)
    warp_reduce_md(m, d);
    // After reduction: m = global max, d = Σ exp(x_j − m) over all j

    // ── Pass 2: normalise ─────────────────────────────────────────────────────
    for (int j = lane; j < N; j += 32) {
        y_row[j] = expf(x_row[j] - m) / d;
    }
}

// ─────────────────────────────────────────────────────────────────────────────

void SoftmaxCudaOnline::setup(const WorkloadConfig& cfg) {
    init_host_data(cfg);
    compute_reference();
    alloc_device();
    upload_inputs();
    CUDA_CHECK(cudaMemset(d_Y_, 0, static_cast<size_t>(M_) * N_ * sizeof(float)));
}

void SoftmaxCudaOnline::run() {
    const dim3 block(32, kWarpsPerBlock);
    const dim3 grid((M_ + kWarpsPerBlock - 1) / kWarpsPerBlock);
    softmax_online_kernel<<<grid, block>>>(d_X_, d_Y_, M_, N_);
}

bool SoftmaxCudaOnline::verify() {
    CUDA_CHECK(cudaDeviceSynchronize());
    download_output();
    return SoftmaxWorkload::verify();
}

void SoftmaxCudaOnline::teardown() {
    free_device();
    h_X_.clear();
    h_Y_ref_.clear();
    h_Y_out_.clear();
}

} // namespace kernelx
