#include "workloads/attention/impl/cuda_naive.hpp"
#include "utils/cuda_utils.hpp"
#include <cfloat>

// ─────────────────────────────────────────────────────────────────────────────
// AttentionCudaNaive — three-kernel GPU attention, materialises full N×N matrix.
//
// Kernel 1 — qk_kernel:  S[i,j] = dot(Q[i], K[j]) / sqrt(D)
//   Grid: (ceil(N/32), ceil(N/32))   Block: (32, 32)
//   Each thread computes one S[i,j].  Inner loop iterates over D.
//
// Kernel 2 — softmax_kernel:  P[i,:] = softmax(S[i,:])
//   Grid: (ceil(N/32),)   Block: (32, 32)   kWarpsPerBlock=32 warps, 32 threads each
//   One warp per query row; online (m,d) warp reduction via __shfl_down_sync.
//
// Kernel 3 — pv_kernel:  O[i,d] = sum_j P[i,j] * V[j,d]
//   Grid: (ceil(N/32), ceil(D/32))   Block: (32, 32)
//   Each thread accumulates one O[i,d].  Inner loop iterates over N.
// ─────────────────────────────────────────────────────────────────────────────

namespace kernelx {

// ── Kernel 1: QK^T / sqrt(D) ─────────────────────────────────────────────────
__global__ void qk_kernel(const float* __restrict__ Q,
                          const float* __restrict__ K,
                          float*       __restrict__ S,
                          int N, int D, float scale) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;  // query index i
    const int col = blockIdx.x * blockDim.x + threadIdx.x;  // key   index j
    if (row >= N || col >= N) return;

    float dot = 0.0f;
    for (int d = 0; d < D; ++d)
        dot += Q[row * D + d] * K[col * D + d];
    S[row * N + col] = dot * scale;
}

// ── Kernel 2: row-wise softmax with online warp reduction ────────────────────
//   Block: (32, kWarpsPerBlock)  — thread (tid, wid) where wid = warp index
//   Each warp processes one row of S; warp-stride pass then warp-reduce.
static constexpr int kSoftmaxWarpsPerBlock = 4;

__global__ void softmax_kernel(float* __restrict__ S, int N) {
    const int row = blockIdx.x * kSoftmaxWarpsPerBlock + threadIdx.y;
    if (row >= N) return;

    const int tid = threadIdx.x;         // lane within warp [0,32)
    float* s_row = S + static_cast<ptrdiff_t>(row) * N;

    // Pass 1: online (m, d) scan over all N columns
    float m = -FLT_MAX, d = 0.0f;
    for (int j = tid; j < N; j += 32) {
        const float x = s_row[j];
        const float m_new = fmaxf(m, x);
        d = d * expf(m - m_new) + expf(x - m_new);
        m = m_new;
    }

    // Warp-level reduction of (m, d) using butterfly shuffle
    for (int offset = 16; offset >= 1; offset >>= 1) {
        const float m_other = __shfl_down_sync(0xFFFFFFFF, m, offset);
        const float d_other = __shfl_down_sync(0xFFFFFFFF, d, offset);
        const float m_new   = fmaxf(m, m_other);
        d = d * expf(m - m_new) + d_other * expf(m_other - m_new);
        m = m_new;
    }
    // Broadcast final (m, d) from lane 0 to all lanes
    m = __shfl_sync(0xFFFFFFFF, m, 0);
    d = __shfl_sync(0xFFFFFFFF, d, 0);

    // Pass 2: normalise in place
    for (int j = tid; j < N; j += 32)
        s_row[j] = expf(s_row[j] - m) / d;
}

// ── Kernel 3: O = P × V ───────────────────────────────────────────────────────
__global__ void pv_kernel(const float* __restrict__ P,
                          const float* __restrict__ V,
                          float*       __restrict__ O,
                          int N, int D) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;  // query index i
    const int col = blockIdx.x * blockDim.x + threadIdx.x;  // output dim d
    if (row >= N || col >= D) return;

    float acc = 0.0f;
    for (int j = 0; j < N; ++j)
        acc += P[row * N + j] * V[j * D + col];
    O[row * D + col] = acc;
}

// ── Setup / Run / Teardown / Verify ──────────────────────────────────────────
void AttentionCudaNaive::setup(const WorkloadConfig& cfg) {
    init_host_data(cfg);
    compute_reference();
    alloc_device();

    const size_t s_bytes = static_cast<size_t>(M_) * M_ * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_S_, s_bytes));

    upload_inputs();
}

void AttentionCudaNaive::run() {
    const int N = M_;
    const int D = D_;

    // ── Kernel 1: S = QK^T / sqrt(D)
    {
        const dim3 block(32, 32);
        const dim3 grid((N + 31) / 32, (N + 31) / 32);
        qk_kernel<<<grid, block>>>(d_Q_, d_K_, d_S_, N, D, scale());
        CUDA_CHECK(cudaGetLastError());
    }

    // ── Kernel 2: P = softmax(S)
    {
        const dim3 block(32, kSoftmaxWarpsPerBlock);
        const dim3 grid((N + kSoftmaxWarpsPerBlock - 1) / kSoftmaxWarpsPerBlock);
        softmax_kernel<<<grid, block>>>(d_S_, N);
        CUDA_CHECK(cudaGetLastError());
    }

    // ── Kernel 3: O = P × V
    {
        const dim3 block(32, 32);
        const dim3 grid((D + 31) / 32, (N + 31) / 32);
        pv_kernel<<<grid, block>>>(d_S_, d_V_, d_O_, N, D);
        CUDA_CHECK(cudaGetLastError());
    }

    download_output();
}

bool AttentionCudaNaive::verify() { return AttentionWorkload::verify(); }

void AttentionCudaNaive::teardown() {
    if (d_S_) { cudaFree(d_S_); d_S_ = nullptr; }
    free_device();
    h_Q_.clear(); h_K_.clear(); h_V_.clear();
    h_O_ref_.clear(); h_O_out_.clear();
}

} // namespace kernelx
