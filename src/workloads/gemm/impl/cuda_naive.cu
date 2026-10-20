#include "workloads/gemm/impl/cuda_naive.hpp"
#include "utils/cuda_utils.hpp"

namespace kernelx {

// ─────────────────────────────────────────────────────────────────────────────
// gemm_naive_kernel
//
// One thread computes one element of the output matrix C = A × B.
//
// Thread (row, col) computes:
//   C[row][col] = Σ_{k=0}^{K-1}  A[row * K + k]  *  B[k * N + col]
//
// Memory access analysis (informing Nsight Speed-of-Light interpretation):
//
//   A[row][k]:  row is fixed per thread, k varies in the loop.
//               Within a warp (32 threads with consecutive col, same row):
//               all threads read the *same* A element each iteration → broadcast.
//               Across iterations: stride-1 for fixed row → cache-friendly.
//
//   B[k][col]:  col varies across a warp's threads (consecutive).
//               For fixed k: B[k][col] is stride-1 → fully coalesced load.
//               Across iterations: k increments → row-to-row stride-N → L2 reuse
//               depends on N fitting in cache.
//
// Net effect: B accesses coalesce well; A accesses are broadcast (also efficient).
// The dominant cost is the K-loop re-reading from global/L2 memory with no
// data reuse across threads — arithmetic intensity ≈ 2K / (2K·4) = 0.25 FLOP/B.
// This places the kernel well below the compute ceiling on the roofline model.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void gemm_naive_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
          float* __restrict__ C,
    int M, int N, int K)
{
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        acc += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = acc;
}

// ─────────────────────────────────────────────────────────────────────────────

void GemmCudaNaive::setup(const WorkloadConfig& cfg) {
    init_host_data(cfg);
    compute_reference();   // CPU ground truth for the verification corner
    alloc_device();
    upload_inputs();
    // Zero-initialise d_C_ so stale values don't corrupt partial results
    CUDA_CHECK(cudaMemset(d_C_, 0, static_cast<size_t>(M_) * N_ * sizeof(float)));
}

void GemmCudaNaive::run() {
    // Grid covers the full M × N output; guard condition handles non-multiples
    const dim3 block(kBlockDim, kBlockDim);
    const dim3 grid(
        (N_ + kBlockDim - 1) / kBlockDim,
        (M_ + kBlockDim - 1) / kBlockDim
    );
    gemm_naive_kernel<<<grid, block>>>(d_A_, d_B_, d_C_, M_, N_, K_);
    // No cudaDeviceSynchronize — BenchmarkRunner uses CUDA events to time this
}

bool GemmCudaNaive::verify() {
    // Sync before download to ensure the last timed kernel has completed
    CUDA_CHECK(cudaDeviceSynchronize());
    download_output();
    return GemmWorkload::verify();
}

void GemmCudaNaive::teardown() {
    free_device();
    h_A_.clear();
    h_B_.clear();
    h_C_out_.clear();
    h_C_ref_.clear();
}

} // namespace kernelx
