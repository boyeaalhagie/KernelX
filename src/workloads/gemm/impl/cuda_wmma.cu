#include "workloads/gemm/impl/cuda_wmma.hpp"
#include "utils/cuda_utils.hpp"

#include <mma.h>
#include <stdexcept>

using namespace nvcuda;

namespace kernelx {

// ─────────────────────────────────────────────────────────────────────────────
// fp32_to_fp16_kernel — bulk FP32 → FP16 conversion on device.
//
// Called once during setup() to produce the fp16 A and B buffers.
// Grid-stride loop handles arbitrary sizes without occupancy constraints.
// ─────────────────────────────────────────────────────────────────────────────
__global__ static void fp32_to_fp16_kernel(
    const float* __restrict__ src,
    half*        __restrict__ dst,
    int n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
             i < n;
             i += blockDim.x * gridDim.x)
    {
        dst[i] = __float2half(src[i]);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// gemm_wmma_kernel<BM, BN, BK, WarpTM, WarpTN>
//
// WMMA Tensor Core GEMM.  Each thread block computes a BM×BN tile of C.
// Within the block, 16 warps are arranged in a (WarpsM × WarpsN) grid;
// each warp owns a (WarpTM × WarpTN) array of 16×16 WMMA fragments and
// accumulates them in FP32.
//
// Algorithm per block:
//   For each K tile of depth BK:
//     1. All 512 threads collaboratively load sA[BM][BK] and sB[BK][BN]
//        from global memory A and B in a coalesced pattern.
//     2. __syncthreads()
//     3. Each warp iterates over its WarpTM × WarpTN output fragments.
//        For each (wm_f, wn_f) in [0,WarpTM) × [0,WarpTN):
//          a. load_matrix_sync(a_frag, sA + ..., BK)
//          b. load_matrix_sync(b_frag, sB + ..., BN)
//          c. mma_sync(acc[wm_f][wn_f], a_frag, b_frag, acc[wm_f][wn_f])
//     4. __syncthreads()  (before next tile overwrites sA/sB)
//   After all K tiles:
//   For each (wm_f, wn_f): store_matrix_sync(C + ..., acc[wm_f][wn_f], N)
//
// Shared memory:
//   sA: half[BM][BK]  (128×16 × 2 bytes = 4 KB)
//   sB: half[BK][BN]  (16×128 × 2 bytes = 4 KB)
//   Total: 8 KB/block
//
// Constraints:
//   • M, N, K must be multiples of 16  (WMMA fragment size)
//   • BM, BN must be multiples of WarpTM*16 and WarpTN*16 respectively
//   • All threads in a warp must execute mma_sync — no early-exit inside
// ─────────────────────────────────────────────────────────────────────────────
template <int BM, int BN, int BK, int WarpTM, int WarpTN>
__global__ void gemm_wmma_kernel(
    const half*  __restrict__ A,    // [M × K], fp16, row-major
    const half*  __restrict__ B,    // [K × N], fp16, row-major
          float* __restrict__ C,    // [M × N], fp32, row-major
    int M, int N, int K)
{
    static_assert(BM  % 16 == 0, "BM must be divisible by 16 (WMMA fragment height)");
    static_assert(BN  % 16 == 0, "BN must be divisible by 16 (WMMA fragment width)");
    static_assert(BK  % 16 == 0, "BK must be divisible by 16 (WMMA fragment depth)");

    constexpr int WARPS_N = BN / (WarpTN * 16);  // warps along N within block

    // ── Warp identity ─────────────────────────────────────────────────────────
    const int warp_id = threadIdx.x / 32;
    const int warp_m  = warp_id / WARPS_N;   // warp row within block
    const int warp_n  = warp_id % WARPS_N;   // warp col within block

    // ── Block tile origin in C ────────────────────────────────────────────────
    const int block_m = blockIdx.y * BM;
    const int block_n = blockIdx.x * BN;

    // ── Shared memory tiles ───────────────────────────────────────────────────
    __shared__ half sA[BM][BK];
    __shared__ half sB[BK][BN];

    // ── Per-warp accumulator fragments ───────────────────────────────────────
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[WarpTM][WarpTN];
    #pragma unroll
    for (int wm = 0; wm < WarpTM; ++wm)
        #pragma unroll
        for (int wn = 0; wn < WarpTN; ++wn)
            wmma::fill_fragment(acc[wm][wn], 0.0f);

    const int tid     = threadIdx.x;
    const int nthreads = blockDim.x;   // BM/WarpTM/16 * BN/WarpTN/16 * 32 * WarpTM*WarpTN

    // ── K-tile loop ───────────────────────────────────────────────────────────
    const int num_tiles = K / BK;   // M, N, K guaranteed multiples of 16; BK=16

    for (int kt = 0; kt < num_tiles; ++kt) {

        // ── Load sA: A[block_m : block_m+BM][kt*BK : kt*BK+BK] ──────────────
        // Linearised index: row = i / BK,  col = i % BK
        // Global address: A[(block_m + row) * K + (kt*BK + col)]
        //
        // Coalescing: consecutive tids access consecutive fp16 elements along
        // the K dimension (cols) → stride-1 access within each row.  With
        // BK=16 fp16 = 32 bytes, one warp covers 64 consecutive bytes → two
        // 32-byte memory transactions, good L2 utilisation.
        for (int i = tid; i < BM * BK; i += nthreads) {
            const int row = i / BK;
            const int col = i % BK;
            sA[row][col] = A[(block_m + row) * K + (kt * BK + col)];
        }

        // ── Load sB: B[kt*BK : kt*BK+BK][block_n : block_n+BN] ──────────────
        for (int i = tid; i < BK * BN; i += nthreads) {
            const int row = i / BN;
            const int col = i % BN;
            sB[row][col] = B[(kt * BK + row) * N + (block_n + col)];
        }

        __syncthreads();

        // ── WMMA accumulation ─────────────────────────────────────────────────
        // For BK=16 there is exactly one WMMA K-fragment per tile.
        // For larger BK this inner loop would iterate BK/16 times.
        for (int k_frag = 0; k_frag < BK; k_frag += 16) {

            #pragma unroll
            for (int wm = 0; wm < WarpTM; ++wm) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;

                // A sub-fragment: sA[warp_m*WarpTM*16 + wm*16][k_frag]
                // Stride = BK (number of fp16 elements per row of sA)
                const int sA_row = warp_m * WarpTM * 16 + wm * 16;
                wmma::load_matrix_sync(a_frag, &sA[sA_row][k_frag], BK);

                #pragma unroll
                for (int wn = 0; wn < WarpTN; ++wn) {
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;

                    // B sub-fragment: sB[k_frag][warp_n*WarpTN*16 + wn*16]
                    // Stride = BN (number of fp16 elements per row of sB)
                    const int sB_col = warp_n * WarpTN * 16 + wn * 16;
                    wmma::load_matrix_sync(b_frag, &sB[k_frag][sB_col], BN);

                    wmma::mma_sync(acc[wm][wn], a_frag, b_frag, acc[wm][wn]);
                }
            }
        }

        __syncthreads();
    }

    // ── Store output tiles to C (fp32) ────────────────────────────────────────
    // M, N guaranteed multiples of 16 → no boundary check needed.
    #pragma unroll
    for (int wm = 0; wm < WarpTM; ++wm) {
        #pragma unroll
        for (int wn = 0; wn < WarpTN; ++wn) {
            const int out_m = block_m + warp_m * WarpTM * 16 + wm * 16;
            const int out_n = block_n + warp_n * WarpTN * 16 + wn * 16;
            wmma::store_matrix_sync(
                &C[out_m * N + out_n],
                acc[wm][wn],
                N,
                wmma::mem_row_major);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────

void GemmCudaWmma::setup(const WorkloadConfig& cfg) {
    // WMMA fragment dimensions require all problem dims to be multiples of 16.
    if (cfg.m % 16 != 0 || cfg.n % 16 != 0 || cfg.k % 16 != 0) {
        throw std::invalid_argument(
            "cuda_wmma requires M, N, K to be multiples of 16 "
            "(WMMA fragment dimension)");
    }

    // Base class: allocate fp32 host data, reference, and fp32 device buffers.
    // d_A_, d_B_ are fp32 — used only as an upload staging area.
    // d_C_ is fp32 — WMMA accumulates into fp32 and writes here directly.
    init_host_data(cfg);
    compute_reference();
    alloc_device();
    upload_inputs();
    CUDA_CHECK(cudaMemset(d_C_, 0, static_cast<size_t>(M_) * N_ * sizeof(float)));

    // Allocate fp16 device buffers for A and B.
    const size_t nA = static_cast<size_t>(M_) * K_;
    const size_t nB = static_cast<size_t>(K_) * N_;
    CUDA_CHECK(cudaMalloc(&d_A_fp16_, nA * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_B_fp16_, nB * sizeof(half)));

    // Convert fp32 device buffers → fp16 on device.
    constexpr int kConvBlock = 256;
    const int gridA = (static_cast<int>(nA) + kConvBlock - 1) / kConvBlock;
    const int gridB = (static_cast<int>(nB) + kConvBlock - 1) / kConvBlock;
    fp32_to_fp16_kernel<<<gridA, kConvBlock>>>(d_A_, d_A_fp16_, static_cast<int>(nA));
    fp32_to_fp16_kernel<<<gridB, kConvBlock>>>(d_B_, d_B_fp16_, static_cast<int>(nB));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void GemmCudaWmma::run() {
    const dim3 block(kBlockSize);
    const dim3 grid(
        (N_ + kBN - 1) / kBN,
        (M_ + kBM - 1) / kBM);

    gemm_wmma_kernel<kBM, kBN, kBK, kWarpTM, kWarpTN><<<grid, block>>>(
        d_A_fp16_, d_B_fp16_, d_C_, M_, N_, K_);
}

bool GemmCudaWmma::verify() {
    CUDA_CHECK(cudaDeviceSynchronize());
    download_output();

    // FP16 inputs accumulate more rounding error than FP32.
    // For random inputs in [-0.5, 0.5] at K=4096, expected relative error ≈ 4–5 %.
    // 10 % threshold provides comfortable margin (consistent with CUTLASS TF32).
    for (int i = 0; i < kVerifyDim; ++i) {
        for (int j = 0; j < kVerifyDim; ++j) {
            const float ref  = h_C_ref_[static_cast<size_t>(i) * kVerifyDim + j];
            const float got  = h_C_out_[static_cast<size_t>(i) * N_ + j];
            const float denom = std::max(std::abs(ref), 1e-4f);
            if (std::abs(got - ref) / denom > kRelTolFp16) {
                return false;
            }
        }
    }
    return true;
}

void GemmCudaWmma::teardown() {
    if (d_A_fp16_) { cudaFree(d_A_fp16_); d_A_fp16_ = nullptr; }
    if (d_B_fp16_) { cudaFree(d_B_fp16_); d_B_fp16_ = nullptr; }
    free_device();
    h_A_.clear();
    h_B_.clear();
    h_C_out_.clear();
    h_C_ref_.clear();
}

} // namespace kernelx
