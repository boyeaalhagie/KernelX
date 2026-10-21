#include "workloads/gemm/impl/cuda_tuned.hpp"
#include "utils/cuda_utils.hpp"

namespace kernelx {

// ─────────────────────────────────────────────────────────────────────────────
// gemm_tuned_kernel<BM, BN, BK, TM, TN>
//
// Register-blocked GEMM.  Thread (tx, ty) computes a TM×TN sub-tile of C.
//
// Outer loop: tiles along K (stride BK).
//   For each tile:
//     1. Load BM×BK elements of A into sA[BM][BK]      (2 KB for defaults)
//     2. Load BK×BN elements of B into sB[BK][BN]      (2 KB for defaults)
//     3. __syncthreads()
//     4. Inner loop over k in [0, BK):
//          a. Load sA column k into a_reg[TM] (register array)
//          b. Load sB row    k into b_reg[TN] (register array)
//          c. Outer product: acc[i][j] += a_reg[i] * b_reg[j]
//     5. __syncthreads()
//
// The key difference from cuda_tiled: step 4a loads TM values once from
// shared memory and reuses each TN times in the inner-j loop.  Symmetrically
// for b_reg.  This is the "register reuse" that reduces shared-memory traffic
// per FLOP compared to V2.
// ─────────────────────────────────────────────────────────────────────────────
template <int BM, int BN, int BK, int TM, int TN>
__global__ void gemm_tuned_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
          float* __restrict__ C,
    int M, int N, int K)
{
    static_assert(BM % TM == 0, "BM must be divisible by TM");
    static_assert(BN % TN == 0, "BN must be divisible by TN");

    // ── Thread identity ───────────────────────────────────────────────────────
    const int tx  = threadIdx.x;   // [0, BN/TN)
    const int ty  = threadIdx.y;   // [0, BM/TM)
    const int tid = ty * (BN / TN) + tx;   // linear index within block

    // ── This block's output tile origin in C ─────────────────────────────────
    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    // ── Shared memory ─────────────────────────────────────────────────────────
    __shared__ float sA[BM][BK];
    __shared__ float sB[BK][BN];

    // ── Accumulator registers — the TM×TN output sub-tile ────────────────────
    float acc[TM][TN] = {};

    // ── Number of elements each thread loads per shared-memory stage ──────────
    constexpr int NTHREADS  = (BM / TM) * (BN / TN);
    constexpr int A_PER_THD = BM * BK / NTHREADS;   // elements of A per thread
    constexpr int B_PER_THD = BK * BN / NTHREADS;   // elements of B per thread

    const int num_tiles = (K + BK - 1) / BK;

    for (int t = 0; t < num_tiles; ++t) {
        // ── Load sA ───────────────────────────────────────────────────────────
        // Thread tid loads A_PER_THD elements, strided by NTHREADS in the
        // linearised BM×BK tile.
        //
        // Coalescing note: with BK=8, consecutive tids access consecutive
        // positions in the linearised tile.  Within a single row, accesses
        // to A are stride-1 (good); warp-level coalescing is partial because
        // a warp spans multiple rows of A when BK < warp_size.
        // CUTLASS resolves this with async copies and 128-byte vectorisation.
        #pragma unroll
        for (int i = 0; i < A_PER_THD; ++i) {
            const int elem      = tid * A_PER_THD + i;
            const int local_row = elem / BK;
            const int local_col = elem % BK;
            const int global_row = block_row + local_row;
            const int global_col = t * BK + local_col;
            sA[local_row][local_col] =
                (global_row < M && global_col < K)
                ? A[global_row * K + global_col] : 0.0f;
        }

        // ── Load sB ───────────────────────────────────────────────────────────
        // Symmetric to sA: thread tid loads B_PER_THD elements from B.
        // B[k][col]: for fixed k, consecutive cols are stride-1 → coalesced
        // within each row.  Same partial-warp caveat as sA.
        #pragma unroll
        for (int i = 0; i < B_PER_THD; ++i) {
            const int elem      = tid * B_PER_THD + i;
            const int local_row = elem / BN;
            const int local_col = elem % BN;
            const int global_row = t * BK + local_row;
            const int global_col = block_col + local_col;
            sB[local_row][local_col] =
                (global_row < K && global_col < N)
                ? B[global_row * N + global_col] : 0.0f;
        }

        __syncthreads();

        // ── Register-blocked accumulation ─────────────────────────────────────
        // For each k in [0, BK):
        //   • Load the TM-element column of sA into a_reg[TM]
        //   • Load the TN-element row   of sB into b_reg[TN]
        //   • Compute TM×TN outer product into acc[][]
        //
        // Each a_reg[i] is reused TN times; each b_reg[j] is reused TM times.
        // This is the register reuse that separates V3 from V2.
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            float a_reg[TM], b_reg[TN];

            #pragma unroll
            for (int i = 0; i < TM; ++i)
                a_reg[i] = sA[ty * TM + i][k];

            #pragma unroll
            for (int j = 0; j < TN; ++j)
                b_reg[j] = sB[k][tx * TN + j];

            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int j = 0; j < TN; ++j)
                    acc[i][j] += a_reg[i] * b_reg[j];
        }

        __syncthreads();
    }

    // ── Write output tile ─────────────────────────────────────────────────────
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int row = block_row + ty * TM + i;
            const int col = block_col + tx * TN + j;
            if (row < M && col < N)
                C[row * N + col] = acc[i][j];
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────

void GemmCudaTuned::setup(const WorkloadConfig& cfg) {
    init_host_data(cfg);
    compute_reference();
    alloc_device();
    upload_inputs();
    CUDA_CHECK(cudaMemset(d_C_, 0, static_cast<size_t>(M_) * N_ * sizeof(float)));
}

void GemmCudaTuned::run() {
    const dim3 block(kBlockX, kBlockY);
    const dim3 grid(
        (N_ + kBN - 1) / kBN,
        (M_ + kBM - 1) / kBM
    );
    gemm_tuned_kernel<kBM, kBN, kBK, kTM, kTN><<<grid, block>>>(
        d_A_, d_B_, d_C_, M_, N_, K_);
}

bool GemmCudaTuned::verify() {
    CUDA_CHECK(cudaDeviceSynchronize());
    download_output();
    return GemmWorkload::verify();
}

void GemmCudaTuned::teardown() {
    free_device();
    h_A_.clear();
    h_B_.clear();
    h_C_out_.clear();
    h_C_ref_.clear();
}

} // namespace kernelx
