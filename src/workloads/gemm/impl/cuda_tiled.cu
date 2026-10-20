#include "workloads/gemm/impl/cuda_tiled.hpp"
#include "utils/cuda_utils.hpp"

namespace kernelx {

// ─────────────────────────────────────────────────────────────────────────────
// gemm_tiled_kernel<TILE>
//
// Each thread block computes a TILE×TILE tile of the output matrix C.
// Threads in the block cooperate to load tiles of A and B into shared memory,
// then each thread accumulates its dot product from shared memory.
//
// Grid  : ⌈N/TILE⌉ × ⌈M/TILE⌉  blocks
// Block : TILE × TILE            threads  (blockDim.x = blockDim.y = TILE)
//
// Thread (ty=threadIdx.y, tx=threadIdx.x) computes:
//   C[blockIdx.y*TILE + ty][blockIdx.x*TILE + tx]
//
// K-loop: iterate over ceil(K/TILE) tiles along the K dimension.
//   For each tile t:
//     1. Load sA[ty][tx] ← A[row][t*TILE + tx]   (row of A, column of tile)
//     2. Load sB[ty][tx] ← B[t*TILE + ty][col]   (row of tile, column of B)
//     3. __syncthreads() — all threads in block have finished loading
//     4. Accumulate: acc += sA[ty][k] * sB[k][tx] for k in [0, TILE)
//     5. __syncthreads() — prevent next tile's loads from overwriting current tile
//
// Boundary handling: guard conditions zero-pad the shared memory load for
// tiles that extend past the matrix edge, so K need not be a multiple of TILE.
// ─────────────────────────────────────────────────────────────────────────────
template <int TILE>
__global__ void gemm_tiled_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
          float* __restrict__ C,
    int M, int N, int K)
{
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    const int row = blockIdx.y * TILE + threadIdx.y;
    const int col = blockIdx.x * TILE + threadIdx.x;

    float acc = 0.0f;

    const int num_tiles = (K + TILE - 1) / TILE;
    for (int t = 0; t < num_tiles; ++t) {
        // ── Load tile of A ────────────────────────────────────────────────────
        // Thread (ty, tx) loads A[row][t*TILE + tx].
        // Across a warp (consecutive tx, same ty): accesses A[row][t*TILE + 0..31]
        //   → stride-1 → fully coalesced global load.
        const int a_col = t * TILE + threadIdx.x;
        sA[threadIdx.y][threadIdx.x] =
            (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;

        // ── Load tile of B ────────────────────────────────────────────────────
        // Thread (ty, tx) loads B[t*TILE + ty][col].
        // Across a warp (consecutive tx, same ty): accesses B[t*TILE+ty][col+0..31]
        //   → stride-1 → fully coalesced global load.
        const int b_row = t * TILE + threadIdx.y;
        sB[threadIdx.y][threadIdx.x] =
            (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

        __syncthreads();

        // ── Accumulate from shared memory ─────────────────────────────────────
        // sA[ty][k]: all threads in warp have same ty, varying k in serial loop
        //   → same bank each iteration → broadcast, no conflict.
        // sB[k][tx]: all threads in warp have different tx, same k
        //   → 32 consecutive 4-byte words → one bank per thread → no conflict.
        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            acc += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }

        // Guard: do not allow threads to start loading the next tile until
        // all threads have finished reading the current one.
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

// ─────────────────────────────────────────────────────────────────────────────

void GemmCudaTiled::setup(const WorkloadConfig& cfg) {
    init_host_data(cfg);
    compute_reference();
    alloc_device();
    upload_inputs();
    CUDA_CHECK(cudaMemset(d_C_, 0, static_cast<size_t>(M_) * N_ * sizeof(float)));
}

void GemmCudaTiled::run() {
    const dim3 block(kTile, kTile);
    const dim3 grid(
        (N_ + kTile - 1) / kTile,
        (M_ + kTile - 1) / kTile
    );
    gemm_tiled_kernel<kTile><<<grid, block>>>(d_A_, d_B_, d_C_, M_, N_, K_);
}

bool GemmCudaTiled::verify() {
    CUDA_CHECK(cudaDeviceSynchronize());
    download_output();
    return GemmWorkload::verify();
}

void GemmCudaTiled::teardown() {
    free_device();
    h_A_.clear();
    h_B_.clear();
    h_C_out_.clear();
    h_C_ref_.clear();
}

} // namespace kernelx
