#pragma once

#include "workloads/gemm/gemm_workload.hpp"

namespace kernelx {

// ─────────────────────────────────────────────────────────────────────────────
// GemmCudaTiled — tiled GEMM using shared memory.
//
// This is Version 2 in the GEMM implementation progression.
//
// ── What changes from cuda_naive ────────────────────────────────────────────
// cuda_naive: each thread independently reads K floats from A and K floats
// from B, entirely from global/L2.  Every value is loaded once per output
// element → no data reuse between threads.
//
// cuda_tiled: threads in the same block cooperate to load a TILE×TILE
// submatrix of A and B into shared memory once, then each thread accumulates
// from shared memory for the duration of that tile.  Each global load is
// reused TILE times before being evicted.
//
// ── Arithmetic intensity analysis ───────────────────────────────────────────
// naive:  2·K FLOPs / (2·K·4 bytes) = 0.25 FLOP/byte  (memory-bound)
// tiled:  2·K FLOPs / (2·(K/TILE)·4 bytes × some reuse)
//         In the idealised model: each thread does 2·K FLOPs
//         and the block loads 2·TILE·K/TILE = 2·K floats shared across
//         TILE² threads → arithmetic intensity ≈ TILE/4 FLOP/byte.
//         For TILE=32: ~8 FLOP/byte  (approaching compute-bound territory)
//
// ── Shared memory layout ────────────────────────────────────────────────────
// sA[TILE][TILE]:  row = threadIdx.y, col = threadIdx.x
//   Load:  sA[ty][tx] ← A[row][tile*TILE + tx]   — coalesced across warp
//   Use:   sA[ty][k]  for fixed ty, varying k    — broadcast, no conflict
//
// sB[TILE][TILE]:  row = threadIdx.y, col = threadIdx.x
//   Load:  sB[ty][tx] ← B[tile*TILE + ty][col]   — coalesced across warp
//   Use:   sB[k][tx]  for varying tx, fixed k    — stride-1, no bank conflict
//
// ── What to look for in Nsight after V2 ─────────────────────────────────────
//   • Global memory load traffic should drop ~TILE× vs naive
//   • Shared memory utilisation should appear in Memory Workload Analysis
//   • Speed-of-Light compute utilisation should rise; memory should fall
//   • If still memory-bound: L1/shared hit rate < expected → inspect tile size
//
// ── What V3 (cuda_tuned) will add ───────────────────────────────────────────
//   Register blocking (thread coarsening): each thread computes a 2×2 or 4×4
//   output sub-tile, amortising __syncthreads() overhead and hiding more
//   latency.  Double-buffering the shared memory loads to overlap computation
//   with the next tile prefetch.
// ─────────────────────────────────────────────────────────────────────────────
class GemmCudaTiled final : public GemmWorkload {
public:
    // Tile dimension.  Both the thread block and the shared memory tiles are
    // TILE×TILE.  Must evenly divide the warp (32); 16 and 32 are common choices.
    //   TILE=16: 2×16×16×4 = 2 KB shared/block  →  more concurrent blocks
    //   TILE=32: 2×32×32×4 = 8 KB shared/block  →  higher arithmetic intensity
    static constexpr int kTile = 32;

    void setup(const WorkloadConfig& cfg) override;
    void run()                            override;
    void teardown()                       override;
    bool verify()                         override;

    std::string implementation_name() const override { return "cuda_tiled"; }
    Backend     backend()             const override { return Backend::CUDA; }
};

} // namespace kernelx
