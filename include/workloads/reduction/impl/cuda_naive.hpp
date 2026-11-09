#pragma once
#include "workloads/reduction/reduction_workload.hpp"

namespace kernelx {

// ─────────────────────────────────────────────────────────────────────────────
// ReductionCudaNaive — shared-memory tree reduction (one block per row).
//
// Algorithm (NVIDIA "Optimizing Parallel Reduction" §4 — no idle first step):
//   1. Each thread accumulates N / blockDim.x elements in a register (stride
//      = blockDim.x) — eliminates the idle-thread problem of the classic 2×N
//      load.
//   2. 256-way binary tree in shared memory:
//      for (int s = blockDim.x/2; s > 0; s >>= 1) {
//          if (tid < s) s_sum[tid] += s_sum[tid + s];
//          __syncthreads();
//      }
//   3. Thread 0 writes s_sum[0] to Y[blockIdx.x].
//
// Grid / block:
//   grid  = (M,)       — one block per row
//   block = (256,)     — 8 warps, 256 threads
//
// Shared memory per block: 256 × 4 = 1 KB (well within any SM limit).
//
// Performance ceiling:
//   This kernel is memory-bound: AI ≈ 0.25 FLOP/byte (vs H100 ridge 17.9).
//   At N=4096, each row reads 16 KB; throughput ≈ HBM bandwidth / 16.
//
// Known cost: __syncthreads() at every tree level (8 barriers for 256 threads).
//   The warp-shuffle variant (cuda_warp) replaces the first 5 barriers with
//   __shfl_down_sync, cutting synchronisation overhead.
// ─────────────────────────────────────────────────────────────────────────────
class ReductionCudaNaive final : public ReductionWorkload {
public:
    void setup(const WorkloadConfig& cfg) override;
    void run()                            override;
    void teardown()                       override;
    bool verify()                         override;

    std::string implementation_name() const override { return "cuda_naive"; }
    Backend     backend()             const override { return Backend::CUDA; }
};

} // namespace kernelx
