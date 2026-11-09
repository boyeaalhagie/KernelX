#pragma once
#include "workloads/reduction/reduction_workload.hpp"

namespace kernelx {

// ─────────────────────────────────────────────────────────────────────────────
// ReductionCudaWarp — warp-shuffle reduction (one block per row).
//
// Key idea: replace all intra-warp __syncthreads() barriers with
// __shfl_down_sync() butterfly steps, then reduce across warps with a minimal
// shared-memory stage.
//
// Algorithm (kBlockDim=256, 8 warps per block):
//   1. Each thread accumulates N / kBlockDim elements in a register (identical
//      to the naive version — same memory access pattern).
//   2. Intra-warp butterfly (5 steps, no barrier):
//        for (int offset = 16; offset >= 1; offset >>= 1)
//            acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
//      After this, lane 0 of each warp holds the warp's partial sum.
//   3. Lane 0 of each warp writes its partial sum to s_partial[warp_id].
//      __syncthreads() once to make all 8 partial sums visible.
//   4. Warp 0 reads the 8 partial sums, reduces them with another butterfly.
//   5. Thread 0 writes s_partial[0] to Y[blockIdx.x].
//
// Synchronisation cost: 1 __syncthreads() (vs 8 for the naive version).
//   For 256 threads the saving is 7 barriers/block — measurable at small N
//   where the reduction phase dominates over HBM latency.
//
// At large N (≥ 4096), both kernels are HBM-bandwidth-bound and perform
// similarly.  At small N (≤ 128), the warp kernel is 1.5–2× faster because
// the per-row latency is dominated by tree overhead, not data movement.
//
// Grid / block: same as cuda_naive — grid=(M,), block=(kBlockDim=256).
// Shared memory per block: kBlockDim / kWarpSize = 8 floats = 32 bytes.
// ─────────────────────────────────────────────────────────────────────────────
class ReductionCudaWarp final : public ReductionWorkload {
public:
    void setup(const WorkloadConfig& cfg) override;
    void run()                            override;
    void teardown()                       override;
    bool verify()                         override;

    std::string implementation_name() const override { return "cuda_warp"; }
    Backend     backend()             const override { return Backend::CUDA; }
};

} // namespace kernelx
