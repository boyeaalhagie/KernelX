#pragma once

#include "workloads/softmax/softmax_workload.hpp"

namespace kernelx {

// SoftmaxCudaOnline — one warp per row, online (single-pass) algorithm.
//
// This is Version 2 in the softmax implementation progression.  It addresses
// two limitations of cuda_naive:
//   1. Memory bandwidth: fuses the max and sum reductions into a single data
//      pass using the online update due to Milakov & Gimelshein (2018).
//   2. Parallelism: 32 threads cooperate on each row via warp-level shuffles,
//      enabling coalesced loads and latency hiding through warp divergence.
//
// Online softmax update (per new element x, running state (m, d)):
//   m' = max(m, x)
//   d' = d · exp(m − m') + exp(x − m')
//
// This maintains m as the running maximum and d as the running
// sum-of-exponentials relative to that maximum, in a single scan.
//
// Kernel structure:
//   Block: (32, kWarpsPerBlock)   — one warp per row; multiple warps per block
//   Grid : ⌈M / kWarpsPerBlock⌉ × 1
//
//   Pass 1 — each lane sweeps its stripe of N/32 elements, computing local
//            (m, d) with the online update; then a warp-reduce via
//            __shfl_down_sync merges all 32 (m, d) pairs to lane 0 and
//            broadcasts the global (m, d) to all lanes.
//   Pass 2 — each lane writes its stripe: Y[j] = exp(X[j] - m) / d.
//
// Memory analysis:
//   Traffic per row: 2 reads × N floats + 1 write × N floats = 3N×4 = 12N bytes.
//   Arithmetic intensity: 4N / 12N ≈ 0.33 FLOP/byte — 2× better than naive.
//   Coalescing: within a warp, threads read X[row*N + j], X[row*N + j+1], ...
//   stride-32, not stride-1.  For large N the access is effectively streaming
//   across a row; for N a multiple of 32 all warp-width loads are coalesced.
//
// References:
//   Milakov, M. & Gimelshein, N. (2018). "Online normalizer calculation for
//   softmax." arXiv:1805.02867.

class SoftmaxCudaOnline final : public SoftmaxWorkload {
public:
    static constexpr int kWarpsPerBlock = 4;   // tunable; 4 × 32 = 128 threads

    void setup(const WorkloadConfig& cfg) override;
    void run()                            override;
    void teardown()                       override;
    bool verify()                         override;

    std::string implementation_name() const override { return "cuda_online"; }
    Backend     backend()             const override { return Backend::CUDA; }
};

} // namespace kernelx
