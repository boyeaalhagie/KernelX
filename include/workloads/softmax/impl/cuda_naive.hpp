#pragma once

#include "workloads/softmax/softmax_workload.hpp"

namespace kernelx {

// SoftmaxCudaNaive — one CUDA thread per row, three global-memory passes.
//
// This is Version 1 in the softmax implementation progression.  It establishes
// a GPU baseline and exposes the dominant bottleneck before any optimisation.
//
// Kernel structure:
//   Grid : ⌈M / kBlockDim⌉ × 1
//   Block: kBlockDim × 1   (256 threads, one per row)
//
//   Each thread independently performs the three-pass softmax over its row:
//     Pass 1 — read N floats: find max
//     Pass 2 — read N floats: exp(x - max), accumulate sum, write N floats
//     Pass 3 — read N floats: divide by sum, write N floats
//
// Memory analysis:
//   Traffic per row: 4 reads × N floats + 2 writes × N floats = 6N×4 = 24N bytes.
//   Arithmetic per row: 4N FLOPs  →  arithmetic intensity ≈ 0.17 FLOP/byte.
//   At H100 HBM bandwidth (3350 GB/s): peak ≈ 570 GFLOP/s — deeply memory-bound.
//   At T4 HBM bandwidth (300 GB/s):    peak ≈  51 GFLOP/s.
//
// Performance notes:
//   For small N (< ~1024), the kernel is latency-bound (few warps active per SM).
//   For large N (> 32K), 256 active threads cover ~256 rows simultaneously, but
//   each thread serialises 3 full-row scans — no parallelism within a row.
//   Both limitations are addressed by cuda_online (warp-per-row + single pass).

class SoftmaxCudaNaive final : public SoftmaxWorkload {
public:
    static constexpr int kBlockDim = 256;

    void setup(const WorkloadConfig& cfg) override;
    void run()                            override;
    void teardown()                       override;
    bool verify()                         override;

    std::string implementation_name() const override { return "cuda_naive"; }
    Backend     backend()             const override { return Backend::CUDA; }
};

} // namespace kernelx
