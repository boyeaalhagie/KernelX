#pragma once

#include "workloads/gemm/gemm_workload.hpp"

namespace kernelx {

// ─────────────────────────────────────────────────────────────────────────────
// GemmCudaNaive — one CUDA thread per output element, global memory only.
//
// This is Version 1 in the GEMM implementation progression.  Its purpose is
// to establish a GPU baseline and expose the dominant bottleneck before any
// optimisation is applied.
//
// Kernel structure:
//   Grid : ⌈N/kBlockDim⌉ × ⌈M/kBlockDim⌉   (col-major grid, row-major blocks)
//   Block: kBlockDim × kBlockDim              (256 threads)
//
//   Thread (row, col) accumulates:
//     C[row][col] = Σ_{k=0}^{K-1}  A[row][k] · B[k][col]
//
// Expected hardware behaviour (confirmed with Nsight Speed-of-Light):
//   • Each thread reads 2K floats — O(K) global-memory traffic per element
//   • A[row][*] access: stride-1 across k  →  sequential per-thread, but
//     threads in the same warp access different rows  →  NOT coalesced
//   • B[*][col] access: stride-N across k  →  strided, high L2 pressure
//   • No data reuse between threads  →  arithmetic intensity ≈ 2/8 FLOP/byte
//   • Bottleneck: memory bandwidth (roofline: well below compute ceiling)
//
// These properties make cuda_naive the ideal starting point for the
// tiled-GEMM analysis in Version 2.
// ─────────────────────────────────────────────────────────────────────────────
class GemmCudaNaive final : public GemmWorkload {
public:
    static constexpr int kBlockDim = 16;  // 16×16 = 256 threads — fits in one warp group

    void setup(const WorkloadConfig& cfg) override;
    void run()                            override;
    void teardown()                       override;
    bool verify()                         override;  // downloads d_C_ then delegates to base

    std::string implementation_name() const override { return "cuda_naive"; }
    Backend     backend()             const override { return Backend::CUDA; }
};

} // namespace kernelx
