#pragma once

#include "workloads/workload.hpp"
#include <vector>

namespace kernelx {

// ─────────────────────────────────────────────────────────────────────────────
// ReductionWorkload — shared base for all Reduction implementations.
//
// Problem: given input X [M × N], compute row-wise sum:
//   Y[i] = Σ_j  X[i][j]    for i in [0, M)
//
// WorkloadConfig mapping:
//   cfg.m = M   (number of rows)
//   cfg.n = N   (row length — determines data-movement volume)
//   cfg.k       (unused; always stored as 1 in ExperimentRecord)
//
// FLOP count: N − 1 additions per row ≈ N per row → flops() = M · N.
//
// Memory analysis:
//   Reads:  M · N · 4 bytes
//   Writes: M · 4 bytes  (one scalar per row)
//   Total ≈ M · N · 4 bytes  (write term negligible for large N)
//   Arithmetic intensity ≈ 1 FLOP / 4 bytes = 0.25 FLOP/byte
//   → Deep in the memory-bound regime on every GPU architecture.
//   → Ideal workload for demonstrating HBM bandwidth efficiency.
//
// Verification: compare GPU Y[i] against CPU reference for first kVerifyRows
// rows.  Tolerance kRelTol = 1e-3f (fp32 accumulation order differs).
// ─────────────────────────────────────────────────────────────────────────────
class ReductionWorkload : public Workload {
public:
    ~ReductionWorkload() override;

    bool   verify() override;
    double flops()  const override;
    std::string workload_name() const override { return "reduction"; }

protected:
    static constexpr int   kVerifyRows = 64;
    static constexpr float kRelTol     = 1e-3f;

    int M_ = 0, N_ = 0;

    // Host buffers
    std::vector<float> h_X_;        // input  [M × N]
    std::vector<float> h_Y_ref_;    // reference sums [M]
    std::vector<float> h_Y_out_;    // GPU output sums [M]

    // Device buffers — non-null for GPU implementations
    float* d_X_ = nullptr;
    float* d_Y_ = nullptr;

    // Setup helpers
    void init_host_data(const WorkloadConfig& cfg);
    void compute_reference();
    void alloc_device();
    void upload_inputs();
    void download_output();
    void free_device();
};

} // namespace kernelx
