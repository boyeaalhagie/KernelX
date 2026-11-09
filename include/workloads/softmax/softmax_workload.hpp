#pragma once

#include "workloads/workload.hpp"
#include <vector>

namespace kernelx {

// SoftmaxWorkload — shared base for all Softmax implementations.
//
// Problem: given input X [M × N], compute row-wise softmax:
//   Y[i][j] = exp(X[i][j] - max_i) / Σ_k exp(X[i][k] - max_i)
//
// WorkloadConfig mapping:
//   cfg.m = M   (number of rows — batch × heads for attention)
//   cfg.n = N   (row length — sequence length or vocabulary size)
//   cfg.k       (unused; always stored as 1 in ExperimentRecord)
//
// FLOP count: 4 FLOPs per element (subtract, exp, add-to-sum, divide).
// exp is counted as 1 FLOP (maps to hardware __expf with --use_fast_math).
// Total: 4 * M * N FLOPs.
//
// Verification strategy:
//   The CPU reference is computed for kVerifyRows rows only (first 32).
//   verify() compares the GPU output for those rows against the reference.
//   Tolerance kRelTol = 1e-3f accounts for fast-math exp error and
//   floating-point accumulation differences in large sums.

class SoftmaxWorkload : public Workload {
public:
    ~SoftmaxWorkload() override;

    bool   verify() override;
    double flops()  const override;
    std::string workload_name() const override { return "softmax"; }

protected:
    // Rows of reference computed (and verified) — bounded to keep setup fast.
    static constexpr int   kVerifyRows = 32;
    static constexpr float kRelTol     = 1e-3f;

    int M_ = 0, N_ = 0;

    // Host buffers
    std::vector<float> h_X_;        // input  [M × N]
    std::vector<float> h_Y_ref_;    // reference output [kVerifyRows × N]
    std::vector<float> h_Y_out_;    // implementation output [M × N] (downloaded)

    // Device buffers — non-null only for GPU implementations
    float* d_X_ = nullptr;
    float* d_Y_ = nullptr;

    // Setup helpers (called by subclasses in their setup())
    void init_host_data(const WorkloadConfig& cfg);
    void compute_reference();   // fills h_Y_ref_ for first kVerifyRows rows
    void alloc_device();
    void upload_inputs();       // h_X_ → d_X_
    void download_output();     // d_Y_ → h_Y_out_
    void free_device();
};

} // namespace kernelx
