#pragma once

#include "workloads/workload.hpp"
#include <vector>

namespace kernelx {

// ─────────────────────────────────────────────────────────────────────────────
// LayerNormWorkload — shared base for all LayerNorm implementations.
//
// Problem: given input X [M × N], learned scale γ [N] and bias β [N]:
//
//   μ_i  = (1/N) Σ_j X[i][j]
//   σ²_i = (1/N) Σ_j (X[i][j] − μ_i)²
//   Y[i][j] = γ[j] · (X[i][j] − μ_i) / sqrt(σ²_i + ε)  +  β[j]
//
// WorkloadConfig mapping:
//   cfg.m = M   (number of rows — tokens in a batch, typically batch × seq_len)
//   cfg.n = N   (row length — hidden dimension, e.g. 768, 1024, 4096)
//   cfg.k       (unused; always stored as 1 in ExperimentRecord)
//
// FLOP count (per row):
//   mean:    N adds + 1 div              ≈ N
//   variance: N subs + N squares + N adds + 1 div + 1 sqrt  ≈ 4N
//   normalise+scale+bias: N subs + N muls + N muls + N adds ≈ 4N
//   Total: 8 FLOPs/element  →  flops() = 8 · M · N
//
// Verification: first kVerifyRows=32 rows against CPU two-pass reference.
// Tolerance kRelTol = 1e-3f (fast-math rsqrtf error + accumulation).
// ─────────────────────────────────────────────────────────────────────────────
class LayerNormWorkload : public Workload {
public:
    static constexpr float kEpsilon    = 1e-5f;

    ~LayerNormWorkload() override;

    bool   verify() override;
    double flops()  const override;
    std::string workload_name() const override { return "layernorm"; }

protected:
    static constexpr int   kVerifyRows = 32;
    static constexpr float kRelTol     = 1e-3f;

    int M_ = 0, N_ = 0;

    // Host buffers
    std::vector<float> h_X_;        // input      [M × N]
    std::vector<float> h_gamma_;    // scale      [N]
    std::vector<float> h_beta_;     // bias       [N]
    std::vector<float> h_Y_ref_;    // reference  [kVerifyRows × N]
    std::vector<float> h_Y_out_;    // output     [M × N]

    // Device buffers — non-null for GPU implementations
    float* d_X_     = nullptr;
    float* d_gamma_ = nullptr;
    float* d_beta_  = nullptr;
    float* d_Y_     = nullptr;

    // ── Setup helpers ─────────────────────────────────────────────────────────
    void init_host_data(const WorkloadConfig& cfg);
    void compute_reference();
    void alloc_device();
    void upload_inputs();
    void download_output();
    void free_device();
};

} // namespace kernelx
