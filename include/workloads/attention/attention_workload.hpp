#pragma once

#include "workloads/workload.hpp"
#include <vector>

namespace kernelx {

// ─────────────────────────────────────────────────────────────────────────────
// AttentionWorkload — shared base for all single-head attention implementations.
//
// Problem: given query Q [N×D], key K [N×D], value V [N×D]:
//
//   S   = Q · K^T / sqrt(D)      [N×N]  attention scores (pre-softmax)
//   P   = softmax(S, dim=1)       [N×N]  attention weights
//   O   = P · V                   [N×D]  output
//
// WorkloadConfig mapping:
//   cfg.m = N   (sequence length — tokens per head)
//   cfg.n = D   (head dimension, typically 64 or 128)
//   cfg.k       (unused; always stored as 1)
//
// Scale: 1/sqrt(D) applied before softmax for stable gradients.
//
// FLOP count:
//   QK^T matmul: 2·N²·D FLOPs
//   Softmax:     5·N²   FLOPs  (max+sub+exp+sum+div per element)
//   PV matmul:   2·N²·D FLOPs
//   Total ≈ 4·N²·D + 5·N² ≈ 4·N²·D  (D >> 1 for typical D=64)
//   flops() returns 4·N²·D (the dominant matmul term).
//
// Memory traffic comparison (informs roofline analysis):
//   cuda_naive: reads Q,K,V (3·N·D), writes N×N attn, reads N×N attn,
//               writes O  →  O(N²) HBM traffic
//   cuda_flash: reads Q,K,V once, writes O once, tiles K,V in SRAM
//               →  O(N·D) HBM traffic, N² computation in SRAM
//
// Verification: first kVerifyRows=16 query rows against CPU reference.
// Tolerance kRelTol = 1e-3f (softmax + fast-math exp accumulation error).
// ─────────────────────────────────────────────────────────────────────────────
class AttentionWorkload : public Workload {
public:
    ~AttentionWorkload() override;

    bool   verify() override;
    double flops()  const override;
    std::string workload_name() const override { return "attention"; }

protected:
    static constexpr int   kVerifyRows = 16;
    static constexpr float kRelTol     = 1e-2f;  // relaxed: online softmax + fast-math

    int M_ = 0;   // sequence length N
    int D_ = 0;   // head dimension

    // Host buffers
    std::vector<float> h_Q_;        // query  [N×D]
    std::vector<float> h_K_;        // key    [N×D]
    std::vector<float> h_V_;        // value  [N×D]
    std::vector<float> h_O_ref_;    // reference output  [kVerifyRows×D]
    std::vector<float> h_O_out_;    // implementation output [N×D]

    // Device buffers — non-null for GPU implementations
    float* d_Q_ = nullptr;
    float* d_K_ = nullptr;
    float* d_V_ = nullptr;
    float* d_O_ = nullptr;

    float scale() const;   // = 1.0f / sqrtf(static_cast<float>(D_))

    void init_host_data(const WorkloadConfig& cfg);
    void compute_reference();
    void alloc_device();
    void upload_inputs();
    void download_output();
    void free_device();
};

} // namespace kernelx
