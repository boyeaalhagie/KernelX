#pragma once
#include "workloads/attention/attention_workload.hpp"
namespace kernelx {

// AttentionCudaNaive — three-kernel GPU attention, materialises full N×N matrix.
//
// Version 1: direct GPU port of the standard attention algorithm.
// Three separate kernel launches:
//   1. qk_kernel:      S[N×N] = Q[N×D] × K^T[D×N] / sqrt(D)
//   2. softmax_kernel: P[N×N] = row-wise softmax(S)  (online warp-per-row)
//   3. pv_kernel:      O[N×D] = P[N×N] × V[N×D]
//
// Memory traffic (per attention call, N tokens, head dim D):
//   reads  Q, K: 2·N·D·4 bytes  (qk_kernel)
//   writes S:    N²·4 bytes     (qk_kernel)
//   reads  S:    N²·4 bytes     (softmax_kernel)
//   writes P:    N²·4 bytes     (softmax_kernel)
//   reads  P, V: N²·4 + N·D·4  (pv_kernel)
//   writes O:    N·D·4 bytes
//   Total ≈ 2·N²·4 + 2·N²·4 = 4·N² floats for the N×N matrix alone.
//   At N=2048, D=64: ~32 MB round-trip through HBM for the attention matrix.
//
// This is the bottleneck FlashAttention eliminates by tiling K, V in SRAM.

class AttentionCudaNaive final : public AttentionWorkload {
public:
    void setup(const WorkloadConfig& cfg) override;
    void run()                            override;
    void teardown()                       override;
    bool verify()                         override;
    std::string implementation_name() const override { return "cuda_naive"; }
    Backend     backend()             const override { return Backend::CUDA; }

private:
    float* d_S_ = nullptr;   // attention score matrix [N×N]
};

} // namespace kernelx
