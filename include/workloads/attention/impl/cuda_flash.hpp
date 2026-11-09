#pragma once
#include "workloads/attention/attention_workload.hpp"
namespace kernelx {

// AttentionCudaFlash — FlashAttention-1 tiled GPU attention.
//
// Version 2: eliminates O(N²) HBM traffic by keeping K and V tiles in SRAM.
// Algorithm (Dao et al., 2022):
//   - Outer loop: query rows (one warp per row via blockIdx.x).
//   - Inner loop: K/V tiles of size kBc columns.
//   - Online state per query: (m, l, o_acc) where
//       m     = running max of QK^T scores seen so far
//       l     = running normalisation denominator (partition function)
//       o_acc = running weighted sum of V rows (unnormalised)
//   - Tile update (Milakov-Gimelshein rescaling):
//       m_new      = max(m, tile_max)
//       l_new      = l * exp(m - m_new) + sum_j exp(S[j] - m_new)
//       o_acc_new  = o_acc * exp(m - m_new) + sum_j exp(S[j] - m_new) * V[j]
//   - Final: O[i] = o_acc / l_new
//
// Memory traffic (per attention call, N tokens, head dim D):
//   reads  Q, K, V: 3·N·D·4 bytes  (each loaded once)
//   writes O:       N·D·4 bytes
//   Total = 4·N·D·4 bytes  (vs 4·N²·4 for cuda_naive at large N)
//   At N=2048, D=64: ~2 MB vs ~32 MB — 16× reduction.
//
// Tile geometry:
//   kBc = 32  (K/V tile columns, one per lane within the warp)
//   kD  = 64  (head dimension; assumed fixed for register efficiency)
//   Block: (32,)   Grid: (N,)   — one warp per query row

class AttentionCudaFlash final : public AttentionWorkload {
public:
    void setup(const WorkloadConfig& cfg) override;
    void run()                            override;
    void teardown()                       override;
    bool verify()                         override;
    std::string implementation_name() const override { return "cuda_flash"; }
    Backend     backend()             const override { return Backend::CUDA; }
};

} // namespace kernelx
