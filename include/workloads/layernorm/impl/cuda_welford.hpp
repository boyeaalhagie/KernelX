#pragma once

#include "workloads/layernorm/layernorm_workload.hpp"

namespace kernelx {

// ─────────────────────────────────────────────────────────────────────────────
// LayerNormCudaWelford — one warp per row, Welford online mean+variance.
//
// Version 2 in the LayerNorm progression.  Addresses both limitations of
// cuda_naive: occupancy and memory traffic.
//
// Algorithm — Welford's online mean+variance (Welford, 1962):
//   For each new element x, given running (count, mean, M2):
//     count++
//     delta  = x − mean
//     mean  += delta / count
//     delta2 = x − mean         ← uses the updated mean
//     M2    += delta · delta2
//   variance = M2 / count
//
//   Merge of two accumulators (n_a, μ_a, M2_a) and (n_b, μ_b, M2_b):
//     n     = n_a + n_b
//     δ     = μ_b − μ_a
//     μ     = (n_a·μ_a + n_b·μ_b) / n
//     M2    = M2_a + M2_b + δ² · n_a · n_b / n
//
//   This computes mean and variance in a single pass over X — 2× less HBM
//   traffic than the two-pass approach (mean-only + variance-only passes).
//
// Kernel structure:
//   Block: (32, kWarpsPerBlock)   — threadIdx.x = lane, threadIdx.y = warp
//   Grid : ⌈M / kWarpsPerBlock⌉ × 1
//
//   Pass 1 — Welford scan (lane sweeps N/32 elements, warp-merge via shuffles):
//     Each lane accumulates local (count, mean, M2) with the sequential update.
//     A tree reduction using __shfl_down_sync merges all 32 partial accumulators
//     into the global (mean, variance) in lane 0, then broadcasts to all lanes.
//
//   Pass 2 — normalise + scale + bias (each lane writes its N/32 elements).
//
// Memory analysis (per row):
//   Traffic: 2 reads × N floats + 1 write × N floats = 3N×4 = 12N bytes.
//   (γ and β typically remain in L1 for the full grid.)
//   Arithmetic intensity: 8N / 12N ≈ 0.67 FLOP/byte — 1.67× better than naive.
//
// References:
//   Welford, B.P. (1962). "Note on a method for calculating corrected sums of
//   squares and products." Technometrics 4(3):419–420.
//   Chan, T.F. et al. (1979). "Updating formulae and a pairwise algorithm for
//   computing sample variances." Technical Report Stan-CS-79-773.
// ─────────────────────────────────────────────────────────────────────────────
class LayerNormCudaWelford final : public LayerNormWorkload {
public:
    static constexpr int kWarpsPerBlock = 4;

    void setup(const WorkloadConfig& cfg) override;
    void run()                            override;
    void teardown()                       override;
    bool verify()                         override;

    std::string implementation_name() const override { return "cuda_welford"; }
    Backend     backend()             const override { return Backend::CUDA; }
};

} // namespace kernelx
