#pragma once

#include "workloads/layernorm/layernorm_workload.hpp"

namespace kernelx {

// ─────────────────────────────────────────────────────────────────────────────
// LayerNormCudaNaive — one CUDA thread per row, three global-memory passes.
//
// Version 1 in the LayerNorm progression.  Directly mirrors the CPU reference
// on the GPU to show why the naive mapping performs poorly.
//
// Kernel structure:
//   Grid : ⌈M / kBlockDim⌉ × 1
//   Block: kBlockDim × 1   (256 threads, one per row)
//
//   Each thread independently:
//     Pass 1 — read N floats: accumulate sum → mean
//     Pass 2 — read N floats: accumulate (x − μ)² → variance
//     Pass 3 — read N floats + γ[N] + β[N]: normalise, write N floats
//
// Memory analysis:
//   Traffic per row: 4 reads × N floats + 1 write × N = 5N×4 = 20N bytes.
//   (γ and β are read once per row; likely cached in L1 for large M.)
//   Arithmetic intensity: 8N / 20N = 0.4 FLOP/byte — memory-bound.
//
// Limitations (driving cuda_welford design):
//   • One-thread-per-row: M active threads total — terrible occupancy for
//     small M or large N (H100 wants 100K+ active threads).
//   • Three passes: 3× more HBM traffic than the theoretical minimum.
//   • Uncoalesced loads: threads in a warp load different rows → non-sequential
//     access pattern for the same column index across consecutive rows.
// ─────────────────────────────────────────────────────────────────────────────
class LayerNormCudaNaive final : public LayerNormWorkload {
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
