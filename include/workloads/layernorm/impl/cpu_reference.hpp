#pragma once

#include "workloads/layernorm/layernorm_workload.hpp"

namespace kernelx {

// ─────────────────────────────────────────────────────────────────────────────
// LayerNormCpuReference — numerically stable two-pass CPU implementation.
//
// Version 0 in the LayerNorm progression.  Provides the ground truth for
// verification and shows the baseline three-pass algorithm:
//   Pass 1 — scan row: accumulate sum → mean (double precision for accuracy)
//   Pass 2 — scan row: accumulate (x − μ)² → variance
//   Pass 3 — scan row: (x − μ) / √(σ² + ε) · γ + β
//
// Using double for the reduction passes avoids float accumulation error on
// large N (N=4096 at float ε ≈ 1.19e-7 → error ~4096·1.19e-7 ≈ 4.9e-4,
// which would exceed the kRelTol=1e-3 verification threshold at the limits).
// ─────────────────────────────────────────────────────────────────────────────
class LayerNormCpuReference final : public LayerNormWorkload {
public:
    void setup(const WorkloadConfig& cfg) override;
    void run()                            override;
    void teardown()                       override;
    bool verify()                         override;

    std::string implementation_name() const override { return "cpu_reference"; }
    Backend     backend()             const override { return Backend::CPU; }
};

} // namespace kernelx
