#pragma once

#include "workloads/gemm/gemm_workload.hpp"

namespace kernelx {

// ─────────────────────────────────────────────────────────────────────────────
// GemmCpuReference — scalar triple-loop GEMM on the host.
//
// Purpose: correctness baseline.  Not a performance target.
//
// The inner loop is ordered i→k→j (rather than i→j→k) to maximise
// spatial locality on A[i][k] and B[k][j] in the row-major layout:
//   A row stride = K  →  A[i][k] is stride-1 in k
//   B row stride = N  →  B[k][j] is stride-1 in j for fixed k
// This ordering keeps both accesses sequential during the j-loop, making
// it cache-friendly for moderate K without any SIMD or threading.
// ─────────────────────────────────────────────────────────────────────────────
class GemmCpuReference final : public GemmWorkload {
public:
    void setup(const WorkloadConfig& cfg) override;
    void run()                            override;
    void teardown()                       override;
    bool verify()                         override { return true; }  // IS the reference

    std::string implementation_name() const override { return "cpu_reference"; }
    Backend     backend()             const override { return Backend::CPU; }
};

} // namespace kernelx
