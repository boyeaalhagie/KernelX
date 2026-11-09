#pragma once
#include "workloads/reduction/reduction_workload.hpp"

namespace kernelx {

// Sequential row-sum on the CPU — single-threaded double-precision accumulation.
// Provides the correctness baseline; also sets an upper bound on how slow
// non-parallelised code is for this memory-bound operation.
class ReductionCpuReference final : public ReductionWorkload {
public:
    void setup(const WorkloadConfig& cfg) override;
    void run()                            override;
    void teardown()                       override;
    bool verify()                         override;

    std::string implementation_name() const override { return "cpu_reference"; }
    Backend     backend()             const override { return Backend::CPU; }
};

} // namespace kernelx
