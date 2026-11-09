#pragma once
#include "workloads/attention/attention_workload.hpp"
namespace kernelx {

// AttentionCpuReference — standard O(N²) three-pass CPU attention.
// Version 0: establishes correctness ground truth.
// Algorithm: (1) S = QK^T/√D [N×N]  (2) P = softmax(S)  (3) O = PV

class AttentionCpuReference final : public AttentionWorkload {
public:
    void setup(const WorkloadConfig& cfg) override;
    void run()                            override;
    void teardown()                       override;
    bool verify()                         override;
    std::string implementation_name() const override { return "cpu_reference"; }
    Backend     backend()             const override { return Backend::CPU; }
};

} // namespace kernelx
