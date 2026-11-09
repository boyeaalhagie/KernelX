#pragma once

#include "workloads/softmax/softmax_workload.hpp"

namespace kernelx {

// SoftmaxCpuReference — numerically stable two-pass CPU softmax.
//
// This is Version 0 in the softmax implementation progression.  Its purpose
// is to establish a correct ground truth and expose the algorithm's structure
// before any GPU optimisation.
//
// Algorithm (per row):
//   1. Find max(X[i][*])          — avoids exp overflow
//   2. Compute exp(X - max), sum  — numerically stable
//   3. Divide by sum              — normalise
//
// Three passes over the row: typical for simple implementations.
// Memory bandwidth: 2×N×4 bytes read + N×4 bytes write ≈ 12N bytes.
// Arithmetic intensity: 4N / 12N = 0.33 FLOP/byte — memory-bound on any GPU.

class SoftmaxCpuReference final : public SoftmaxWorkload {
public:
    void setup(const WorkloadConfig& cfg) override;
    void run()                            override;
    void teardown()                       override;
    bool verify()                         override;

    std::string implementation_name() const override { return "cpu_reference"; }
    Backend     backend()             const override { return Backend::CPU; }
};

} // namespace kernelx
