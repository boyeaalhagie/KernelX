#pragma once
#include "workloads/datamove/datamove_workload.hpp"

namespace kernelx {

// ─────────────────────────────────────────────────────────────────────────────
// DataMovePinned — H2D transfer using a pinned (page-locked) host buffer.
//
// cudaMallocHost allocates memory that is:
//   - Page-locked: the OS cannot swap it out, so its physical address is stable.
//   - Accessible by the DMA engine directly — no staging copy required.
//   - Mapped into the CUDA address space (accessible from device via
//     cudaHostGetDevicePointer, though we don't use that path here).
//
// Performance:
//   - Eliminates the CPU staging step of pageable transfers.
//   - Approaches the full PCIe unidirectional bandwidth (≈ 64 GB/s on H100).
//   - cudaMallocHost is slower than malloc (OS must wire physical pages) —
//     prefer allocating once at setup, not in run().
//
// Cost:
//   - Pinned memory is a scarce OS resource: too much degrades overall system
//     memory performance (reduces OS page cache, hurts other processes).
//   - Rule of thumb: pin only persistent transfer buffers, not temporary ones.
// ─────────────────────────────────────────────────────────────────────────────
class DataMovePinned final : public DataMoveWorkload {
public:
    void setup(const WorkloadConfig& cfg) override;
    void run()                            override;
    void teardown()                       override;
    bool verify()                         override;

    std::string implementation_name() const override { return "pinned"; }
    Backend     backend()             const override { return Backend::CUDA; }

private:
    float* h_pinned_ = nullptr;   // cudaMallocHost'd host buffer
};

} // namespace kernelx
