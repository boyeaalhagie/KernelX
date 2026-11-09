#pragma once
#include "workloads/datamove/datamove_workload.hpp"

namespace kernelx {

// ─────────────────────────────────────────────────────────────────────────────
// DataMovePageable — H2D transfer using a pageable (malloc'd) host buffer.
//
// CUDA's cudaMemcpy with a pageable source must first stage the data through
// an internal pinned bounce buffer before the DMA engine can initiate the
// PCIe transfer.  This two-step process:
//   1. CPU copies pageable → internal pinned staging buffer
//   2. DMA engine copies pinned staging → device
//
// halves the effective bandwidth vs a direct pinned transfer:
//   - Pageable:  ≈ min(CPU copy BW, PCIe BW / 2) — staging bottleneck
//   - Pinned:    ≈ PCIe BW (direct DMA, no staging)
//
// This is the baseline implementation: easy to write but not optimal.
// ─────────────────────────────────────────────────────────────────────────────
class DataMovePageable final : public DataMoveWorkload {
public:
    void setup(const WorkloadConfig& cfg) override;
    void run()                            override;
    void teardown()                       override;
    bool verify()                         override;

    std::string implementation_name() const override { return "pageable"; }
    Backend     backend()             const override { return Backend::CUDA; }
};

} // namespace kernelx
