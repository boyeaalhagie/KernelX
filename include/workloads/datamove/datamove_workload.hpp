#pragma once

#include "workloads/workload.hpp"
#include <vector>

namespace kernelx {

// ─────────────────────────────────────────────────────────────────────────────
// DataMoveWorkload — shared base for host↔GPU transfer bandwidth benchmarks.
//
// Operation: copy N floats from host to device (H2D).
//   Each call to run() performs one H2D transfer of N floats.
//   verify() issues a D2H copy and compares against the original host buffer.
//
// WorkloadConfig mapping:
//   cfg.n = N   (number of float elements; transfer size = N × 4 bytes)
//   cfg.m, cfg.k — unused; stored as 1 in ExperimentRecord
//
// Implementations:
//   pageable    — host buffer allocated with new/std::vector (pageable DRAM).
//                 CUDA internally stages via a temporary pinned buffer; this
//                 staging adds one extra CPU→pinned copy before the DMA,
//                 halving effective bandwidth vs a pinned transfer.
//   pinned      — host buffer allocated with cudaMallocHost (page-locked DRAM).
//                 The DMA engine transfers directly — no staging, peak PCIe BW.
//   pinned_async — cudaMallocHost + cudaMemcpyAsync on a dedicated stream.
//                 Same PCIe bandwidth as pinned, but the API call is
//                 non-blocking: the CPU can issue compute work while DMA runs.
//                 run() calls cudaStreamSynchronize so the runner's CUDA-event
//                 timing captures the full transfer latency.
//
// Throughput display:
//   flops() returns N × sizeof(float) × 1000, so the "TFLOP/s" column in the
//   bench output reads directly as GB/s (1 TFLOP/s ≡ 1 GB/s here).
//   Peak H100 PCIe 5.0 × 16: ~64 GB/s unidirectional.
//
// Verification: single-precision exact equality — no rounding involved in a
//   memory copy, so tolerance is exactly zero.
// ─────────────────────────────────────────────────────────────────────────────
class DataMoveWorkload : public Workload {
public:
    ~DataMoveWorkload() override;

    bool   verify() override;
    double flops()  const override;
    std::string workload_name() const override { return "datamove"; }

protected:
    int N_ = 0;

    // Pageable host source buffer — always populated for data init / verify ref
    std::vector<float> h_buf_;

    // Device destination buffer — non-null for CUDA implementations
    float* d_buf_ = nullptr;

    // Setup helpers
    void init_host_data(const WorkloadConfig& cfg);
    void alloc_device();
    void free_device();
    // Download d_buf_ → h_check_ for verify()
    void download_check();

private:
    std::vector<float> h_check_;   // scratch for verify download
};

} // namespace kernelx
