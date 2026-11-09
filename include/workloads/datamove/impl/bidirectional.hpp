#pragma once
#include "workloads/datamove/datamove_workload.hpp"

namespace kernelx {

// ─────────────────────────────────────────────────────────────────────────────
// DataMoveBidirectional — simultaneous H2D + D2H on two independent streams.
//
// Motivation:
//   PCIe links are full-duplex: the upstream (H2D) and downstream (D2H) lanes
//   operate independently.  A single cudaMemcpy can saturate only one direction.
//   Two concurrent streams — one H2D, one D2H — can saturate both directions
//   simultaneously, approaching 2× the unidirectional bandwidth in total bytes
//   moved per second.
//
// Implementation:
//   stream0_: H2D   h_src_ → d_src_   (N floats = N×4 bytes H2D)
//   stream1_: D2H   d_dst_ → h_dst_   (N floats = N×4 bytes D2H)
//   Both copies are launched without synchronisation between them; each stream
//   runs its DMA engine independently.  cudaDeviceSynchronize() after both
//   launches ensures run() is synchronous for the benchmark timer.
//
// flops() = 2 × N × sizeof(float) × 1000  (both directions combined)
//   → "TFLOP/s" reads as total GB/s (H2D + D2H summed).
//   At full-duplex saturation: ~110–120 GB/s on PCIe 5.0 × 16 (2 × ~55 GB/s).
//
// Verification:
//   After run(), d_dst_ should contain the data that was in d_dst_ before run()
//   (which was initialised to a known pattern at setup).  verify() downloads
//   d_src_ (should match h_src_) and h_dst_ (should match the pre-run d_dst_
//   pattern).  Both checks confirm both directions transferred correctly.
// ─────────────────────────────────────────────────────────────────────────────
class DataMoveBidirectional final : public DataMoveWorkload {
public:
    void setup(const WorkloadConfig& cfg) override;
    void run()                            override;
    void teardown()                       override;
    bool verify()                         override;

    double flops() const override;   // 2× base: counts both directions
    std::string implementation_name() const override { return "bidirectional"; }
    Backend     backend()             const override { return Backend::CUDA; }

private:
    // Source: H2D transfer (h_src_ → d_src_)
    float*        h_src_    = nullptr;   // cudaMallocHost
    float*        d_src_    = nullptr;   // separate from base d_buf_

    // Destination: D2H transfer (d_dst_ → h_dst_)
    float*        d_dst_    = nullptr;
    float*        h_dst_    = nullptr;   // cudaMallocHost, receives D2H data

    cudaStream_t  stream0_  = nullptr;   // H2D stream
    cudaStream_t  stream1_  = nullptr;   // D2H stream

    std::vector<float> h_d_dst_ref_;     // CPU copy of d_dst_ initial data for verify
};

} // namespace kernelx
