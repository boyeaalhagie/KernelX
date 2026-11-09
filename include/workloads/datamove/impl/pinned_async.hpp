#pragma once
#include "workloads/datamove/datamove_workload.hpp"
#include <cuda_runtime.h>

namespace kernelx {

// ─────────────────────────────────────────────────────────────────────────────
// DataMovePinnedAsync — H2D transfer using cudaMemcpyAsync on a CUDA stream.
//
// Key distinction from DataMovePinned:
//   cudaMemcpy    — blocks the CPU until the transfer completes.
//   cudaMemcpyAsync — enqueues the transfer on a stream and returns immediately;
//                     the CPU can proceed to launch kernels or other work while
//                     the DMA runs concurrently.
//
// When to use async:
//   - Pipelining: overlap the H2D transfer of batch N+1 with the kernel
//     computation on batch N (requires double buffering: two device buffers,
//     two pinned host buffers, two streams).
//   - Multi-stream: dispatch N independent transfers in parallel to saturate
//     the PCIe link in both directions simultaneously.
//
// Raw bandwidth:
//   cudaMemcpyAsync with a pinned buffer uses the same DMA path as the sync
//   version — PCIe bandwidth is identical.  The benefit is latency hiding, not
//   higher throughput for a single stream.
//
// Synchronisation in run():
//   cudaStreamSynchronize(stream_) is called at the end of run() so that the
//   BenchmarkRunner's CUDA event pair captures the full transfer time.
//   In a real pipeline, this sync would not appear inside the hot loop.
// ─────────────────────────────────────────────────────────────────────────────
class DataMovePinnedAsync final : public DataMoveWorkload {
public:
    void setup(const WorkloadConfig& cfg) override;
    void run()                            override;
    void teardown()                       override;
    bool verify()                         override;

    std::string implementation_name() const override { return "pinned_async"; }
    Backend     backend()             const override { return Backend::CUDA; }

private:
    float*        h_pinned_ = nullptr;   // cudaMallocHost'd host buffer
    cudaStream_t  stream_   = nullptr;   // dedicated transfer stream
};

} // namespace kernelx
