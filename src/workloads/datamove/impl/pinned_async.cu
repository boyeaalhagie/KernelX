#include "workloads/datamove/impl/pinned_async.hpp"
#include "utils/cuda_utils.hpp"

#include <cstring>   // memcpy

namespace kernelx {

void DataMovePinnedAsync::setup(const WorkloadConfig& cfg) {
    init_host_data(cfg);
    alloc_device();

    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&h_pinned_),
        static_cast<size_t>(N_) * sizeof(float)));
    std::memcpy(h_pinned_, h_buf_.data(),
        static_cast<size_t>(N_) * sizeof(float));

    // Create a dedicated transfer stream.
    // Using the default stream (0) would work but would serialize with any
    // kernels launched on the default stream — a dedicated stream allows
    // the DMA to overlap with compute on other streams.
    CUDA_CHECK(cudaStreamCreate(&stream_));
}

void DataMovePinnedAsync::run() {
    // Enqueue the H2D transfer on stream_ — returns to CPU immediately.
    // The DMA runs concurrently with subsequent CPU code or kernels on other streams.
    CUDA_CHECK(cudaMemcpyAsync(d_buf_, h_pinned_,
        static_cast<size_t>(N_) * sizeof(float),
        cudaMemcpyHostToDevice, stream_));

    // Sync here so the BenchmarkRunner's CUDA-event stop captures full latency.
    // In a real double-buffered pipeline this sync would not appear in the loop:
    // the CPU would instead launch the next batch's kernel while this DMA runs.
    CUDA_CHECK(cudaStreamSynchronize(stream_));
}

bool DataMovePinnedAsync::verify() { return DataMoveWorkload::verify(); }

void DataMovePinnedAsync::teardown() {
    if (stream_)   { cudaStreamDestroy(stream_);    stream_   = nullptr; }
    if (h_pinned_) { cudaFreeHost(h_pinned_);       h_pinned_ = nullptr; }
    free_device();
    h_buf_.clear();
}

} // namespace kernelx
