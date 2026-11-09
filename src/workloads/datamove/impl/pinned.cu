#include "workloads/datamove/impl/pinned.hpp"
#include "utils/cuda_utils.hpp"

#include <cstring>   // memcpy

namespace kernelx {

void DataMovePinned::setup(const WorkloadConfig& cfg) {
    init_host_data(cfg);    // fills h_buf_ with random data
    alloc_device();

    // Allocate page-locked host buffer and copy initialised data into it.
    // cudaMallocHost is not called in run() — pinning is expensive (OS must
    // wire physical pages) and must not be on the hot path.
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&h_pinned_),
        static_cast<size_t>(N_) * sizeof(float)));
    std::memcpy(h_pinned_, h_buf_.data(),
        static_cast<size_t>(N_) * sizeof(float));
}

void DataMovePinned::run() {
    // Direct DMA transfer: no staging copy required because h_pinned_ has a
    // stable physical address.  Approaches full PCIe unidirectional bandwidth.
    CUDA_CHECK(cudaMemcpy(d_buf_, h_pinned_,
        static_cast<size_t>(N_) * sizeof(float), cudaMemcpyHostToDevice));
}

bool DataMovePinned::verify() { return DataMoveWorkload::verify(); }

void DataMovePinned::teardown() {
    if (h_pinned_) { cudaFreeHost(h_pinned_); h_pinned_ = nullptr; }
    free_device();
    h_buf_.clear();
}

} // namespace kernelx
