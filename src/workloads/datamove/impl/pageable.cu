#include "workloads/datamove/impl/pageable.hpp"
#include "utils/cuda_utils.hpp"

namespace kernelx {

void DataMovePageable::setup(const WorkloadConfig& cfg) {
    init_host_data(cfg);    // fills h_buf_ (pageable std::vector)
    alloc_device();         // cudaMalloc d_buf_
}

void DataMovePageable::run() {
    // cudaMemcpy with a pageable source internally stages through a pinned
    // bounce buffer allocated by the CUDA runtime.  The CPU blocks until
    // the full transfer completes (staging copy + DMA).
    CUDA_CHECK(cudaMemcpy(d_buf_, h_buf_.data(),
        static_cast<size_t>(N_) * sizeof(float), cudaMemcpyHostToDevice));
}

bool DataMovePageable::verify() { return DataMoveWorkload::verify(); }

void DataMovePageable::teardown() {
    free_device();
    h_buf_.clear();
}

} // namespace kernelx
