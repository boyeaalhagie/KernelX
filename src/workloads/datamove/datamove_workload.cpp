#include "workloads/datamove/datamove_workload.hpp"
#include "utils/cuda_utils.hpp"

#include <iostream>
#include <random>

namespace kernelx {

DataMoveWorkload::~DataMoveWorkload() {
    free_device();
}

void DataMoveWorkload::init_host_data(const WorkloadConfig& cfg) {
    N_ = cfg.n;
    h_buf_.resize(static_cast<size_t>(N_));

    // Fixed seed — consistent data across all implementations.
    // Uniform in [-1, 1]: no special values that might mask transfer errors.
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (float& v : h_buf_) v = dist(rng);
}

void DataMoveWorkload::alloc_device() {
    CUDA_CHECK(cudaMalloc(&d_buf_, static_cast<size_t>(N_) * sizeof(float)));
}

void DataMoveWorkload::free_device() {
    if (d_buf_) { cudaFree(d_buf_); d_buf_ = nullptr; }
}

void DataMoveWorkload::download_check() {
    h_check_.resize(static_cast<size_t>(N_));
    CUDA_CHECK(cudaMemcpy(h_check_.data(), d_buf_,
        static_cast<size_t>(N_) * sizeof(float), cudaMemcpyDeviceToHost));
}

bool DataMoveWorkload::verify() {
    download_check();
    for (int i = 0; i < N_; ++i) {
        if (h_check_[i] != h_buf_[i]) {
            std::cerr << "  [verify] mismatch at index=" << i
                      << "  expected=" << h_buf_[i]
                      << "  got="      << h_check_[i] << "\n";
            return false;
        }
    }
    return true;
}

double DataMoveWorkload::flops() const {
    // "flops" is repurposed here to represent bytes transferred × 1000.
    // Result: "TFLOP/s" in the bench output reads directly as GB/s.
    //   displayed = (N * 4 * 1000) / (time_s * 1e12)
    //             = (N * 4) / (time_s * 1e9)
    //             = GB/s
    // Peak H100 PCIe 5.0 ×16 unidirectional ≈ 64 GB/s.
    return static_cast<double>(N_) * sizeof(float) * 1000.0;
}

} // namespace kernelx
