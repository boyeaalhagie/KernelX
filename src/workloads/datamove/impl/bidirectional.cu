#include "workloads/datamove/impl/bidirectional.hpp"
#include "utils/cuda_utils.hpp"

#include <cstring>
#include <iostream>

namespace kernelx {

void DataMoveBidirectional::setup(const WorkloadConfig& cfg) {
    init_host_data(cfg);   // fills h_buf_ with random data (used as source reference)
    const size_t bytes = static_cast<size_t>(N_) * sizeof(float);

    // ── H2D side: pinned source ───────────────────────────────────────────────
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&h_src_), bytes));
    std::memcpy(h_src_, h_buf_.data(), bytes);
    CUDA_CHECK(cudaMalloc(&d_src_, bytes));

    // ── D2H side: pre-fill d_dst_ with a known pattern, receive into h_dst_ ──
    // Use a different seed so d_dst_ data is distinct from h_src_ data.
    h_d_dst_ref_.resize(static_cast<size_t>(N_));
    for (int i = 0; i < N_; ++i)
        h_d_dst_ref_[i] = static_cast<float>(i % 1024) * 0.001f;

    CUDA_CHECK(cudaMalloc(&d_dst_, bytes));
    CUDA_CHECK(cudaMemcpy(d_dst_, h_d_dst_ref_.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&h_dst_), bytes));

    // ── Two independent streams ───────────────────────────────────────────────
    CUDA_CHECK(cudaStreamCreate(&stream0_));
    CUDA_CHECK(cudaStreamCreate(&stream1_));
}

void DataMoveBidirectional::run() {
    const size_t bytes = static_cast<size_t>(N_) * sizeof(float);

    // Launch both transfers concurrently — PCIe full-duplex.
    // No dependency between stream0_ and stream1_: the hardware can schedule
    // both DMA engines simultaneously (H2D upstream + D2H downstream).
    CUDA_CHECK(cudaMemcpyAsync(d_src_, h_src_, bytes, cudaMemcpyHostToDevice, stream0_));
    CUDA_CHECK(cudaMemcpyAsync(h_dst_, d_dst_, bytes, cudaMemcpyDeviceToHost, stream1_));

    // Wait for both transfers.  cudaDeviceSynchronize is broader than needed
    // but avoids stream ordering complexity; both streams finish before return.
    CUDA_CHECK(cudaDeviceSynchronize());
}

bool DataMoveBidirectional::verify() {
    const size_t bytes = static_cast<size_t>(N_) * sizeof(float);

    // Check H2D: d_src_ should now equal h_src_
    std::vector<float> d_src_check(N_);
    CUDA_CHECK(cudaMemcpy(d_src_check.data(), d_src_, bytes, cudaMemcpyDeviceToHost));
    for (int i = 0; i < N_; ++i) {
        if (d_src_check[i] != h_src_[i]) {
            std::cerr << "  [verify bidir H2D] mismatch at i=" << i
                      << "  expected=" << h_src_[i]
                      << "  got=" << d_src_check[i] << "\n";
            return false;
        }
    }

    // Check D2H: h_dst_ should now equal h_d_dst_ref_
    for (int i = 0; i < N_; ++i) {
        if (h_dst_[i] != h_d_dst_ref_[i]) {
            std::cerr << "  [verify bidir D2H] mismatch at i=" << i
                      << "  expected=" << h_d_dst_ref_[i]
                      << "  got=" << h_dst_[i] << "\n";
            return false;
        }
    }
    return true;
}

double DataMoveBidirectional::flops() const {
    // Both directions: 2 × N × 4 bytes.  ×1000 so "TFLOP/s" reads as total GB/s.
    return 2.0 * static_cast<double>(N_) * sizeof(float) * 1000.0;
}

void DataMoveBidirectional::teardown() {
    if (stream0_) { cudaStreamDestroy(stream0_); stream0_ = nullptr; }
    if (stream1_) { cudaStreamDestroy(stream1_); stream1_ = nullptr; }
    if (h_src_)   { cudaFreeHost(h_src_);        h_src_   = nullptr; }
    if (h_dst_)   { cudaFreeHost(h_dst_);        h_dst_   = nullptr; }
    if (d_src_)   { cudaFree(d_src_);            d_src_   = nullptr; }
    if (d_dst_)   { cudaFree(d_dst_);            d_dst_   = nullptr; }
    h_buf_.clear();
    h_d_dst_ref_.clear();
}

} // namespace kernelx
