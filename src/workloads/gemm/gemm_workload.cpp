#include "workloads/gemm/gemm_workload.hpp"
#include "utils/cuda_utils.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>

namespace kernelx {

GemmWorkload::~GemmWorkload() {
    free_device();
}

void GemmWorkload::init_host_data(const WorkloadConfig& cfg) {
    M_     = cfg.m;
    N_     = cfg.n;
    K_     = cfg.k;
    dtype_ = cfg.dtype;

    h_A_.resize(static_cast<size_t>(M_) * K_);
    h_B_.resize(static_cast<size_t>(K_) * N_);
    h_C_out_.resize(static_cast<size_t>(M_) * N_, 0.0f);

    // Fixed seed — results must be reproducible across runs and implementations
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    for (float& v : h_A_) v = dist(rng);
    for (float& v : h_B_) v = dist(rng);
}

void GemmWorkload::compute_reference() {
    // We only verify the top-left kVerifyDim × kVerifyDim corner of C.
    // For production sizes (4096³) a full CPU reference would take ~30 minutes
    // on a single core.  The corner is large enough to catch indexing and
    // accumulation bugs in any practical implementation.
    const int rows = std::min(M_, kVerifyDim);
    const int cols = std::min(N_, kVerifyDim);
    h_C_ref_.assign(static_cast<size_t>(rows) * cols, 0.0f);

    for (int i = 0; i < rows; ++i) {
        for (int k = 0; k < K_; ++k) {
            const float a_ik = h_A_[i * K_ + k];
            for (int j = 0; j < cols; ++j) {
                h_C_ref_[i * cols + j] += a_ik * h_B_[k * N_ + j];
            }
        }
    }
}

void GemmWorkload::alloc_device() {
    CUDA_CHECK(cudaMalloc(&d_A_, static_cast<size_t>(M_) * K_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B_, static_cast<size_t>(K_) * N_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C_, static_cast<size_t>(M_) * N_ * sizeof(float)));
}

void GemmWorkload::upload_inputs() {
    CUDA_CHECK(cudaMemcpy(d_A_, h_A_.data(),
        static_cast<size_t>(M_) * K_ * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B_, h_B_.data(),
        static_cast<size_t>(K_) * N_ * sizeof(float), cudaMemcpyHostToDevice));
}

void GemmWorkload::download_output() {
    CUDA_CHECK(cudaMemcpy(h_C_out_.data(), d_C_,
        static_cast<size_t>(M_) * N_ * sizeof(float), cudaMemcpyDeviceToHost));
}

void GemmWorkload::free_device() {
    // Null-guard prevents double-free if teardown() and ~GemmWorkload() both run
    if (d_A_) { cudaFree(d_A_); d_A_ = nullptr; }
    if (d_B_) { cudaFree(d_B_); d_B_ = nullptr; }
    if (d_C_) { cudaFree(d_C_); d_C_ = nullptr; }
}

bool GemmWorkload::verify() {
    const int rows = std::min(M_, kVerifyDim);
    const int cols = std::min(N_, kVerifyDim);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            const float ref = h_C_ref_[i * cols + j];
            const float out = h_C_out_[i * N_ + j];
            // Relative error with absolute floor to handle near-zero values
            const float denom = std::max(std::abs(ref), 1e-4f);
            const float rel   = std::abs(out - ref) / denom;
            if (rel > kRelTol) {
                std::cerr << "  [verify] mismatch at (" << i << "," << j << ")"
                          << "  ref=" << ref << "  got=" << out
                          << "  rel_err=" << rel << "\n";
                return false;
            }
        }
    }
    return true;
}

double GemmWorkload::flops() const {
    // Each output element requires K multiply-adds: 2·M·N·K FLOPs total
    return 2.0 * static_cast<double>(M_) * N_ * K_;
}

} // namespace kernelx
