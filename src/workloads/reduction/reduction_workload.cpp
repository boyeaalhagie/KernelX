#include "workloads/reduction/reduction_workload.hpp"
#include "utils/cuda_utils.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>

namespace kernelx {

ReductionWorkload::~ReductionWorkload() {
    free_device();
}

void ReductionWorkload::init_host_data(const WorkloadConfig& cfg) {
    M_ = cfg.m;
    N_ = cfg.n;

    h_X_.resize(static_cast<size_t>(M_) * N_);
    h_Y_out_.resize(static_cast<size_t>(M_), 0.0f);

    // Fixed seed — reproducible across all implementations.
    // Values in [-1, 1] keep partial sums well within fp32 range even at N=131072.
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (float& v : h_X_) v = dist(rng);
}

void ReductionWorkload::compute_reference() {
    const int rows = std::min(M_, kVerifyRows);
    h_Y_ref_.resize(static_cast<size_t>(rows));

    // Double-precision accumulation avoids catastrophic cancellation for large N.
    for (int i = 0; i < rows; ++i) {
        const float* x = h_X_.data() + static_cast<size_t>(i) * N_;
        double sum = 0.0;
        for (int j = 0; j < N_; ++j) sum += x[j];
        h_Y_ref_[i] = static_cast<float>(sum);
    }
}

void ReductionWorkload::alloc_device() {
    CUDA_CHECK(cudaMalloc(&d_X_, static_cast<size_t>(M_) * N_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Y_, static_cast<size_t>(M_)      * sizeof(float)));
}

void ReductionWorkload::upload_inputs() {
    CUDA_CHECK(cudaMemcpy(d_X_, h_X_.data(),
        static_cast<size_t>(M_) * N_ * sizeof(float), cudaMemcpyHostToDevice));
}

void ReductionWorkload::download_output() {
    CUDA_CHECK(cudaMemcpy(h_Y_out_.data(), d_Y_,
        static_cast<size_t>(M_) * sizeof(float), cudaMemcpyDeviceToHost));
}

void ReductionWorkload::free_device() {
    if (d_X_) { cudaFree(d_X_); d_X_ = nullptr; }
    if (d_Y_) { cudaFree(d_Y_); d_Y_ = nullptr; }
}

bool ReductionWorkload::verify() {
    const int rows = std::min(M_, kVerifyRows);

    for (int i = 0; i < rows; ++i) {
        const float ref = h_Y_ref_[i];
        const float got = h_Y_out_[i];
        // Use absolute tolerance scaled by N — fp32 sum of N random values in
        // [-1,1] has rounding error ≈ sqrt(N) · eps_fp32 ≈ sqrt(N) · 1.2e-7.
        // kRelTol = 1e-3 is well above that for all tested N.
        const float denom = std::max(std::abs(ref), 1.0f);
        const float rel   = std::abs(got - ref) / denom;
        if (rel > kRelTol) {
            std::cerr << "  [verify] mismatch at row=" << i
                      << "  ref=" << ref << "  got=" << got
                      << "  rel_err=" << rel << "\n";
            return false;
        }
    }
    return true;
}

double ReductionWorkload::flops() const {
    // N additions per row (N − 1 strictly, but N is the conventional count).
    return static_cast<double>(M_) * N_;
}

} // namespace kernelx
