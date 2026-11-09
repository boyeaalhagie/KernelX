#include "workloads/softmax/softmax_workload.hpp"
#include "utils/cuda_utils.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>

namespace kernelx {

SoftmaxWorkload::~SoftmaxWorkload() {
    free_device();
}

void SoftmaxWorkload::init_host_data(const WorkloadConfig& cfg) {
    M_ = cfg.m;
    N_ = cfg.n;

    h_X_.resize(static_cast<size_t>(M_) * N_);
    h_Y_out_.resize(static_cast<size_t>(M_) * N_, 0.0f);

    // Fixed seed — reproducible across runs and implementations.
    // Values in [-3, 3] produce a mix of large and small softmax weights,
    // exercising both the numerically safe and numerically challenging paths.
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-3.0f, 3.0f);
    for (float& v : h_X_) v = dist(rng);
}

void SoftmaxWorkload::compute_reference() {
    // Compute reference output for the first kVerifyRows rows only.
    // For large N (e.g. 131072), computing all M rows on CPU would be slow.
    const int rows = std::min(M_, kVerifyRows);
    h_Y_ref_.resize(static_cast<size_t>(rows) * N_);

    for (int i = 0; i < rows; ++i) {
        const float* x  = h_X_.data() + static_cast<size_t>(i) * N_;
        float*       yr = h_Y_ref_.data() + static_cast<size_t>(i) * N_;

        // Pass 1: numerically stable max
        float max_val = -std::numeric_limits<float>::infinity();
        for (int j = 0; j < N_; ++j)
            max_val = std::max(max_val, x[j]);

        // Pass 2: exp(x - max) and accumulate sum
        float sum = 0.0f;
        for (int j = 0; j < N_; ++j) {
            yr[j] = std::exp(x[j] - max_val);
            sum += yr[j];
        }

        // Pass 3: normalise
        const float inv_sum = 1.0f / sum;
        for (int j = 0; j < N_; ++j)
            yr[j] *= inv_sum;
    }
}

void SoftmaxWorkload::alloc_device() {
    CUDA_CHECK(cudaMalloc(&d_X_, static_cast<size_t>(M_) * N_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Y_, static_cast<size_t>(M_) * N_ * sizeof(float)));
}

void SoftmaxWorkload::upload_inputs() {
    CUDA_CHECK(cudaMemcpy(d_X_, h_X_.data(),
        static_cast<size_t>(M_) * N_ * sizeof(float), cudaMemcpyHostToDevice));
}

void SoftmaxWorkload::download_output() {
    CUDA_CHECK(cudaMemcpy(h_Y_out_.data(), d_Y_,
        static_cast<size_t>(M_) * N_ * sizeof(float), cudaMemcpyDeviceToHost));
}

void SoftmaxWorkload::free_device() {
    if (d_X_) { cudaFree(d_X_); d_X_ = nullptr; }
    if (d_Y_) { cudaFree(d_Y_); d_Y_ = nullptr; }
}

bool SoftmaxWorkload::verify() {
    const int rows = std::min(M_, kVerifyRows);

    for (int i = 0; i < rows; ++i) {
        const float* ref = h_Y_ref_.data() + static_cast<size_t>(i) * N_;
        const float* out = h_Y_out_.data() + static_cast<size_t>(i) * N_;
        for (int j = 0; j < N_; ++j) {
            const float denom = std::max(std::abs(ref[j]), 1e-7f);
            const float rel   = std::abs(out[j] - ref[j]) / denom;
            if (rel > kRelTol) {
                std::cerr << "  [verify] mismatch at row=" << i << " col=" << j
                          << "  ref=" << ref[j] << "  got=" << out[j]
                          << "  rel_err=" << rel << "\n";
                return false;
            }
        }
    }
    return true;
}

double SoftmaxWorkload::flops() const {
    // 4 FLOPs per element: subtract (shift by max), exp, add-to-sum, divide.
    // exp is counted as 1 FLOP (maps to hardware __expf with --use_fast_math).
    return 4.0 * static_cast<double>(M_) * N_;
}

} // namespace kernelx
