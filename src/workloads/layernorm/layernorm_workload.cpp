#include "workloads/layernorm/layernorm_workload.hpp"
#include "utils/cuda_utils.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>

namespace kernelx {

LayerNormWorkload::~LayerNormWorkload() {
    free_device();
}

void LayerNormWorkload::init_host_data(const WorkloadConfig& cfg) {
    M_ = cfg.m;
    N_ = cfg.n;

    h_X_.resize(static_cast<size_t>(M_) * N_);
    h_Y_out_.resize(static_cast<size_t>(M_) * N_, 0.0f);
    h_gamma_.resize(N_);
    h_beta_.resize(N_);

    // Fixed seed — reproducible across all implementations.
    std::mt19937 rng(42);
    // Input: uniform in [-3, 3] — representative of pre-normalisation activations
    std::uniform_real_distribution<float> x_dist(-3.0f, 3.0f);
    for (float& v : h_X_) v = x_dist(rng);

    // γ in [0.5, 1.5], β in [-0.1, 0.1] — non-trivial but near-identity transform
    std::uniform_real_distribution<float> g_dist(0.5f, 1.5f);
    std::uniform_real_distribution<float> b_dist(-0.1f, 0.1f);
    for (float& v : h_gamma_) v = g_dist(rng);
    for (float& v : h_beta_)  v = b_dist(rng);
}

void LayerNormWorkload::compute_reference() {
    const int rows = std::min(M_, kVerifyRows);
    h_Y_ref_.resize(static_cast<size_t>(rows) * N_);

    for (int i = 0; i < rows; ++i) {
        const float* x  = h_X_.data()     + static_cast<size_t>(i) * N_;
        float*       yr = h_Y_ref_.data() + static_cast<size_t>(i) * N_;

        // Pass 1: mean
        double sum = 0.0;
        for (int j = 0; j < N_; ++j) sum += x[j];
        const float mean = static_cast<float>(sum / N_);

        // Pass 2: variance
        double var_sum = 0.0;
        for (int j = 0; j < N_; ++j) {
            const double d = x[j] - mean;
            var_sum += d * d;
        }
        const float inv_std = 1.0f / std::sqrt(static_cast<float>(var_sum / N_) + kEpsilon);

        // Pass 3: normalise + scale + bias
        for (int j = 0; j < N_; ++j)
            yr[j] = h_gamma_[j] * (x[j] - mean) * inv_std + h_beta_[j];
    }
}

void LayerNormWorkload::alloc_device() {
    CUDA_CHECK(cudaMalloc(&d_X_,     static_cast<size_t>(M_) * N_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gamma_, static_cast<size_t>(N_)       * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_beta_,  static_cast<size_t>(N_)       * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Y_,     static_cast<size_t>(M_) * N_ * sizeof(float)));
}

void LayerNormWorkload::upload_inputs() {
    CUDA_CHECK(cudaMemcpy(d_X_, h_X_.data(),
        static_cast<size_t>(M_) * N_ * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gamma_, h_gamma_.data(),
        static_cast<size_t>(N_) * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_beta_, h_beta_.data(),
        static_cast<size_t>(N_) * sizeof(float), cudaMemcpyHostToDevice));
}

void LayerNormWorkload::download_output() {
    CUDA_CHECK(cudaMemcpy(h_Y_out_.data(), d_Y_,
        static_cast<size_t>(M_) * N_ * sizeof(float), cudaMemcpyDeviceToHost));
}

void LayerNormWorkload::free_device() {
    if (d_X_)     { cudaFree(d_X_);     d_X_     = nullptr; }
    if (d_gamma_) { cudaFree(d_gamma_); d_gamma_ = nullptr; }
    if (d_beta_)  { cudaFree(d_beta_);  d_beta_  = nullptr; }
    if (d_Y_)     { cudaFree(d_Y_);     d_Y_     = nullptr; }
}

bool LayerNormWorkload::verify() {
    const int rows = std::min(M_, kVerifyRows);

    for (int i = 0; i < rows; ++i) {
        const float* ref = h_Y_ref_.data() + static_cast<size_t>(i) * N_;
        const float* out = h_Y_out_.data() + static_cast<size_t>(i) * N_;
        for (int j = 0; j < N_; ++j) {
            const float denom = std::max(std::abs(ref[j]), 1e-5f);
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

double LayerNormWorkload::flops() const {
    // Per element: mean (1) + variance (3) + normalise+scale+bias (4) = 8 FLOPs
    // rsqrtf counts as 1 FLOP (hardware instruction with --use_fast_math).
    return 8.0 * static_cast<double>(M_) * N_;
}

} // namespace kernelx
