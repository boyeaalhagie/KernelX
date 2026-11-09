#include "workloads/attention/attention_workload.hpp"
#include "utils/cuda_utils.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>

namespace kernelx {

AttentionWorkload::~AttentionWorkload() {
    free_device();
}

float AttentionWorkload::scale() const {
    return 1.0f / std::sqrt(static_cast<float>(D_));
}

void AttentionWorkload::init_host_data(const WorkloadConfig& cfg) {
    M_ = cfg.m;   // sequence length N
    D_ = cfg.n;   // head dimension

    h_Q_.resize(static_cast<size_t>(M_) * D_);
    h_K_.resize(static_cast<size_t>(M_) * D_);
    h_V_.resize(static_cast<size_t>(M_) * D_);
    h_O_out_.resize(static_cast<size_t>(M_) * D_, 0.0f);

    // Fixed seed — reproducible across all implementations.
    // Values in [-1, 1] represent normalised activations (post layer-norm).
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (float& v : h_Q_) v = dist(rng);
    for (float& v : h_K_) v = dist(rng);
    for (float& v : h_V_) v = dist(rng);
}

void AttentionWorkload::compute_reference() {
    // Reference: standard O(N²) attention for the first kVerifyRows queries.
    // Uses a temporary N-element buffer for each query's attention row.
    const int rows = std::min(M_, kVerifyRows);
    const int N    = M_;
    const float sc = scale();

    h_O_ref_.assign(static_cast<size_t>(rows) * D_, 0.0f);

    std::vector<float> s_row(N);
    for (int i = 0; i < rows; ++i) {
        const float* q = h_Q_.data() + static_cast<size_t>(i) * D_;
        float*       o = h_O_ref_.data() + static_cast<size_t>(i) * D_;

        // S[i, :] = Q[i] · K^T · scale
        for (int j = 0; j < N; ++j) {
            const float* k = h_K_.data() + static_cast<size_t>(j) * D_;
            float dot = 0.0f;
            for (int d = 0; d < D_; ++d) dot += q[d] * k[d];
            s_row[j] = dot * sc;
        }

        // Softmax over s_row (numerically stable)
        float max_s = -std::numeric_limits<float>::infinity();
        for (int j = 0; j < N; ++j) max_s = std::max(max_s, s_row[j]);
        float sum = 0.0f;
        for (int j = 0; j < N; ++j) { s_row[j] = std::exp(s_row[j] - max_s); sum += s_row[j]; }
        for (int j = 0; j < N; ++j) s_row[j] /= sum;

        // O[i] = P[i] · V
        for (int j = 0; j < N; ++j) {
            const float* v = h_V_.data() + static_cast<size_t>(j) * D_;
            for (int d = 0; d < D_; ++d) o[d] += s_row[j] * v[d];
        }
    }
}

void AttentionWorkload::alloc_device() {
    const size_t nd = static_cast<size_t>(M_) * D_ * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_Q_, nd));
    CUDA_CHECK(cudaMalloc(&d_K_, nd));
    CUDA_CHECK(cudaMalloc(&d_V_, nd));
    CUDA_CHECK(cudaMalloc(&d_O_, nd));
}

void AttentionWorkload::upload_inputs() {
    const size_t nd = static_cast<size_t>(M_) * D_ * sizeof(float);
    CUDA_CHECK(cudaMemcpy(d_Q_, h_Q_.data(), nd, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K_, h_K_.data(), nd, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V_, h_V_.data(), nd, cudaMemcpyHostToDevice));
}

void AttentionWorkload::download_output() {
    CUDA_CHECK(cudaMemcpy(h_O_out_.data(), d_O_,
        static_cast<size_t>(M_) * D_ * sizeof(float), cudaMemcpyDeviceToHost));
}

void AttentionWorkload::free_device() {
    if (d_Q_) { cudaFree(d_Q_); d_Q_ = nullptr; }
    if (d_K_) { cudaFree(d_K_); d_K_ = nullptr; }
    if (d_V_) { cudaFree(d_V_); d_V_ = nullptr; }
    if (d_O_) { cudaFree(d_O_); d_O_ = nullptr; }
}

bool AttentionWorkload::verify() {
    const int rows = std::min(M_, kVerifyRows);
    for (int i = 0; i < rows; ++i) {
        const float* ref = h_O_ref_.data() + static_cast<size_t>(i) * D_;
        const float* out = h_O_out_.data() + static_cast<size_t>(i) * D_;
        for (int d = 0; d < D_; ++d) {
            const float denom = std::max(std::abs(ref[d]), 1e-5f);
            const float rel   = std::abs(out[d] - ref[d]) / denom;
            if (rel > kRelTol) {
                std::cerr << "  [verify] mismatch at row=" << i << " d=" << d
                          << "  ref=" << ref[d] << "  got=" << out[d]
                          << "  rel_err=" << rel << "\n";
                return false;
            }
        }
    }
    return true;
}

double AttentionWorkload::flops() const {
    // Dominant cost: two N×N×D matmuls (QK^T and PV), each 2·N²·D FLOPs.
    // Softmax (5·N²) is negligible for D=64 and is omitted for a clean metric.
    return 4.0 * static_cast<double>(M_) * M_ * D_;
}

} // namespace kernelx
