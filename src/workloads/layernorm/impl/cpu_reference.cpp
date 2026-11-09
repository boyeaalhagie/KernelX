#include "workloads/layernorm/impl/cpu_reference.hpp"

#include <cmath>

namespace kernelx {

void LayerNormCpuReference::setup(const WorkloadConfig& cfg) {
    init_host_data(cfg);
    compute_reference();
}

void LayerNormCpuReference::run() {
    for (int i = 0; i < M_; ++i) {
        const float* x = h_X_.data()     + static_cast<size_t>(i) * N_;
        float*       y = h_Y_out_.data() + static_cast<size_t>(i) * N_;

        // Pass 1: mean (double accumulation for numerical stability)
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

        // Pass 3: normalise, scale, bias
        for (int j = 0; j < N_; ++j)
            y[j] = h_gamma_[j] * (x[j] - mean) * inv_std + h_beta_[j];
    }
}

bool LayerNormCpuReference::verify() {
    return LayerNormWorkload::verify();
}

void LayerNormCpuReference::teardown() {
    h_X_.clear(); h_gamma_.clear(); h_beta_.clear();
    h_Y_ref_.clear(); h_Y_out_.clear();
}

} // namespace kernelx
