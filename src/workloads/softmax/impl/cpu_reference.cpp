#include "workloads/softmax/impl/cpu_reference.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace kernelx {

void SoftmaxCpuReference::setup(const WorkloadConfig& cfg) {
    init_host_data(cfg);
    compute_reference();   // h_Y_ref_ is both reference and the output we verify
}

void SoftmaxCpuReference::run() {
    for (int i = 0; i < M_; ++i) {
        const float* x = h_X_.data() + static_cast<size_t>(i) * N_;
        float*       y = h_Y_out_.data() + static_cast<size_t>(i) * N_;

        // Pass 1: max for numerical stability
        float max_val = -std::numeric_limits<float>::infinity();
        for (int j = 0; j < N_; ++j)
            max_val = std::max(max_val, x[j]);

        // Pass 2: exp(x - max) and sum
        float sum = 0.0f;
        for (int j = 0; j < N_; ++j) {
            y[j] = std::exp(x[j] - max_val);
            sum += y[j];
        }

        // Pass 3: normalise
        const float inv_sum = 1.0f / sum;
        for (int j = 0; j < N_; ++j)
            y[j] *= inv_sum;
    }
}

bool SoftmaxCpuReference::verify() {
    return SoftmaxWorkload::verify();
}

void SoftmaxCpuReference::teardown() {
    h_X_.clear();
    h_Y_ref_.clear();
    h_Y_out_.clear();
}

} // namespace kernelx
