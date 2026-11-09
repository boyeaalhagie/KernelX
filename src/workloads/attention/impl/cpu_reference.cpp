#include "workloads/attention/impl/cpu_reference.hpp"
#include <cmath>
#include <limits>
#include <vector>

namespace kernelx {

void AttentionCpuReference::setup(const WorkloadConfig& cfg) {
    init_host_data(cfg);
    compute_reference();
}

void AttentionCpuReference::run() {
    const int   N  = M_;
    const float sc = scale();
    std::vector<float> s_row(N);

    for (int i = 0; i < N; ++i) {
        const float* q = h_Q_.data() + static_cast<size_t>(i) * D_;
        float*       o = h_O_out_.data() + static_cast<size_t>(i) * D_;
        for (int d = 0; d < D_; ++d) o[d] = 0.0f;

        for (int j = 0; j < N; ++j) {
            const float* k = h_K_.data() + static_cast<size_t>(j) * D_;
            float dot = 0.0f;
            for (int d = 0; d < D_; ++d) dot += q[d] * k[d];
            s_row[j] = dot * sc;
        }

        float max_s = -std::numeric_limits<float>::infinity();
        for (int j = 0; j < N; ++j) max_s = std::max(max_s, s_row[j]);
        float sum = 0.0f;
        for (int j = 0; j < N; ++j) { s_row[j] = std::exp(s_row[j] - max_s); sum += s_row[j]; }
        for (int j = 0; j < N; ++j) s_row[j] /= sum;

        for (int j = 0; j < N; ++j) {
            const float* v = h_V_.data() + static_cast<size_t>(j) * D_;
            for (int d = 0; d < D_; ++d) o[d] += s_row[j] * v[d];
        }
    }
}

bool AttentionCpuReference::verify() { return AttentionWorkload::verify(); }

void AttentionCpuReference::teardown() {
    h_Q_.clear(); h_K_.clear(); h_V_.clear();
    h_O_ref_.clear(); h_O_out_.clear();
}

} // namespace kernelx
