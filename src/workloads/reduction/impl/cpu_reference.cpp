#include "workloads/reduction/impl/cpu_reference.hpp"

namespace kernelx {

void ReductionCpuReference::setup(const WorkloadConfig& cfg) {
    init_host_data(cfg);
    compute_reference();
}

void ReductionCpuReference::run() {
    for (int i = 0; i < M_; ++i) {
        const float* x = h_X_.data() + static_cast<size_t>(i) * N_;
        double sum = 0.0;
        for (int j = 0; j < N_; ++j) sum += x[j];
        h_Y_out_[i] = static_cast<float>(sum);
    }
}

void ReductionCpuReference::teardown() {
    h_X_.clear();
    h_Y_ref_.clear();
    h_Y_out_.clear();
}

bool ReductionCpuReference::verify() { return ReductionWorkload::verify(); }

} // namespace kernelx
