#include "workloads/gemm/impl/cpu_reference.hpp"
#include <algorithm>

namespace kernelx {

void GemmCpuReference::setup(const WorkloadConfig& cfg) {
    init_host_data(cfg);
    // No device buffers.  No reference pre-computation — this IS the reference.
}

void GemmCpuReference::run() {
    // i→k→j loop order maximises cache reuse in row-major layout:
    //   A[i][k]: sequential in k for fixed i  →  stride-1 loads on A
    //   B[k][j]: sequential in j for fixed k  →  stride-1 loads on B
    // Rewriting as i→j→k would make B[k][j] stride-N, thrashing the cache.
    std::fill(h_C_out_.begin(), h_C_out_.end(), 0.0f);
    for (int i = 0; i < M_; ++i) {
        for (int k = 0; k < K_; ++k) {
            const float a_ik = h_A_[i * K_ + k];
            for (int j = 0; j < N_; ++j) {
                h_C_out_[i * N_ + j] += a_ik * h_B_[k * N_ + j];
            }
        }
    }
}

void GemmCpuReference::teardown() {
    h_A_.clear();
    h_B_.clear();
    h_C_out_.clear();
    h_C_ref_.clear();
}

} // namespace kernelx
