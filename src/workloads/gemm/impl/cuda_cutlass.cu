#include "workloads/gemm/impl/cuda_cutlass.hpp"
#include "utils/cuda_utils.hpp"

#include "cutlass/gemm/device/gemm.h"

#include <cmath>
#include <stdexcept>
#include <string>

namespace kernelx {

namespace {

// ── CUTLASS Gemm type alias — FP32 SIMT ──────────────────────────────────────
//
// GemmSimt targets sm_75 (T4) and any architecture that lacks TF32 Tensor
// Cores.  CUTLASS's default tile shapes and software pipeline for OpClassSimt
// are still a step up from V3's hand-written code: autotuned thread-block
// shapes, vectorised 128-bit loads where alignment allows, and predicated
// bounds checking without branch divergence.
//
// This type is safe to compile for any architecture because it uses only
// standard FP32 multiply-accumulate instructions.  GemmTF32 (Sm80,
// OpClassTensorOp) is kept in a separate translation unit
// (cuda_cutlass_tf32.cu) compiled exclusively for sm_80+ to avoid CUTLASS
// internal specialisation errors when sm_75 is in CMAKE_CUDA_ARCHITECTURES.

using GemmSimt = cutlass::gemm::device::Gemm<
    float, cutlass::layout::RowMajor,   // A: row-major FP32
    float, cutlass::layout::RowMajor,   // B: row-major FP32
    float, cutlass::layout::RowMajor,   // C/D: row-major FP32
    float,                               // accumulator: FP32
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm75
>;

template <typename GemmOp>
static void launch(int M, int N, int K, const float* A, const float* B, float* C) {
    typename GemmOp::Arguments args{
        {M, N, K},
        {A, K},         // A ref: row-major, leading dim = K
        {B, N},         // B ref: row-major, leading dim = N
        {C, N},         // C source (beta=0 → unused)
        {C, N},         // D output
        {1.0f, 0.0f}    // alpha=1, beta=0  →  D = A×B
    };
    GemmOp op;
    cutlass::Status status = op(args);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error(
            std::string("CUTLASS GEMM failed: ") + cutlassGetStatusString(status));
    }
}

} // anonymous namespace

// ── Forward declaration for TF32 path (cuda_cutlass_tf32.cu, sm_80+ only) ────
#ifdef KERNELX_CUTLASS_TF32
namespace kernelx_internal {
    void launch_cutlass_tf32(int M, int N, int K,
                              const float* A, const float* B, float* C);
}
#endif

// ─────────────────────────────────────────────────────────────────────────────

void GemmCudaCutlass::setup(const WorkloadConfig& cfg) {
    int dev;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&sm_major_, cudaDevAttrComputeCapabilityMajor, dev));

    init_host_data(cfg);
    compute_reference();
    alloc_device();
    upload_inputs();
    CUDA_CHECK(cudaMemset(d_C_, 0, static_cast<size_t>(M_) * N_ * sizeof(float)));
}

void GemmCudaCutlass::run() {
#ifdef KERNELX_CUTLASS_TF32
    if (sm_major_ >= 8) {
        kernelx_internal::launch_cutlass_tf32(M_, N_, K_, d_A_, d_B_, d_C_);
        return;
    }
#endif
    launch<GemmSimt>(M_, N_, K_, d_A_, d_B_, d_C_);
}

bool GemmCudaCutlass::verify() {
    CUDA_CHECK(cudaDeviceSynchronize());
    download_output();

    // Select tolerance based on the math path that actually ran:
    //   SIMT  (sm_75): strict FP32 — same threshold as other hand-written kernels
    //   TF32  (sm_80+): relaxed  — TF32's 10-bit mantissa accumulates ~6 % error
    //                              at K=4096; 10 % gives comfortable margin
    const float tol = (sm_major_ >= 8) ? kRelTolTF32 : kRelTolSimt;

    for (int i = 0; i < kVerifyDim; ++i) {
        for (int j = 0; j < kVerifyDim; ++j) {
            float ref = h_C_ref_[static_cast<size_t>(i) * N_ + j];
            float got = h_C_out_[static_cast<size_t>(i) * N_ + j];
            float denom = std::max(std::abs(ref), 1e-4f);
            if (std::abs(got - ref) / denom > tol)
                return false;
        }
    }
    return true;
}

void GemmCudaCutlass::teardown() {
    free_device();
    h_A_.clear();
    h_B_.clear();
    h_C_out_.clear();
    h_C_ref_.clear();
}

} // namespace kernelx
