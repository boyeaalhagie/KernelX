#include "workloads/gemm/impl/cuda_cutlass.hpp"
#include "utils/cuda_utils.hpp"

#include "cutlass/gemm/device/gemm.h"

#include <stdexcept>
#include <cmath>
#include <string>

namespace kernelx {

namespace {

// ── CUTLASS Gemm type aliases ─────────────────────────────────────────────────
//
// GemmSimt  — FP32 SIMT math.  Targets sm_75 (T4).  CUTLASS automates tile
//             shape selection, software pipelining, and predicated bounds
//             checking — improvements over V3's hand-written tiling that are
//             visible in Nsight Compute as higher L2 hit rate and fewer
//             warp stalls.
//
// GemmTF32  — TF32 Tensor Core math.  Targets sm_80+ (H100).
//             Each FP32 input operand is rounded to 10-bit mantissa TF32
//             before the mma.sync.aligned.m16n8k8 instruction executes.
//             This is the same math mode cuBLAS uses by default
//             (CUBLAS_MATH_DEFAULT / CUBLAS_TF32_TENSOR_OP_MATH).
//             The 3-stage software pipeline and async shared-memory copies
//             (cp.async) hide global memory latency behind compute.

using GemmSimt = cutlass::gemm::device::Gemm<
    float, cutlass::layout::RowMajor,   // A: row-major FP32
    float, cutlass::layout::RowMajor,   // B: row-major FP32
    float, cutlass::layout::RowMajor,   // C/D: row-major FP32
    float,                               // accumulator: FP32
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm75
>;

using GemmTF32 = cutlass::gemm::device::Gemm<
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80
>;

template <typename GemmOp>
static void launch(int M, int N, int K, const float* A, const float* B, float* C) {
    typename GemmOp::Arguments args{
        {M, N, K},
        {A, K},         // A ref: row-major, leading dim = K
        {B, N},         // B ref: row-major, leading dim = N
        {C, N},         // C source (beta=0 → unused)
        {C, N},         // D output: row-major, leading dim = N
        {1.0f, 0.0f}    // alpha=1, beta=0  →  D = A×B
    };
    GemmOp op;
    cutlass::Status status = op(args);   // runs on default stream (nullptr)
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error(
            std::string("CUTLASS GEMM failed: ") + cutlassGetStatusString(status));
    }
}

} // anonymous namespace

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
    if (sm_major_ >= 8)
        launch<GemmTF32>(M_, N_, K_, d_A_, d_B_, d_C_);
    else
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
