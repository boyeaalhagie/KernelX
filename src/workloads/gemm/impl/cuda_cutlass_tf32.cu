// cuda_cutlass_tf32.cu — CUTLASS TF32 Tensor Core GEMM (sm_80+ only)
//
// This translation unit is compiled EXCLUSIVELY for sm_80 and sm_90
// (see CMakeLists.txt: set_source_files_properties ... CUDA_ARCHITECTURES).
// It must never be compiled for sm_75 or below: CUTLASS's MmaMultistage
// specialisations for OpClassTensorOp + Sm80 reference TF32 mma.sync
// instructions that do not exist on earlier architectures, causing
// incomplete-type errors at compile time.
//
// GemmCudaCutlass::run() forward-declares launch_cutlass_tf32 behind the
// KERNELX_CUTLASS_TF32 preprocessor guard (defined by CMakeLists when any
// sm_80+ architecture is in CMAKE_CUDA_ARCHITECTURES) and calls it only
// after a runtime sm_major >= 8 check.

#include "cutlass/gemm/device/gemm.h"

#include <stdexcept>
#include <string>

namespace kernelx_internal {

// ── CUTLASS Gemm type alias — TF32 Tensor Cores ───────────────────────────────
//
// GemmTF32 targets sm_80+ (A100, H100).  Before each mma.sync multiply-
// accumulate, both FP32 input operands are rounded to 10-bit mantissa TF32
// format.  The same rounding is performed by cuBLAS in its default math mode
// (CUBLAS_MATH_DEFAULT / CUBLAS_TF32_TENSOR_OP_MATH since CUDA 11).
//
// CUTLASS selects a 3-stage software pipeline with cp.async (asynchronous
// global → shared copy) for Sm80 + OpClassTensorOp.  This overlaps the
// global memory load for tile k with the mma.sync computation on tile k-1,
// hiding most of the HBM latency and pushing SM utilisation to ~80–90 % of
// peak on H100.

using GemmTF32 = cutlass::gemm::device::Gemm<
    float, cutlass::layout::RowMajor,   // A: row-major FP32 (rounded to TF32)
    float, cutlass::layout::RowMajor,   // B: row-major FP32 (rounded to TF32)
    float, cutlass::layout::RowMajor,   // C/D: row-major FP32
    float,                               // accumulator: FP32
    cutlass::arch::OpClassTensorOp,     // TF32 Tensor Cores
    cutlass::arch::Sm80                 // sm_80 and sm_90 (Ampere / Hopper)
>;

void launch_cutlass_tf32(int M, int N, int K,
                          const float* A, const float* B, float* C) {
    GemmTF32::Arguments args{
        {M, N, K},
        {A, K},         // A ref: row-major, leading dim = K
        {B, N},         // B ref: row-major, leading dim = N
        {C, N},         // C source (beta=0 → unused)
        {C, N},         // D output
        {1.0f, 0.0f}    // alpha=1, beta=0  →  D = A×B
    };
    GemmTF32 op;
    cutlass::Status status = op(args);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error(
            std::string("CUTLASS TF32 GEMM failed: ") + cutlassGetStatusString(status));
    }
}

} // namespace kernelx_internal
