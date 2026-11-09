// cuda_cutlass_tf32.cu — CUTLASS TF32 Tensor Core GEMM (sm_80+ only)
//
// This translation unit is compiled EXCLUSIVELY for sm_80 and sm_90
// (see CMakeLists.txt: set_source_files_properties ... CUDA_ARCHITECTURES).
//
// Key configuration note:
//   The default InstructionShape selected by CUTLASS for
//   Gemm<float, ..., OpClassTensorOp, Sm80> is GemmShape<16,8,16>,
//   which is the FP16 Tensor Core shape.  For TF32 the correct shape
//   is GemmShape<16,8,8> (mma.sync.aligned.m16n8k8.f32.tf32.tf32.f32).
//   Specifying tile shapes explicitly avoids the incomplete-type errors
//   that arise from DefaultGemmConfiguration selecting the wrong path.

#include "cutlass/gemm/device/gemm.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"

#include <stdexcept>
#include <string>

namespace kernelx {
namespace kernelx_internal {

// ── CUTLASS Gemm type alias — TF32 Tensor Cores ───────────────────────────────
//
// Tile shape selection for sm_80 TF32:
//   ThreadblockShape <128, 128, 32>  — 128×128 output tile per block, K-stage=32
//   WarpShape        <64,  64,  32>  — 64×64 output per warp
//   InstructionShape <16,  8,   8>   — TF32 mma.sync instruction (NOT <16,8,16>)
//   Stages = 3                       — 3-stage async cp.async pipeline
//
// The 3-stage pipeline (cp.async, sm_80+) overlaps the global-memory load
// for tile k with mma.sync computation on tile k-1, hiding most HBM latency.
// This is what pushes H100 utilisation to ~80 % of peak vs ~35 % for V3.

using GemmTF32 = cutlass::gemm::device::Gemm<
    float, cutlass::layout::RowMajor,                              // A
    float, cutlass::layout::RowMajor,                              // B
    float, cutlass::layout::RowMajor,                              // C/D
    float,                                                          // accumulator
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,                        // threadblock tile
    cutlass::gemm::GemmShape<64,  64,  32>,                        // warp tile
    cutlass::gemm::GemmShape<16,  8,   8>,                         // TF32 instruction
    cutlass::epilogue::thread::LinearCombination<float, 4, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3                                                               // pipeline stages
>;

void launch_cutlass_tf32(int M, int N, int K,
                          const float* A, const float* B, float* C) {
    GemmTF32::Arguments args{
        {M, N, K},
        {A, K},         // A: row-major, leading dim = K
        {B, N},         // B: row-major, leading dim = N
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
} // namespace kernelx
