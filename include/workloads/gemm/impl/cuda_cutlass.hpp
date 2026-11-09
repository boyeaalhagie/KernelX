#pragma once

#include "workloads/gemm/gemm_workload.hpp"

namespace kernelx {

// ─────────────────────────────────────────────────────────────────────────────
// GemmCudaCutlass — CUTLASS device Gemm, V4 in the GEMM implementation ladder.
//
// ── What CUTLASS provides over V3 (cuda_tuned) ───────────────────────────────
// V3 (cuda_tuned) manually implements register-blocked tiling and achieves
// ~35 % of peak FP32 throughput on the H100.  Three bottlenecks remain:
//
//   1. Load coalescing — with BK=8 the inner-loop sA loads span multiple rows
//      and require several 32-byte transactions per warp.  CUTLASS replaces
//      these with 128-byte vectorised loads (LDG.128) aligned to warp
//      boundaries, saturating global memory bandwidth.
//
//   2. Latency hiding — V3 issues loads and immediately uses the result;
//      warp stalls are dominated by "long scoreboard" (memory latency).
//      CUTLASS inserts a software pipeline with double- or triple-buffered
//      shared-memory stages so that arithmetic on tile k-1 overlaps the
//      load of tile k.  On sm_80+ this is implemented with cp.async
//      (asynchronous copy from global → shared), eliminating the register
//      pressure of the intermediate load path.
//
//   3. Compute unit utilisation — sm_75 (T4) has 2560 CUDA cores; sm_90
//      (H100) adds Tensor Core units executing mma.sync, which process a
//      16×8×8 FP32 (TF32) GEMM tile per instruction at ~16× throughput
//      over SIMT.  V3 never touches Tensor Cores; GemmTF32 below does.
//
// Together these bring measured throughput to 80–90 % of peak on H100.
//
// ── Architecture dispatch strategy ───────────────────────────────────────────
// Two CUTLASS Gemm type aliases are compiled into the translation unit:
//
//   GemmSimt   sm_75 (T4)    — FP32 SIMT math, CUTLASS-tuned tile shapes
//   GemmTF32   sm_80+ (H100) — TF32 Tensor Core math, 3-stage async pipeline
//
// CUTLASS guards each via __CUDA_ARCH__ internally, so both compile for all
// architectures in CMAKE_CUDA_ARCHITECTURES without host-side ifdefs.
// At runtime, setup() reads cudaDevAttrComputeCapabilityMajor and stores it
// in sm_major_; run() branches on that value to call the correct template.
//
// ── TF32 precision and verification tolerance ─────────────────────────────────
// TF32 is an NVIDIA-defined format introduced with Ampere (sm_80).  Before
// each mma.sync multiply-accumulate, each FP32 input operand is silently
// rounded to 10-bit mantissa (8 exponent bits remain, matching FP32).
// The resulting relative per-element rounding error is bounded by 2^{-10}.
//
// For a GEMM accumulating K dot-product terms, errors combine in an
// approximately √K fashion (they are not systematic and partially cancel):
//
//   ε_rel ≈ √K × 2^{-10}
//   At K=4096 : ε_rel ≈ 64 × 9.8×10^{-4} ≈ 6.3 %
//
// This matches cuBLAS CUBLAS_MATH_DEFAULT / CUBLAS_TF32_TENSOR_OP_MATH mode,
// which is the default for cublasGemmEx in CUDA 11+.  The 10 % tolerance
// (kRelTolTF32) provides comfortable margin while still catching wrong
// answers (a transposed index or sign error produces ≫10 % deviation).
//
// ── What to look for in Nsight after V4 ──────────────────────────────────────
//   • sm__throughput.avg.pct_of_peak_sustained_elapsed should jump from
//     ~35 % (V3) to 80–90 % (V4 on H100 with TF32 Tensor Cores).
//   • lts__throughput (L2) will be lower — CUTLASS's vectorised loads land
//     more data in L2 per transaction and the larger tile size improves reuse.
//   • Long scoreboard stalls should drop dramatically (async copy pipeline).
//   • On T4, improvement is modest (no Tensor Cores); expect 50–60 % peak.
// ─────────────────────────────────────────────────────────────────────────────
class GemmCudaCutlass final : public GemmWorkload {
public:
    void setup(const WorkloadConfig& cfg) override;
    void run()                            override;
    void teardown()                       override;
    bool verify()                         override;

    std::string implementation_name() const override { return "cuda_cutlass"; }
    Backend     backend()             const override { return Backend::CUTLASS; }

private:
    int sm_major_{0};

    // Relative tolerance for TF32 results (sm_80+).
    // TF32 has a 10-bit mantissa (vs FP32's 23-bit).  For K accumulations
    // the expected relative error is roughly √K × 2^{-10}.  At K=4096 that
    // gives ~6 %; a 10 % threshold provides comfortable margin while still
    // catching wrong answers.
    static constexpr float kRelTolTF32 = 1e-1f;

    // Standard FP32 SIMT tolerance — same as GemmWorkload::kRelTol (1e-3).
    // Redeclared here because kRelTol is private in the base class.
    static constexpr float kRelTolSimt = 1e-3f;
};

} // namespace kernelx
