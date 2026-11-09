#pragma once

#include "workloads/gemm/gemm_workload.hpp"
#include <cuda_fp16.h>

namespace kernelx {

// ─────────────────────────────────────────────────────────────────────────────
// GemmCudaWmma — WMMA Tensor Core GEMM, V4 in the GEMM implementation ladder.
//
// ── What changes from cuda_tuned (V3) ───────────────────────────────────────
// cuda_tuned operates exclusively on FP32 scalar CUDA cores.  Even with
// register blocking, it is bounded by the scalar FP32 throughput of the SM
// (512 FP32 FMA/s per H100 SM, 132 SMs → ~60 TFLOP/s peak).
//
// V4 uses nvcuda::wmma — the Warp Matrix Multiply-Accumulate API — to dispatch
// each warp's inner-loop computation directly to the Tensor Core unit.  On
// Volta/Turing/Ampere/Hopper, one mma.sync instruction processes a 16×16×16
// FP16 multiply-accumulate tile in a single clock cycle, delivering ~16×
// higher throughput per instruction than scalar FP32 FMA.
//
// ── The WMMA programming model ───────────────────────────────────────────────
// The WMMA API operates at warp granularity — all 32 lanes of a warp
// collectively own one fragment, and every lane must call mma_sync.
//
// Three fragment types are involved:
//   matrix_a   <16, 16, 16, half, row_major>   — 16×16 slice of A (fp16)
//   matrix_b   <16, 16, 16, half, row_major>   — 16×16 slice of B (fp16)
//   accumulator<16, 16, 16, float>              — 16×16 partial sum (fp32)
//
// Workflow per warp, per K-tile:
//   1. load_matrix_sync(a_frag, sA_ptr, BK)    — load from shared memory
//   2. load_matrix_sync(b_frag, sB_ptr, BN)
//   3. mma_sync(acc, a_frag, b_frag, acc)      — 16×16×16 FP16 GEMM into FP32
//
// ── Block / grid dimensions ───────────────────────────────────────────────────
// Block tile: BM=128 × BN=128,  K-depth: BK=16
// Each warp computes a 2×2 arrangement of 16×16 fragments (32×32 output).
//   Warps in M direction: BM / (kWarpTM * 16) = 128 / 32 = 4
//   Warps in N direction: BN / (kWarpTN * 16) = 128 / 32 = 4
//   Total warps per block:  4 × 4 = 16  →  512 threads
//
// Grid: ⌈N/BN⌉ × ⌈M/BM⌉
//
// ── Shared memory ─────────────────────────────────────────────────────────────
// sA[BM][BK] = half[128][16] = 4 KB
// sB[BK][BN] = half[16][128] = 4 KB
// Total: 8 KB/block  →  good occupancy (H100 has 228 KB shared mem/SM)
//
// ── Precision and verification tolerance ─────────────────────────────────────
// Inputs are converted FP32 → FP16 at upload time.  FP16 has a 10-bit
// mantissa (~3 decimal digits).  Accumulated over K=4096 terms the expected
// relative error vs a pure FP32 reference is roughly 4–5 %.
// kRelTolFp16 = 0.10 (10 %) provides comfortable margin.
//
// ── What to look for in Nsight after V4 ──────────────────────────────────────
//   • sm__mma_in_use should show Tensor Core utilisation > 0
//   • Expected throughput: 50–150 TFLOP/s — far below CUTLASS because this
//     kernel lacks the double-buffered cp.async pipeline.  The gap between
//     V4 (manual WMMA) and V5 (CUTLASS) illustrates how much software
//     pipeline engineering matters on top of raw Tensor Core access.
//   • Long scoreboard stalls will remain — the kernel stalls waiting for the
//     global → shared load to complete before mma_sync can proceed.
//
// ── Constraint ───────────────────────────────────────────────────────────────
// M, N, and K must each be a multiple of 16 (the WMMA fragment dimension).
// setup() throws std::invalid_argument if this is violated.
// All benchmark sizes (512, 1024, 2048, 4096) satisfy this constraint.
// ─────────────────────────────────────────────────────────────────────────────
class GemmCudaWmma final : public GemmWorkload {
public:
    // Block tile: 128×128 output per thread block, 16 K-elements per stage
    static constexpr int kBM = 128;
    static constexpr int kBN = 128;
    static constexpr int kBK = 16;

    // Each warp covers kWarpTM × kWarpTN WMMA fragments (each 16×16)
    // → 2×2 = 32×32 output elements per warp
    static constexpr int kWarpTM = 2;
    static constexpr int kWarpTN = 2;

    // Warp grid within the block
    static constexpr int kWarpsM    = kBM / (kWarpTM * 16);  // 4
    static constexpr int kWarpsN    = kBN / (kWarpTN * 16);  // 4
    static constexpr int kWarps     = kWarpsM * kWarpsN;     // 16
    static constexpr int kBlockSize = kWarps * 32;            // 512 threads

    void setup(const WorkloadConfig& cfg) override;
    void run()                            override;
    void teardown()                       override;
    bool verify()                         override;

    std::string implementation_name() const override { return "cuda_wmma"; }
    Backend     backend()             const override { return Backend::CUDA; }

private:
    // FP16 device buffers — derived from d_A_/d_B_ after FP32→FP16 conversion
    half* d_A_fp16_{nullptr};
    half* d_B_fp16_{nullptr};

    // FP16 precision: ~10-bit mantissa, ~4–5 % error at K=4096 vs FP32 ref.
    // 10 % threshold is consistent with CUTLASS TF32 tolerance.
    static constexpr float kRelTolFp16 = 0.10f;
};

} // namespace kernelx
