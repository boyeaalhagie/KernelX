#pragma once

#include "workloads/gemm/gemm_workload.hpp"

namespace kernelx {

// ─────────────────────────────────────────────────────────────────────────────
// GemmCudaTuned — register-blocked GEMM with thread coarsening.
//
// This is Version 3 in the GEMM implementation progression.
//
// ── What changes from cuda_tiled ─────────────────────────────────────────────
// cuda_tiled (V2): each thread computes 1 output element.  The inner loop
// reads sA[ty][k] and sB[k][tx] from shared memory for each k, accumulating
// into a single register.  sA/sB hits are fast, but:
//   • BK syncthreads per output element — high synchronisation overhead
//   • Low register reuse: sA[ty][k] is read once and discarded
//
// cuda_tuned (V3): each thread computes a TM×TN sub-tile of outputs.
// For each k in the inner loop, we load TM values from sA and TN values
// from sB into per-thread register arrays, then compute the TM×TN outer
// product.  Each shared-memory load is reused TN or TM times:
//
//   V2:  sA[ty][k] used once  → 1 FLOP per shared load
//   V3:  a_reg[i]  used TN times  → TN FLOPs per shared load
//        b_reg[j]  used TM times  → TM FLOPs per shared load
//
// ── Arithmetic intensity ──────────────────────────────────────────────────────
// V2 (TILE=32):    ≈ TILE/4        =  8 FLOP/byte (shared → compute)
// V3 (TM=TN=4):   ≈ TILE*TM/4     = BK*TM/4 FLOP/byte per A-load
//                  ≈ TILE*TN/4     = BK*TN/4 FLOP/byte per B-load
//                  With BK=8, TM=TN=4: still ~8 FLOP/byte from shared,
//                  BUT we halve the thread count (4× fewer syncthreads
//                  per output tile), so the compute-to-sync ratio improves.
//
// ── Thread / block / grid dimensions ─────────────────────────────────────────
// Thread block: (BN/TN) × (BM/TM) threads
//   Default BM=BN=64, BK=8, TM=TN=4 → 16×16 = 256 threads
//
// Thread (tx, ty) computes the output sub-tile:
//   rows [blockIdx.y*BM + ty*TM,  blockIdx.y*BM + ty*TM + TM)
//   cols [blockIdx.x*BN + tx*TN,  blockIdx.x*BN + tx*TN + TN)
//
// Grid: ⌈N/BN⌉ × ⌈M/BM⌉
//
// ── Shared memory ─────────────────────────────────────────────────────────────
// sA[BM][BK]:  BM=64, BK=8  → 2 KB
// sB[BK][BN]:  BK=8,  BN=64 → 2 KB
// Total: 4 KB/block  →  up to 8 concurrent blocks on T4 (32 KB shared/SM)
//
// ── Register usage ────────────────────────────────────────────────────────────
// acc[TM][TN] = 16 floats          (accumulator tile)
// a_reg[TM]   =  4 floats          (column of sA for current k)
// b_reg[TN]   =  4 floats          (row   of sB for current k)
// Indices + loop vars              ≈ 8 floats
// Estimated total: ~28–32 registers per thread  →  good occupancy
//
// ── What to look for in Nsight after V3 ──────────────────────────────────────
//   • Warp stall reasons should shift: fewer "wait" stalls (sync overhead),
//     more "math throttle" stalls  → we are more compute-bound
//   • Registers per thread in the PTX info line (--ptxas-options=-v)
//   • sm__throughput should rise vs V2; if not, check register spilling
//
// ── Loading coalescing note (gap addressed by V4 CUTLASS) ─────────────────────
// With BK=8 < warp_size=32, a single warp's sA load spans multiple rows
// of A, requiring multiple memory transactions.  CUTLASS addresses this via
// async copy pipelines and 128-byte vectorised loads.  V3 intentionally
// keeps loading simple to isolate the register-blocking improvement.
// ─────────────────────────────────────────────────────────────────────────────
class GemmCudaTuned final : public GemmWorkload {
public:
    // Block tile dimensions — the region of C each thread block computes
    static constexpr int kBM = 64;   // rows per block tile
    static constexpr int kBN = 64;   // cols per block tile
    static constexpr int kBK =  8;   // K-depth per shared-memory stage

    // Thread tile dimensions — outputs computed per thread
    static constexpr int kTM = 4;    // output rows per thread
    static constexpr int kTN = 4;    // output cols per thread

    // Derived: thread block size
    static constexpr int kBlockX = kBN / kTN;   // 16
    static constexpr int kBlockY = kBM / kTM;   // 16

    void setup(const WorkloadConfig& cfg) override;
    void run()                            override;
    void teardown()                       override;
    bool verify()                         override;

    std::string implementation_name() const override { return "cuda_tuned"; }
    Backend     backend()             const override { return Backend::CUDA; }
};

} // namespace kernelx
