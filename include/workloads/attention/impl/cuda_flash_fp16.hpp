#pragma once
#include "workloads/attention/attention_workload.hpp"
#include <cuda_fp16.h>

namespace kernelx {

// ─────────────────────────────────────────────────────────────────────────────
// AttentionCudaFlashFp16 — mixed-precision FlashAttention-1.
//
// Identical tiled algorithm to cuda_flash (kBr=kBc=32, kD=64, 1024 threads),
// with one key difference: Q, K, V, O in global memory are __half (FP16).
//
// Precision strategy:
//   Global memory  : __half  (2 bytes/element — halves HBM traffic)
//   Shared memory  : __half  (2× less L1/shared traffic in dot-product loop)
//   Dot product    : __half2 vectorised loads → float FMAs (2 elems/instr)
//   Register accum : float   (m, l, o_reg[2] — numerical stability for softmax)
//
// Expected speedup vs cuda_flash (FP32):
//   HBM traffic: 2× reduction (half bytes) → approaches 2× speedup at large N
//   where the kernel is HBM-bound.  At N=4096 the break-even was
//   already close for FP32; FP16 is strictly bandwidth-limited.
//
// Memory layout (device): row-major [N × kD], stored as __half.
// Host buffers stay float (init_host_data, compute_reference from base class).
// Conversion: CPU-side float→half on upload, half→float on download.
//
// Verification: compare against float-precision CPU reference.
//   Tolerance kRelTolFp16 = 2e-2f (FP16 epsilon ≈ 1e-3 → accumulated error
//   over N=2048 softmax + matmul can reach 1–2%).
// ─────────────────────────────────────────────────────────────────────────────
class AttentionCudaFlashFp16 final : public AttentionWorkload {
public:
    static constexpr float kRelTolFp16 = 2e-2f;

    void setup(const WorkloadConfig& cfg) override;
    void run()                            override;
    void teardown()                       override;
    bool verify()                         override;

    std::string implementation_name() const override { return "cuda_flash_fp16"; }
    Backend     backend()             const override { return Backend::CUDA; }

private:
    // FP16 device buffers — replace the float d_Q_ / d_K_ / d_V_ / d_O_ from base
    __half* d_Q_h_ = nullptr;
    __half* d_K_h_ = nullptr;
    __half* d_V_h_ = nullptr;
    __half* d_O_h_ = nullptr;

    void alloc_device_fp16();
    void upload_inputs_fp16();
    void download_output_fp16();
    void free_device_fp16();
};

} // namespace kernelx
