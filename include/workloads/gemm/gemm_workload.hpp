#pragma once

#include "workloads/workload.hpp"
#include <vector>

namespace kernelx {

// ─────────────────────────────────────────────────────────────────────────────
// GemmWorkload — shared base for all GEMM C = A × B implementations.
//
// Owns host-side buffers (A, B, C_ref, C_out) and provides helpers for:
//   • random data initialisation (fixed seed for reproducibility)
//   • computing a CPU ground-truth reference over a bounded submatrix
//   • allocating / uploading / downloading / freeing device buffers
//   • comparing implementation output against the reference
//
// GPU subclasses allocate d_A_, d_B_, d_C_ via alloc_device(); CPU
// subclasses leave them null.
//
// Verification strategy:
//   Computing a full CPU reference for a 4096³ GEMM would take minutes.
//   We instead verify the top-left kVerifyDim × kVerifyDim corner of C,
//   which is sufficient to catch indexing and precision bugs while keeping
//   setup time negligible.  Pass --no-verify to skip entirely.
// ─────────────────────────────────────────────────────────────────────────────
class GemmWorkload : public Workload {
public:
    ~GemmWorkload() override;

    bool   verify() override;
    double flops()  const override;
    std::string workload_name() const override { return "gemm"; }

protected:
    // Maximum rows/cols for which we compute the CPU reference.
    static constexpr int kVerifyDim = 64;

    int         M_ = 0, N_ = 0, K_ = 0;
    std::string dtype_;

    // Host buffers — fp32 throughout (fp16 paths added in cuda_tiled and beyond)
    std::vector<float> h_A_;
    std::vector<float> h_B_;
    std::vector<float> h_C_ref_;   // ground truth: kVerifyDim × kVerifyDim corner
    std::vector<float> h_C_out_;   // full M × N output (copied back for verification)

    // Device buffers — non-null only for GPU implementations
    float* d_A_ = nullptr;
    float* d_B_ = nullptr;
    float* d_C_ = nullptr;

    // ── Setup helpers (called by subclasses) ─────────────────────────────────
    void init_host_data(const WorkloadConfig& cfg);
    void compute_reference();   // fills h_C_ref_ (kVerifyDim × kVerifyDim)
    void alloc_device();
    void upload_inputs();       // h_A_, h_B_ → d_A_, d_B_
    void download_output();     // d_C_ → h_C_out_
    void free_device();

private:
    static constexpr float kRelTol = 1e-3f;
};

} // namespace kernelx
