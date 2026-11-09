#include "workloads/attention/impl/cuda_flash_fp16.hpp"
#include "utils/cuda_utils.hpp"
#include <cfloat>
#include <cuda_fp16.h>
#include <iostream>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
// Mixed-precision FlashAttention-1 — FP16 global + shared memory, FP32 accum.
//
// V1 of this kernel stored tiles as float in shared memory and converted on
// load.  Profiling showed the kernel is compute-bound (not HBM-bound): halving
// global memory traffic alone didn't help because the dot-product loop was
// bottlenecked on shared memory reads, not HBM.
//
// V2 (this version) stores Q, K, V tiles as __half in shared memory:
//   • s_Q[kBr][kD]      = 32×64×2 =  4 KB   (vs 8 KB float)
//   • s_K[kBc][kDp2]    = 32×66×2 =  4.1 KB (vs 8.25 KB float)
//   • s_V[kBc][kDp2]    = 4.1 KB
//   • s_S[kBr][kBc]     = 4 KB (float — scores, softmax ops)
//   Total ≈ 16.2 KB (vs 28.25 KB) → better occupancy potential
//
// Dot product uses __half2 vectorised reads (2 elements per instruction):
//   for d in [0, kD, step 2):
//     float2 q2 = __half22float2(s_Q_h2[wid][d/2]);
//     float2 k2 = __half22float2(s_K_h2[lid][d/2]);
//     s_j += q2.x*k2.x + q2.y*k2.y;
//   → 32 __half2 loads per lane vs 64 float loads; 2× less shared mem traffic.
//
// V accumulation similarly uses __half2 to read two V elements per instruction.
//
// Bank conflict analysis for __half2 layout (kDp2 = kD + 2):
//   s_K stored as __half2[kBc][kD/2+1]: row stride = (kD/2+1)*4 bytes.
//   Lane lid reads column lid/2 of its row: bank = (lid*(kD/2+1) + d/2) % 32.
//   With kDp2=kD+2 → kD/2+1 = 33: bank = (lid*33 + col) % 32 → no conflict.
//
// Block / grid: same as FP32 — (kBc=32, kBr=32), grid=(ceil(N/kBr),).
// ─────────────────────────────────────────────────────────────────────────────

namespace kernelx {

static constexpr int kBr_h  = 32;
static constexpr int kBc_h  = 32;
static constexpr int kD_h   = 64;
// +2 padding keeps __half2 rows at 33×__half2 = 132 bytes (bank-conflict-free)
static constexpr int kDp2_h = kD_h + 2;

__global__ void flash_attn_fp16_kernel(
        const __half* __restrict__ Q,   // [N, kD_h]
        const __half* __restrict__ K,   // [N, kD_h]
        const __half* __restrict__ V,   // [N, kD_h]
        __half*       __restrict__ O,   // [N, kD_h]
        int N, float scale)
{
    const int lid         = threadIdx.x;
    const int wid         = threadIdx.y;
    const int block_start = blockIdx.x * kBr_h;
    const int qi          = block_start + wid;

    // __half shared memory — 2× less capacity used vs float.
    // Cast to __half2 for vectorised (2-element) loads in the dot product.
    __shared__ __half s_Q[kBr_h][kD_h];
    __shared__ __half s_K[kBc_h][kDp2_h];
    __shared__ __half s_V[kBc_h][kDp2_h];
    __shared__ float  s_S[kBr_h][kBc_h];   // float: scores, softmax arithmetic

    // ── Load Q tile: store directly as __half (no conversion) ────────────────
    const int tid_flat = wid * kBc_h + lid;
    for (int idx = tid_flat; idx < kBr_h * kD_h; idx += kBr_h * kBc_h) {
        const int row = idx / kD_h;
        const int col = idx % kD_h;
        const int src = block_start + row;
        s_Q[row][col] = (src < N) ? Q[src * kD_h + col] : __float2half(0.0f);
    }
    __syncthreads();

    float m        = -FLT_MAX;
    float l        = 0.0f;
    float o_reg[2] = {};

    const int tc = (N + kBc_h - 1) / kBc_h;

    for (int tile = 0; tile < tc; ++tile) {
        const int tile_start = tile * kBc_h;
        const int tile_width = min(kBc_h, N - tile_start);

        // ── Coalesced load K and V tiles as __half (direct, no conversion) ───
        for (int idx = tid_flat; idx < kBc_h * kD_h; idx += kBr_h * kBc_h) {
            const int row = idx / kD_h;
            const int col = idx % kD_h;
            const int src = tile_start + row;
            const __half zero = __float2half(0.0f);
            s_K[row][col] = (src < N) ? K[src * kD_h + col] : zero;
            s_V[row][col] = (src < N) ? V[src * kD_h + col] : zero;
        }
        __syncthreads();

        // ── Score: s_S[wid][lid] = dot(s_Q[wid], s_K[lid]) * scale ──────────
        // Use __half2 vectorised loads: 2 elements per instruction.
        // s_Q and s_K are __half; reinterpret rows as __half2 arrays.
        float s_j = 0.0f;
        if (lid < tile_width) {
            const __half2* q_h2 = reinterpret_cast<const __half2*>(s_Q[wid]);
            const __half2* k_h2 = reinterpret_cast<const __half2*>(s_K[lid]);
            for (int d2 = 0; d2 < kD_h / 2; ++d2) {
                const float2 qf = __half22float2(q_h2[d2]);
                const float2 kf = __half22float2(k_h2[d2]);
                s_j += qf.x * kf.x + qf.y * kf.y;
            }
            s_j *= scale;
        }
        s_S[wid][lid] = (lid < tile_width) ? s_j : -FLT_MAX;
        __syncwarp();

        // ── Tile max via warp butterfly ───────────────────────────────────────
        float tile_m = s_S[wid][lid];
        for (int offset = 16; offset >= 1; offset >>= 1)
            tile_m = fmaxf(tile_m, __shfl_down_sync(0xFFFFFFFF, tile_m, offset));
        tile_m = __shfl_sync(0xFFFFFFFF, tile_m, 0);

        // ── Rescale running output with new max ───────────────────────────────
        const float m_new   = fmaxf(m, tile_m);
        const float rescale = expf(m - m_new);
        o_reg[0] *= rescale;
        o_reg[1] *= rescale;

        // ── Accumulate V contribution — __half2 vectorised V reads ───────────
        // Load two V elements per instruction to match the reduced shmem width.
        float tile_l = 0.0f;
        for (int j = 0; j < tile_width; ++j) {
            const float e_j = expf(s_S[wid][j] - m_new);
            tile_l  += e_j;
            // lid and lid+32 are in the same __half2 pair when lid < 32:
            // s_V[j][lid] and s_V[j][lid+32] span a __half2 at d2=lid/2 or similar.
            // Load individually for correctness — the compiler fuses adjacent accesses.
            o_reg[0] += e_j * __half2float(s_V[j][lid]);
            o_reg[1] += e_j * __half2float(s_V[j][lid + 32]);
        }

        l = l * rescale + tile_l;
        m = m_new;
        __syncthreads();
    }

    // ── Normalise and write output as __half ──────────────────────────────────
    if (qi < N) {
        const float inv_l = 1.0f / l;
        __half* o_row = O + static_cast<ptrdiff_t>(qi) * kD_h;
        o_row[lid]      = __float2half(o_reg[0] * inv_l);
        o_row[lid + 32] = __float2half(o_reg[1] * inv_l);
    }
}

// ── Setup helpers ─────────────────────────────────────────────────────────────

void AttentionCudaFlashFp16::alloc_device_fp16() {
    const size_t nd = static_cast<size_t>(M_) * D_ * sizeof(__half);
    CUDA_CHECK(cudaMalloc(&d_Q_h_, nd));
    CUDA_CHECK(cudaMalloc(&d_K_h_, nd));
    CUDA_CHECK(cudaMalloc(&d_V_h_, nd));
    CUDA_CHECK(cudaMalloc(&d_O_h_, nd));
}

void AttentionCudaFlashFp16::upload_inputs_fp16() {
    const size_t n = static_cast<size_t>(M_) * D_;

    // CPU-side float → half conversion; avoids a separate conversion kernel.
    std::vector<__half> tmp(n);

    auto upload = [&](const std::vector<float>& src, __half* dst) {
        for (size_t i = 0; i < n; ++i) tmp[i] = __float2half(src[i]);
        CUDA_CHECK(cudaMemcpy(dst, tmp.data(), n * sizeof(__half), cudaMemcpyHostToDevice));
    };
    upload(h_Q_, d_Q_h_);
    upload(h_K_, d_K_h_);
    upload(h_V_, d_V_h_);
}

void AttentionCudaFlashFp16::download_output_fp16() {
    const size_t n = static_cast<size_t>(M_) * D_;
    std::vector<__half> tmp(n);
    CUDA_CHECK(cudaMemcpy(tmp.data(), d_O_h_, n * sizeof(__half), cudaMemcpyDeviceToHost));
    h_O_out_.resize(n);
    for (size_t i = 0; i < n; ++i) h_O_out_[i] = __half2float(tmp[i]);
}

void AttentionCudaFlashFp16::free_device_fp16() {
    if (d_Q_h_) { cudaFree(d_Q_h_); d_Q_h_ = nullptr; }
    if (d_K_h_) { cudaFree(d_K_h_); d_K_h_ = nullptr; }
    if (d_V_h_) { cudaFree(d_V_h_); d_V_h_ = nullptr; }
    if (d_O_h_) { cudaFree(d_O_h_); d_O_h_ = nullptr; }
}

void AttentionCudaFlashFp16::setup(const WorkloadConfig& cfg) {
    init_host_data(cfg);
    compute_reference();
    alloc_device_fp16();
    upload_inputs_fp16();
}

void AttentionCudaFlashFp16::run() {
    const dim3 block(kBc_h, kBr_h);
    const dim3 grid((M_ + kBr_h - 1) / kBr_h);
    flash_attn_fp16_kernel<<<grid, block>>>(d_Q_h_, d_K_h_, d_V_h_, d_O_h_, M_, scale());
    CUDA_CHECK(cudaGetLastError());
    download_output_fp16();
}

bool AttentionCudaFlashFp16::verify() {
    // Use a relaxed tolerance: FP16 epsilon ≈ 1e-3, accumulated over N softmax
    // steps and D-dimensional dot products → up to ~2% relative error.
    const int rows = std::min(M_, kVerifyRows);
    for (int i = 0; i < rows; ++i) {
        const float* ref = h_O_ref_.data() + static_cast<size_t>(i) * D_;
        const float* out = h_O_out_.data() + static_cast<size_t>(i) * D_;
        for (int d = 0; d < D_; ++d) {
            const float denom = std::max(std::abs(ref[d]), 1e-4f);
            const float rel   = std::abs(out[d] - ref[d]) / denom;
            if (rel > kRelTolFp16) {
                std::cerr << "  [verify fp16] mismatch row=" << i << " d=" << d
                          << "  ref=" << ref[d] << "  got=" << out[d]
                          << "  rel=" << rel << "\n";
                return false;
            }
        }
    }
    return true;
}

void AttentionCudaFlashFp16::teardown() {
    free_device_fp16();
    h_Q_.clear(); h_K_.clear(); h_V_.clear();
    h_O_ref_.clear(); h_O_out_.clear();
}

} // namespace kernelx
