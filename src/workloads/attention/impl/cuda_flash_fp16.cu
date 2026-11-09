#include "workloads/attention/impl/cuda_flash_fp16.hpp"
#include "utils/cuda_utils.hpp"
#include <cfloat>
#include <cuda_fp16.h>

// ─────────────────────────────────────────────────────────────────────────────
// Mixed-precision FlashAttention-1 — FP16 global memory, FP32 accumulators.
//
// Algorithm is identical to cuda_flash (kBr=kBc=32, one warp per query row),
// with these changes:
//   • Global loads:  __half2float() on each element as it enters shared memory.
//   • Global stores: __float2half() on each output element before writing O.
//   • Everything else (s_Q, s_K, s_V, s_S, m, l, o_reg) stays float.
//
// HBM traffic (N=4096, D=64, kBr=32):
//   FP32 baseline: K reads = N²·D·4/kBr = 128 MB, same for V, Q + O ≈ 2 MB
//     → Total ≈ 258 MB
//   FP16 this kernel: same tile structure but sizeof(__half)=2 → ≈ 129 MB
//   → ~2× bandwidth saving at large N.
//
// Block structure: (kBc=32, kBr=32) = 1024 threads / 32 warps.
//   threadIdx.y = wid (warp, owns query row wid within block)
//   threadIdx.x = lid (lane, owns key column / output dim)
// Grid: (ceil(N/kBr),)
// ─────────────────────────────────────────────────────────────────────────────

namespace kernelx {

static constexpr int kBr_h  = 32;
static constexpr int kBc_h  = 32;
static constexpr int kD_h   = 64;
static constexpr int kDp1_h = kD_h + 1;   // +1 bank-conflict padding

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

    // Shared memory in float — convert from __half on load
    __shared__ float s_Q[kBr_h][kD_h];
    __shared__ float s_K[kBc_h][kDp1_h];
    __shared__ float s_V[kBc_h][kDp1_h];
    __shared__ float s_S[kBr_h][kBc_h];

    // ── Load Q tile (kBr × kD = 2048 elements, 1024 threads → 2 each) ────────
    const int tid_flat = wid * kBc_h + lid;
    for (int idx = tid_flat; idx < kBr_h * kD_h; idx += kBr_h * kBc_h) {
        const int row = idx / kD_h;
        const int col = idx % kD_h;
        const int src = block_start + row;
        s_Q[row][col] = (src < N) ? __half2float(Q[src * kD_h + col]) : 0.0f;
    }
    __syncthreads();

    float m        = -FLT_MAX;
    float l        = 0.0f;
    float o_reg[2] = {};

    const int tc = (N + kBc_h - 1) / kBc_h;

    for (int tile = 0; tile < tc; ++tile) {
        const int tile_start = tile * kBc_h;
        const int tile_width = min(kBc_h, N - tile_start);

        // ── Coalesced load K and V tiles (convert __half → float) ────────────
        for (int idx = tid_flat; idx < kBc_h * kD_h; idx += kBr_h * kBc_h) {
            const int row = idx / kD_h;
            const int col = idx % kD_h;
            const int src = tile_start + row;
            s_K[row][col] = (src < N) ? __half2float(K[src * kD_h + col]) : 0.0f;
            s_V[row][col] = (src < N) ? __half2float(V[src * kD_h + col]) : 0.0f;
        }
        __syncthreads();

        // ── Score: s_S[wid][lid] = dot(s_Q[wid], s_K[lid]) * scale ──────────
        float s_j = 0.0f;
        if (lid < tile_width) {
            for (int d = 0; d < kD_h; ++d)
                s_j += s_Q[wid][d] * s_K[lid][d];
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

        // ── Accumulate V contribution ─────────────────────────────────────────
        float tile_l = 0.0f;
        for (int j = 0; j < tile_width; ++j) {
            const float e_j = expf(s_S[wid][j] - m_new);
            tile_l  += e_j;
            o_reg[0] += e_j * s_V[j][lid];
            o_reg[1] += e_j * s_V[j][lid + 32];
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
