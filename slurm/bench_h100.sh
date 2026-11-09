#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# KernelX — ROSIE H100 benchmark job
#
# H100 (Hopper, sm_90) is the primary target for production benchmarks.
# Peak FP32 throughput: ~60 TFLOP/s
# Memory bandwidth:     ~3350 GB/s (HBM3)
#
# Submit:   sbatch slurm/bench_h100.sh
# Monitor:  squeue -u boyea@ad.msoe.edu
# Output:   tail -f job_<JOBID>.out
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name="kernelx_bench"
#SBATCH --output=job_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=boyea@ad.msoe.edu
#SBATCH --partition=dgxh100
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=32G

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR}"
BUILD_DIR="${REPO_ROOT}/build_h100"
BENCH="${BUILD_DIR}/bench"

module load cuda/12.1
export CUDACXX="$(which nvcc)"

echo "=== Environment ==="
srun hostname
srun nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
date
echo ""

echo "=== Build ==="
# Remove stale CMake cache — prevents old source-file lists from persisting
# when CMakeLists.txt changes (e.g. adding per-arch source restrictions).
rm -f "${BUILD_DIR}/CMakeCache.txt"
rm -rf "${BUILD_DIR}/CMakeFiles"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"
cmake "${REPO_ROOT}" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90
make -j"$(nproc)"
cd "${REPO_ROOT}"
echo ""

echo "=== GEMM Benchmarks (H100) ==="

SIZES=(512 1024 2048 4096)
IMPLS=(cuda_naive cuda_tiled cuda_tuned cuda_cutlass)
WARMUP=20
ITERS=100

for SIZE in "${SIZES[@]}"; do
    for IMPL in "${IMPLS[@]}"; do
        echo "--- GEMM ${IMPL}  M=N=K=${SIZE} ---"
        srun "${BENCH}" \
            --workload gemm \
            --implementation "${IMPL}" \
            --m "${SIZE}" --n "${SIZE}" --k "${SIZE}" \
            --dtype fp32 \
            --warmup "${WARMUP}" \
            --iterations "${ITERS}" \
            --no-verify
    done
done

echo ""
echo "=== LayerNorm Benchmarks (H100) ==="
# M=1024 tokens; N = hidden dimension (768 → 8192 covers BERT → LLaMA-70B)
LN_SIZES=(768 1024 2048 4096 8192)
LN_IMPLS=(cuda_naive cuda_welford)

for N_SIZE in "${LN_SIZES[@]}"; do
    for IMPL in "${LN_IMPLS[@]}"; do
        echo "--- LayerNorm ${IMPL}  M=1024 N=${N_SIZE} ---"
        srun "${BENCH}" \
            --workload layernorm \
            --implementation "${IMPL}" \
            --m 1024 --n "${N_SIZE}" --k 1 \
            --dtype fp32 \
            --warmup "${WARMUP}" \
            --iterations "${ITERS}" \
            --no-verify
    done
done

echo ""
echo "=== Softmax Benchmarks (H100) ==="
# M=1024 rows (batch × heads); N varies from 1K to 128K (seq len / vocab size)
SOFTMAX_SIZES=(1024 4096 8192 32768 131072)
SOFTMAX_IMPLS=(cuda_naive cuda_online)

for N_SIZE in "${SOFTMAX_SIZES[@]}"; do
    for IMPL in "${SOFTMAX_IMPLS[@]}"; do
        echo "--- Softmax ${IMPL}  M=1024 N=${N_SIZE} ---"
        srun "${BENCH}" \
            --workload softmax \
            --implementation "${IMPL}" \
            --m 1024 --n "${N_SIZE}" --k 1 \
            --dtype fp32 \
            --warmup "${WARMUP}" \
            --iterations "${ITERS}" \
            --no-verify
    done
done

echo ""
echo "=== Attention Benchmarks (H100) ==="
# N = sequence length; D = head dimension (fixed at 64, typical for multi-head attention)
ATTN_N_SIZES=(512 1024 2048 4096 8192)
ATTN_IMPLS=(cuda_naive cuda_flash)

for N_SIZE in "${ATTN_N_SIZES[@]}"; do
    for IMPL in "${ATTN_IMPLS[@]}"; do
        echo "--- Attention ${IMPL}  N=${N_SIZE} D=64 ---"
        srun "${BENCH}" \
            --workload attention \
            --implementation "${IMPL}" \
            --m "${N_SIZE}" --n 64 --k 1 \
            --dtype fp32 \
            --warmup "${WARMUP}" \
            --iterations "${ITERS}" \
            --no-verify
    done
done

echo ""
echo "=== GEMM Benchmarks — Triton V5 (H100) ==="
# Triton needs PyTorch.  Set up a venv on first run; reuse on subsequent runs.
# --system-site-packages picks up any cluster-wide torch/triton if present.
VENV="${REPO_ROOT}/.venv_triton"
_triton_ready=false

if [ -f "${VENV}/bin/python3" ] && "${VENV}/bin/python3" -c "import triton, torch" 2>/dev/null; then
    _triton_ready=true
else
    echo "  Setting up Triton venv at ${VENV} ..."
    # --clear wipes any broken state from previous attempts (e.g. cross-node
    # Python version mismatch when T4 and H100 jobs ran concurrently).
    python3 -m venv --clear --system-site-packages "${VENV}" \
        && "${VENV}/bin/pip" install --quiet \
               triton torch --index-url https://download.pytorch.org/whl/cu121 \
        && _triton_ready=true \
        || echo "  [warn] Triton venv setup failed — skipping Triton benchmarks"
fi

if $_triton_ready; then
    for SIZE in "${SIZES[@]}"; do
        echo "--- GEMM triton  M=N=K=${SIZE} ---"
        srun "${VENV}/bin/python3" "${REPO_ROOT}/scripts/bench_gemm_triton.py" \
            --m "${SIZE}" --n "${SIZE}" --k "${SIZE}" \
            --warmup "${WARMUP}" \
            --iterations "${ITERS}" \
            --no-verify \
            --output "${REPO_ROOT}/results/experiments.jsonl" || true
    done
else
    echo "  [skip] triton or torch not importable in ${VENV}"
fi

echo ""
echo "=== Done ==="
date
echo "Results written to ${REPO_ROOT}/results/experiments.jsonl"
