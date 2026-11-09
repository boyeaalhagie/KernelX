#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# KernelX — ROSIE T4 benchmark job
#
# T4 is used for dev/test and smaller-scale benchmarks.
# Peak FP32 throughput: ~8.1 TFLOP/s  (vs V100: ~14, H100: ~60)
# Memory bandwidth:     ~300 GB/s      (vs V100: ~900, H100: ~3350)
#
# Submit:   sbatch slurm/bench_t4.sh
# Monitor:  squeue -u boyea@ad.msoe.edu
# Output:   tail -f job_<JOBID>.out
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name="kernelx_bench_t4"
#SBATCH --output=job_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=boyea@ad.msoe.edu
#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --gres=gpu:t4:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=16G

set -euo pipefail

# SLURM_SUBMIT_DIR is set by SLURM to the directory where sbatch was invoked
REPO_ROOT="${SLURM_SUBMIT_DIR}"
BUILD_DIR="${REPO_ROOT}/build_t4"
BENCH="${BUILD_DIR}/bench"

module load cuda/12.9
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
cmake "${REPO_ROOT}" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75
make -j"$(nproc)"
cd "${REPO_ROOT}"
echo ""

echo "=== GEMM Benchmarks (T4) ==="

# Smaller sizes for T4 — 4096³ at low bandwidth will take a while
SIZES=(512 1024 2048)
IMPLS=(cuda_naive cuda_tiled cuda_tuned cuda_cutlass)
WARMUP=10
ITERS=50

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
echo "=== LayerNorm Benchmarks (T4) ==="
LN_SIZES=(768 1024 2048 4096)
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
echo "=== Softmax Benchmarks (T4) ==="
SOFTMAX_SIZES=(1024 4096 8192 32768)
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
echo "=== Attention Benchmarks (T4) ==="
ATTN_N_SIZES=(512 1024 2048)
ATTN_IMPLS=(cuda_naive cuda_flash cuda_flash_fp16)

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
echo "=== Reduction Benchmarks (T4) ==="
RED_N_SIZES=(256 1024 4096 16384 65536)
RED_IMPLS=(cuda_naive cuda_warp)

for N_SIZE in "${RED_N_SIZES[@]}"; do
    for IMPL in "${RED_IMPLS[@]}"; do
        echo "--- Reduction ${IMPL}  M=1024 N=${N_SIZE} ---"
        srun "${BENCH}" \
            --workload reduction \
            --implementation "${IMPL}" \
            --m 1024 --n "${N_SIZE}" --k 1 \
            --dtype fp32 \
            --warmup "${WARMUP}" \
            --iterations "${ITERS}" \
            --no-verify
    done
done

echo ""
echo "=== GEMM Benchmarks — Triton V5 (T4) ==="
VENV="${REPO_ROOT}/.venv_triton"
_triton_ready=false

if [ -f "${VENV}/bin/python3" ] && "${VENV}/bin/python3" -c "import triton, torch" 2>/dev/null; then
    _triton_ready=true
else
    echo "  Setting up Triton venv at ${VENV} ..."
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
echo "=== Data Movement Benchmarks (T4) ==="
# T4 uses PCIe 3.0 × 16: ~16 GB/s unidirectional (vs H100 PCIe 5.0 ~64 GB/s).
# Cap at 256 MB to keep job time reasonable.
DM_N_SIZES=(1048576 4194304 16777216 67108864)
DM_IMPLS=(pageable pinned pinned_async bidirectional)

for N_SIZE in "${DM_N_SIZES[@]}"; do
    for IMPL in "${DM_IMPLS[@]}"; do
        SIZE_MB=$(( N_SIZE * 4 / 1024 / 1024 ))
        echo "--- DataMove ${IMPL}  N=${N_SIZE} (${SIZE_MB} MB) ---"
        srun "${BENCH}" \
            --workload datamove \
            --implementation "${IMPL}" \
            --m 1 --n "${N_SIZE}" --k 1 \
            --dtype fp32 \
            --warmup "${WARMUP}" \
            --iterations "${ITERS}" \
            --no-verify
    done
done

echo ""
echo "=== Done ==="
date
echo "Results written to ${REPO_ROOT}/results/experiments.jsonl"
