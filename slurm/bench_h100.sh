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
echo "=== GEMM Benchmarks — Triton V5 (H100) ==="
# Triton needs PyTorch.  Set up a venv on first run; reuse on subsequent runs.
# --system-site-packages picks up any cluster-wide torch/triton if present.
VENV="${REPO_ROOT}/.venv_triton"
# Check importability rather than directory existence — handles the case where
# the venv was set up by a T4 job (which has internet access) and we just reuse it.
if [ ! -f "${VENV}/bin/python3" ] || ! "${VENV}/bin/python3" -c "import triton, torch" 2>/dev/null; then
    echo "  Setting up Triton venv at ${VENV} ..."
    python3 -m venv --system-site-packages "${VENV}"
    # cu121 wheel is compatible with CUDA 12.x runtime
    "${VENV}/bin/pip" install --quiet \
        triton \
        torch --index-url https://download.pytorch.org/whl/cu121 \
        || echo "  [warn] pip install incomplete — Triton benchmarks may be skipped"
fi

if "${VENV}/bin/python3" -c "import triton, torch" 2>/dev/null; then
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
