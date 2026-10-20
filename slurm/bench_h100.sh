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

set -euo pipefail

# Load CUDA toolkit — must match the target architecture (H100 = sm_90 → CUDA 12+)
module load cuda/12.5

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${REPO_ROOT}/build"
BENCH="${BUILD_DIR}/bench"

# ── Build ─────────────────────────────────────────────────────────────────────
echo "=== Environment ==="
srun hostname
srun nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
date
echo ""

echo "=== Build ==="
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"
cmake "${REPO_ROOT}" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90
make -j"$(nproc)"
cd "${REPO_ROOT}"
echo ""

# ── GEMM sweep ────────────────────────────────────────────────────────────────
echo "=== GEMM Benchmarks (H100) ==="

SIZES=(512 1024 2048 4096)
IMPLS=(cuda_naive cuda_tiled)
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
echo "=== Done ==="
date
echo "Results written to ${REPO_ROOT}/results/experiments.jsonl"
