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
echo "=== Done ==="
date
echo "Results written to ${REPO_ROOT}/results/experiments.jsonl"
