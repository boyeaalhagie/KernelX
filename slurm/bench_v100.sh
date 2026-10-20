#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# KernelX — ROSIE V100 benchmark job
#
# NOTE: The dgx1 nodes (V100) currently run CUDA 10.1 and CMake 3.16.3, which
# are below the project's minimum requirements (CUDA 11+, CMake 3.18+).
# A Singularity container with a modern toolchain is required.
#
# TODO: build a container image with CUDA 12.x + CMake 3.28 and update this
# script to use:
#   singularity exec --nv -B /data:/data <image.sif> ./build.sh
#
# Submit:   sbatch slurm/bench_v100.sh
# Monitor:  squeue -u boyea@ad.msoe.edu
# Output:   tail -f job_<JOBID>.out
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name="kernelx_bench_v100"
#SBATCH --output=job_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=boyea@ad.msoe.edu
#SBATCH --partition=dgx
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-gpu=8

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR}"

echo "=== Environment ==="
srun hostname
srun nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
date
echo ""

echo "=== Toolchain check ==="
cmake --version
nvcc --version 2>/dev/null || echo "nvcc: not found"
echo ""

echo "ERROR: V100 nodes require CUDA 11+ and CMake 3.18+."
echo "Current node has CUDA 10.1 and CMake 3.16.3."
echo "Rebuild this script to use a Singularity container."
echo "See slurm/bench_v100.sh for details."
exit 1
