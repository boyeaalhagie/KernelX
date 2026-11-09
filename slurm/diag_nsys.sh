#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# KernelX — nsys capability diagnostic
#
# Tests whether Nsight Systems can profile CUDA kernels on ROSIE H100.
# nsys uses CUDA API callbacks (not hardware perf counters), so it may
# work even with perf_event_paranoid=4 / ERR_NVGPUCTRPERM.
#
# Submit:  sbatch slurm/diag_nsys.sh
# Output:  tail -f diag_nsys_<JOBID>.out
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name="kx_nsys_diag"
#SBATCH --output=diag_nsys_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=boyea@ad.msoe.edu
#SBATCH --partition=dgxh100
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR}"
BUILD_DIR="${REPO_ROOT}/build_h100"
BENCH="${BUILD_DIR}/bench"
NSYS="${HOME}/cuda_local/bin/nsys"

module load cuda/12.1

echo "=== Environment ==="
hostname
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
echo "perf_event_paranoid: $(cat /proc/sys/kernel/perf_event_paranoid)"
echo "nsys: ${NSYS}"
"${NSYS}" --version | head -1
date
echo ""

echo "=== nsys capability test ==="
# --trace=cuda captures CUDA API calls and kernel launches via driver callbacks
# --stats=true prints a summary table to stdout without needing a GUI
# --output=/tmp/... avoids cluttering the repo
echo "--- gemm / cuda_naive  M=1024 ---"
timeout 60s "${NSYS}" profile \
    --trace=cuda \
    --stats=true \
    --output=/tmp/nsys_diag_$$ \
    --force-overwrite=true \
    "${BENCH}" \
        --workload gemm \
        --implementation cuda_naive \
        --m 1024 --n 1024 --k 1024 \
        --dtype fp32 \
        --warmup 2 --iterations 5 \
        --no-verify --no-save \
2>&1
echo "nsys exit: $?"
echo ""

echo "--- attention / cuda_flash  N=1024 ---"
timeout 60s "${NSYS}" profile \
    --trace=cuda \
    --stats=true \
    --output=/tmp/nsys_diag2_$$ \
    --force-overwrite=true \
    "${BENCH}" \
        --workload attention \
        --implementation cuda_flash \
        --m 1024 --n 64 --k 1 \
        --dtype fp32 \
        --warmup 2 --iterations 5 \
        --no-verify --no-save \
2>&1
echo "nsys exit: $?"

rm -f /tmp/nsys_diag_$$.* /tmp/nsys_diag2_$$.*
echo ""
echo "=== Done ==="
date
