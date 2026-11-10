#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# KernelX — Nsight Compute profiling job (H100)
#
# Runs ncu on representative kernels for each workload, parses CSV output
# via scripts/parse_ncu.py, and appends profile records to
# profiles/ncu_records.jsonl.
#
# Submit:  sbatch slurm/profile_h100.sh
# Output:  tail -f profile_h100_<JOBID>.out
# Results: profiles/ncu_records.jsonl
#
# Captured metrics per kernel:
#   Speed-of-Light:   compute %, HBM %, L1 %, L2 %
#   FLOPs:            fadd + fmul + 2×ffma (→ arithmetic intensity)
#   HBM traffic:      dram_bytes_read + dram_bytes_write
#   Occupancy:        achieved %, register/shmem/warp limits
#   Warp stalls:      long_scoreboard, math_pipe_throttle, mio_throttle
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name="kx_profile"
#SBATCH --output=profile_h100_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=your_username@your_institution.edu
#SBATCH --partition=dgxh100
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=32G

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR}"
BUILD_DIR="${REPO_ROOT}/build_h100"
BENCH="${BUILD_DIR}/bench"
PARSE="${REPO_ROOT}/scripts/parse_ncu.py"
PROFILES_DIR="${REPO_ROOT}/profiles"

# Singularity container that includes ncu (/usr/local/bin/ncu).
# On ROSIE, ncu does not exist on the host — it lives inside the CUDA/PyTorch
# container.  Set SIF_IMAGE to the same .sif you used for InferX, or any
# container built from nvcr.io/nvidia/pytorch:24.xx-py3 (includes full toolkit).
SIF_IMAGE="${SIF_IMAGE:-/data/ai_club/containers/pytorch_24.08.sif}"
NCU_BIN="/usr/local/bin/ncu"

# Wrapper: run ncu inside the container with GPU passthrough and repo mounted.
# --clock-control none  — skips GPU clock locking, which requires root on HPC.
ncu() {
    singularity exec --nv \
        -B "${REPO_ROOT}:${REPO_ROOT}" \
        -B /tmp:/tmp \
        "${SIF_IMAGE}" \
        "${NCU_BIN}" --clock-control none "$@"
}

module load cuda/12.1
export CUDACXX="$(which nvcc)"

echo "=== Environment ==="
srun hostname
GPU_NAME=$(srun nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "GPU: ${GPU_NAME}"
echo "nvcc: $(which nvcc 2>/dev/null || echo NOT FOUND)"
echo "SIF:  ${SIF_IMAGE}"
date
echo ""

echo "=== Build ==="
rm -f "${BUILD_DIR}/CMakeCache.txt"
rm -rf "${BUILD_DIR}/CMakeFiles"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"
cmake "${REPO_ROOT}" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90
make -j"$(nproc)"
cd "${REPO_ROOT}"
echo ""

# Verify the container image exists
if [ ! -f "${SIF_IMAGE}" ]; then
    echo "[error] Singularity image not found: ${SIF_IMAGE}"
    echo "        Set SIF_IMAGE=/path/to/your/container.sif and resubmit."
    exit 1
fi

echo "=== ncu version ==="
ncu --version | head -3
echo ""

echo "=== ncu capability check ==="
echo "perf_event_paranoid: $(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo unknown)"
echo "ncu --query-metrics (10s timeout):"
timeout 10s ncu --query-metrics 2>&1 | head -3 \
    || echo "  [timed out or failed]"
echo ""

mkdir -p "${PROFILES_DIR}"

# ── Metric set ────────────────────────────────────────────────────────────────
# Speed-of-Light: compute, HBM, L1, L2
# FLOPs:          fadd, fmul, ffma (2× weight)
# HBM traffic:    bytes read + write  (→ arithmetic intensity)
# Occupancy:      achieved %, register/shmem/warp limits
# Warp stalls:    long_scoreboard (memory dep), math_pipe_throttle, mio_throttle
NCU_METRICS="sm__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__throughput.avg.pct_of_peak_sustained_elapsed,\
lts__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__sass_thread_inst_executed_op_fadd_pred_on.sum,\
sm__sass_thread_inst_executed_op_fmul_pred_on.sum,\
sm__sass_thread_inst_executed_op_ffma_pred_on.sum,\
dram__bytes_read.sum,\
dram__bytes_write.sum,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
launch__occupancy_limit_registers,\
launch__occupancy_limit_shared_mem,\
launch__occupancy_limit_warps,\
smsp__warp_issue_stall_long_scoreboard_per_warp_active.pct,\
smsp__warp_issue_stall_math_pipe_throttle_per_warp_active.pct,\
smsp__warp_issue_stall_mio_throttle_per_warp_active.pct"

# ── Helper: profile one bench invocation ─────────────────────────────────────
# Usage: profile_kernel <workload> <impl> <m> <n> <k> [dtype]
profile_kernel() {
    local workload=$1
    local impl=$2
    local m=$3
    local n=$4
    local k=$5
    local dtype=${6:-fp32}

    echo "--- ncu  ${workload} / ${impl}  M=${m} N=${n} K=${k} ---"

    # Run with warmup=0, iterations=1 so ncu captures exactly one kernel set.
    # --target-processes all requires CAP_PERFMON on Linux 5.8+; try without it
    # first (bench is ncu's direct target, not a child), then fall back with it.
    # 120s timeout guards against permission-induced hangs.
    local ncu_tmp
    ncu_tmp="$(mktemp /tmp/ncu_XXXXXX.csv)"

    timeout 120s ncu \
        --metrics "${NCU_METRICS}" \
        --csv --page raw \
        "${BENCH}" \
            --workload  "${workload}" \
            --implementation "${impl}" \
            --m "${m}" --n "${n}" --k "${k}" \
            --dtype "${dtype}" \
            --warmup 0 --iterations 1 \
            --no-verify --no-save \
    > "${ncu_tmp}" 2>&1
    local ncu_exit=$?

    # If no kernel data, retry with --target-processes all
    if ! grep -q "^\"" "${ncu_tmp}" 2>/dev/null && [ ${ncu_exit} -eq 0 ]; then
        timeout 120s ncu \
            --metrics "${NCU_METRICS}" \
            --csv --page raw \
            --target-processes all \
            "${BENCH}" \
                --workload  "${workload}" \
                --implementation "${impl}" \
                --m "${m}" --n "${n}" --k "${k}" \
                --dtype "${dtype}" \
                --warmup 0 --iterations 1 \
                --no-verify --no-save \
        > "${ncu_tmp}" 2>&1 || true
    fi

    echo "  [ncu raw] $(wc -l < "${ncu_tmp}") lines, first 5:"
    head -5 "${ncu_tmp}" | sed 's/^/    /'

    python3 "${PARSE}" \
        --workload  "${workload}" \
        --implementation "${impl}" \
        --m "${m}" --n "${n}" --k "${k}" \
        --dtype "${dtype}" \
        --gpu  "${GPU_NAME}" \
        --output "${PROFILES_DIR}/ncu_records.jsonl" \
        < "${ncu_tmp}" \
    || echo "  [warn] profiling failed for ${workload}/${impl} — skipping"

    rm -f "${ncu_tmp}"

    echo ""
}

# ─────────────────────────────────────────────────────────────────────────────
# GEMM — profile N=4096 for the key variants
# (N=4096 is the regime where bandwidth vs compute tradeoffs are clearest)
# ─────────────────────────────────────────────────────────────────────────────
echo "=== Profiling GEMM (N=4096) ==="
profile_kernel gemm cuda_naive   4096 4096 4096
profile_kernel gemm cuda_tiled   4096 4096 4096
profile_kernel gemm cuda_tuned   4096 4096 4096
profile_kernel gemm cuda_cutlass 4096 4096 4096

# ─────────────────────────────────────────────────────────────────────────────
# Softmax — M=1024 rows, N=4096 (saturates HBM bandwidth for cuda_online)
# ─────────────────────────────────────────────────────────────────────────────
echo "=== Profiling Softmax (M=1024, N=4096) ==="
profile_kernel softmax cuda_naive  1024 4096 1
profile_kernel softmax cuda_online 1024 4096 1

# ─────────────────────────────────────────────────────────────────────────────
# LayerNorm — M=1024 rows, N=4096
# ─────────────────────────────────────────────────────────────────────────────
echo "=== Profiling LayerNorm (M=1024, N=4096) ==="
profile_kernel layernorm cuda_naive   1024 4096 1
profile_kernel layernorm cuda_welford 1024 4096 1

# ─────────────────────────────────────────────────────────────────────────────
# Attention — N=2048 (memory vs compute tradeoff is visible at this size)
# ─────────────────────────────────────────────────────────────────────────────
echo "=== Profiling Attention (N=2048, D=64) ==="
profile_kernel attention cuda_naive 2048 64 1
profile_kernel attention cuda_flash 2048 64 1

# ─────────────────────────────────────────────────────────────────────────────
# Reduction — N=4096 (HBM-bound) and N=256 (shows barrier-overhead regime)
# ─────────────────────────────────────────────────────────────────────────────
echo "=== Profiling Reduction ==="
profile_kernel reduction cuda_naive  1024  256 1
profile_kernel reduction cuda_warp   1024  256 1
profile_kernel reduction cuda_naive  1024 4096 1
profile_kernel reduction cuda_warp   1024 4096 1

echo "=== Done ==="
date
echo "Profile records written to ${PROFILES_DIR}/ncu_records.jsonl"
echo "Run: python scripts/analyze.py --workload gemm"
