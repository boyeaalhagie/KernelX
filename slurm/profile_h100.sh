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
PARSE="${REPO_ROOT}/scripts/parse_ncu.py"
PROFILES_DIR="${REPO_ROOT}/profiles"

module load cuda/12.1
export CUDACXX="$(which nvcc)"

# ncu (Nsight Compute CLI) is part of the full CUDA toolkit installer but is
# NOT installed as a system package on ROSIE — only nvcc is available via apt.
# Attempt to find it in common HPC locations before giving up.
_find_ncu() {
    # 1. User-local toolkit install (installed via runfile --toolkitpath)
    [ -x "${HOME}/cuda_local/bin/ncu" ] && { echo "${HOME}/cuda_local/bin/ncu"; return; }

    # 2. Alongside nvcc (standard toolkit layout)
    local nvcc_dir
    nvcc_dir="$(dirname "$(which nvcc 2>/dev/null)")"
    [ -x "${nvcc_dir}/ncu" ] && { echo "${nvcc_dir}/ncu"; return; }

    # 3. CUDA_HOME set by the module
    [ -x "${CUDA_HOME}/bin/ncu" ] && { echo "${CUDA_HOME}/bin/ncu"; return; }

    # 4. Versioned installations common on cluster shared filesystems
    for d in /usr/local/cuda/bin /usr/local/cuda-12*/bin /opt/cuda/bin; do
        [ -x "${d}/ncu" ] && { echo "${d}/ncu"; return; }
    done

    return 1
}
NCU="$(_find_ncu)" || NCU=""

echo "=== Environment ==="
srun hostname
GPU_NAME=$(srun nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "GPU: ${GPU_NAME}"
echo "nvcc: $(which nvcc 2>/dev/null || echo NOT FOUND)"
echo "ncu:  ${NCU:-NOT FOUND}"
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

# Verify ncu is available
if [ -z "${NCU}" ]; then
    echo "[error] ncu (Nsight Compute CLI) not found on this node."
    echo "        ncu requires a full CUDA toolkit installation, not just the"
    echo "        system nvcc package.  On ROSIE, contact sysadmin to install"
    echo "        cuda-tools-12-1 or run via Singularity with a full CUDA image."
    echo ""
    echo "        Diagnostic — nvcc dir contents:"
    ls "$(dirname "$(which nvcc 2>/dev/null)")" 2>/dev/null | tr '\n' ' '
    echo ""
    exit 1
fi
echo "=== ncu version ==="
"${NCU}" --version | head -3
echo ""

# Quick capability check — ncu needs perf_event_paranoid <= 2 or CAP_SYS_ADMIN.
# Run a trivial profile; if it errors we report clearly rather than silently skipping.
echo "=== ncu capability check ==="
"${NCU}" --metrics "sm__throughput.avg.pct_of_peak_sustained_elapsed" \
    --csv --page raw \
    /bin/true 2>&1 | head -5
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
    # stderr goes to stdout (2>&1) so ncu errors appear in the job log.
    "${NCU}" \
        --metrics "${NCU_METRICS}" \
        --csv --page raw \
        "${BENCH}" \
            --workload  "${workload}" \
            --implementation "${impl}" \
            --m "${m}" --n "${n}" --k "${k}" \
            --dtype "${dtype}" \
            --warmup 0 --iterations 1 \
            --no-verify --no-save \
    2>&1 \
    | python3 "${PARSE}" \
        --workload  "${workload}" \
        --implementation "${impl}" \
        --m "${m}" --n "${n}" --k "${k}" \
        --dtype "${dtype}" \
        --gpu  "${GPU_NAME}" \
        --output "${PROFILES_DIR}/ncu_records.jsonl" \
    || echo "  [warn] profiling failed for ${workload}/${impl} — skipping"

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
