#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# KernelX — Nsight Systems profiling job (H100)
#
# nsys uses CUDA driver API callbacks — works without hardware perf counters
# (perf_event_paranoid=4 / ERR_NVGPUCTRPERM does NOT block nsys).
#
# Captured data per kernel:
#   cuda_gpu_kern_sum  — kernel avg/med/min/max time, instance count
#   cuda_api_sum       — CUDA API call overhead
#   cuda_gpu_mem_time_sum / cuda_gpu_mem_size_sum — H2D/D2H bandwidth
#
# Output: profiles/nsys_records.jsonl
# Analyse: python scripts/analyze.py --workload gemm
#
# Submit:  sbatch slurm/profile_nsys.sh
# Output:  tail -f profile_nsys_<JOBID>.out
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name="kx_nsys"
#SBATCH --output=profile_nsys_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=your_username@your_institution.edu
#SBATCH --partition=dgxh100
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR}"
BUILD_DIR="${REPO_ROOT}/build_h100"
BENCH="${BUILD_DIR}/bench"
PARSE="${REPO_ROOT}/scripts/parse_nsys.py"
PROFILES_DIR="${REPO_ROOT}/profiles"
NSYS="${HOME}/cuda_local/bin/nsys"

module load cuda/12.1
export CUDACXX="$(which nvcc)"

echo "=== Environment ==="
hostname
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "GPU: ${GPU_NAME}"
echo "nsys: ${NSYS}"
"${NSYS}" --version | head -1
echo "perf_event_paranoid: $(cat /proc/sys/kernel/perf_event_paranoid)"
date
echo ""

echo "=== Build ==="
rm -f "${BUILD_DIR}/CMakeCache.txt"
rm -rf "${BUILD_DIR}/CMakeFiles"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"
cmake "${REPO_ROOT}" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90 -DCMAKE_CUDA_FLAGS="-lineinfo"
make -j"$(nproc)"
cd "${REPO_ROOT}"
echo ""

mkdir -p "${PROFILES_DIR}"

# ── Helper ────────────────────────────────────────────────────────────────────
# profile_kernel <workload> <impl> <m> <n> <k> [dtype]
profile_kernel() {
    local workload=$1
    local impl=$2
    local m=$3
    local n=$4
    local k=$5
    local dtype=${6:-fp32}

    echo "--- nsys  ${workload} / ${impl}  M=${m} N=${n} K=${k} ---"

    local nsys_tmp
    nsys_tmp="$(mktemp /tmp/nsys_XXXXXX)"

    # warmup=3 iterations=10: nsys stats will cover all 10 timed launches;
    # the summary table averages them, giving stable timing.
    # Note: --quiet is intentionally omitted — it suppresses --stats=true output.
    timeout 90s "${NSYS}" profile \
        --trace=cuda \
        --stats=true \
        --output="${nsys_tmp}" \
        --force-overwrite=true \
        "${BENCH}" \
            --workload  "${workload}" \
            --implementation "${impl}" \
            --m "${m}" --n "${n}" --k "${k}" \
            --dtype "${dtype}" \
            --warmup 3 --iterations 10 \
            --no-verify --no-save \
    2>&1 \
    | python3 "${PARSE}" \
        --workload  "${workload}" \
        --implementation "${impl}" \
        --m "${m}" --n "${n}" --k "${k}" \
        --dtype "${dtype}" \
        --gpu  "${GPU_NAME}" \
        --output "${PROFILES_DIR}/nsys_records.jsonl" \
    || echo "  [warn] nsys profiling failed for ${workload}/${impl} — skipping"

    rm -f "${nsys_tmp}".* "${nsys_tmp}"
}

# ─────────────────────────────────────────────────────────────────────────────
# GEMM — full progression at N=4096
# ─────────────────────────────────────────────────────────────────────────────
echo "=== Profiling GEMM (N=4096) ==="
profile_kernel gemm cuda_naive   4096 4096 4096
profile_kernel gemm cuda_tiled   4096 4096 4096
profile_kernel gemm cuda_tuned   4096 4096 4096
profile_kernel gemm cuda_wmma    4096 4096 4096
profile_kernel gemm cuda_cutlass 4096 4096 4096

# ─────────────────────────────────────────────────────────────────────────────
# Softmax — M=1024 rows, N=4096 (HBM-saturating size)
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
# Attention — N=2048 and N=4096 (shows FlashAttention advantage)
# ─────────────────────────────────────────────────────────────────────────────
echo "=== Profiling Attention ==="
profile_kernel attention cuda_naive      2048 64 1
profile_kernel attention cuda_flash      2048 64 1
profile_kernel attention cuda_flash_fp16 2048 64 1
profile_kernel attention cuda_naive      4096 64 1
profile_kernel attention cuda_flash      4096 64 1
profile_kernel attention cuda_flash_fp16 4096 64 1

# ─────────────────────────────────────────────────────────────────────────────
# Reduction — N=256 (barrier-overhead regime) and N=4096 (HBM-bound)
# ─────────────────────────────────────────────────────────────────────────────
echo "=== Profiling Reduction ==="
profile_kernel reduction cuda_naive  1024  256 1
profile_kernel reduction cuda_warp   1024  256 1
profile_kernel reduction cuda_naive  1024 4096 1
profile_kernel reduction cuda_warp   1024 4096 1

# ─────────────────────────────────────────────────────────────────────────────
# Data Movement — largest size (HBM-saturating)
# ─────────────────────────────────────────────────────────────────────────────
echo "=== Profiling Data Movement ==="
profile_kernel datamove pageable     1 67108864 1
profile_kernel datamove pinned       1 67108864 1
profile_kernel datamove pinned_async 1 67108864 1
profile_kernel datamove bidirectional 1 67108864 1

echo ""
echo "=== Done ==="
date
echo "Profile records written to ${PROFILES_DIR}/nsys_records.jsonl"
echo "Run: python scripts/analyze.py --workload gemm"
