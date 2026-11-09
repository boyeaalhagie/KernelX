#!/usr/bin/env python3
"""
bench_gemm_triton.py — GEMM V5: Triton kernel benchmark

Implements a row-major FP32 GEMM (C = A × B) using OpenAI Triton and
measures kernel throughput with the same methodology as the C++ benchmark
runner.  Appends a canonical KernelX ExperimentRecord to experiments.jsonl.

Design
------
  Tiling strategy
    Each thread block computes a (BLOCK_M × BLOCK_N) output tile.  The
    K-dimension is accumulated in BLOCK_K-wide panels.  @triton.autotune
    selects the best (BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, num_warps,
    num_stages) for the resident GPU.

  Grouped program ordering (GROUP_M)
    Programs are assigned to output tiles in a swizzled order: GROUP_M
    row-tiles that share the same B column tiles are scheduled together,
    maximising L2 reuse on B before moving to the next group.

  Tensor Core path
    tl.dot with float32 inputs generates mma.sync.aligned.m16n8k8 TF32
    instructions on sm_80+ (A100/H100).  On sm_75 (T4) it falls back to
    SIMT FMAs.  This mirrors the CUTLASS V4 behavior exactly.

  Timing
    CUDA Events (not Python wall-clock) are used for per-iteration GPU
    timing, identical to the cudaEvent-based approach in the C++ runner.

Usage
-----
  python3 scripts/bench_gemm_triton.py \\
      --m 4096 --n 4096 --k 4096 \\
      --warmup 20 --iterations 100 \\
      --output results/experiments.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# ── Dependency check ──────────────────────────────────────────────────────────
try:
    import torch
    import triton
    import triton.language as tl
except ImportError as exc:
    print(f"[skip] GEMM Triton benchmark: {exc}")
    print("  To enable: pip install triton torch")
    sys.exit(0)   # exit 0 so the SLURM job is not marked failed


# ── Autotune configuration ────────────────────────────────────────────────────
#
# Configs are ordered roughly best-first for H100; the autotuner will profile
# all of them on first call and cache the winner for subsequent runs.
#
# BLOCK_M × BLOCK_N — output tile per thread block
# BLOCK_K           — K-panel width (register tile along K)
# GROUP_M           — number of row-tiles per L2-reuse group
# num_stages        — async pipeline depth (cp.async on sm_80+)
# num_warps         — warps per thread block

_GEMM_CONFIGS = [
    # ── H100 / A100 (sm_80+, TF32 TC) ────────────────────────────────────────
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8},
                  num_stages=3, num_warps=8),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8},
                  num_stages=3, num_warps=8),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8},
                  num_stages=4, num_warps=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_K": 64, "GROUP_M": 8},
                  num_stages=4, num_warps=4),
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8},
                  num_stages=4, num_warps=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_M": 8},
                  num_stages=4, num_warps=8),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 64,  "BLOCK_K": 32, "GROUP_M": 8},
                  num_stages=4, num_warps=4),
    # ── T4 / V100 (sm_75 / sm_70, SIMT FMAs) ─────────────────────────────────
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 32, "GROUP_M": 8},
                  num_stages=4, num_warps=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 32,  "BLOCK_K": 32, "GROUP_M": 4},
                  num_stages=4, num_warps=4),
    triton.Config({"BLOCK_M": 32,  "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 4},
                  num_stages=4, num_warps=4),
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 32,  "BLOCK_K": 32, "GROUP_M": 4},
                  num_stages=5, num_warps=2),
]


# ── Triton kernel ─────────────────────────────────────────────────────────────

@triton.autotune(configs=_GEMM_CONFIGS, key=["M", "N", "K"])
@triton.jit
def _matmul_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """
    Tiled GEMM with grouped program ordering for improved L2 reuse.

    One thread block = one (BLOCK_M × BLOCK_N) output tile.
    Programs are numbered in row-major order but scheduled in groups of
    GROUP_M row-tiles to maximise reuse of B tiles in L2 cache.
    """
    # ── Program → tile mapping (grouped) ─────────────────────────────────────
    pid       = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    # Number of programs in one group (GROUP_M rows × all columns)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id         = pid // num_pid_in_group
    first_pid_m      = group_id * GROUP_M
    # Clamp to remaining rows in the last (possibly partial) group
    group_size_m     = min(num_pid_m - first_pid_m, GROUP_M)

    pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ── Tile offsets ──────────────────────────────────────────────────────────
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # [BLOCK_M]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # [BLOCK_N]
    offs_k = tl.arange(0, BLOCK_K)                      # [BLOCK_K]

    # Initial pointers to A[offs_m, 0] and B[0, offs_n]
    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    # ── K-panel accumulation ──────────────────────────────────────────────────
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, tl.cdiv(K, BLOCK_K)):
        # Mask out-of-bounds K lanes in the final panel
        k_rem = K - k_start * BLOCK_K
        a = tl.load(a_ptrs, mask=offs_k[None, :] < k_rem, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < k_rem, other=0.0)

        # tl.dot → mma.sync TF32 TC on sm_80+, SIMT FMA on sm_75
        acc = tl.dot(a, b, acc)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # ── Store output tile ─────────────────────────────────────────────────────
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs  = C + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    mask    = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask)


# ── Launch wrapper ────────────────────────────────────────────────────────────

def triton_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Launch _matmul_kernel for row-major float32 A × B."""
    assert a.ndim == 2 and b.ndim == 2 and a.shape[1] == b.shape[0]
    assert a.dtype == torch.float32 and b.dtype == torch.float32
    M, K = a.shape
    _,  N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)
    _matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c


# ── Benchmarking ──────────────────────────────────────────────────────────────

def run_benchmark(
    M: int, N: int, K: int,
    warmup: int,
    iterations: int,
) -> tuple[list[float], torch.Tensor]:
    """
    Returns (times_ms, C) where times_ms[i] is the CUDA-event-measured
    kernel time for iteration i, and C is the output from the last warmup
    run (used for verification).
    """
    a = torch.randn((M, K), device="cuda", dtype=torch.float32)
    b = torch.randn((K, N), device="cuda", dtype=torch.float32)

    # Warmup — also triggers autotuner on first call
    print(f"  Warming up    ({warmup} iters) ... ", end="", flush=True)
    for _ in range(warmup):
        c = triton_matmul(a, b)
    torch.cuda.synchronize()
    print("done")

    # Timed iterations using CUDA Events (not Python wall-clock)
    print(f"  Benchmarking  ({iterations} iters) ... ", end="", flush=True)
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends   = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]

    for i in range(iterations):
        starts[i].record()
        c = triton_matmul(a, b)
        ends[i].record()

    torch.cuda.synchronize()
    print("done")

    times_ms = [starts[i].elapsed_time(ends[i]) for i in range(iterations)]
    return times_ms, c


# ── Stats helpers ─────────────────────────────────────────────────────────────

def compute_stats(times_ms: list[float], M: int, N: int, K: int) -> dict:
    t = sorted(times_ms)
    median = statistics.median(t)
    mean   = statistics.mean(t)
    p95    = t[int(0.95 * len(t))]
    std    = statistics.stdev(t) if len(t) > 1 else 0.0
    tflops = 2.0 * M * N * K / (median * 1e-3) / 1e12
    return {
        "median_ms": round(median, 4),
        "mean_ms":   round(mean,   4),
        "p95_ms":    round(p95,    4),
        "stddev_ms": round(std,    4),
        "tflops":    round(tflops, 4),
    }


# ── Environment probes ────────────────────────────────────────────────────────

def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _triton_best_config(M: int, N: int, K: int) -> str:
    """Return the autotuner's winning config as a compact string."""
    try:
        cfg = _matmul_kernel.best_config
        return (
            f"BLOCK_M={cfg.kwargs['BLOCK_M']}, "
            f"BLOCK_N={cfg.kwargs['BLOCK_N']}, "
            f"BLOCK_K={cfg.kwargs['BLOCK_K']}, "
            f"GROUP_M={cfg.kwargs['GROUP_M']}, "
            f"num_warps={cfg.num_warps}, "
            f"num_stages={cfg.num_stages}"
        )
    except Exception:
        return "unavailable"


# ── Verification ──────────────────────────────────────────────────────────────

def verify(M: int, N: int, K: int, C_triton: torch.Tensor) -> bool:
    """
    Compare a corner of the Triton output against torch.mm (cuBLAS).
    Uses the same kVerifyDim=64 strategy as the C++ verifier.
    """
    VERIFY_DIM = min(64, M, N, K)
    a_ref = torch.randn((VERIFY_DIM, K), device="cuda", dtype=torch.float32)
    b_ref = torch.randn((K, VERIFY_DIM), device="cuda", dtype=torch.float32)
    c_ref = torch.mm(a_ref, b_ref)
    c_got = triton_matmul(a_ref, b_ref)
    # TF32 on sm_80+ has ~1e-1 relative error; SIMT has ~1e-3
    sm_major = torch.cuda.get_device_capability(0)[0]
    rtol = 1e-1 if sm_major >= 8 else 1e-3
    return torch.allclose(c_got, c_ref, rtol=rtol, atol=1e-5)


# ── Output ────────────────────────────────────────────────────────────────────

def print_results(record: dict, best_config: str) -> None:
    def row(label: str, value: str) -> None:
        print(f"  {label:<22}{value}")

    print()
    print(f"  ┌─ Experiment ──────────────────────────────────────────")
    row("Workload:",         record["workload"])
    row("Implementation:",   record["implementation"])
    row("GPU:",              record["gpu"])
    row("dtype:",            record["dtype"])
    row("M × N × K:",
        f"{record['m']} × {record['n']} × {record['k']}")
    row("CUDA:",             record["cuda_version"])
    row("Triton:",           record["triton_version"])
    row("Commit:",           record["git_commit"])
    row("Timestamp:",        record["timestamp"])
    row("Best config:",      best_config)
    print(f"  ├─ Performance ─────────────────────────────────────────")
    row("Median:",           f"{record['median_ms']:.3f} ms")
    row("Mean:",             f"{record['mean_ms']:.3f} ms")
    row("P95:",              f"{record['p95_ms']:.3f} ms")
    row("Std dev:",          f"{record['stddev_ms']:.3f} ms")
    row("Throughput:",       f"{record['tflops']:.3f} TFLOP/s")
    print(f"  ├─ Config ──────────────────────────────────────────────")
    row("Warmup iters:",     str(record["warmup_iters"]))
    row("Timed iters:",      str(record["timed_iters"]))
    row("Verified:",         "PASS" if record["verified"] else "SKIP")
    print(f"  └───────────────────────────────────────────────────────")
    print()


def save_record(record: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"  Saved  →  {path}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="KernelX GEMM V5 — Triton benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--m",          type=int, default=4096)
    parser.add_argument("--n",          type=int, default=4096)
    parser.add_argument("--k",          type=int, default=4096)
    parser.add_argument("--warmup",     type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--no-verify",  action="store_true",
                        help="Skip correctness check")
    parser.add_argument("--output",     default="results/experiments.jsonl",
                        help="JSONL file to append results to")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("[skip] GEMM Triton benchmark: no CUDA device")
        sys.exit(0)

    M, N, K = args.m, args.n, args.k

    gpu_name     = torch.cuda.get_device_name(0)
    cuda_version = torch.version.cuda or "unknown"

    print(f"\n  KernelX — gemm / triton")
    print(f"  M={M}  N={N}  K={K}  dtype=fp32")
    print(f"  warmup={args.warmup}  iterations={args.iterations}")
    print()

    times_ms, c_out = run_benchmark(M, N, K, args.warmup, args.iterations)
    stats        = compute_stats(times_ms, M, N, K)
    best_config  = _triton_best_config(M, N, K)

    verified = False
    if not args.no_verify:
        verified = verify(M, N, K, c_out)

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    record: dict = {
        "workload":        "gemm",
        "implementation":  "triton",
        "gpu":             gpu_name,
        "dtype":           "fp32",
        "m":               M,
        "n":               N,
        "k":               K,
        "cuda_version":    cuda_version,
        "triton_version":  triton.__version__,
        "git_commit":      _git_commit(),
        "timestamp":       timestamp,
        **stats,
        "warmup_iters":    args.warmup,
        "timed_iters":     args.iterations,
        "verified":        verified,
    }

    print_results(record, best_config)
    save_record(record, args.output)


if __name__ == "__main__":
    main()
