#!/usr/bin/env python3
"""
gpu_compare.py — compare the same implementation across GPU generations.

Reads results/experiments.jsonl and produces a cross-GPU performance table
for a given workload and implementation.  Useful for answering:

  "How does cuda_naive GEMM scale across T4 → V100 → H100?"
  "Does the roofline bottleneck change between architectures?"

Usage
-----
  python scripts/gpu_compare.py --implementation cuda_naive --workload gemm
  python scripts/gpu_compare.py --implementation cuda_naive --m 2048 --n 2048 --k 2048
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict


# Approximate hardware specs — used to contextualise throughput numbers.
# Values are representative FP32 peaks and HBM bandwidth (not authoritative).
GPU_SPECS = {
    "T4":   {"tflops_fp32": 8.1,  "bandwidth_gbs": 300},
    "V100": {"tflops_fp32": 14.0, "bandwidth_gbs": 900},
    "A100": {"tflops_fp32": 19.5, "bandwidth_gbs": 2000},
    "H100": {"tflops_fp32": 60.0, "bandwidth_gbs": 3350},
}


def load_records(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def gpu_shortname(full_name: str) -> str:
    for key in GPU_SPECS:
        if key in full_name:
            return key
    return full_name.split()[1] if len(full_name.split()) > 1 else full_name


def pct_of_peak(tflops: float, gpu_short: str) -> str:
    spec = GPU_SPECS.get(gpu_short)
    if spec is None:
        return "  ?"
    pct = tflops / spec["tflops_fp32"] * 100.0
    return f"{pct:5.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-GPU performance comparison for KernelX experiments."
    )
    parser.add_argument("--file",           default="results/experiments.jsonl")
    parser.add_argument("--implementation", default=None, help="Filter by implementation")
    parser.add_argument("--workload",       default="gemm")
    parser.add_argument("--m",              type=int, default=None)
    parser.add_argument("--n",              type=int, default=None)
    parser.add_argument("--k",              type=int, default=None)
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f"error: {path} not found", file=sys.stderr)
        sys.exit(1)

    records = load_records(str(path))

    # Filter
    def keep(r: dict) -> bool:
        if r.get("workload") != args.workload:
            return False
        if args.implementation and r.get("implementation") != args.implementation:
            return False
        if args.m and r.get("m") != args.m:
            return False
        if args.n and r.get("n") != args.n:
            return False
        if args.k and r.get("k") != args.k:
            return False
        return True

    records = [r for r in records if keep(r)]

    if not records:
        print("No matching records found.", file=sys.stderr)
        sys.exit(1)

    # Group by (implementation, m, n, k, dtype) then by GPU — keep latest per GPU
    groups: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for r in records:
        key  = (r["implementation"], r["m"], r["n"], r["k"], r["dtype"])
        gpu  = gpu_shortname(r.get("gpu", "?"))
        groups[key][gpu] = r   # later records overwrite earlier ones

    for (impl, m, n, k, dtype), by_gpu in sorted(groups.items()):
        print(f"\n  {impl}  |  {args.workload}  M={m} N={n} K={k}  dtype={dtype}\n")

        col = 10
        header = (
            f"  {'GPU':<{col}}"
            f"{'Median':>10}"
            f"{'TFLOP/s':>10}"
            f"{'% Peak':>10}"
            f"  Commit"
        )
        print(header)
        print("  " + "─" * (len(header) - 2))

        # Sort by throughput descending
        for gpu, r in sorted(by_gpu.items(),
                             key=lambda kv: kv[1]["tflops"], reverse=True):
            pct = pct_of_peak(r["tflops"], gpu)
            print(
                f"  {gpu:<{col}}"
                f"{r['median_ms']:>9.3f}ms"
                f"{r['tflops']:>10.3f}"
                f"{pct:>10}"
                f"  {r.get('git_commit', '?')}"
            )

        # Speedup relative to slowest
        if len(by_gpu) >= 2:
            vals = sorted(by_gpu.values(), key=lambda r: r["tflops"])
            slowest = vals[0]
            fastest = vals[-1]
            ratio = fastest["tflops"] / slowest["tflops"]
            print(
                f"\n  {gpu_shortname(fastest['gpu'])} is {ratio:.1f}× faster than "
                f"{gpu_shortname(slowest['gpu'])} on this workload."
            )

    print()


if __name__ == "__main__":
    main()
