#!/usr/bin/env python3
"""
compare.py — compare two KernelX experiment records.

Reads results/experiments.jsonl (or a supplied path) and prints a side-by-side
performance comparison between two implementations for the same workload and
problem size.

Usage
-----
  # Most recent run of each implementation (auto-matched by workload + M/N/K)
  python scripts/compare.py --a cuda_naive --b cuda_tiled

  # Explicit JSONL file
  python scripts/compare.py --file results/experiments.jsonl --a cuda_naive --b cuda_tiled

  # Filter by workload and size
  python scripts/compare.py --a cuda_naive --b cuda_tiled \\
      --workload gemm --m 4096 --n 4096 --k 4096
"""

import argparse
import json
import sys
from pathlib import Path


def load_records(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def match(record: dict, impl: str, workload: str | None,
          m: int | None, n: int | None, k: int | None) -> bool:
    if record.get("implementation") != impl:
        return False
    if workload and record.get("workload") != workload:
        return False
    if m and record.get("m") != m:
        return False
    if n and record.get("n") != n:
        return False
    if k and record.get("k") != k:
        return False
    return True


def find_latest(records: list[dict], impl: str, workload: str | None,
                m: int | None, n: int | None, k: int | None) -> dict | None:
    matched = [r for r in records if match(r, impl, workload, m, n, k)]
    return matched[-1] if matched else None


def fmt_ms(v: float) -> str:
    return f"{v:.3f} ms"


def fmt_tflops(v: float) -> str:
    return f"{v:.3f} TFLOP/s"


def speedup_str(a_ms: float, b_ms: float) -> str:
    ratio = a_ms / b_ms
    direction = "faster" if ratio > 1.0 else "slower"
    return f"{ratio:.2f}× {direction}"


def print_comparison(a: dict, b: dict) -> None:
    col = 26

    def row(label: str, va: str, vb: str) -> None:
        print(f"  {label:<{col}} {va:<20} {vb}")

    print()
    print(f"  {'':>{col}} {a['implementation']:<20} {b['implementation']}")
    print(f"  {'─' * (col + 44)}")
    row("GPU:",          a.get("gpu", "?"),   b.get("gpu", "?"))
    row("dtype:",        a.get("dtype", "?"), b.get("dtype", "?"))
    row("M × N × K:",    f"{a['m']}×{a['n']}×{a['k']}", f"{b['m']}×{b['n']}×{b['k']}")
    row("Commit:",       a.get("git_commit", "?"), b.get("git_commit", "?"))
    row("Timestamp:",    a.get("timestamp", "?"), b.get("timestamp", "?"))
    print(f"  {'─' * (col + 44)}")
    row("Median:",       fmt_ms(a["median_ms"]),   fmt_ms(b["median_ms"]))
    row("Mean:",         fmt_ms(a["mean_ms"]),     fmt_ms(b["mean_ms"]))
    row("P95:",          fmt_ms(a["p95_ms"]),      fmt_ms(b["p95_ms"]))
    row("Std dev:",      fmt_ms(a["stddev_ms"]),   fmt_ms(b["stddev_ms"]))
    row("Throughput:",   fmt_tflops(a["tflops"]),  fmt_tflops(b["tflops"]))
    print(f"  {'─' * (col + 44)}")

    speedup = speedup_str(a["median_ms"], b["median_ms"])
    tflops_delta = b["tflops"] - a["tflops"]
    print(f"  {'Speedup (B vs A):':>{col}} {speedup}")
    print(f"  {'Throughput delta:':>{col}} {tflops_delta:+.3f} TFLOP/s")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two KernelX experiment records.")
    parser.add_argument("--file",     default="results/experiments.jsonl",
                        help="Path to JSONL results file")
    parser.add_argument("--a",        required=True, help="Implementation A (baseline)")
    parser.add_argument("--b",        required=True, help="Implementation B (candidate)")
    parser.add_argument("--workload", default=None,  help="Filter by workload name")
    parser.add_argument("--m",        type=int, default=None)
    parser.add_argument("--n",        type=int, default=None)
    parser.add_argument("--k",        type=int, default=None)
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f"error: results file not found: {path}", file=sys.stderr)
        sys.exit(1)

    records = load_records(str(path))
    rec_a = find_latest(records, args.a, args.workload, args.m, args.n, args.k)
    rec_b = find_latest(records, args.b, args.workload, args.m, args.n, args.k)

    if rec_a is None:
        print(f"error: no record found for implementation '{args.a}'", file=sys.stderr)
        sys.exit(1)
    if rec_b is None:
        print(f"error: no record found for implementation '{args.b}'", file=sys.stderr)
        sys.exit(1)

    print_comparison(rec_a, rec_b)


if __name__ == "__main__":
    main()
