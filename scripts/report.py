#!/usr/bin/env python3
"""
report.py — performance report from KernelX experiment records.

Reads results/experiments.jsonl and prints a ranked summary table.
Optionally checks for regressions against a baseline implementation.

Usage
-----
  python scripts/report.py                              # all experiments
  python scripts/report.py --workload gemm             # filter by workload
  python scripts/report.py --workload gemm --last 10   # last 10 experiments
  python scripts/report.py --regression-check          # flag regressions vs baseline
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional


REGRESSION_THRESHOLD = 0.05   # 5% slowdown triggers a warning


def load_records(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def filter_records(records: list[dict],
                   workload: Optional[str],
                   last: Optional[int]) -> list[dict]:
    if workload:
        records = [r for r in records if r.get("workload") == workload]
    if last:
        records = records[-last:]
    return records


def print_table(records: list[dict]) -> None:
    if not records:
        print("  No records to display.")
        return

    # Group by (workload, m, n, k, dtype) for display
    col_impl  = max(len(r.get("implementation", "")) for r in records) + 2
    col_impl  = max(col_impl, 16)

    header = (
        f"  {'Implementation':<{col_impl}}"
        f"{'Median':>10}"
        f"{'P95':>10}"
        f"{'TFLOP/s':>10}"
        f"{'Verified':>10}"
        f"  {'Commit':<10}"
        f"  Timestamp"
    )
    sep = "  " + "─" * (len(header) - 2)

    print()
    print(header)
    print(sep)

    for r in records:
        verified_str = "PASS" if r.get("verified", False) else "SKIP"
        print(
            f"  {r.get('implementation', '?'):<{col_impl}}"
            f"{r.get('median_ms', 0.0):>9.3f}ms"
            f"{r.get('p95_ms', 0.0):>9.3f}ms"
            f"{r.get('tflops', 0.0):>10.3f}"
            f"{verified_str:>10}"
            f"  {r.get('git_commit', '?'):<10}"
            f"  {r.get('timestamp', '?')}"
        )

    print()


def regression_check(records: list[dict], baseline_impl: str) -> bool:
    """
    For each (workload, m, n, k, dtype) configuration, compare the most
    recent record against the most recent record for baseline_impl.
    Print a warning if median latency increased by more than REGRESSION_THRESHOLD.
    Returns True if any regression is detected.
    """
    # Build baseline map: config → baseline median_ms
    baselines: dict[tuple, float] = {}
    for r in records:
        if r.get("implementation") == baseline_impl:
            key = (r["workload"], r["m"], r["n"], r["k"], r["dtype"])
            baselines[key] = r["median_ms"]   # last occurrence wins

    if not baselines:
        print(f"  [warn] no records found for baseline '{baseline_impl}'")
        return False

    regression_found = False
    for r in records:
        if r.get("implementation") == baseline_impl:
            continue
        key = (r["workload"], r["m"], r["n"], r["k"], r["dtype"])
        if key not in baselines:
            continue
        base_ms  = baselines[key]
        curr_ms  = r["median_ms"]
        delta    = (curr_ms - base_ms) / base_ms
        if delta > REGRESSION_THRESHOLD:
            print(
                f"  [REGRESSION] {r['implementation']} on {r['workload']} "
                f"{r['m']}×{r['n']}×{r['k']} {r['dtype']}:  "
                f"{base_ms:.3f} ms  →  {curr_ms:.3f} ms  "
                f"(+{delta * 100:.1f}%)"
            )
            regression_found = True

    if not regression_found:
        print("  [OK] No regressions detected.")
    return regression_found


def main() -> None:
    parser = argparse.ArgumentParser(description="KernelX performance report")
    parser.add_argument("--file",      default="results/experiments.jsonl",
                        help="Path to JSONL results file")
    parser.add_argument("--workload",  default=None, help="Filter by workload")
    parser.add_argument("--last",      type=int, default=None,
                        help="Show only the last N records")
    parser.add_argument("--regression-check", action="store_true",
                        help="Flag regressions against the baseline implementation")
    parser.add_argument("--baseline",  default="cpu_reference",
                        help="Baseline implementation for regression check (default: cpu_reference)")
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f"error: results file not found: {path}", file=sys.stderr)
        sys.exit(1)

    records = load_records(str(path))
    filtered = filter_records(records, args.workload, args.last)

    if args.workload:
        print(f"\n  Workload: {args.workload}  ({len(filtered)} records)\n")
    else:
        print(f"\n  All workloads  ({len(filtered)} records)\n")

    print_table(filtered)

    if args.regression_check:
        print(f"  Regression check (baseline: {args.baseline}):")
        had_regression = regression_check(filtered, args.baseline)
        print()
        sys.exit(1 if had_regression else 0)


if __name__ == "__main__":
    main()
