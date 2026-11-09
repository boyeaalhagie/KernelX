#!/usr/bin/env python3
"""
parse_nsys.py — parse nsys --stats=true output into KernelX JSONL records.

Reads nsys stats tables from stdin (or --input file), extracts:
  cuda_gpu_kern_sum  → main kernel timing (avg/med/min/max ns, instances)
  cuda_api_sum       → total CUDA API overhead (ns)
  cuda_gpu_mem_time_sum → H2D / D2H transfer time (ns)
  cuda_gpu_mem_size_sum → H2D / D2H transfer bytes

Writes one JSON record to --output (appended) and also prints a
human-readable summary to stdout.

Usage
-----
  nsys profile --trace=cuda --stats=true --output=/tmp/nsys_$$ <bench> ... \\
      2>/dev/null | python3 scripts/parse_nsys.py \\
          --workload gemm --implementation cuda_naive \\
          --m 4096 --n 4096 --k 4096 --dtype fp32 \\
          --gpu "NVIDIA H100 80GB HBM3" \\
          --output profiles/nsys_records.jsonl
"""

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
import subprocess


# ── Table parser ──────────────────────────────────────────────────────────────

def _strip_commas(s: str) -> str:
    return s.replace(",", "")


def _parse_stats_table(lines: list[str], section_tag: str) -> list[dict]:
    """
    Extract rows from an nsys --stats=true table identified by section_tag.
    Returns list of dicts keyed by column header (lowercase, spaces→underscore).
    """
    # Find section header line e.g. "Executing 'cuda_gpu_kern_sum' stats report"
    start = None
    for i, line in enumerate(lines):
        if section_tag in line and "stats report" in line:
            start = i
            break
    if start is None:
        return []

    # Skip blank lines to find the header row (contains column names)
    header_line = None
    sep_line = None
    for i in range(start + 1, min(start + 10, len(lines))):
        line = lines[i].strip()
        if not line:
            continue
        if header_line is None:
            header_line = lines[i]
        elif sep_line is None and set(line.replace(" ", "")) <= set("-"):
            sep_line = lines[i]
            data_start = i + 1
            break

    if header_line is None or sep_line is None:
        return []

    # Determine column boundaries from separator dashes
    col_spans = []
    in_col = False
    col_start = 0
    for j, ch in enumerate(sep_line):
        if ch == "-" and not in_col:
            col_start = j
            in_col = True
        elif ch == " " and in_col:
            col_spans.append((col_start, j))
            in_col = False
    if in_col:
        col_spans.append((col_start, len(sep_line)))

    # Extract header names
    headers = []
    for s, e in col_spans:
        h = header_line[s:e].strip().lower().replace(" ", "_").replace("(", "").replace(")", "").replace("%", "pct")
        headers.append(h)

    # Parse data rows until blank line or next section
    rows = []
    for i in range(data_start, len(lines)):
        line = lines[i]
        if not line.strip():
            break
        if line.strip().startswith("[") and "stats report" in line:
            break
        row = {}
        for k, (s, e) in enumerate(col_spans):
            val = line[s:e].strip() if e <= len(line) else line[s:].strip()
            # Last column (Name/Operation) may extend beyond sep width
            if k == len(col_spans) - 1:
                val = line[col_spans[-1][0]:].strip()
            row[headers[k]] = _strip_commas(val)
        if any(row.values()):
            rows.append(row)

    return rows


# ── Main kernel identification ────────────────────────────────────────────────

def _main_kernel(kern_rows: list[dict]) -> dict | None:
    """
    Pick the primary compute kernel from cuda_gpu_kern_sum rows.
    Prefer kernelx:: namespace; fall back to highest time%.
    """
    if not kern_rows:
        return None

    # Try kernelx:: namespace first
    kx = [r for r in kern_rows if "kernelx::" in r.get("name", "")]
    if kx:
        # Pick the one with most total time
        return max(kx, key=lambda r: float(r.get("time_pct", 0) or 0))

    # Fallback: highest time%
    return max(kern_rows, key=lambda r: float(r.get("time_pct", 0) or 0))


# ── Parse and build record ────────────────────────────────────────────────────

def parse(text: str, args: argparse.Namespace) -> dict:
    lines = text.splitlines()

    kern_rows = _parse_stats_table(lines, "cuda_gpu_kern_sum")
    api_rows  = _parse_stats_table(lines, "cuda_api_sum")
    mem_time  = _parse_stats_table(lines, "cuda_gpu_mem_time_sum")
    mem_size  = _parse_stats_table(lines, "cuda_gpu_mem_size_sum")

    main = _main_kernel(kern_rows)

    def _ns(row, field):
        try:
            return int(float(row.get(field, 0) or 0))
        except (ValueError, TypeError):
            return 0

    def _float(row, field):
        try:
            return float(row.get(field, 0) or 0)
        except (ValueError, TypeError):
            return 0.0

    kernel_avg_ns  = _ns(main, "avg_ns") if main else 0
    kernel_med_ns  = _ns(main, "med_ns") if main else 0
    kernel_min_ns  = _ns(main, "min_ns") if main else 0
    kernel_max_ns  = _ns(main, "max_ns") if main else 0
    kernel_instances = int(_float(main, "instances")) if main else 0
    kernel_name    = main.get("name", "") if main else ""

    # API overhead: sum of all cuda API calls minus the kernel launches
    api_total_ns = sum(_ns(r, "total_time_ns") for r in api_rows)

    # Memory transfers
    h2d_time_ns = d2h_time_ns = 0
    h2d_bytes = d2h_bytes = 0
    for r in mem_time:
        op = r.get("operation", "")
        if "HtoD" in op:
            h2d_time_ns += _ns(r, "total_time_ns")
        elif "DtoH" in op:
            d2h_time_ns += _ns(r, "total_time_ns")
    for r in mem_size:
        op = r.get("operation", "")
        mb = _float(r, "total_mb")
        if "HtoD" in op:
            h2d_bytes += int(mb * 1e6)
        elif "DtoH" in op:
            d2h_bytes += int(mb * 1e6)

    # git commit
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        commit = "unknown"

    return {
        "workload":          args.workload,
        "implementation":    args.implementation,
        "gpu":               args.gpu,
        "dtype":             args.dtype,
        "m":                 args.m,
        "n":                 args.n,
        "k":                 args.k,
        "git_commit":        commit,
        "timestamp":         datetime.now(timezone.utc).isoformat(),
        # Kernel timing (ns) — median of repeated launches, excluding warmup
        "kernel_name":       kernel_name,
        "kernel_avg_ns":     kernel_avg_ns,
        "kernel_med_ns":     kernel_med_ns,
        "kernel_min_ns":     kernel_min_ns,
        "kernel_max_ns":     kernel_max_ns,
        "kernel_instances":  kernel_instances,
        # API overhead (ns) — total CUDA API call time (malloc, launch, memcpy)
        "api_total_ns":      api_total_ns,
        # Memory transfers
        "h2d_bytes":         h2d_bytes,
        "d2h_bytes":         d2h_bytes,
        "h2d_time_ns":       h2d_time_ns,
        "d2h_time_ns":       d2h_time_ns,
    }


def print_summary(rec: dict) -> None:
    print(f"\n  [nsys] {rec['workload']} / {rec['implementation']}")
    if rec["kernel_name"]:
        short = rec["kernel_name"].split("(")[0].split("::")[-1]
        print(f"  Kernel:        {short}")
        print(f"  Avg:           {rec['kernel_avg_ns']/1e6:.3f} ms  "
              f"(min {rec['kernel_min_ns']/1e6:.3f}  "
              f"max {rec['kernel_max_ns']/1e6:.3f})  "
              f"× {rec['kernel_instances']} launches")
    else:
        print("  Kernel:        [not found in nsys output]")
    if rec["api_total_ns"]:
        print(f"  API overhead:  {rec['api_total_ns']/1e6:.3f} ms total")
    if rec["h2d_bytes"]:
        bw = rec["h2d_bytes"] / max(rec["h2d_time_ns"], 1) # GB/s
        print(f"  H2D:           {rec['h2d_bytes']/1e6:.1f} MB  "
              f"in {rec['h2d_time_ns']/1e6:.3f} ms  ({bw:.1f} GB/s)")
    if rec["d2h_bytes"]:
        bw = rec["d2h_bytes"] / max(rec["d2h_time_ns"], 1)
        print(f"  D2H:           {rec['d2h_bytes']/1e6:.1f} MB  "
              f"in {rec['d2h_time_ns']/1e6:.3f} ms  ({bw:.1f} GB/s)")
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Parse nsys --stats=true output")
    p.add_argument("--input",          default="-",   help="nsys stats file (default: stdin)")
    p.add_argument("--output",         required=True, help="JSONL output path")
    p.add_argument("--workload",       required=True)
    p.add_argument("--implementation", required=True)
    p.add_argument("--gpu",            required=True)
    p.add_argument("--dtype",          default="fp32")
    p.add_argument("--m",  type=int,   required=True)
    p.add_argument("--n",  type=int,   required=True)
    p.add_argument("--k",  type=int,   required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.input == "-":
        text = sys.stdin.read()
    else:
        text = Path(args.input).read_text()

    if not text.strip():
        print("  [parse_nsys] warning: empty input", file=sys.stderr)
        sys.exit(1)

    rec = parse(text, args)
    print_summary(rec)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"  Saved → {out_path}")


if __name__ == "__main__":
    main()
