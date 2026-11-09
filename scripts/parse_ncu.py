#!/usr/bin/env python3
"""
parse_ncu.py — parse ncu --csv --page raw output into KernelX profile records.

Reads ncu CSV from stdin (or --input), enriches with workload metadata passed
as CLI args, computes derived metrics, and appends JSON records to an output
JSONL file (default: profiles/ncu_records.jsonl).

Usage (called from slurm/profile_h100.sh):
  ncu --metrics "..." --csv --page raw ./bench [bench_flags] 2>/dev/null \\
      | python scripts/parse_ncu.py \\
            --workload gemm --implementation cuda_naive \\
            --m 4096 --n 4096 --k 4096 --dtype fp32 \\
            --gpu "NVIDIA H100 80GB HBM3"

Output record fields:
  type, workload, implementation, m, n, k, dtype, gpu, timestamp, git_commit,
  kernel_name, kernel_time_ms,
  metrics: {
    compute_sol_pct, dram_sol_pct, l1_sol_pct, l2_sol_pct,
    achieved_occupancy_pct,
    flops, dram_bytes, arithmetic_intensity_flop_per_byte,
    stall_long_scoreboard_pct, stall_math_pipe_throttle_pct,
    stall_mio_throttle_pct,
    occupancy_limit  (register|shared_mem|warps|none)
  },
  derived: { bottleneck (compute|memory|l2), limiting_resource }
"""

import argparse
import csv
import io
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


# ── Metric name constants (must match the --metrics list in profile_h100.sh) ─
_M_COMPUTE_SOL  = "sm__throughput.avg.pct_of_peak_sustained_elapsed"
_M_DRAM_SOL     = "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed"
_M_L1_SOL       = "l1tex__throughput.avg.pct_of_peak_sustained_elapsed"
_M_L2_SOL       = "lts__throughput.avg.pct_of_peak_sustained_elapsed"
_M_FADD         = "sm__sass_thread_inst_executed_op_fadd_pred_on.sum"
_M_FMUL         = "sm__sass_thread_inst_executed_op_fmul_pred_on.sum"
_M_FFMA         = "sm__sass_thread_inst_executed_op_ffma_pred_on.sum"
_M_DRAM_RD      = "dram__bytes_read.sum"
_M_DRAM_WR      = "dram__bytes_write.sum"
_M_OCCUPANCY    = "sm__warps_active.avg.pct_of_peak_sustained_active"
_M_OCC_REG      = "launch__occupancy_limit_registers"
_M_OCC_SMEM     = "launch__occupancy_limit_shared_mem"
_M_OCC_WARPS    = "launch__occupancy_limit_warps"
_M_STALL_LSB    = "smsp__warp_issue_stall_long_scoreboard_per_warp_active.pct"
_M_STALL_MATH   = "smsp__warp_issue_stall_math_pipe_throttle_per_warp_active.pct"
_M_STALL_MIO    = "smsp__warp_issue_stall_mio_throttle_per_warp_active.pct"


def _parse_float(s: str) -> float:
    """Parse a numeric string that may contain thousands-separator commas."""
    s = s.strip().replace(",", "")
    # Handle ncu's "<empty>" or "N/A" gracefully
    if not s or s in ("-", "N/A", "n/a", ""):
        return 0.0
    # Strip trailing unit labels if present ("%" or "byte" etc.)
    s = re.sub(r"\s+.*$", "", s)
    return float(s)


def _get_kernel_time_ms(kernel_time_str: str) -> float:
    """Parse ncu Kernel Time field, e.g. '0.000250 s' or '250.3 us'."""
    s = kernel_time_str.strip()
    m = re.match(r"([0-9eE.+\-]+)\s*(s|ms|us|ns)?", s)
    if not m:
        return 0.0
    val = float(m.group(1))
    unit = m.group(2) or "s"
    return val * {"s": 1000.0, "ms": 1.0, "us": 1e-3, "ns": 1e-6}[unit]


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def parse_ncu_csv(text: str) -> list[dict]:
    """
    Parse ncu --csv --page raw output.

    Returns a list of dicts: one per kernel invocation.
    Each dict maps metric_name → (value_float, unit_str).
    Also includes special keys '__kernel_name__' and '__kernel_time_ms__'.
    """
    # Strip ==PROF== and other non-CSV lines; find the header row
    csv_lines = []
    in_csv = False
    for line in text.splitlines():
        if line.startswith('"ID"'):   # CSV header
            in_csv = True
        if in_csv:
            csv_lines.append(line)

    if not csv_lines:
        return []

    reader = csv.DictReader(io.StringIO("\n".join(csv_lines)))

    # Group rows by (ID, Kernel Name) → metric dict
    kernels: dict[tuple, dict] = {}
    kernel_times: dict[tuple, float] = {}

    for row in reader:
        try:
            kid          = row.get("ID", "0").strip()
            kernel_name  = row.get("Kernel Name", "").strip()
            kernel_time  = row.get("Kernel Time", "0 s").strip()
            metric_name  = row.get("Metric Name", "").strip()
            metric_unit  = row.get("Metric Unit", "").strip()
            metric_value = row.get("Metric Value", "0").strip()
        except (KeyError, AttributeError):
            continue

        key = (kid, kernel_name)
        if key not in kernels:
            kernels[key] = {}
            kernel_times[key] = _get_kernel_time_ms(kernel_time)

        if metric_name:
            kernels[key][metric_name] = (_parse_float(metric_value), metric_unit)

    return [
        {"__kernel_name__": k[1],
         "__kernel_time_ms__": kernel_times[k],
         **{name: val for name, (val, _) in metrics.items()}}
        for k, metrics in kernels.items()
    ]


def build_record(raw: dict, args: argparse.Namespace) -> dict:
    """Convert one raw ncu kernel dict into a structured KernelX profile record."""

    def g(name: str) -> float:
        return raw.get(name, 0.0)

    # FLOPs: each ffma counts as 2 operations
    flops = int(g(_M_FADD) + g(_M_FMUL) + 2.0 * g(_M_FFMA))
    dram_bytes = int(g(_M_DRAM_RD) + g(_M_DRAM_WR))
    arith_intensity = flops / dram_bytes if dram_bytes > 0 else 0.0

    # Occupancy limiting factor
    occ_reg   = g(_M_OCC_REG)
    occ_smem  = g(_M_OCC_SMEM)
    occ_warps = g(_M_OCC_WARPS)
    achieved  = g(_M_OCCUPANCY)
    limits = {"register": occ_reg, "shared_mem": occ_smem, "warps": occ_warps}
    # The binding constraint is the lowest non-zero theoretical limit
    active_limits = {k: v for k, v in limits.items() if v > 0}
    occ_limit = min(active_limits, key=active_limits.get) if active_limits else "none"

    # Bottleneck: highest SOL wins
    compute_sol = g(_M_COMPUTE_SOL)
    dram_sol    = g(_M_DRAM_SOL)
    l2_sol      = g(_M_L2_SOL)
    top_sol     = max(compute_sol, dram_sol, l2_sol)
    if top_sol == compute_sol:
        bottleneck = "compute"
        limiting   = "FP32 math throughput"
    elif top_sol == dram_sol:
        bottleneck = "memory"
        limiting   = "HBM bandwidth"
    else:
        bottleneck = "l2"
        limiting   = "L2 bandwidth"

    return {
        "type":           "ncu_profile",
        "workload":       args.workload,
        "implementation": args.implementation,
        "m":              args.m,
        "n":              args.n,
        "k":              args.k,
        "dtype":          args.dtype,
        "gpu":            args.gpu,
        "timestamp":      datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit":     _git_commit(),
        "kernel_name":    raw["__kernel_name__"],
        "kernel_time_ms": round(raw["__kernel_time_ms__"], 6),
        "metrics": {
            "compute_sol_pct":               round(compute_sol, 2),
            "dram_sol_pct":                  round(dram_sol, 2),
            "l1_sol_pct":                    round(g(_M_L1_SOL), 2),
            "l2_sol_pct":                    round(l2_sol, 2),
            "achieved_occupancy_pct":        round(achieved, 2),
            "flops":                         flops,
            "dram_bytes":                    dram_bytes,
            "arithmetic_intensity":          round(arith_intensity, 4),
            "occ_limit_register_pct":        round(occ_reg, 2),
            "occ_limit_shared_mem_pct":      round(occ_smem, 2),
            "occ_limit_warps_pct":           round(occ_warps, 2),
            "stall_long_scoreboard_pct":     round(g(_M_STALL_LSB), 2),
            "stall_math_pipe_throttle_pct":  round(g(_M_STALL_MATH), 2),
            "stall_mio_throttle_pct":        round(g(_M_STALL_MIO), 2),
        },
        "derived": {
            "bottleneck":        bottleneck,
            "limiting_resource": limiting,
            "occupancy_limit":   occ_limit,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse ncu --csv --page raw output into KernelX profile records"
    )
    parser.add_argument("--workload",       required=True)
    parser.add_argument("--implementation", required=True)
    parser.add_argument("--m",   type=int,  required=True)
    parser.add_argument("--n",   type=int,  required=True)
    parser.add_argument("--k",   type=int,  required=True)
    parser.add_argument("--dtype",          default="fp32")
    parser.add_argument("--gpu",            default="unknown")
    parser.add_argument("--input",          default="-",
                        help="ncu CSV file; '-' reads stdin (default)")
    parser.add_argument("--output",         default="profiles/ncu_records.jsonl",
                        help="JSONL file to append records to")
    args = parser.parse_args()

    # Read ncu CSV
    if args.input == "-":
        text = sys.stdin.read()
    else:
        text = Path(args.input).read_text()

    raw_kernels = parse_ncu_csv(text)
    if not raw_kernels:
        print(f"  [parse_ncu] warning: no kernel records found in ncu output",
              file=sys.stderr)
        return

    # Build output records
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records = [build_record(raw, args) for raw in raw_kernels]
    with open(out_path, "a") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    for rec in records:
        kname = rec["kernel_name"][:60]
        b = rec["derived"]["bottleneck"]
        c = rec["metrics"]["compute_sol_pct"]
        d = rec["metrics"]["dram_sol_pct"]
        ai = rec["metrics"]["arithmetic_intensity"]
        print(f"  [ncu] {kname}  compute={c:.1f}%  dram={d:.1f}%  "
              f"AI={ai:.2f}  → {b}-bound")


if __name__ == "__main__":
    main()
