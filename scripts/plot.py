#!/usr/bin/env python3
"""
plot.py — KernelX throughput visualisation.

Generates one throughput-vs-size chart per workload from
results/experiments.jsonl, saving PNGs to results/plots/.

For each workload the "size" axis is the most discriminating dimension:
  gemm       — M (= N = K, square matrices)
  softmax    — N (row length)
  layernorm  — N (hidden dimension)
  reduction  — N (row length)
  attention  — M (sequence length N)
  datamove   — N (element count → MB on secondary x-axis)

The y-axis is TFLOP/s (or GB/s for datamove — same column, different units).

Only records from the most recent git commit are plotted so the chart
reflects the current state of the codebase.  Pass --all-commits to overlay
every run (useful for regression visualisation).

Usage
-----
  python scripts/plot.py                        # all workloads, latest commit
  python scripts/plot.py --workload gemm        # single workload
  python scripts/plot.py --all-commits          # overlay all historical runs
  python scripts/plot.py --gpu "NVIDIA H100 80GB HBM3"  # filter by GPU
  python scripts/plot.py --out results/plots    # custom output directory
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")          # non-interactive backend — no display needed
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import numpy as np
except ImportError:
    sys.exit(
        "plot.py requires matplotlib and numpy.\n"
        "  pip install matplotlib numpy"
    )

# ── Workload metadata ─────────────────────────────────────────────────────────

_WORKLOAD_CFG = {
    "gemm": {
        "size_field":  "m",
        "size_label":  "Matrix size M (= N = K)",
        "y_label":     "Throughput (TFLOP/s)",
        "title":       "GEMM — Throughput vs Matrix Size",
        "rooflines": {
            "NVIDIA H100 80GB HBM3": {"compute": 60.0,  "label": "H100 FP32 peak (60 TFLOP/s)"},
            "NVIDIA T4":             {"compute": 8.1,   "label": "T4 FP32 peak (8.1 TFLOP/s)"},
        },
    },
    "softmax": {
        "size_field":  "n",
        "size_label":  "Row length N",
        "y_label":     "Throughput (TFLOP/s)",
        "title":       "Softmax — Throughput vs Row Length",
        "rooflines": {},
    },
    "layernorm": {
        "size_field":  "n",
        "size_label":  "Hidden dimension N",
        "y_label":     "Throughput (TFLOP/s)",
        "title":       "LayerNorm — Throughput vs Hidden Dimension",
        "rooflines": {},
    },
    "reduction": {
        "size_field":  "n",
        "size_label":  "Row length N",
        "y_label":     "Throughput (TFLOP/s)",
        "title":       "Reduction — Throughput vs Row Length",
        "rooflines": {},
    },
    "attention": {
        "size_field":  "m",
        "size_label":  "Sequence length N",
        "y_label":     "Throughput (TFLOP/s)",
        "title":       "Attention — Throughput vs Sequence Length",
        "rooflines": {},
    },
    "datamove": {
        "size_field":  "n",
        "size_label":  "Transfer size (MB)",
        "y_label":     "Bandwidth (GB/s)",
        "title":       "Data Movement — Bandwidth vs Transfer Size",
        "rooflines": {
            "NVIDIA H100 80GB HBM3": {"compute": 64.0,  "label": "H100 PCIe 5.0 peak (~64 GB/s)"},
            "NVIDIA T4":             {"compute": 16.0,  "label": "T4 PCIe 3.0 peak (~16 GB/s)"},
        },
    },
}

# Colour cycle — distinct enough for up to 6 implementations per workload.
_COLOURS = [
    "#1f77b4",   # blue
    "#ff7f0e",   # orange
    "#2ca02c",   # green
    "#d62728",   # red
    "#9467bd",   # purple
    "#8c564b",   # brown
    "#e377c2",   # pink
    "#7f7f7f",   # grey
]

_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_records(path: Path) -> list[dict]:
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def latest_commit(records: list[dict]) -> str:
    """Return the git commit hash that appears most recently by timestamp."""
    by_ts = sorted(records, key=lambda r: r.get("timestamp", ""), reverse=True)
    return by_ts[0]["git_commit"] if by_ts else ""


def filter_records(
    records: list[dict],
    workload: str,
    gpu: str | None,
    commit: str | None,
) -> list[dict]:
    out = [r for r in records if r["workload"] == workload]
    if gpu:
        out = [r for r in out if r.get("gpu", "").startswith(gpu)]
    if commit:
        out = [r for r in out if r["git_commit"] == commit]
    return out


# ── Plotting ──────────────────────────────────────────────────────────────────

def _size_value(record: dict, size_field: str, workload: str) -> float:
    """Return the x-axis value for a record."""
    raw = record[size_field]
    if workload == "datamove":
        # Convert element count → MB (float32 = 4 bytes)
        return raw * 4 / (1024 * 1024)
    return float(raw)


def _y_value(record: dict, workload: str) -> float:
    """Return the y-axis value (TFLOP/s or GB/s) for a record."""
    if workload == "datamove":
        # tflops column encodes GB/s for datamove (see DataMoveWorkload::flops())
        return record["tflops"]
    return record["tflops"]


def plot_workload(
    records: list[dict],
    workload: str,
    out_dir: Path,
    all_commits: bool,
    gpu_filter: str | None,
) -> Path:
    cfg = _WORKLOAD_CFG[workload]
    size_field = cfg["size_field"]

    # Group by (implementation, commit) or just implementation when single-commit
    # Key: (impl, commit_short) if all_commits else (impl,)
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        commit_short = r["git_commit"][:7]
        key = (r["implementation"], commit_short) if all_commits else (r["implementation"],)
        groups[key].append(r)

    if not groups:
        print(f"  [skip] no records for workload={workload}", file=sys.stderr)
        return None

    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.patch.set_facecolor("#fafafa")
    ax.set_facecolor("#fafafa")

    colour_map: dict[str, str] = {}
    marker_map: dict[str, str] = {}
    colour_idx = 0

    for key in sorted(groups):
        impl = key[0]
        label = impl if not all_commits else f"{impl} @ {key[1]}"

        # Assign stable colour/marker per implementation name
        if impl not in colour_map:
            colour_map[impl] = _COLOURS[colour_idx % len(_COLOURS)]
            marker_map[impl] = _MARKERS[colour_idx % len(_MARKERS)]
            colour_idx += 1

        recs = sorted(groups[key], key=lambda r: _size_value(r, size_field, workload))
        xs = [_size_value(r, size_field, workload) for r in recs]
        ys = [_y_value(r, workload) for r in recs]

        ax.plot(
            xs, ys,
            marker=marker_map[impl],
            color=colour_map[impl],
            linewidth=1.8,
            markersize=6,
            label=label,
            alpha=0.9,
        )

    # Roofline ceiling(s)
    gpu_name = records[0].get("gpu", "") if records else ""
    for gpu_key, roof in cfg.get("rooflines", {}).items():
        if gpu_name.startswith(gpu_key.split()[1]):  # match "H100" or "T4"
            ax.axhline(
                y=roof["compute"],
                linestyle="--",
                linewidth=1.2,
                color="#555555",
                alpha=0.6,
                label=roof["label"],
            )

    # Axes
    ax.set_xscale("log", base=2)
    ax.set_xlabel(cfg["size_label"], fontsize=11)
    ax.set_ylabel(cfg["y_label"], fontsize=11)
    ax.set_title(cfg["title"], fontsize=13, fontweight="bold", pad=10)

    # X-axis: show actual values rather than powers of 2
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{int(x):,}" if x >= 1 else f"{x:.3g}"
    ))
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())

    ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.grid(True, which="minor", linestyle=":",  linewidth=0.3, alpha=0.3)

    ax.legend(fontsize=9, loc="upper left", framealpha=0.85)

    # GPU annotation
    if gpu_name:
        ax.annotate(
            gpu_name,
            xy=(0.99, 0.02), xycoords="axes fraction",
            ha="right", va="bottom",
            fontsize=8, color="#666666",
        )

    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{workload}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="KernelX throughput visualisation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--file", default="results/experiments.jsonl",
        help="Path to JSONL results file (default: results/experiments.jsonl)",
    )
    p.add_argument(
        "--workload", default=None,
        choices=list(_WORKLOAD_CFG) + [None],
        help="Plot a single workload (default: all)",
    )
    p.add_argument(
        "--gpu", default=None,
        help="Filter records by GPU name prefix (e.g. 'NVIDIA H100')",
    )
    p.add_argument(
        "--all-commits", action="store_true",
        help="Overlay all historical commits rather than only the latest",
    )
    p.add_argument(
        "--out", default="results/plots",
        help="Output directory for PNG files (default: results/plots)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    results_path = Path(args.file)
    if not results_path.exists():
        sys.exit(f"Results file not found: {results_path}")

    records = load_records(results_path)
    if not records:
        sys.exit("No records found.")

    commit = None if args.all_commits else latest_commit(records)
    if commit and not args.all_commits:
        print(f"Plotting latest commit: {commit[:7]}")

    out_dir = Path(args.out)
    workloads = [args.workload] if args.workload else list(_WORKLOAD_CFG)

    for wl in workloads:
        wl_records = filter_records(records, wl, args.gpu, commit)
        if not wl_records:
            print(f"  [{wl}] no records — skipping")
            continue
        path = plot_workload(wl_records, wl, out_dir, args.all_commits, args.gpu)
        if path:
            print(f"  [{wl}] → {path}")

    print(f"\nDone. Plots saved to {out_dir}/")


if __name__ == "__main__":
    main()
