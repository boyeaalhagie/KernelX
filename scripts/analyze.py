#!/usr/bin/env python3
"""
analyze.py — KernelX performance engineering report.

Reads benchmark records (results/experiments.jsonl) and Nsight Compute
profile records (profiles/ncu_records.jsonl), then prints the full
performance analysis described in KernelX §20:

  - Timing table with speedup vs baseline
  - Nsight Speed-of-Light (compute / HBM / L1 / L2)
  - Roofline position (arithmetic intensity vs hardware ridge point)
  - Occupancy and limiting factor
  - Dominant warp stall type
  - Engineering interpretation: bottleneck, observed behaviour, next step

Usage
-----
  python scripts/analyze.py --workload gemm
  python scripts/analyze.py --workload softmax --m 1024 --n 4096
  python scripts/analyze.py --workload attention --n 2048
  python scripts/analyze.py --workload gemm --impl cuda_naive   # focus one impl
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional


# ── Hardware constants (H100 SXM5) ───────────────────────────────────────────
# Used for roofline ridge-point calculation.
# Source: NVIDIA H100 Datasheet / Whitepaper
_HW = {
    "NVIDIA H100 80GB HBM3": {
        "peak_fp32_tflops": 60.0,
        "peak_hbm_gb_s":    3350.0,
    },
    "NVIDIA T4": {
        "peak_fp32_tflops": 8.1,
        "peak_hbm_gb_s":    300.0,
    },
    "NVIDIA V100": {
        "peak_fp32_tflops": 14.0,
        "peak_hbm_gb_s":    900.0,
    },
}
_HW_DEFAULT = {"peak_fp32_tflops": 60.0, "peak_hbm_gb_s": 3350.0}

# ── Engineering insights ──────────────────────────────────────────────────────
# Human-readable interpretation for each (workload, implementation) pair.
# "observed": what the ncu data shows and why
# "next":     the specific hypothesis that motivates the next implementation
_INSIGHTS: dict[tuple, dict] = {
    ("gemm", "cuda_naive"): {
        "observed": (
            "Every thread independently reads one row of A and one column of B "
            "with no data reuse.  Inner-loop memory traffic = O(2·N) global loads "
            "per output element → O(2·N³) total.  High L2 miss rate."
        ),
        "next": (
            "cuda_tiled tiles A and B into shared memory (32×32 blocks), reusing "
            "each loaded element across the tile.  Global traffic drops from "
            "O(2·N) to O(2·N/tile) per output element."
        ),
    },
    ("gemm", "cuda_tiled"): {
        "observed": (
            "Shared-memory tiling reduces global traffic and improves L2 hit rate.  "
            "Throughput is still register/occupancy-limited: 32×32 tiles leave "
            "insufficient registers for aggressive loop unrolling."
        ),
        "next": (
            "cuda_tuned: larger tile (128×128), 8×8 register-blocked accumulator, "
            "vectorised float4 loads, fully unrolled inner loop.  Pushes toward "
            "the compute ceiling for large matrices."
        ),
    },
    ("gemm", "cuda_tuned"): {
        "observed": (
            "Register blocking lifts arithmetic intensity above the ridge point.  "
            "Kernel becomes compute-bound for N≥2048.  FP32 FMA utilisation "
            "approaches 35–40% of peak — limited by instruction throughput, "
            "not memory."
        ),
        "next": (
            "cuda_cutlass uses TF32 mma.sync Tensor Cores (8× theoretical FP32 "
            "throughput) with CUTLASS's software-pipelined shared-memory stages.  "
            "Expect 3–8× further improvement."
        ),
    },
    ("gemm", "cuda_cutlass"): {
        "observed": (
            "CUTLASS TF32 Tensor Cores via mma.sync.aligned.m16n8k8.  "
            "CUTLASS's multi-stage software pipeline (cp.async + ldmatrix) "
            "overlaps HBM loads with Tensor Core math.  Near peak compute SOL.  "
            "Remaining gap from ceiling = pipeline stall between stages."
        ),
        "next": (
            "Profile with '--set full' to inspect Tensor Core utilisation "
            "and async copy pipeline efficiency.  FP16 would double throughput "
            "again (120 TFLOP/s theoretical)."
        ),
    },
    ("softmax", "cuda_naive"): {
        "observed": (
            "1024-thread blocks process one N-element row: 2 HBM passes (max scan "
            "then exp/sum/div).  For N≤4096, blocks are overprovisioned — most "
            "threads idle waiting for the global max broadcast.  HBM SOL <5%."
        ),
        "next": (
            "cuda_online assigns one warp (32 threads) per row with online (m,d) "
            "reduction (Milakov & Gimelshein 2018).  Eliminates the separate max "
            "pass; warp butterfly reduces without __syncthreads.  2× less HBM "
            "traffic, far better occupancy."
        ),
    },
    ("softmax", "cuda_online"): {
        "observed": (
            "Warp-per-row online softmax hits the HBM bandwidth ceiling for large N.  "
            "Two HBM passes (read X, write Y) are unavoidable with this design.  "
            "At N=4096 achieved ~97% of theoretical HBM3 bandwidth."
        ),
        "next": (
            "To eliminate both passes, fuse softmax into the QK^T kernel — "
            "this is exactly what FlashAttention does (cuda_flash): K, V stay in "
            "SRAM, the softmax intermediate never touches HBM."
        ),
    },
    ("layernorm", "cuda_naive"): {
        "observed": (
            "Two-pass design: first compute mean (read X), then compute variance "
            "(read X again).  3 HBM passes per row (read, read, write).  "
            "Arithmetic intensity ≈ 8·N / (3·N·4) = 0.67 FLOP/byte — below ridge."
        ),
        "next": (
            "cuda_welford: single-pass Welford online accumulation (Welford 1962, "
            "Chan 1979) computes mean and M₂ simultaneously.  2 HBM passes instead "
            "of 3; warp butterfly reduction with no shared memory or __syncthreads."
        ),
    },
    ("layernorm", "cuda_welford"): {
        "observed": (
            "Single-pass Welford nearly halves HBM traffic vs naive.  At N=4096 "
            "achieves ~83% HBM3 bandwidth.  Throughput drops at N=8192 due to "
            "gamma/beta (learned affine parameters) causing L2 pressure — they "
            "are reloaded every row and don't fit in L1 for large N."
        ),
        "next": (
            "Persistent-thread design: a fixed block processes many rows, keeping "
            "gamma/beta in L1/registers.  Or: batch normalisation instead of "
            "per-row LN if the downstream model supports it."
        ),
    },
    ("attention", "cuda_naive"): {
        "observed": (
            "Three-kernel pipeline materialises the full N×N attention matrix S.  "
            "Memory traffic: write S (N²), read S (N²), write P (N²), read P (N²) "
            "= 4·N²·4 bytes for the scratch buffer alone.  At N=4096: 256 MB.  "
            "Memory traffic scales as O(N²) — the fundamental bottleneck."
        ),
        "next": (
            "cuda_flash (FlashAttention-1, Dao 2022): tiles K and V in SRAM, "
            "online softmax across tiles keeps (m, l, o_acc) in registers.  "
            "No N×N scratch buffer.  HBM traffic drops to O(N·D), enabling "
            "much larger sequence lengths within the same wall-clock budget."
        ),
    },
    ("attention", "cuda_flash"): {
        "observed": (
            "Single-pass FlashAttention with kBr=kBc=32 (32 warps/block, 1024 threads).  "
            "K/V loaded once per block per tile, reused across kBr=32 query rows.  "
            "No N×N scratch buffer.  At N=8192: 8.6× faster than cuda_naive.  "
            "HBM traffic ≈ naive for small N (D/kBr=2 is the break-even), "
            "advantage grows quadratically with N."
        ),
        "next": (
            "FP16 inputs + Tensor Core accumulation: flash_fp16 would yield "
            "~8× additional compute throughput.  Multi-head batching: combine H "
            "heads across the batch dimension for better SM utilisation."
        ),
    },
    ("reduction", "cuda_naive"): {
        "observed": (
            "Shared-memory tree reduction: 8 __syncthreads() barriers per row for "
            "kBlockDim=256.  At large N (≥4096) the kernel is HBM-bandwidth-bound "
            "(AI ≈ 0.25 FLOP/byte vs H100 ridge 17.9): barrier overhead is "
            "amortised over many DRAM loads and both implementations are equal.  "
            "At N=256 the row fits in L2 and the 8-barrier tree dominates latency."
        ),
        "next": (
            "cuda_warp: replace the 5 intra-warp barrier steps with __shfl_down_sync "
            "butterfly — only 1 __syncthreads() remains (inter-warp partial sums).  "
            "At small N the warp kernel is 1.5–2× faster; at large N both are "
            "HBM-bound and performance converges."
        ),
    },
    ("reduction", "cuda_warp"): {
        "observed": (
            "Warp-shuffle reduction: 5 __shfl_down_sync steps (no barrier) reduce "
            "each warp to a partial sum; 1 __syncthreads() synchronises the 8 "
            "partial sums; warp 0 finishes with another 3-step butterfly.  "
            "At N≥4096 stall_long_scoreboard dominates (HBM latency), identical "
            "to the naive version — confirming the memory-bound ceiling.  "
            "Stall_math_pipe_throttle is effectively zero: 0.25 FLOP/byte keeps "
            "the compute pipes idle > 90% of the time."
        ),
        "next": (
            "To exceed HBM bandwidth, fuse reduction with an adjacent operator "
            "(e.g. layer-norm mean + variance in a single pass, or "
            "attention score max + sum online).  Fusion hides reduction latency "
            "inside a compute-heavier pass, raising the effective AI."
        ),
    },
}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def filter_bench(records: list[dict],
                 workload: str,
                 m: Optional[int],
                 n: Optional[int],
                 k: Optional[int]) -> list[dict]:
    out = [r for r in records
           if r.get("type") != "ncu_profile"
           and r.get("workload") == workload]
    if m is not None:
        out = [r for r in out if r.get("m") == m]
    if n is not None:
        out = [r for r in out if r.get("n") == n]
    if k is not None:
        out = [r for r in out if r.get("k") == k]
    return out


def filter_ncu(records: list[dict],
               workload: str,
               m: Optional[int],
               n: Optional[int],
               k: Optional[int]) -> list[dict]:
    out = [r for r in records if r.get("type") == "ncu_profile"
           and r.get("workload") == workload]
    if m is not None:
        out = [r for r in out if r.get("m") == m]
    if n is not None:
        out = [r for r in out if r.get("n") == n]
    if k is not None:
        out = [r for r in out if r.get("k") == k]
    return out


def latest_per_impl(records: list[dict]) -> dict[str, dict]:
    """Return the most recent record for each implementation."""
    out: dict[str, dict] = {}
    for r in records:
        impl = r.get("implementation", "?")
        out[impl] = r   # later records overwrite earlier ones
    return out


def latest_ncu_per_impl(records: list[dict]) -> dict[str, list[dict]]:
    """Return the most recent ncu records for each implementation (may be multi-kernel)."""
    # Group by impl, track the timestamp of the latest run
    by_impl: dict[str, dict[str, list]] = {}
    for r in records:
        impl = r.get("implementation", "?")
        ts   = r.get("timestamp", "")
        if impl not in by_impl or ts > max(by_impl[impl].keys()):
            by_impl[impl] = {ts: []}
        if ts >= max(by_impl[impl].keys()):
            by_impl[impl].setdefault(ts, []).append(r)
    out = {}
    for impl, ts_dict in by_impl.items():
        latest_ts = max(ts_dict.keys())
        out[impl] = ts_dict[latest_ts]
    return out


# ── Report sections ───────────────────────────────────────────────────────────

W = 68   # report width

def banner(title: str) -> str:
    return f"\n  {'═' * W}\n  {title}\n  {'═' * W}"


def section(title: str) -> str:
    return f"\n  {title}\n  {'─' * W}"


def print_timing_table(bench_map: dict[str, dict], workload: str) -> None:
    """Print benchmark timing table with speedup vs baseline."""
    if not bench_map:
        print("  No benchmark records found.")
        return

    # Sort by tflops ascending (slowest first = baseline at top)
    ranked = sorted(bench_map.values(), key=lambda r: r.get("tflops", 0.0))
    baseline_ms = ranked[0].get("median_ms", 1.0) if ranked else 1.0
    baseline_impl = ranked[0].get("implementation", "?") if ranked else "?"

    # Column widths
    cw = max(len(r.get("implementation", "")) for r in ranked) + 2
    cw = max(cw, 18)

    hdr = (f"  {'Implementation':<{cw}}"
           f"{'Median':>10}"
           f"{'P95':>10}"
           f"{'TFLOP/s':>10}"
           f"{'Speedup':>10}"
           f"  {'GPU':<36}"
           f"  Commit")
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))

    for r in ranked:
        impl    = r.get("implementation", "?")
        med     = r.get("median_ms", 0.0)
        p95     = r.get("p95_ms", 0.0)
        tflops  = r.get("tflops", 0.0)
        gpu     = r.get("gpu", "?")[:34]
        commit  = r.get("git_commit", "?")

        if impl == baseline_impl:
            speedup_str = "baseline"
        else:
            spd = baseline_ms / med if med > 0 else 0.0
            speedup_str = f"{spd:.1f}×"

        print(f"  {impl:<{cw}}"
              f"{med:>9.3f}ms"
              f"{p95:>9.3f}ms"
              f"{tflops:>10.3f}"
              f"{speedup_str:>10}"
              f"  {gpu:<36}"
              f"  {commit}")


def print_ncu_section(ncu_records: list[dict], gpu: str) -> None:
    """Print Nsight Compute Speed-of-Light, roofline, occupancy, stalls."""
    if not ncu_records:
        print("\n  No Nsight profile data.  Run: sbatch slurm/profile_h100.sh")
        return

    hw = _HW.get(gpu, _HW_DEFAULT)
    ridge = (hw["peak_fp32_tflops"] * 1e12) / (hw["peak_hbm_gb_s"] * 1e9)  # FLOP/byte

    for rec in ncu_records:
        kname   = rec.get("kernel_name", "?")
        # Trim mangled name for readability
        kshort  = kname.split("(")[0].split("::")[-1][:50]
        metrics = rec.get("metrics", {})
        derived = rec.get("derived", {})

        compute_sol = metrics.get("compute_sol_pct", 0.0)
        dram_sol    = metrics.get("dram_sol_pct", 0.0)
        l1_sol      = metrics.get("l1_sol_pct", 0.0)
        l2_sol      = metrics.get("l2_sol_pct", 0.0)
        occ         = metrics.get("achieved_occupancy_pct", 0.0)
        occ_limit   = derived.get("occupancy_limit", "?")
        bottleneck  = derived.get("bottleneck", "?")
        ai          = metrics.get("arithmetic_intensity", 0.0)
        stall_lsb   = metrics.get("stall_long_scoreboard_pct", 0.0)
        stall_math  = metrics.get("stall_math_pipe_throttle_pct", 0.0)
        stall_mio   = metrics.get("stall_mio_throttle_pct", 0.0)

        # Determine top stall
        stalls = {
            "long_scoreboard (memory dep)": stall_lsb,
            "math_pipe_throttle (compute)": stall_math,
            "mio_throttle (mem instr)":     stall_mio,
        }
        top_stall_name = max(stalls, key=stalls.get)
        top_stall_pct  = stalls[top_stall_name]

        # Roofline interpretation
        if ai >= ridge:
            roofline = f"compute-bound  (AI={ai:.2f} ≥ ridge {ridge:.1f} FLOP/byte)"
        else:
            roofline = f"memory-bound   (AI={ai:.2f} < ridge {ridge:.1f} FLOP/byte)"

        print(f"\n  Kernel: {kshort}")
        print(f"  {'Speed-of-Light':<30}  {'Compute':>8}  {'HBM':>8}  {'L2':>8}  {'L1':>8}")
        print(f"  {'':─<30}  {'':─>8}  {'':─>8}  {'':─>8}  {'':─>8}".replace("─", "─"))
        print(f"  {'':30}  {compute_sol:>7.1f}%  {dram_sol:>7.1f}%  "
              f"{l2_sol:>7.1f}%  {l1_sol:>7.1f}%")
        print()
        print(f"  Roofline:          {roofline}")
        print(f"  Bottleneck:        {bottleneck}-bound  "
              f"(peak FP32 {hw['peak_fp32_tflops']:.0f} TFLOP/s, "
              f"HBM {hw['peak_hbm_gb_s']:.0f} GB/s)")
        print()
        print(f"  Occupancy:         {occ:.1f}%  (limited by {occ_limit})")
        print(f"  Top warp stall:    {top_stall_name}  {top_stall_pct:.1f}%")


def print_insights(workload: str, impl: str) -> None:
    """Print engineering interpretation for the given (workload, impl) pair."""
    key = (workload, impl)
    info = _INSIGHTS.get(key)
    if not info:
        return

    print()
    print(f"  Primary bottleneck:  {info.get('observed', '—')}")
    print()
    print(f"  Next investigation:  {info.get('next', '—')}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="KernelX performance engineering report"
    )
    parser.add_argument("--workload",  required=True,
                        help="Workload to report (gemm, softmax, layernorm, attention)")
    parser.add_argument("--m",   type=int, default=None, help="Filter by M dimension")
    parser.add_argument("--n",   type=int, default=None, help="Filter by N dimension")
    parser.add_argument("--k",   type=int, default=None, help="Filter by K dimension")
    parser.add_argument("--impl",      default=None,
                        help="Implementation to show Nsight detail for (default: all)")
    parser.add_argument("--file",      default="results/experiments.jsonl")
    parser.add_argument("--ncu-file",  default="profiles/ncu_records.jsonl")
    args = parser.parse_args()

    bench_all = load_jsonl(Path(args.file))
    ncu_all   = load_jsonl(Path(args.ncu_file))

    bench_filtered = filter_bench(bench_all, args.workload, args.m, args.n, args.k)
    ncu_filtered   = filter_ncu(ncu_all,   args.workload, args.m, args.n, args.k)

    bench_map  = latest_per_impl(bench_filtered)
    ncu_by_impl = latest_ncu_per_impl(ncu_filtered)

    # Determine GPU from records for ridge-point calculation
    gpu = "NVIDIA H100 80GB HBM3"
    if bench_map:
        gpu = next(iter(bench_map.values())).get("gpu", gpu)

    # Determine display dimensions
    sample = next(iter(bench_map.values())) if bench_map else {}
    m_str = str(sample.get("m", args.m or "?"))
    n_str = str(sample.get("n", args.n or "?"))
    k_str = str(sample.get("k", args.k or "?"))
    dtype = sample.get("dtype", "fp32")

    dim_str = f"M={m_str} N={n_str}" if args.workload != "gemm" \
              else f"M=N=K={m_str}"

    title = f"{args.workload.upper()}  {dim_str}  {dtype}  — {gpu}"
    print(banner(title))

    # ── 1. Timing table ───────────────────────────────────────────────────────
    print(section("Benchmark Results"))
    print()
    print_timing_table(bench_map, args.workload)

    # ── 2. Nsight Compute sections ────────────────────────────────────────────
    impls_to_show = ([args.impl] if args.impl else list(ncu_by_impl.keys()))

    for impl in impls_to_show:
        ncu_recs = ncu_by_impl.get(impl, [])
        print(section(f"Nsight Compute Analysis — {impl}"))
        print_ncu_section(ncu_recs, gpu)
        print_insights(args.workload, impl)

    print()
    print(f"  {'═' * W}")
    print()


if __name__ == "__main__":
    main()
