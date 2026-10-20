# KernelX

**Kernel Performance Engineering Suite** — a GPU benchmarking, profiling, and optimization platform for NVIDIA hardware.

## What this is

KernelX answers two questions for any GPU workload:

> **Why is this kernel performing the way it is?**
> **What change actually makes it faster?**

It is structured as a performance engineering platform, not a collection of CUDA programs. Given a workload, KernelX establishes a reproducible baseline, profiles execution with Nsight Compute and Nsight Systems, identifies the limiting hardware resource through Speed-of-Light and Roofline analysis, implements optimized variants, and quantitatively evaluates the result.

The goal is not the fastest number. It is the explanation.

---

## Architecture

```
KernelX/
├── include/
│   ├── benchmark/          # BenchmarkRunner, ExperimentRecord
│   ├── workloads/          # Workload interface + per-workload headers
│   │   └── gemm/impl/      # cpu_reference, cuda_naive, ...
│   └── utils/              # CUDA error checking, device queries
├── src/
│   ├── main.cpp            # bench CLI entry point
│   ├── benchmark/          # runner.cpp, record.cpp
│   └── workloads/gemm/     # gemm_workload.cpp + impl/*.cu
├── results/                # Canonical JSONL experiment records
├── scripts/
│   ├── compare.py          # Side-by-side implementation comparison
│   └── report.py           # Performance table + regression detection
└── slurm/
    └── bench_h100.sh       # ROSIE H100 job template
```

The four project layers map directly to this structure:

| Layer | Location |
|-------|----------|
| Workloads | `src/workloads/` |
| Implementations | `src/workloads/*/impl/` |
| Performance analysis | Nsight Compute / Systems (profiling scripts TBD) |
| Engineering infrastructure | `src/benchmark/`, `scripts/`, `slurm/`, `results/` |

---

## Workload / Implementation Matrix

| Workload | Implementation | Status | Notes |
|----------|---------------|--------|-------|
| GEMM | `cpu_reference` | ✓ | Correctness baseline; ikj loop order |
| GEMM | `cuda_naive` | ✓ | V1: one thread per element, global memory only |
| GEMM | `cuda_tiled` | ✓ | V2: shared-memory tiling, TILE=32 |
| GEMM | `cuda_tuned` | Planned | V3: register blocking, occupancy tuning |
| GEMM | `cutlass` | Planned | V4: CUTLASS GEMM primitives |
| GEMM | `triton` | Planned | V5: Triton kernel, comparison vs CUDA |
| Softmax | `cpu_reference` | Planned | Numerically stable (online algorithm) |
| Softmax | `cuda_online` | Planned | Single-pass online softmax |
| LayerNorm | `cpu_reference` | Planned | |
| LayerNorm | `cuda_fused` | Planned | Fused mean/var/norm pass |
| Attention | `cuda_unfused` | Planned | QK^T → Softmax → V, unfused |
| Attention | `cuda_fused` | Planned | FlashAttention-style tiling |

---

## Quick Start

```bash
# Configure and build (Release with -lineinfo for Nsight source mapping)
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
cd ..

# GEMM — CPU reference (correctness baseline, small size)
./build/bench --workload gemm --implementation cpu_reference \
              --m 512 --n 512 --k 512

# GEMM — naive CUDA (GPU baseline, full benchmark)
./build/bench --workload gemm --implementation cuda_naive \
              --m 4096 --n 4096 --k 4096 --dtype fp32 \
              --warmup 20 --iterations 100

# Skip the CPU reference computation for large matrices
./build/bench --workload gemm --implementation cuda_naive \
              --m 4096 --n 4096 --k 4096 --no-verify

# Compare two implementations
python scripts/compare.py --a cpu_reference --b cuda_naive \
                          --workload gemm --m 512 --n 512 --k 512

# Report all GEMM experiments
python scripts/report.py --workload gemm

# Regression detection (exit code 1 if any >5% slowdown)
python scripts/report.py --workload gemm --regression-check
```

---

## Experiment Records

Every run appends a canonical, machine-readable record to `results/experiments.jsonl`:

```json
{
  "workload": "gemm",
  "implementation": "cuda_naive",
  "gpu": "NVIDIA H100 80GB HBM3",
  "dtype": "fp32",
  "m": 4096, "n": 4096, "k": 4096,
  "cuda_version": "12.2",
  "git_commit": "a3f9c1d",
  "timestamp": "2024-10-20T14:32:01Z",
  "median_ms": 8.21,
  "mean_ms": 8.34,
  "p95_ms": 9.10,
  "stddev_ms": 0.42,
  "tflops": 16.71,
  "warmup_iters": 20,
  "timed_iters": 100,
  "verified": true
}
```

Records are immutable. Comparisons, regressions, and reports are all derived from this file — no manual spreadsheets.

The JSONL format loads directly into pandas:

```python
import pandas as pd
df = pd.read_json("results/experiments.jsonl", lines=True)
```

---

## Performance Analysis Approach

Each kernel is investigated with three lenses, in order:

### 1. Speed-of-Light
Is the kernel compute-bound or memory-bound?
Nsight Compute's `sm__throughput` and `l1tex__throughput` metrics show utilization as a percentage of peak. A kernel achieving 90% of memory bandwidth ceiling with 10% compute utilization is memory-bound — tiling is the right lever, not instruction-level optimizations.

### 2. Roofline
Where does the kernel sit on the performance / arithmetic-intensity plot relative to the ridge point?
A GEMM with arithmetic intensity well below the ridge has room to improve through data reuse (shared memory, register blocking). Moving it rightward on the roofline is the optimization objective, not moving it upward arbitrarily.

### 3. Memory Workload
L1/L2 hit rates, global load efficiency, coalescing.
Nsight Compute's Memory Workload Analysis section breaks down traffic per cache level. For `cuda_naive`, we expect near-zero L1 reuse and high L2 pressure — this hypothesis is confirmed, not assumed.

The optimization hypothesis comes from measurement, not intuition.

---

## Running on ROSIE (MSOE HPC)

Three GPU generations are available, each with a dedicated job script:

| Script | Partition | GPU | sm | FP32 Peak | Bandwidth |
|--------|-----------|-----|----|-----------|-----------|
| `slurm/bench_h100.sh` | `dgxh100` | H100 | 90 | ~60 TFLOP/s | ~3350 GB/s |
| `slurm/bench_v100.sh` | `dgx` | V100 | 70 | ~14 TFLOP/s | ~900 GB/s |
| `slurm/bench_t4.sh` | `teaching` | T4 | 75 | ~8 TFLOP/s | ~300 GB/s |

```bash
# Submit to all three (results accumulate in the same JSONL file)
sbatch slurm/bench_h100.sh
sbatch slurm/bench_v100.sh
sbatch slurm/bench_t4.sh

# Monitor
squeue -u boyea@ad.msoe.edu

# Live output
tail -f job_<JOBID>.out

# Cross-GPU comparison once results are in
python scripts/gpu_compare.py --implementation cuda_naive --workload gemm
```

Because every record includes the `gpu` field, results from all three GPUs accumulate in the same `results/experiments.jsonl` and can be filtered or compared by device.

---

## Technology Stack

| Category | Technologies |
|----------|-------------|
| Kernel implementations | CUDA C++, CUTLASS (planned), Triton (planned) |
| Profiling | Nsight Compute, Nsight Systems |
| Build | CMake 3.18+, CUDA Toolkit |
| Analysis | Python, NumPy, Pandas, Matplotlib |
| Infrastructure | Git, SLURM |
| Cluster | MSOE ROSIE (H100 / V100 / T4 nodes) |

---

## Timing Methodology

GPU kernels are timed with CUDA events on the default stream:

```
cudaEventRecord(start)
kernel<<<grid, block>>>(...)
cudaEventRecord(stop)
cudaEventSynchronize(stop)
cudaEventElapsedTime(&ms, start, stop)
```

This measures device-side kernel time with ~0.5 µs resolution, independent of host-side scheduling jitter. CPU kernels use `std::chrono::high_resolution_clock`.

The reported statistic is the **median** over all timed iterations, not the mean. The median is robust against outliers from OS preemptions, ECC scrubbing, or thermal throttling on early iterations — all common on shared HPC nodes.
