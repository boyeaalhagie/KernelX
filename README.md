# KernelX

**GPU Kernel Performance Engineering Platform** — a benchmarking, profiling, and optimization framework for CUDA kernels on NVIDIA hardware.

## The question this project answers

> **Why is this kernel performing the way it is, and what change actually makes it faster?**

KernelX is not a collection of CUDA programs. It is a performance engineering system: given a workload, it establishes a reproducible baseline, profiles with Nsight Systems, identifies the hardware bottleneck, implements optimized variants, and quantitatively measures the result. The goal is the explanation, not the fastest number.

---

## Results

All benchmarks run on NVIDIA H100 80GB HBM3 (Hopper, sm_90) and Tesla T4 (Turing, sm_75) on MSOE ROSIE HPC.

### GEMM — M=N=K=4096, fp32

| Implementation | Kernel time | TFLOP/s | Speedup | What changes |
|---|---|---|---|---|
| `cuda_naive` | 26.4 ms | 5.2 | 1× | Global memory only — L2 miss-dominated |
| `cuda_tiled` | 15.6 ms | 8.8 | 1.7× | Shared-memory tiling — 32× data reuse per tile |
| `cuda_tuned` | 6.5 ms | 21.1 | 4.1× | Register blocking — eliminates redundant smem reads |
| `cuda_wmma` | 4.9 ms | 27.9 | 5.4× | Manual Tensor Core access via `nvcuda::wmma` (fp16) |
| Triton | 1.0 ms | 139.5 | 26.8× | Python DSL autotuner — 82% of CUTLASS |
| **CUTLASS** | **0.81 ms** | **169.3** | **32.6×** | TF32 Tensor Cores + `cp.async` double-buffering |

### Other workloads

| Workload | Implementations | Key result |
|---|---|---|
| Softmax | naive → online (1-pass) | 300× improvement — numerically stable and faster |
| LayerNorm | naive → Welford | 180× — fused single-pass eliminates 2 HBM passes |
| Attention | naive → FlashAttention (fp32, fp16) | 7.9× on T4, 4.1× on H100 — HBM bandwidth changes the benefit |
| Reduction | naive → warp shuffle | HBM-bound at N≥4096; warp shuffles eliminate atomic contention |
| Data Movement | pageable → pinned → bidirectional | 5.6× pageable→pinned; 86.8 GB/s bidirectional |

---

## Architecture

```
KernelX/
├── include/workloads/          # Workload base + per-impl headers (fully documented)
├── src/
│   ├── main.cpp                # bench CLI — single binary, --implementation flag selects kernel
│   ├── benchmark/              # CUDA-event timing, JSONL record serialisation
│   └── workloads/              # 6 workloads × N implementations
│       ├── gemm/impl/          # cuda_naive, cuda_tiled, cuda_tuned, cuda_wmma, cuda_cutlass
│       ├── softmax/impl/       # cuda_naive, cuda_online
│       ├── layernorm/impl/     # cuda_naive, cuda_welford
│       ├── attention/impl/     # cuda_naive, cuda_flash, cuda_flash_fp16
│       ├── reduction/impl/     # cuda_naive, cuda_warp
│       └── datamove/impl/      # pageable, pinned, pinned_async, bidirectional
├── scripts/
│   ├── analyze.py              # Benchmark + nsys report for any workload
│   ├── plot.py                 # Throughput charts → results/plots/
│   ├── parse_nsys.py           # Parses nsys --stats=true tables → JSONL
│   └── bench_gemm_triton.py    # Triton GEMM autotuner
├── slurm/
│   ├── bench_h100.sh           # Full benchmark suite — H100 (sm_90)
│   ├── bench_t4.sh             # Full benchmark suite — T4 (sm_75)
│   ├── profile_nsys.sh         # Nsight Systems sweep — all workloads
│   └── diag_*.sh               # Environment diagnostics
└── results/
    └── experiments.jsonl       # Canonical records (gitignored — generated at runtime)
```

---

## Quick start

### Prerequisites

- CUDA Toolkit 11.8+ (12.x recommended)
- CMake 3.18+
- GCC 9+ or Clang 10+ (C++17)
- Python 3.9+ (stdlib only — no pip installs needed for `analyze.py`)

### Clone

```bash
git clone https://github.com/boyeaalhagie/KernelX.git
cd KernelX

# CUTLASS headers (header-only, sparse clone — ~30 MB)
git clone --depth 1 --branch v3.4.1 \
          --filter=blob:none --sparse \
          https://github.com/NVIDIA/cutlass.git third_party/cutlass
cd third_party/cutlass && git sparse-checkout set include && cd ../..
```

### Build

Replace `XX` with your GPU's compute capability:

| GPU | sm |
|---|---|
| H100, H200 | 90 |
| A100, A6000 | 80 |
| RTX 3090 | 86 |
| RTX 4090 | 89 |
| T4, RTX 2080 | 75 |
| V100 | 70 |
| GTX 1080 | 61 (no WMMA) |

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=XX
cmake --build build -j$(nproc)
```

### Run

```bash
# Single benchmark
./build/bench --workload gemm --implementation cuda_tuned \
              --m 4096 --n 4096 --k 4096 --warmup 20 --iterations 100

# Full GEMM progression
for IMPL in cuda_naive cuda_tiled cuda_tuned cuda_wmma cuda_cutlass; do
  ./build/bench --workload gemm --implementation $IMPL \
                --m 4096 --n 4096 --k 4096 --warmup 20 --iterations 100
done

# Analyze results
python scripts/analyze.py --workload gemm
```

### On a SLURM cluster

```bash
# Edit the --mail-user line in slurm/*.sh to your email, then:
sbatch slurm/bench_h100.sh      # full H100 suite
sbatch slurm/bench_t4.sh        # full T4 suite
sbatch slurm/profile_nsys.sh    # Nsight Systems profiling
```

---

## Experiment records

Every run appends a machine-readable record to `results/experiments.jsonl`:

```json
{
  "workload": "gemm",
  "implementation": "cuda_tuned",
  "gpu": "NVIDIA H100 80GB HBM3",
  "dtype": "fp32",
  "m": 4096, "n": 4096, "k": 4096,
  "git_commit": "b6d74d3",
  "timestamp": "2026-11-09T23:28:24Z",
  "median_ms": 6.515,
  "mean_ms": 6.521,
  "p95_ms": 6.539,
  "stddev_ms": 0.012,
  "tflops_fp32": 21.096,
  "warmup_iters": 20,
  "timed_iters": 100,
  "verified": true
}
```

Records from every GPU accumulate in the same file and are filtered by `gpu` field for comparison.

---

## GPU compatibility

| Feature | Pascal (sm_61) | Volta (sm_70) | Turing (sm_75) | Ampere (sm_80+) | Hopper (sm_90) |
|---|---|---|---|---|---|
| All workloads except WMMA | ✓ | ✓ | ✓ | ✓ | ✓ |
| `cuda_wmma` (Tensor Cores) | — | ✓ | ✓ | ✓ | ✓ |
| CUTLASS TF32 path | — | — | — | ✓ | ✓ |

---

## Technology stack

CUDA C++17, CUTLASS 3.4.1, Triton, CMake, Python, Nsight Systems, SLURM
