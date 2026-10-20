#pragma once

#include "workloads/workload.hpp"
#include <vector>

namespace kernelx {

// ─────────────────────────────────────────────────────────────────────────────
// BenchmarkStats — timing summary produced by BenchmarkRunner.
// ─────────────────────────────────────────────────────────────────────────────
struct BenchmarkStats {
    double median_ms  = 0.0;
    double mean_ms    = 0.0;
    double p95_ms     = 0.0;
    double stddev_ms  = 0.0;
    double tflops     = 0.0;
    bool   verified   = false;
    int    warmup     = 0;
    int    iterations = 0;
};

// ─────────────────────────────────────────────────────────────────────────────
// BenchmarkRunner — executes warmup + timed iterations and computes statistics.
//
// Timing methodology:
//   GPU (CUDA/CUTLASS/TRITON): CUDA events on the default stream.
//     cudaEventRecord(start) → workload.run() → cudaEventRecord(stop)
//     → cudaEventSynchronize(stop) → cudaEventElapsedTime()
//     This measures device-side kernel time with ~0.5 µs resolution,
//     independent of host-side scheduling jitter.
//
//   CPU: std::chrono::high_resolution_clock bracketing workload.run().
//
// Reported statistic: median (not mean) — robust against outliers from OS
// scheduling preemptions or ECC scrubbing on the first few iterations.
// ─────────────────────────────────────────────────────────────────────────────
class BenchmarkRunner {
public:
    BenchmarkStats run(Workload& workload,
                       const WorkloadConfig& cfg,
                       int warmup,
                       int iterations);

private:
    static double time_once_gpu(Workload& w);
    static double time_once_cpu(Workload& w);
    static double percentile(std::vector<double>& v, double p);
    static double stddev(const std::vector<double>& v, double mean);
};

} // namespace kernelx
