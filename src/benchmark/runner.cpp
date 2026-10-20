#include "benchmark/runner.hpp"
#include "utils/cuda_utils.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>

namespace kernelx {

BenchmarkStats BenchmarkRunner::run(Workload&             workload,
                                    const WorkloadConfig& /*cfg*/,
                                    int                   warmup,
                                    int                   iterations)
{
    const bool is_gpu = (workload.backend() != Workload::Backend::CPU);

    // ── Warmup ────────────────────────────────────────────────────────────────
    std::cout << "  Warming up    (" << warmup << " iters) ... ";
    std::cout.flush();
    for (int i = 0; i < warmup; ++i) workload.run();
    std::cout << "done\n";

    // ── Timed iterations ──────────────────────────────────────────────────────
    std::cout << "  Benchmarking  (" << iterations << " iters) ... ";
    std::cout.flush();

    std::vector<double> times(iterations);
    for (int i = 0; i < iterations; ++i) {
        times[i] = is_gpu ? time_once_gpu(workload) : time_once_cpu(workload);
    }
    std::cout << "done\n\n";

    // ── Statistics ────────────────────────────────────────────────────────────
    std::sort(times.begin(), times.end());
    const double mean = std::accumulate(times.begin(), times.end(), 0.0)
                        / static_cast<double>(iterations);

    BenchmarkStats s;
    s.warmup     = warmup;
    s.iterations = iterations;
    s.median_ms  = percentile(times, 50.0);
    s.p95_ms     = percentile(times, 95.0);
    s.mean_ms    = mean;
    s.stddev_ms  = stddev(times, mean);
    s.tflops     = workload.flops() / (s.median_ms * 1e-3) / 1e12;

    return s;
}

// ── Private helpers ───────────────────────────────────────────────────────────

double BenchmarkRunner::time_once_gpu(Workload& w) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start, /*stream=*/nullptr));
    w.run();
    CUDA_CHECK(cudaEventRecord(stop, /*stream=*/nullptr));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return static_cast<double>(ms);
}

double BenchmarkRunner::time_once_cpu(Workload& w) {
    using Clock = std::chrono::high_resolution_clock;
    const auto t0 = Clock::now();
    w.run();
    const auto t1 = Clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

double BenchmarkRunner::percentile(std::vector<double>& v, double p) {
    // v must already be sorted
    if (v.empty()) return 0.0;
    const double idx  = (p / 100.0) * (static_cast<double>(v.size()) - 1.0);
    const size_t lo   = static_cast<size_t>(idx);
    const double frac = idx - static_cast<double>(lo);
    if (lo + 1 >= v.size()) return v.back();
    return v[lo] * (1.0 - frac) + v[lo + 1] * frac;
}

double BenchmarkRunner::stddev(const std::vector<double>& v, double mean) {
    if (v.size() < 2) return 0.0;
    double sq = 0.0;
    for (const double x : v) sq += (x - mean) * (x - mean);
    return std::sqrt(sq / static_cast<double>(v.size() - 1));
}

} // namespace kernelx
