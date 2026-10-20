#pragma once

#include "benchmark/runner.hpp"
#include "workloads/workload.hpp"
#include <string>

namespace kernelx {

// ─────────────────────────────────────────────────────────────────────────────
// ExperimentRecord — the canonical, immutable result of one benchmark run.
//
// Every field that could affect performance is captured: GPU identity, driver
// version, CUDA version, compiler flags are not yet included but git_commit
// provides reproducibility via the build system.
//
// Records are appended as newline-delimited JSON to results/experiments.jsonl.
// This makes the file both human-readable and directly loadable with pandas:
//   df = pd.read_json("results/experiments.jsonl", lines=True)
// ─────────────────────────────────────────────────────────────────────────────
struct ExperimentRecord {
    // ── Identity ──────────────────────────────────────────────────────────────
    std::string workload;
    std::string implementation;
    std::string gpu;
    std::string dtype;

    // ── Problem size ──────────────────────────────────────────────────────────
    int m = 0, n = 0, k = 0;

    // ── Environment ───────────────────────────────────────────────────────────
    std::string cuda_version;
    std::string git_commit;
    std::string timestamp;   // ISO 8601 UTC

    // ── Performance ───────────────────────────────────────────────────────────
    double median_ms  = 0.0;
    double mean_ms    = 0.0;
    double p95_ms     = 0.0;
    double stddev_ms  = 0.0;
    double tflops     = 0.0;

    // ── Run parameters ────────────────────────────────────────────────────────
    int  warmup_iters = 0;
    int  timed_iters  = 0;
    bool verified     = false;

    // ── API ───────────────────────────────────────────────────────────────────

    /// Build a record from a completed workload run and its timing stats.
    static ExperimentRecord build(const Workload&       w,
                                  const WorkloadConfig& cfg,
                                  const BenchmarkStats& stats);

    /// Serialise to compact single-line JSON (no trailing newline).
    std::string to_json() const;

    /// Print a formatted summary table to stdout.
    void print() const;

    /// Append this record as one line to `path` (creates file if absent).
    void save(const std::string& path = "results/experiments.jsonl") const;
};

} // namespace kernelx
