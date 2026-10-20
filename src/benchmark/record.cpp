#include "benchmark/record.hpp"
#include "utils/cuda_utils.hpp"

#include <chrono>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace kernelx {

// ── Internal helpers ──────────────────────────────────────────────────────────

static std::string shell_capture(const char* cmd) {
    FILE* pipe = popen(cmd, "r");
    if (!pipe) return "unknown";
    char buf[64] = {};
    if (std::fgets(buf, sizeof(buf), pipe) == nullptr) {
        pclose(pipe);
        return "unknown";
    }
    pclose(pipe);
    std::string s(buf);
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
    return s.empty() ? "unknown" : s;
}

static std::string utc_timestamp() {
    const auto now  = std::chrono::system_clock::now();
    const auto time = std::chrono::system_clock::to_time_t(now);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&time));
    return buf;
}

// ─────────────────────────────────────────────────────────────────────────────

ExperimentRecord ExperimentRecord::build(const Workload&       w,
                                         const WorkloadConfig& cfg,
                                         const BenchmarkStats& stats)
{
    ExperimentRecord r;
    r.workload       = w.workload_name();
    r.implementation = w.implementation_name();
    r.gpu            = device_name();
    r.dtype          = cfg.dtype;
    r.m              = cfg.m;
    r.n              = cfg.n;
    r.k              = cfg.k;
    r.cuda_version   = cuda_runtime_version();
    r.git_commit     = shell_capture("git rev-parse --short HEAD 2>/dev/null");
    r.timestamp      = utc_timestamp();
    r.median_ms      = stats.median_ms;
    r.mean_ms        = stats.mean_ms;
    r.p95_ms         = stats.p95_ms;
    r.stddev_ms      = stats.stddev_ms;
    r.tflops         = stats.tflops;
    r.warmup_iters   = stats.warmup;
    r.timed_iters    = stats.iterations;
    r.verified       = stats.verified;
    return r;
}

std::string ExperimentRecord::to_json() const {
    std::ostringstream o;
    o << std::fixed << std::setprecision(4);

    auto str = [](const std::string& s) { return "\"" + s + "\""; };
    auto b   = [](bool v)               { return v ? "true" : "false"; };

    o << "{"
      << "\"workload\":"       << str(workload)       << ","
      << "\"implementation\":" << str(implementation) << ","
      << "\"gpu\":"            << str(gpu)             << ","
      << "\"dtype\":"          << str(dtype)           << ","
      << "\"m\":"              << m                    << ","
      << "\"n\":"              << n                    << ","
      << "\"k\":"              << k                    << ","
      << "\"cuda_version\":"   << str(cuda_version)   << ","
      << "\"git_commit\":"     << str(git_commit)      << ","
      << "\"timestamp\":"      << str(timestamp)       << ","
      << "\"median_ms\":"      << median_ms            << ","
      << "\"mean_ms\":"        << mean_ms              << ","
      << "\"p95_ms\":"         << p95_ms               << ","
      << "\"stddev_ms\":"      << stddev_ms            << ","
      << "\"tflops\":"         << tflops               << ","
      << "\"warmup_iters\":"   << warmup_iters         << ","
      << "\"timed_iters\":"    << timed_iters          << ","
      << "\"verified\":"       << b(verified)
      << "}";
    return o.str();
}

void ExperimentRecord::print() const {
    auto row = [](const std::string& label, const std::string& value) {
        std::cout << "  " << std::left << std::setw(22) << label << value << "\n";
    };
    auto rowf = [&](const std::string& label, double value,
                    const std::string& unit, int prec = 3) {
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(prec) << value << " " << unit;
        row(label, ss.str());
    };

    std::cout << "  ┌─ Experiment ──────────────────────────────────────────\n";
    row("Workload:",        workload);
    row("Implementation:",  implementation);
    row("GPU:",             gpu);
    row("dtype:",           dtype);
    row("M × N × K:",       std::to_string(m) + " × " +
                            std::to_string(n) + " × " + std::to_string(k));
    row("CUDA:",            cuda_version);
    row("Commit:",          git_commit);
    row("Timestamp:",       timestamp);
    std::cout << "  ├─ Performance ─────────────────────────────────────────\n";
    rowf("Median:",         median_ms,  "ms");
    rowf("Mean:",           mean_ms,    "ms");
    rowf("P95:",            p95_ms,     "ms");
    rowf("Std dev:",        stddev_ms,  "ms");
    rowf("Throughput:",     tflops,     "TFLOP/s");
    std::cout << "  ├─ Config ──────────────────────────────────────────────\n";
    row("Warmup iters:",    std::to_string(warmup_iters));
    row("Timed iters:",     std::to_string(timed_iters));
    row("Verified:",        verified ? "PASS" : "SKIP");
    std::cout << "  └───────────────────────────────────────────────────────\n\n";
}

void ExperimentRecord::save(const std::string& path) const {
    // Create parent directory if it doesn't exist
    std::filesystem::create_directories(
        std::filesystem::path(path).parent_path());

    std::ofstream f(path, std::ios::app);
    if (!f) {
        std::cerr << "  [warn] could not open " << path << " for writing\n";
        return;
    }
    f << to_json() << "\n";
}

} // namespace kernelx
