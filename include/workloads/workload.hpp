#pragma once

#include <string>

namespace kernelx {

// ─────────────────────────────────────────────────────────────────────────────
// WorkloadConfig — problem dimensions and data type.
// Passed to every Workload at setup time.
// ─────────────────────────────────────────────────────────────────────────────
struct WorkloadConfig {
    int         m     = 4096;
    int         n     = 4096;
    int         k     = 4096;
    std::string dtype = "fp32";   // "fp32" | "fp16"
};

// ─────────────────────────────────────────────────────────────────────────────
// Workload — abstract interface for a single benchmarkable computation.
//
// Lifecycle imposed by BenchmarkRunner:
//   1. setup(cfg)      — allocate buffers, initialise data, upload to device
//   2. run()   [×N]    — execute one iteration (must be synchronous)
//   3. verify()        — compare output to reference (call after run, before teardown)
//   4. teardown()      — free all resources
//
// Timing contract:
//   GPU implementations: run() launches a kernel on the default stream with no
//   explicit device sync.  The runner times it with CUDA events.
//   CPU implementations: run() does CPU work; the runner times it with
//   std::chrono::high_resolution_clock.
// ─────────────────────────────────────────────────────────────────────────────
class Workload {
public:
    enum class Backend { CPU, CUDA, CUTLASS, TRITON };

    virtual ~Workload() = default;

    virtual void setup(const WorkloadConfig& cfg)   = 0;
    virtual void run()                              = 0;
    virtual void teardown()                         = 0;
    virtual bool verify()                           = 0;

    /// Arithmetic operations per call to run() — used to compute TFLOP/s.
    virtual double flops() const                    = 0;

    virtual std::string workload_name()        const = 0;
    virtual std::string implementation_name()  const = 0;
    virtual Backend     backend()              const = 0;
};

} // namespace kernelx
