#include "benchmark/record.hpp"
#include "benchmark/runner.hpp"
#include "workloads/gemm/impl/cpu_reference.hpp"
#include "workloads/gemm/impl/cuda_naive.hpp"
#include "workloads/gemm/impl/cuda_tiled.hpp"
#include "workloads/workload.hpp"

#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

// ─────────────────────────────────────────────────────────────────────────────
// CLI argument parsing
// ─────────────────────────────────────────────────────────────────────────────
struct CliArgs {
    std::string workload       = "gemm";
    std::string implementation = "cuda_naive";
    int         m              = 4096;
    int         n              = 4096;
    int         k              = 4096;
    std::string dtype          = "fp32";
    int         warmup         = 20;
    int         iterations     = 100;
    bool        save           = true;
    bool        verify         = true;
};

static void print_usage(const char* prog) {
    std::cout <<
        "Usage: " << prog << " [options]\n\n"
        "Options:\n"
        "  --workload       <name>   Workload to run          (default: gemm)\n"
        "  --implementation <name>   Implementation variant   (default: cuda_naive)\n"
        "  --m              <int>    Matrix rows (M)          (default: 4096)\n"
        "  --n              <int>    Matrix cols (N)          (default: 4096)\n"
        "  --k              <int>    Inner dimension (K)      (default: 4096)\n"
        "  --dtype          <type>   Data type: fp32 | fp16   (default: fp32)\n"
        "  --warmup         <int>    Warmup iterations        (default: 20)\n"
        "  --iterations     <int>    Timed iterations         (default: 100)\n"
        "  --no-save                 Do not append to results file\n"
        "  --no-verify               Skip correctness check\n"
        "  --help                    Print this message\n\n"
        "Available workloads / implementations:\n"
        "  gemm / cpu_reference\n"
        "  gemm / cuda_naive\n"
        "  gemm / cuda_tiled\n\n"
        "Example:\n"
        "  ./bench --workload gemm --implementation cuda_naive \\\n"
        "          --m 4096 --n 4096 --k 4096 --dtype fp32   \\\n"
        "          --warmup 20 --iterations 100\n";
}

static CliArgs parse_args(int argc, char** argv) {
    CliArgs args;
    for (int i = 1; i < argc; ++i) {
        std::string flag = argv[i];

        auto require_next = [&]() -> std::string {
            if (i + 1 >= argc)
                throw std::invalid_argument("flag " + flag + " requires a value");
            return argv[++i];
        };

        if      (flag == "--workload")       args.workload       = require_next();
        else if (flag == "--implementation") args.implementation = require_next();
        else if (flag == "--m")              args.m              = std::stoi(require_next());
        else if (flag == "--n")              args.n              = std::stoi(require_next());
        else if (flag == "--k")              args.k              = std::stoi(require_next());
        else if (flag == "--dtype")          args.dtype          = require_next();
        else if (flag == "--warmup")         args.warmup         = std::stoi(require_next());
        else if (flag == "--iterations")     args.iterations     = std::stoi(require_next());
        else if (flag == "--no-save")        args.save           = false;
        else if (flag == "--no-verify")      args.verify         = false;
        else if (flag == "--help") { print_usage(argv[0]); std::exit(0); }
        else throw std::invalid_argument("unknown argument: " + flag);
    }
    return args;
}

// ─────────────────────────────────────────────────────────────────────────────
// Workload factory
// ─────────────────────────────────────────────────────────────────────────────
static std::unique_ptr<kernelx::Workload> make_workload(
    const std::string& workload,
    const std::string& impl)
{
    if (workload == "gemm") {
        if (impl == "cpu_reference") return std::make_unique<kernelx::GemmCpuReference>();
        if (impl == "cuda_naive")    return std::make_unique<kernelx::GemmCudaNaive>();
        if (impl == "cuda_tiled")    return std::make_unique<kernelx::GemmCudaTiled>();
        throw std::invalid_argument("unknown GEMM implementation: '" + impl + "'");
    }
    throw std::invalid_argument("unknown workload: '" + workload + "'");
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    CliArgs args;
    try {
        args = parse_args(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        print_usage(argv[0]);
        return 1;
    }

    std::unique_ptr<kernelx::Workload> workload;
    try {
        workload = make_workload(args.workload, args.implementation);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what()
                  << "\nRun with --help to see available options.\n";
        return 1;
    }

    const kernelx::WorkloadConfig cfg{args.m, args.n, args.k, args.dtype};

    std::cout << "\n";
    std::cout << "  KernelX — " << args.workload << " / " << args.implementation << "\n";
    std::cout << "  M=" << args.m << "  N=" << args.n << "  K=" << args.k
              << "  dtype=" << args.dtype << "\n";
    std::cout << "  warmup=" << args.warmup << "  iterations=" << args.iterations << "\n\n";

    workload->setup(cfg);

    kernelx::BenchmarkRunner runner;
    kernelx::BenchmarkStats  stats = runner.run(*workload, cfg, args.warmup, args.iterations);

    if (args.verify) {
        stats.verified = workload->verify();
        if (!stats.verified) {
            std::cerr << "\n  [FAIL] Correctness check failed.\n";
        }
    }

    workload->teardown();

    kernelx::ExperimentRecord rec = kernelx::ExperimentRecord::build(*workload, cfg, stats);
    rec.print();

    if (args.save) {
        rec.save();
        std::cout << "  Saved  →  results/experiments.jsonl\n\n";
    }

    // Exit 1 only if verification was explicitly run and failed.
    // Skipped verification (--no-verify) is not a failure.
    return (args.verify && !stats.verified) ? 1 : 0;
}
