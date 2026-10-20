#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <string>

// ─────────────────────────────────────────────────────────────────────────────
// CUDA_CHECK — wraps every CUDA runtime call.
// Prints file/line/error string and aborts on failure.
// ─────────────────────────────────────────────────────────────────────────────
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _err = (call);                                              \
        if (_err != cudaSuccess) {                                              \
            std::fprintf(stderr,                                                \
                "CUDA error at %s:%d  →  %s\n",                                \
                __FILE__, __LINE__, cudaGetErrorString(_err));                  \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

namespace kernelx {

/// Return the name of CUDA device 0 (e.g. "NVIDIA H100 80GB HBM3").
inline std::string device_name() {
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    return prop.name;
}

/// Return the CUDA runtime version as "major.minor" (e.g. "12.2").
inline std::string cuda_runtime_version() {
    int v = 0;
    CUDA_CHECK(cudaRuntimeGetVersion(&v));
    return std::to_string(v / 1000) + "." + std::to_string((v % 1000) / 10);
}

} // namespace kernelx
