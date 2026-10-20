cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(KernelX LANGUAGES CXX CUDA)

# ── Standards ─────────────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD          17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD         17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ── Build type ────────────────────────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ── CUDA compute capabilities ─────────────────────────────────────────────────
# V100=70  T4=75  A100=80  H100=90
# Generating PTX for 90 ensures forward compatibility on future Hopper variants.
set(CMAKE_CUDA_ARCHITECTURES 70 75 80 90)

# ── Compiler flags ────────────────────────────────────────────────────────────
set(CMAKE_CXX_FLAGS_RELEASE  "-O3 -march=native")
set(CMAKE_CUDA_FLAGS         "--use_fast_math -Xptxas=-v")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -lineinfo")   # -lineinfo keeps source mapping for Nsight

# ── Dependencies ─────────────────────────────────────────────────────────────
find_package(CUDAToolkit REQUIRED)

# ── Sources ───────────────────────────────────────────────────────────────────
set(KERNELX_SOURCES
    src/main.cpp
    src/benchmark/runner.cpp
    src/benchmark/record.cpp
    src/workloads/gemm/gemm_workload.cpp
    src/workloads/gemm/impl/cpu_reference.cpp
    src/workloads/gemm/impl/cuda_naive.cu
    src/workloads/gemm/impl/cuda_tiled.cu
)

add_executable(bench ${KERNELX_SOURCES})

target_include_directories(bench PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(bench PRIVATE CUDA::cudart)

target_compile_options(bench PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:
    -Xcompiler=-Wall,-Wextra,-Wno-unused-parameter
  >
  $<$<COMPILE_LANGUAGE:CXX>:
    -Wall -Wextra -Wno-unused-parameter
  >
)

# Export compile commands for IDE / clangd support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

install(TARGETS bench DESTINATION bin)
