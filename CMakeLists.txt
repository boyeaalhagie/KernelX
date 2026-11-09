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
# Guard with NOT DEFINED so that -DCMAKE_CUDA_ARCHITECTURES=<x> on the cmake
# command line (e.g. from SLURM scripts) is not silently overridden by a plain
# set() — a plain set() creates a normal variable that shadows the cache value.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 75 80 90)
endif()

# ── Compiler flags ────────────────────────────────────────────────────────────
set(CMAKE_CXX_FLAGS_RELEASE  "-O3 -march=native")
set(CMAKE_CUDA_FLAGS         "--use_fast_math -Xptxas=-v")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -lineinfo")   # -lineinfo keeps source mapping for Nsight

# ── Dependencies ─────────────────────────────────────────────────────────────
find_package(CUDAToolkit REQUIRED)

# ── CUTLASS (header-only) ─────────────────────────────────────────────────────
# CUTLASS 3.4.1 headers are vendored at third_party/cutlass/include via a
# sparse git clone (see third_party/cutlass/.git).  No internet access is
# required on compute nodes; headers live on the shared home filesystem.
#
# To update: cd third_party/cutlass && git fetch && git checkout <new-tag>
set(CUTLASS_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/third_party/cutlass/include)
if(NOT EXISTS "${CUTLASS_INCLUDE_DIR}/cutlass/cutlass.h")
    message(FATAL_ERROR
        "CUTLASS headers not found at ${CUTLASS_INCLUDE_DIR}.\n"
        "Run from the repo root:\n"
        "  git clone --depth 1 --branch v3.4.1 --filter=blob:none --sparse "
        "https://github.com/NVIDIA/cutlass.git third_party/cutlass\n"
        "  cd third_party/cutlass && git sparse-checkout set include")
endif()

# ── Sources ───────────────────────────────────────────────────────────────────
set(KERNELX_SOURCES
    src/main.cpp
    src/benchmark/runner.cpp
    src/benchmark/record.cpp
    src/workloads/gemm/gemm_workload.cpp
    src/workloads/gemm/impl/cpu_reference.cpp
    src/workloads/gemm/impl/cuda_naive.cu
    src/workloads/gemm/impl/cuda_tiled.cu
    src/workloads/gemm/impl/cuda_tuned.cu
    src/workloads/gemm/impl/cuda_cutlass.cu
    src/workloads/softmax/softmax_workload.cpp
    src/workloads/softmax/impl/cpu_reference.cpp
    src/workloads/softmax/impl/cuda_naive.cu
    src/workloads/softmax/impl/cuda_online.cu
    src/workloads/layernorm/layernorm_workload.cpp
    src/workloads/layernorm/impl/cpu_reference.cpp
    src/workloads/layernorm/impl/cuda_naive.cu
    src/workloads/layernorm/impl/cuda_welford.cu
)

# ── CUTLASS TF32 Tensor Core path (sm_80+ only) ───────────────────────────────
# cuda_cutlass_tf32.cu uses OpClassTensorOp + Sm80 which references TF32
# mma.sync instructions unavailable on sm_75.  Restrict its compilation to
# only the sm_80+ architectures present in CMAKE_CUDA_ARCHITECTURES, and
# define KERNELX_CUTLASS_TF32 so the main cuda_cutlass.cu enables the
# runtime dispatch branch.
set(_SM80_PLUS "")
foreach(_ARCH IN LISTS CMAKE_CUDA_ARCHITECTURES)
    if(_ARCH GREATER_EQUAL 80)
        list(APPEND _SM80_PLUS "${_ARCH}")
    endif()
endforeach()

if(_SM80_PLUS)
    list(APPEND KERNELX_SOURCES src/workloads/gemm/impl/cuda_cutlass_tf32.cu)
    set_source_files_properties(
        src/workloads/gemm/impl/cuda_cutlass_tf32.cu
        PROPERTIES CUDA_ARCHITECTURES "${_SM80_PLUS}"
    )
    # Consumed by cuda_cutlass.cu to enable the TF32 dispatch branch
    add_compile_definitions(KERNELX_CUTLASS_TF32)
    message(STATUS "CUTLASS TF32 path enabled for sm: ${_SM80_PLUS}")
else()
    message(STATUS "CUTLASS TF32 path disabled (no sm_80+ in CMAKE_CUDA_ARCHITECTURES)")
endif()

add_executable(bench ${KERNELX_SOURCES})

target_include_directories(bench PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CUTLASS_INCLUDE_DIR}
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
