cmake_minimum_required(VERSION 3.18)

# Default CUDA architectures.  Each entry is a device the project is
# expected to run on, so the fatbin carries real SASS for it: Ampere
# (sm_80), Ada / L40S (sm_89), Hopper (sm_90), Blackwell datacenter
# (sm_100), and Blackwell consumer / RTX PRO (sm_120).
#
# `120` (vs `120-real`) is the only entry emitting PTX as well as SASS.
# PTX JIT is forward-only — it never lowers onto an older architecture —
# so PTX from the newest entry covers hardware that hasn't shipped yet
# and covers nothing already on this list.  A device missing from the
# list therefore finds no cubin and no JIT path, and simply fails to run.
#
# For fast dev-iteration on a single known GPU, pass
# -DSMALLFIELD_NATIVE_CUDA_ARCH=ON or an explicit
# -DCMAKE_CUDA_ARCHITECTURES=... — then the build collapses to a
# single-arch native compile.  Multi-arch fatbin compile is ~3× longer.
#
# These two settings must run BEFORE project() so the CUDA toolkit
# auto-detect at project-time doesn't override us with sm_75 default.
set(SMALLFIELD_DEFAULT_CUDA_ARCH "80-real;89-real;90-real;100-real;120"
    CACHE STRING "Default fatbin CUDA architectures")
option(SMALLFIELD_NATIVE_CUDA_ARCH
    "Detect a single native CUDA arch via nvidia-smi instead of the multi-arch fatbin"
    OFF)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    if(SMALLFIELD_NATIVE_CUDA_ARCH)
        execute_process(
            COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
            OUTPUT_VARIABLE _gpu_caps OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET RESULT_VARIABLE _smi_result)
        if(_smi_result EQUAL 0 AND _gpu_caps)
            string(REGEX MATCH "[0-9]+\\.[0-9]+" _cap_match "${_gpu_caps}")
            if(_cap_match)
                string(REPLACE "." "" _arch_code "${_cap_match}")
                set(CMAKE_CUDA_ARCHITECTURES "${_arch_code}")
            else()
                set(CMAKE_CUDA_ARCHITECTURES "120")
            endif()
        else()
            set(CMAKE_CUDA_ARCHITECTURES "120")
        endif()
    else()
        set(CMAKE_CUDA_ARCHITECTURES "${SMALLFIELD_DEFAULT_CUDA_ARCH}")
    endif()
endif()

project(SmallField LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

message(STATUS "CMAKE_CUDA_ARCHITECTURES = ${CMAKE_CUDA_ARCHITECTURES}")

# Compiler cache, when one is installed.  Unchanged translation units then cost
# nothing on rebuilds, branch switches and CI restores, which dominates the
# inner loop far more than any one-off compile-speed flag.
find_program(SMALLFIELD_COMPILER_CACHE NAMES ccache sccache)
if(SMALLFIELD_COMPILER_CACHE)
    message(STATUS "Using compiler cache: ${SMALLFIELD_COMPILER_CACHE}")
    set(CMAKE_CXX_COMPILER_LAUNCHER "${SMALLFIELD_COMPILER_CACHE}")
    set(CMAKE_CUDA_COMPILER_LAUNCHER "${SMALLFIELD_COMPILER_CACHE}")
endif()

# Deliberately not passing nvcc `--threads`: it makes one invocation compile its
# architectures concurrently, which is 55 % faster in isolation but no faster here,
# because `-j` already saturates the machine across targets and the two layers of
# parallelism then contend.  Measured 366 s without against 380 s with, 5 arches,
# 16 cores.  Likewise no precompiled header — nvcc flattens `#include` directives
# away before the host compiler runs, so a `.gch` can never be consulted.

# Point `core.hooksPath` at `.githooks` here — configuring the build arms the `pre-commit` /
# `commit-msg` checks, each of which carries its own rationale in that file.  Skipped when built
# as a subproject, since mutating a consumer's Git config is not this project's business, and
# without `.git`, as in a tarball build.
if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.git")
    execute_process(
        COMMAND git config core.hooksPath .githooks
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        RESULT_VARIABLE _smallfield_hooks_status OUTPUT_QUIET ERROR_QUIET)
    if(_smallfield_hooks_status EQUAL 0)
        message(STATUS "smallfield: Git hooks enabled (core.hooksPath -> .githooks)")
    endif()
endif()

# Enable -Wall -Wextra globally on non-MSVC toolchains.
# CUDA device code gets the host warnings forwarded through -Xcompiler.
# We intentionally avoid -Werror: warnings should surface but not block builds.
if(NOT MSVC)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wall>
        $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
        $<$<COMPILE_LANGUAGE:CXX>:-Wconversion>
        $<$<COMPILE_LANGUAGE:CXX>:-Wsign-conversion>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wconversion,-Wsign-conversion>)
endif()

# Default build type: Release with host-native vectorization and CUDA fast-math.
# The default CMake `Release` uses `-O3 -DNDEBUG` for CXX; we additionally enable
# `-march=native` so the AVX-512 on Xeon 6776P is actually used, and
# `--use_fast_math` on nvcc which maps `expf`/`logf`/`sqrtf`/etc. to their hardware
# intrinsic variants.  Users can override with `-DCMAKE_BUILD_TYPE=Debug` or by
# passing their own `-DCMAKE_CXX_FLAGS_RELEASE=...`.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    message(STATUS "Defaulting to Release build (use -DCMAKE_BUILD_TYPE=... to override)")
endif()
# Append our performance flags to the Release preset without clobbering any the
# user may have set.  AppendStringsToCache would be cleaner; we do the explicit
# form because CMake doesn't provide it for language flag variables.
foreach(_lang CXX CUDA)
    # CMake-supplied default for *_FLAGS_RELEASE is already "-O3 -DNDEBUG" for CXX
    # and "-O3 -DNDEBUG" passed through to host for CUDA.  We add -march=native
    # (CXX + forwarded to host via -Xcompiler for CUDA) and --use_fast_math (CUDA only).
endforeach()
if(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    if(NOT CMAKE_CXX_FLAGS_RELEASE MATCHES "-march=")
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")
    endif()
    if(NOT CMAKE_CXX_FLAGS_RELWITHDEBINFO MATCHES "-march=")
        set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -march=native")
    endif()
    # `--generate-line-info` (= -lineinfo) is workstream GP-0: makes ncu's
    # source-counter attribution map stalls back to specific source lines in
    # etkdg_pipeline.cuh / dg_forcefield.cuh / mmff94_functor.cuh.  No
    # optimisation impact — pure debug info that survives -O3.
    # `--expt-relaxed-constexpr` lets `__host__ __device__` functions call any
    # `constexpr` function including the libstdc++ <algorithm> facilities
    # (`std::lower_bound`, `std::equal_range`) we use in the MMFF device
    # lookups.  `--extended-lambda` enables lambdas passed as comparators
    # inside those same lookups.  Required for the MG-2/3 GPU kernels.
    if(NOT CMAKE_CUDA_FLAGS_RELEASE MATCHES "use_fast_math")
        set(CMAKE_CUDA_FLAGS_RELEASE
            "${CMAKE_CUDA_FLAGS_RELEASE} --use_fast_math -Xcompiler=-march=native -Xptxas=-O3 --generate-line-info --expt-relaxed-constexpr --extended-lambda")
    endif()
    if(NOT CMAKE_CUDA_FLAGS_RELWITHDEBINFO MATCHES "use_fast_math")
        set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO
            "${CMAKE_CUDA_FLAGS_RELWITHDEBINFO} --use_fast_math -Xcompiler=-march=native -Xptxas=-O3 --generate-line-info --expt-relaxed-constexpr --extended-lambda")
    endif()
    if(NOT CMAKE_CUDA_FLAGS MATCHES "expt-relaxed-constexpr")
        set(CMAKE_CUDA_FLAGS
            "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --extended-lambda")
    endif()
endif()

# RDKit from conda-forge.
# The conda rdkit-config.cmake requires Boost "headers" component which
# confuses CMake < 3.30's FindBoost. Instead of fighting cmake, we set up
# the required imported targets manually when RDKIT_PREFIX is given.
set(RDKit_FOUND FALSE)
if(DEFINED RDKIT_PREFIX)
    set(_rdkit_inc "${RDKIT_PREFIX}/include/rdkit")
    set(_rdkit_lib "${RDKIT_PREFIX}/lib")
    set(_boost_inc "${RDKIT_PREFIX}/include")

    if(EXISTS "${_rdkit_inc}/GraphMol/GraphMol.h")
        set(RDKit_FOUND TRUE)
        message(STATUS "Using RDKit from ${RDKIT_PREFIX}")

        # helper: create an imported shared library target for one RDKit component
        macro(rdkit_add_lib target_name lib_name)
            if(NOT TARGET ${target_name})
                find_library(_${lib_name}_path NAMES ${lib_name}
                    PATHS "${_rdkit_lib}" NO_DEFAULT_PATH)
                if(_${lib_name}_path)
                    add_library(${target_name} SHARED IMPORTED)
                    set_target_properties(${target_name} PROPERTIES
                        IMPORTED_LOCATION "${_${lib_name}_path}"
                        INTERFACE_INCLUDE_DIRECTORIES "${_rdkit_inc};${_boost_inc}")
                else()
                    message(WARNING "RDKit library ${lib_name} not found in ${_rdkit_lib}")
                endif()
            endif()
        endmacro()

        rdkit_add_lib(RDKit::SmilesParse RDKitSmilesParse)
        rdkit_add_lib(RDKit::GraphMol RDKitGraphMol)
        rdkit_add_lib(RDKit::RDGeneral RDKitRDGeneral)
        rdkit_add_lib(RDKit::ForceField RDKitForceField)
        rdkit_add_lib(RDKit::ForceFieldHelpers RDKitForceFieldHelpers)
        rdkit_add_lib(RDKit::DistGeomHelpers RDKitDistGeomHelpers)
        rdkit_add_lib(RDKit::DistGeometry RDKitDistGeometry)
        rdkit_add_lib(RDKit::Depictor RDKitDepictor)
        rdkit_add_lib(RDKit::MolTransforms RDKitMolTransforms)
        rdkit_add_lib(RDKit::SubstructMatch RDKitSubstructMatch)
        rdkit_add_lib(RDKit::Alignment RDKitAlignment)
        rdkit_add_lib(RDKit::EigenSolvers RDKitEigenSolvers)
        rdkit_add_lib(RDKit::ChemTransforms RDKitChemTransforms)
        rdkit_add_lib(RDKit::Descriptors RDKitDescriptors)
        rdkit_add_lib(RDKit::MolStandardize RDKitMolStandardize)

    else()
        message(WARNING "RDKit headers not found at ${_rdkit_inc}. "
            "Install rdkit-dev: micromamba install -c conda-forge rdkit-dev")
    endif()
endif()
if(NOT RDKit_FOUND)
    message(WARNING "RDKit not found. C++ tests will not be built. "
        "Pass -DRDKIT_PREFIX=/path/to/conda/env")
endif()

# `{fmt}` — used everywhere for formatted and colored logging.  The same
# conda-forge environment that supplies RDKit ships it, so search there first.
# A system `libfmt` whose headers disagree with the RDKit prefix otherwise wins
# and the mismatch only surfaces as an undefined `fmt::v*::vprint` at link time.
if(DEFINED RDKIT_PREFIX)
    list(PREPEND CMAKE_PREFIX_PATH "${RDKIT_PREFIX}")
endif()
find_package(fmt REQUIRED CONFIG)

# OpenMP for CPU parallelism
find_package(OpenMP REQUIRED)

# jemalloc for allocation-heavy CPU paths (RDKit SMILES parsing, MMFF parameter lookups).
find_library(JEMALLOC_LIB NAMES jemalloc)
if(JEMALLOC_LIB)
    message(STATUS "Found jemalloc: ${JEMALLOC_LIB}")
else()
    message(STATUS "jemalloc not found — consider installing for better allocation performance")
endif()

# Host-side link-time optimization for CPU-bound paths.
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
endif()

# CUDA runtime libraries via CMake's modern CUDAToolkit module.  This picks up
# curand / cusolver / cudart wherever the active CUDA install keeps them
# (/usr/local/cuda/lib64, /usr/local/cuda-XX.Y/targets/x86_64-linux/lib, or a
# conda-forge prefix) and exposes them as imported targets that carry the
# correct include paths, link ordering, and rpath.  Using CUDA::* targets
# instead of raw find_library calls also means cross-version upgrades (13.0 →
# 13.1 → 14.x) do not require CMakeLists edits.
find_package(CUDAToolkit REQUIRED)

if(RDKit_FOUND)
    # RDKit targets needed by the library
    set(RDKIT_LIBS
        RDKit::SmilesParse RDKit::GraphMol RDKit::RDGeneral
        RDKit::ForceFieldHelpers RDKit::ForceField
        RDKit::DistGeomHelpers RDKit::DistGeometry
        RDKit::Depictor RDKit::SubstructMatch
        RDKit::Alignment RDKit::EigenSolvers RDKit::MolTransforms
        RDKit::ChemTransforms RDKit::Descriptors
        RDKit::MolStandardize)

    function(configure_smallfield_library target_name library_type)
        add_library(${target_name} ${library_type}
            src/smallfield/api/smallfield.cu)
        set_source_files_properties(
            src/smallfield/api/smallfield.cu
            PROPERTIES LANGUAGE CUDA)
        target_include_directories(${target_name} PUBLIC ${CMAKE_SOURCE_DIR}/src)
        target_link_libraries(${target_name} PUBLIC
            ${RDKIT_LIBS}
            CUDA::curand
            CUDA::cudart
            OpenMP::OpenMP_CXX
            fmt::fmt)
        # Pass -fopenmp to the host compiler through nvcc so #pragma omp inside
        # etkdg_prepare_batch.hpp (included by smallfield.cu) actually parallelizes.
        # Without this the OpenMP pragma is silently ignored by nvcc's host pass
        # and prepare_molecule_batch runs single-threaded.
        #
        # --expt-relaxed-constexpr lets `constexpr` host functions be called from
        # `__device__` code, which is what NumKong's tensor_view / vector_view
        # accessors rely on.  --extended-lambda permits __host__ __device__ lambdas
        # to cross the same boundary.  Both are required to consume NumKong from
        # any .cuh that runs on the GPU.
        target_compile_options(${target_name} PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>
            $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
            $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
        set_target_properties(${target_name} PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            POSITION_INDEPENDENT_CODE ON)
    endfunction()

    # SmallField libraries (CUDA + RDKit preprocessing)
    configure_smallfield_library(smallfield_lib STATIC)
    configure_smallfield_library(smallfield_python_lib SHARED)

    enable_testing()
    add_subdirectory(test)
    add_subdirectory(bench)
    add_subdirectory(cli)
endif()
