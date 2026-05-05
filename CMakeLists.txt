cmake_minimum_required(VERSION 3.18)
project(SmallField LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Named fallback for the CUDA architecture used when nvidia-smi probing fails.
# SM 12.0 targets Blackwell RTX PRO, the primary dev/test GPU for SmallField.
set(SMALLFIELD_DEFAULT_CUDA_ARCH "120" CACHE STRING "Fallback CUDA SM architecture if auto-detection fails")

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

# detect GPU architecture at build time
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # detect highest GPU compute capability via nvidia-smi
    execute_process(
        COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
        OUTPUT_VARIABLE _gpu_caps OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET RESULT_VARIABLE _smi_result)
    if(_smi_result EQUAL 0 AND _gpu_caps)
        # take the first line (first GPU)
        string(REGEX MATCH "[0-9]+\\.[0-9]+" _cap_match "${_gpu_caps}")
        if(_cap_match)
            string(REPLACE "." "" _arch_code "${_cap_match}")
            set(CMAKE_CUDA_ARCHITECTURES "${_arch_code}")
            message(STATUS "Detected GPU compute capability: ${_cap_match} (SM ${_arch_code})")
        else()
            set(CMAKE_CUDA_ARCHITECTURES "${SMALLFIELD_DEFAULT_CUDA_ARCH}")
            message(STATUS "Could not parse GPU capability, defaulting to SM ${SMALLFIELD_DEFAULT_CUDA_ARCH}")
        endif()
    else()
        set(CMAKE_CUDA_ARCHITECTURES "${SMALLFIELD_DEFAULT_CUDA_ARCH}")
        message(STATUS "nvidia-smi not found, defaulting to SM ${SMALLFIELD_DEFAULT_CUDA_ARCH}")
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

        rdkit_add_lib(RDKit::SmilesParse      RDKitSmilesParse)
        rdkit_add_lib(RDKit::GraphMol         RDKitGraphMol)
        rdkit_add_lib(RDKit::RDGeneral        RDKitRDGeneral)
        rdkit_add_lib(RDKit::ForceField       RDKitForceField)
        rdkit_add_lib(RDKit::ForceFieldHelpers RDKitForceFieldHelpers)
        rdkit_add_lib(RDKit::DistGeomHelpers  RDKitDistGeomHelpers)
        rdkit_add_lib(RDKit::DistGeometry     RDKitDistGeometry)
        rdkit_add_lib(RDKit::Depictor         RDKitDepictor)
        rdkit_add_lib(RDKit::MolTransforms    RDKitMolTransforms)
        rdkit_add_lib(RDKit::SubstructMatch   RDKitSubstructMatch)
        rdkit_add_lib(RDKit::Alignment        RDKitAlignment)
        rdkit_add_lib(RDKit::EigenSolvers     RDKitEigenSolvers)
        rdkit_add_lib(RDKit::ChemTransforms   RDKitChemTransforms)
        rdkit_add_lib(RDKit::Descriptors      RDKitDescriptors)

    else()
        message(WARNING "RDKit headers not found at ${_rdkit_inc}. "
                        "Install rdkit-dev: micromamba install -c conda-forge rdkit-dev")
    endif()
endif()
if(NOT RDKit_FOUND)
    message(WARNING "RDKit not found. C++ tests will not be built. "
                    "Pass -DRDKIT_PREFIX=/path/to/conda/env")
endif()

# `{fmt}` — used everywhere for formatted and colored logging.  Conda-forge
# ships it at `$CONDA_PREFIX/lib/cmake/fmt/`; pass `-Dfmt_DIR=<prefix>/lib/cmake/fmt`
# if the package is installed outside CMAKE_PREFIX_PATH.
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

# ThunderKittens — header-only tile-based MMA library used by the MACE-OFF
# inference module.  Default OFF: the existing fp32 reference path stays
# buildable without TK (regression target).  Turn ON for the bf16 / WGMMA
# production build.  Pass `-DTHUNDERKITTENS_DIR=/path/to/ThunderKittens`
# pointing to a clone of https://github.com/HazyResearch/ThunderKittens
# (validated against commit 0b55588d2769dde0c7a9a606ffc015b0ca7a9551).
#
# When TK is on, MACE kernels need `-arch=sm_90a` (Hopper, with the `a`
# suffix that enables WGMMA / TMA features) or `-arch=sm_100a` / `sm_120a`
# on Blackwell.  The build appends `-DSMALLFIELD_USE_THUNDERKITTENS=1` and
# the right arch suffix automatically when TK is requested.
option(SMALLFIELD_USE_THUNDERKITTENS
       "Enable ThunderKittens for the MACE-OFF inference module"
       OFF)

if(SMALLFIELD_USE_THUNDERKITTENS)
    # If THUNDERKITTENS_DIR is not provided, fetch upstream automatically.
    # Pinning to the same commit hash that the kernel sources were validated
    # against (per the comment block above this option declaration) keeps
    # the captured-graph numerics stable across reproductions.
    if(NOT DEFINED THUNDERKITTENS_DIR)
        include(FetchContent)
        set(SMALLFIELD_THUNDERKITTENS_TAG "0b55588d2769dde0c7a9a606ffc015b0ca7a9551"
            CACHE STRING "ThunderKittens git ref to fetch when THUNDERKITTENS_DIR is unset")
        FetchContent_Declare(thunderkittens
            GIT_REPOSITORY https://github.com/HazyResearch/ThunderKittens.git
            GIT_TAG        "${SMALLFIELD_THUNDERKITTENS_TAG}"
            GIT_SHALLOW    OFF) # tag is a hash, shallow doesn't help
        FetchContent_MakeAvailable(thunderkittens)
        set(THUNDERKITTENS_DIR "${thunderkittens_SOURCE_DIR}" CACHE PATH "" FORCE)
        message(STATUS "ThunderKittens fetched to ${THUNDERKITTENS_DIR}")
    endif()
    if(NOT EXISTS "${THUNDERKITTENS_DIR}/include/kittens.cuh")
        message(FATAL_ERROR
                "ThunderKittens not found at ${THUNDERKITTENS_DIR}/include/kittens.cuh. "
                "Clone https://github.com/HazyResearch/ThunderKittens and pass its "
                "absolute path via -DTHUNDERKITTENS_DIR=..., or omit the variable "
                "to let CMake FetchContent pull the pinned commit.")
    endif()
    add_library(smallfield_thunderkittens INTERFACE)
    target_include_directories(smallfield_thunderkittens INTERFACE "${THUNDERKITTENS_DIR}/include")
    target_compile_definitions(smallfield_thunderkittens INTERFACE SMALLFIELD_USE_THUNDERKITTENS=1)
    message(STATUS "ThunderKittens enabled (include path: ${THUNDERKITTENS_DIR}/include)")

    # Promote the auto-detected Hopper / Blackwell arch to the `a` variant
    # so WGMMA / TMA intrinsics are available.  TK's persistent-kernel
    # patterns rely on these features; without the `a` suffix the build
    # falls back to non-async loads and the throughput win disappears.
    if(CMAKE_CUDA_ARCHITECTURES)
        set(_archs_with_a "")
        foreach(_arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
            if(NOT "${_arch}" MATCHES "a$")
                list(APPEND _archs_with_a "${_arch}a")
            else()
                list(APPEND _archs_with_a "${_arch}")
            endif()
        endforeach()
        set(CMAKE_CUDA_ARCHITECTURES "${_archs_with_a}" CACHE STRING "" FORCE)
        message(STATUS "ThunderKittens: promoted CUDA archs to ${CMAKE_CUDA_ARCHITECTURES}")
    endif()
else()
    message(STATUS "ThunderKittens disabled — MACE-OFF inference uses the fp32 reference path. "
                   "Pass -DSMALLFIELD_USE_THUNDERKITTENS=ON -DTHUNDERKITTENS_DIR=... to opt in.")
endif()

if(RDKit_FOUND)
    # RDKit targets needed by the library
    set(RDKIT_LIBS
        RDKit::SmilesParse RDKit::GraphMol RDKit::RDGeneral
        RDKit::ForceFieldHelpers RDKit::ForceField
        RDKit::DistGeomHelpers RDKit::DistGeometry
        RDKit::Depictor RDKit::SubstructMatch
        RDKit::Alignment RDKit::EigenSolvers RDKit::MolTransforms
        RDKit::ChemTransforms RDKit::Descriptors)

    function(configure_smallfield_library target_name library_type)
        add_library(${target_name} ${library_type}
            src/smallfield/api/smallfield.cu
            src/smallfield/api/smallfield_mace.cu)
        set_source_files_properties(
            src/smallfield/api/smallfield.cu
            src/smallfield/api/smallfield_mace.cu
            PROPERTIES LANGUAGE CUDA)
        target_include_directories(${target_name} PUBLIC ${CMAKE_SOURCE_DIR}/src)
        target_link_libraries(${target_name} PUBLIC
            ${RDKIT_LIBS}
            CUDA::curand
            CUDA::cusolver
            CUDA::cudart
            OpenMP::OpenMP_CXX
            fmt::fmt)
        # When ThunderKittens is enabled, the MACE-OFF inference TU pulls in
        # TK headers transitively via mace_inference.cuh.  Link the interface
        # target so the include path and SMALLFIELD_USE_THUNDERKITTENS define
        # are visible.
        if(TARGET smallfield_thunderkittens)
            target_link_libraries(${target_name} PUBLIC smallfield_thunderkittens)
        endif()
        # Pass -fopenmp to the host compiler through nvcc so #pragma omp inside
        # etkdg_prepare_batch.hpp (included by smallfield.cu) actually parallelizes.
        # Without this the OpenMP pragma is silently ignored by nvcc's host pass
        # and prepare_molecule_batch runs single-threaded.
        target_compile_options(${target_name} PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>)
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
