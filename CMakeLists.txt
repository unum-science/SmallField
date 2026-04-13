cmake_minimum_required(VERSION 3.18)
project(SmallField LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

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
    if(NOT CMAKE_CUDA_FLAGS_RELEASE MATCHES "use_fast_math")
        set(CMAKE_CUDA_FLAGS_RELEASE
            "${CMAKE_CUDA_FLAGS_RELEASE} --use_fast_math -Xcompiler=-march=native -Xptxas=-O3 --generate-line-info")
    endif()
    if(NOT CMAKE_CUDA_FLAGS_RELWITHDEBINFO MATCHES "use_fast_math")
        set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO
            "${CMAKE_CUDA_FLAGS_RELWITHDEBINFO} --use_fast_math -Xcompiler=-march=native -Xptxas=-O3 --generate-line-info")
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
            set(CMAKE_CUDA_ARCHITECTURES "90")
            message(STATUS "Could not parse GPU capability, defaulting to SM 90")
        endif()
    else()
        set(CMAKE_CUDA_ARCHITECTURES "90")
        message(STATUS "nvidia-smi not found, defaulting to SM 90")
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

        # Boost libs that RDKit links against
        find_library(_boost_serialization NAMES boost_serialization
            PATHS "${_rdkit_lib}" NO_DEFAULT_PATH)
        find_library(_boost_iostreams NAMES boost_iostreams
            PATHS "${_rdkit_lib}" NO_DEFAULT_PATH)
    else()
        message(WARNING "RDKit headers not found at ${_rdkit_inc}. "
                        "Install rdkit-dev: micromamba install -c conda-forge rdkit-dev")
    endif()
endif()
if(NOT RDKit_FOUND)
    message(WARNING "RDKit not found. C++ tests will not be built. "
                    "Pass -DRDKIT_PREFIX=/path/to/conda/env")
endif()

# OpenMP for CPU parallelism
find_package(OpenMP REQUIRED)

# curand library
find_library(CURAND_LIBRARY curand
    HINTS ${CUDAToolkit_LIBRARY_DIR}
          ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}/../lib64
          /usr/local/cuda/lib64)

# cusolver library
find_library(CUSOLVER_LIBRARY cusolver
    HINTS ${CUDAToolkit_LIBRARY_DIR}
          ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}/../lib64
          /usr/local/cuda/lib64)

if(RDKit_FOUND)
    # RDKit targets needed by the library
    set(RDKIT_LIBS
        RDKit::SmilesParse RDKit::GraphMol RDKit::RDGeneral
        RDKit::ForceFieldHelpers RDKit::ForceField
        RDKit::DistGeomHelpers RDKit::DistGeometry
        RDKit::Depictor RDKit::SubstructMatch
        RDKit::Alignment RDKit::EigenSolvers RDKit::MolTransforms
        RDKit::ChemTransforms)

    # SmallField library (CUDA + RDKit preprocessing)
    add_library(smallfield_lib STATIC src/smallfield.cu)
    set_source_files_properties(src/smallfield.cu PROPERTIES LANGUAGE CUDA)
    target_include_directories(smallfield_lib PUBLIC ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(smallfield_lib PUBLIC
        ${RDKIT_LIBS}
        ${CURAND_LIBRARY}
        ${CUSOLVER_LIBRARY}
        cudart
        OpenMP::OpenMP_CXX)
    # Pass -fopenmp to the host compiler through nvcc so #pragma omp inside
    # etkdg_preprocess.hpp (included by smallfield.cu) actually parallelizes.
    # Without this the OpenMP pragma is silently ignored by nvcc's host pass
    # and prepare_molecule_batch runs single-threaded.
    target_compile_options(smallfield_lib PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>)
    set_target_properties(smallfield_lib PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON)

    enable_testing()
    add_subdirectory(test)
    add_subdirectory(bench)
endif()
