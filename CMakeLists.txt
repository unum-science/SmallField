cmake_minimum_required(VERSION 3.18)
project(SmallField LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# detect GPU architecture at build time, fall back to SM 9.0 (H100)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "90")
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

# ForkUnion (header-only submodule)
add_subdirectory(ForkUnion)

# curand library
find_library(CURAND_LIBRARY curand
    HINTS ${CUDAToolkit_LIBRARY_DIR}
          ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}/../lib64
          /usr/local/cuda/lib64)

if(RDKit_FOUND)
    enable_testing()
    add_subdirectory(tests)
endif()
