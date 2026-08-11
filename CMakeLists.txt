cmake_minimum_required(VERSION 3.24)

# Default CUDA architectures.  Each entry is a device the project is expected to run on, so the fatbin carries real SASS
# for it: Ampere (sm_80), Ada / L40S (sm_89), Hopper (sm_90), Blackwell datacenter (sm_100), and Blackwell consumer /
# RTX PRO (sm_120).
#
# `120` (vs `120-real`) is the only entry emitting PTX as well as SASS.  PTX JIT is forward-only -- it never lowers onto
# an older architecture -- so PTX from the newest entry covers hardware that hasn't shipped yet and covers nothing
# already on this list.  A device missing from the list therefore finds no cubin and no JIT path, and simply fails to
# run.
#
# `native` compiles for the local GPU alone and is what the `dev` preset selects: a single architecture is roughly 2.5x
# faster to compile, and a build that never leaves this machine has no use for the other four.  The multi-architecture
# fatbin is what ships and what CI proves.
#
# This must run BEFORE project() so the CUDA toolkit auto-detect at project-time doesn't override it with an sm_75
# default.
set(SMALLFIELD_CUDA_ARCH
    "80-real;89-real;90-real;100-real;120"
    CACHE STRING "CUDA architectures to compile for, or `native` for the local GPU alone"
)
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "${SMALLFIELD_CUDA_ARCH}")
endif ()

file(READ "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" SMALLFIELD_VERSION)
string(STRIP "${SMALLFIELD_VERSION}" SMALLFIELD_VERSION)

project(
    SmallField
    VERSION ${SMALLFIELD_VERSION}
    DESCRIPTION "SmallField: GPU conformer generation with ETKDG and MMFF94"
    HOMEPAGE_URL "https://github.com/unum-science/SmallField"
    LANGUAGES CXX CUDA
)

enable_testing()

message(STATUS "SmallField ${SMALLFIELD_VERSION}, CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE
        Release
        CACHE STRING "Build type"
    )
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
    message(STATUS "Defaulting to Release build (use -DCMAKE_BUILD_TYPE=... to override)")
endif ()

# Compiler cache, when one is installed.  Unchanged translation units then cost nothing on rebuilds, branch switches and
# CI restores, which dominates the inner loop far more than any one-off compile-speed flag.  This only pays off because
# the CUDA targets emit SASS at compile time -- see the interprocedural-optimization note below, since `ccache` can
# never cache a device link.
find_program(SMALLFIELD_COMPILER_CACHE NAMES ccache sccache)
if (SMALLFIELD_COMPILER_CACHE)
    message(STATUS "Using compiler cache: ${SMALLFIELD_COMPILER_CACHE}")
    set(CMAKE_CXX_COMPILER_LAUNCHER "${SMALLFIELD_COMPILER_CACHE}")
    set(CMAKE_CUDA_COMPILER_LAUNCHER "${SMALLFIELD_COMPILER_CACHE}")
endif ()

# Point `core.hooksPath` at `.githooks` here -- configuring the build arms the `pre-commit` / `commit-msg` checks, each
# of which carries its own rationale in that file.  Skipped when built as a subproject, since mutating a consumer's Git
# config is not this project's business, and without `.git`, as in a tarball build.
if (PROJECT_IS_TOP_LEVEL AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
    execute_process(
        COMMAND git config core.hooksPath .githooks
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
        RESULT_VARIABLE _hooks_status
        OUTPUT_QUIET ERROR_QUIET
    )
    if (_hooks_status EQUAL 0)
        message(STATUS "Git hooks enabled (core.hooksPath -> .githooks)")
    endif ()
endif ()

# RDKit from conda-forge or pixi.  The conda `rdkit-config.cmake` requires a Boost "headers" component that confuses
# CMake < 3.30's FindBoost, so the imported targets are set up by hand.
#
# The prefix is resolved in priority order: an explicit `-DRDKIT_PREFIX=`, then an activated conda/micromamba
# environment, then the in-tree pixi environment that `pixi install` creates. Testing the value rather than `DEFINED`
# matters: `-DRDKIT_PREFIX=$CONDA_PREFIX` with no environment active passes the empty string, which `DEFINED` accepts
# and which then searches `/include/rdkit`.
if (NOT RDKIT_PREFIX)
    if (DEFINED ENV{CONDA_PREFIX})
        set(RDKIT_PREFIX "$ENV{CONDA_PREFIX}")
    elseif (EXISTS "${PROJECT_SOURCE_DIR}/.pixi/envs/default/include/rdkit/GraphMol/GraphMol.h")
        set(RDKIT_PREFIX "${PROJECT_SOURCE_DIR}/.pixi/envs/default")
    endif ()
endif ()
set(RDKIT_PREFIX
    "${RDKIT_PREFIX}"
    CACHE PATH "Environment prefix providing RDKit headers and libraries"
)

if (NOT EXISTS "${RDKIT_PREFIX}/include/rdkit/GraphMol/GraphMol.h")
    message(FATAL_ERROR "RDKit headers not found under '${RDKIT_PREFIX}'.  Run `pixi install`, or pass "
                        "-DRDKIT_PREFIX=/path/to/environment."
    )
endif ()
message(STATUS "Using RDKit from ${RDKIT_PREFIX}")

# One imported shared-library target per RDKit component.
function (define_rdkit_library target_name library_name)
    find_library(
        ${library_name}_PATH
        NAMES ${library_name}
        PATHS "${RDKIT_PREFIX}/lib"
        NO_DEFAULT_PATH
    )
    if (NOT ${library_name}_PATH)
        message(FATAL_ERROR "RDKit component ${library_name} not found in ${RDKIT_PREFIX}/lib")
    endif ()
    add_library(${target_name} SHARED IMPORTED GLOBAL)
    set_target_properties(
        ${target_name} PROPERTIES IMPORTED_LOCATION "${${library_name}_PATH}"
                                  INTERFACE_INCLUDE_DIRECTORIES "${RDKIT_PREFIX}/include/rdkit;${RDKIT_PREFIX}/include"
    )
endfunction ()

set(SMALLFIELD_RDKIT_COMPONENTS
    Alignment
    ChemTransforms
    Depictor
    Descriptors
    DistGeomHelpers
    DistGeometry
    EigenSolvers
    ForceField
    ForceFieldHelpers
    GraphMol
    MolStandardize
    MolTransforms
    RDGeneral
    SmilesParse
    SubstructMatch
)
foreach (_component IN LISTS SMALLFIELD_RDKIT_COMPONENTS)
    define_rdkit_library(RDKit::${_component} RDKit${_component})
    list(APPEND SMALLFIELD_RDKIT_TARGETS RDKit::${_component})
endforeach ()

# `{fmt}` -- the same environment that supplies RDKit ships it, so search there first.  A system `libfmt` whose headers
# disagree with the RDKit prefix otherwise wins, and the mismatch surfaces only as an undefined `fmt::v*::vprint` at
# link time.
list(PREPEND CMAKE_PREFIX_PATH "${RDKIT_PREFIX}")
find_package(fmt REQUIRED CONFIG)
find_package(OpenMP REQUIRED)

# CUDA runtime libraries via CMake's CUDAToolkit module, which exposes imported targets carrying the correct include
# paths, link ordering and rpath wherever the active install keeps them.
find_package(CUDAToolkit REQUIRED)

# jemalloc for allocation-heavy CPU paths -- RDKit SMILES parsing and MMFF parameter lookups.
find_library(JEMALLOC_LIB NAMES jemalloc)
if (JEMALLOC_LIB)
    message(STATUS "Found jemalloc: ${JEMALLOC_LIB}")
else ()
    message(STATUS "jemalloc not found -- consider installing it for better allocation performance")
endif ()

# Build settings are interface targets rather than mutations of `CMAKE_<LANG>_FLAGS`.  Global flag variables leak into
# every subdirectory including future dependencies, and the config-conditional forms are invisible to multi-config
# generators, where `CMAKE_BUILD_TYPE` is empty and an `if(CMAKE_BUILD_TYPE STREQUAL "Release")` guard silently drops
# every optimization flag.  Generator expressions are evaluated per configuration and do the right thing under both.

add_library(smallfield_warnings INTERFACE)
if (NOT MSVC)
    # No `-Werror`: warnings should surface without blocking a build.  CUDA device code gets the host warnings forwarded
    # through `-Xcompiler`.
    target_compile_options(
        smallfield_warnings
        INTERFACE $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wconversion;-Wsign-conversion>
                  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wconversion,-Wsign-conversion>
    )
endif ()

option(SMALLFIELD_CUDA_LINEINFO "Emit CUDA line info for `ncu` source attribution" OFF)

add_library(smallfield_settings INTERFACE)
target_link_libraries(smallfield_settings INTERFACE smallfield_warnings)
target_include_directories(smallfield_settings INTERFACE "${PROJECT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/src")
# C++20 in both languages. As a compile feature rather than `CMAKE_CXX_STANDARD` this travels with whatever links the
# settings, including RDKit's headers, which use `constexpr` destructors and fail to parse under anything older.
target_compile_features(smallfield_settings INTERFACE cxx_std_20 cuda_std_20)
if (NOT MSVC)
    # `-march=native` uses the host's widest ISA.  It also makes the resulting archive and `.so` non-redistributable,
    # which is a shipping question rather than a build-speed one.
    target_compile_options(
        smallfield_settings INTERFACE $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release,RelWithDebInfo>>:-march=native>
    )
endif ()
# `--expt-relaxed-constexpr` lets device code call `constexpr` functions not themselves annotated `__host__ __device__`.
# It does not rescue libstdc++'s <algorithm>: `std::lower_bound` silently miscompiles on device, which is why the MMFF
# device lookups use the hand-rolled `lower_bound_u8_` in `mmff94_tables.hpp` instead.  `--extended-lambda` permits the
# comparator lambdas those lookups pass. Both are unconditional -- a Debug build needs them exactly as much as a Release
# one, and scoping them to Release is what previously broke every `.cu` under `test/` and `bench/` in Debug.
#
# `-Xcompiler=-fopenmp` reaches the host pass, without which the `#pragma omp` in the preprocess headers is silently
# ignored and batch preparation runs single-threaded.
target_compile_options(
    smallfield_settings
    INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;--extended-lambda;-Xcompiler=-fopenmp>
              $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release,RelWithDebInfo>>:--use_fast_math;-Xptxas=-O3>
              $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release,RelWithDebInfo>>:-Xcompiler=-march=native>
              $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<BOOL:${SMALLFIELD_CUDA_LINEINFO}>>:--generate-line-info>
)

# Interprocedural optimization is deliberately absent.
#
# `CMAKE_INTERPROCEDURAL_OPTIMIZATION ON` combined with `CUDA_SEPARABLE_COMPILATION ON` switches nvcc from emitting SASS
# to emitting NVVM LTO IR, which defers every architecture's `ptxas` codegen to the device-link step.  That step then
# runs once per executable -- 26 of them -- each regenerating the whole library's SASS.  Measured at 101.9 s per device
# link against 0.25 s for a plain `nvlink`, it was roughly 84 % of all CPU work in the build, and `ccache` cannot cache
# a device link at all.
#
# Device LTO optimizes across translation units, and every target here has at most one CUDA translation unit, so it had
# no input to work on.  Host LTO is separately unattractive: the host link measures 0.64 s, and `-fno-fat-lto-objects`
# leaves GIMPLE rather than machine code in the static archive, which breaks any consumer on a different GCC.

# Every target links the same interface settings, so this absorbs what was repeated at each site. CUDA is the linker
# language because these targets pull in a CUDA static library.
function (define_smallfield_target target_name)
    cmake_parse_arguments(ARG "" "" "SOURCES;LINK;DEFINES" ${ARGN})
    add_executable(${target_name} ${ARG_SOURCES})
    target_link_libraries(
        ${target_name} PRIVATE smallfield_settings OpenMP::OpenMP_CXX ${ARG_LINK}
                               $<$<BOOL:${JEMALLOC_LIB}>:${JEMALLOC_LIB}>
    )
    target_compile_definitions(${target_name} PRIVATE ${ARG_DEFINES})
    set_target_properties(${target_name} PROPERTIES LINKER_LANGUAGE CUDA CUDA_SEPARABLE_COMPILATION OFF)
endfunction ()

# The library is header-only and funnels through one translation unit, so it is compiled once into an object library and
# assembled into both the archive the C++ tools link and the shared object `setup.py` hands to pybind11.  Compiling it
# twice produced byte-identical objects.
#
# `CUDA_SEPARABLE_COMPILATION` is off: relocatable device code is only required when a `__device__` function or
# `__device__`/`__constant__` variable defined in one translation unit is referenced from another.  Every `__constant__`
# here is `static`, and there is no `extern __device__` anywhere, so each target's device code is whole-program within
# its own translation unit.
add_library(smallfield_objects OBJECT src/smallfield/api/smallfield.cu)
target_link_libraries(
    smallfield_objects PUBLIC smallfield_settings ${SMALLFIELD_RDKIT_TARGETS} CUDA::curand CUDA::cudart
                              OpenMP::OpenMP_CXX fmt::fmt
)
set_target_properties(smallfield_objects PROPERTIES CUDA_SEPARABLE_COMPILATION OFF POSITION_INDEPENDENT_CODE ON)

add_library(smallfield_lib STATIC)
target_link_libraries(smallfield_lib PUBLIC smallfield_objects)

# Only `setup.py` consumes the shared object, so a plain `cmake --build` does not pay for it.
add_library(smallfield_python_lib SHARED EXCLUDE_FROM_ALL )
target_link_libraries(smallfield_python_lib PUBLIC smallfield_objects)

add_subdirectory(test)
add_subdirectory(bench)
add_subdirectory(cli)
