# SmallField CMakeLists.txt
#
# This file defines:
#
# * smallfield_objects: the OBJECT compilation of the library. Everything but the torsion corpus is header-only and
#   funnels through `src/smallfield/api/smallfield.cu`, so it is compiled once here and assembled into both libraries
#   below.
# * smallfield_lib: the static archive that the CLI, tests and benchmarks link.
# * smallfield_python_lib: the shared object `setup.py` hands to pybind11. EXCLUDE_FROM_ALL, since nothing but the
#   Python build wants it.
# * smallfield_settings / smallfield_warnings: interface targets carrying the language standard, include roots, warning
#   set and CUDA flags. Every target opts in by linking them, so no build setting is a global variable mutation.
#
# Consumers link the same spelling whether this tree is vendored or installed:
#
#     find_package(smallfield REQUIRED)   # installed
#     add_subdirectory(smallfield)        # vendored
#     target_link_libraries(app PRIVATE smallfield::smallfield)
#
# Options, all declared together below and none of them in a subdirectory:
#
#     SMALLFIELD_BUILD_TESTS        Build the ctest suite            ON when top-level
#     SMALLFIELD_BUILD_BENCHMARKS   Build the benchmarks             ON when top-level
#     SMALLFIELD_BUILD_CLI          Build the two Parquet binaries   OFF, needs Apache Arrow
#     SMALLFIELD_USE_CCACHE         Route compiles through ccache    ON when one is installed
#     SMALLFIELD_USE_JEMALLOC       Link jemalloc                    ON when it is installed
#     SMALLFIELD_USE_NVTX           NVTX ranges for `nsys`           OFF
#     SMALLFIELD_CUDA_ARCH          Architectures, or `native`       the shipped fatbin
#     SMALLFIELD_CUDA_LINEINFO      Line info for `ncu`              OFF
#
# `cmake/rdkit.cmake` builds RDKit's imported targets and is shared with the installed package
# config, since a consumer's project has no `RDKit::*` targets for the export set to reference.
# The version is not generated: `src/smallfield/version.hpp` carries it, and the release workflow
# rewrites it from `VERSION`.

cmake_minimum_required(VERSION 3.24)

file(READ "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" SMALLFIELD_VERSION)
string(STRIP "${SMALLFIELD_VERSION}" SMALLFIELD_VERSION)

# CUDA is enabled further down, once its architectures are known -- see the `enable_language` call.
project(
    smallfield
    VERSION ${SMALLFIELD_VERSION}
    DESCRIPTION "SmallField: GPU conformer generation with ETKDG and MMFF94"
    HOMEPAGE_URL "https://github.com/unum-science/SmallField"
    LANGUAGES CXX
)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

# A vendoring consumer gets the library and nothing else.
option(SMALLFIELD_BUILD_TESTS "Build the ctest suite" ${PROJECT_IS_TOP_LEVEL})
option(SMALLFIELD_BUILD_BENCHMARKS "Build the benchmarks" ${PROJECT_IS_TOP_LEVEL})
option(SMALLFIELD_BUILD_CLI "Build the smallfield_batch and smallfield_descriptors binaries" OFF)

option(SMALLFIELD_USE_CCACHE "Route compilation through ccache or sccache when one is installed" ON)
option(SMALLFIELD_USE_JEMALLOC "Link jemalloc for the allocation-heavy host paths" ON)
option(SMALLFIELD_USE_NVTX "Annotate pipeline roles with NVTX ranges for `nsys` timelines" OFF)

option(SMALLFIELD_CUDA_LINEINFO "Emit CUDA line info so `ncu` can attribute stalls to source lines" OFF)

# Each entry is a device this is expected to run on, so the fatbin carries real SASS for it: Ampere (sm_80), Ada / L40S
# (sm_89), Hopper (sm_90), Blackwell datacenter (sm_100), and Blackwell consumer / RTX PRO (sm_120).
#
# `120` without `-real` is the only entry emitting PTX as well as SASS. PTX JIT is forward-only -- it never lowers onto
# an older architecture -- so PTX from the newest entry covers hardware that has not shipped yet and covers nothing
# already on this list. A device missing from the list finds no cubin and no JIT path, and does not run.
#
# `native` compiles for the local GPU alone and is what the `dev` preset selects: one architecture is roughly 2.5x
# faster to compile, and a build that never leaves this machine has no use for the other four. The multi-architecture
# fatbin is what ships, and CI is what proves it.
set(SMALLFIELD_CUDA_ARCH
    "80-real;89-real;90-real;100-real;120"
    CACHE STRING "CUDA architectures to compile for, or `native` for the local GPU alone"
)

# `-DCMAKE_CUDA_ARCHITECTURES=` is honoured when passed directly, but then it and SMALLFIELD_CUDA_ARCH disagree while
# only one is in effect, so say which won rather than letting the cache entry lie.
if (DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT CMAKE_CUDA_ARCHITECTURES STREQUAL SMALLFIELD_CUDA_ARCH)
    message(STATUS "CMAKE_CUDA_ARCHITECTURES overrides SMALLFIELD_CUDA_ARCH ('${SMALLFIELD_CUDA_ARCH}')")
else ()
    set(CMAKE_CUDA_ARCHITECTURES "${SMALLFIELD_CUDA_ARCH}")
endif ()
enable_language(CUDA)

message(STATUS "SmallField ${PROJECT_VERSION}, CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE
        Release
        CACHE STRING "Build type"
    )
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
    message(STATUS "Defaulting to Release build (use -DCMAKE_BUILD_TYPE=... to override)")
endif ()

# `.clang-tidy` ships with this repository and is useless without a compilation database.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Unchanged translation units cost nothing on rebuilds, branch switches and CI restores, which dominates the inner loop
# far more than any one-off compile-speed flag. This only pays off because the CUDA targets emit SASS at compile time --
# see the interprocedural-optimization note below, since `ccache` can never cache a device link.
if (SMALLFIELD_USE_CCACHE)
    find_program(SMALLFIELD_COMPILER_CACHE NAMES ccache sccache)
    if (SMALLFIELD_COMPILER_CACHE)
        message(STATUS "Using compiler cache: ${SMALLFIELD_COMPILER_CACHE}")
        set(CMAKE_CXX_COMPILER_LAUNCHER "${SMALLFIELD_COMPILER_CACHE}")
        set(CMAKE_CUDA_COMPILER_LAUNCHER "${SMALLFIELD_COMPILER_CACHE}")
    endif ()
endif ()

# Configuring the build arms the `pre-commit` / `commit-msg` checks, each of which carries its own rationale in
# `.githooks`. Skipped when built as a subproject, since mutating a consumer's Git config is not this project's
# business, and without `.git`, as in a tarball build.
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

set(SMALLFIELD_IN_TREE_PIXI_PREFIX "${PROJECT_SOURCE_DIR}/.pixi/envs/default")
include(rdkit)
message(STATUS "Using RDKit from ${RDKIT_PREFIX}")

# `{fmt}` -- the same environment that supplies RDKit ships it, so search there first. A system `libfmt` whose headers
# disagree with the RDKit prefix otherwise wins, and the mismatch surfaces only as an undefined `fmt::v*::vprint` at
# link time.
list(PREPEND CMAKE_PREFIX_PATH "${RDKIT_PREFIX}")
find_package(fmt REQUIRED CONFIG)
find_package(OpenMP REQUIRED)

# CUDA runtime libraries via CMake's CUDAToolkit module, which exposes imported targets carrying the correct include
# paths, link ordering and rpath wherever the active install keeps them.
find_package(CUDAToolkit REQUIRED)

# jemalloc for allocation-heavy host paths -- RDKit SMILES parsing and MMFF parameter lookups.
# `JEMALLOC_LIB` is a cached `find_library` result and outlives the option that produced it, so consumers
# must link the plain variable below rather than the cache entry: turning the option off in an existing
# build tree leaves the cache populated, and every binary would keep `libjemalloc.so.2` in `DT_NEEDED`.
# That is not cosmetic -- compute-sanitizer refuses to start against a jemalloc-linked binary, so the
# option silently failed to do the one thing anyone turns it off for.
if (SMALLFIELD_USE_JEMALLOC)
    find_library(JEMALLOC_LIB NAMES jemalloc)
    if (JEMALLOC_LIB)
        message(STATUS "Found jemalloc: ${JEMALLOC_LIB}")
    else ()
        message(STATUS "jemalloc not found -- consider installing it for better allocation performance")
    endif ()
    set(SMALLFIELD_JEMALLOC_LINK "${JEMALLOC_LIB}")
else ()
    set(SMALLFIELD_JEMALLOC_LINK "")
    message(STATUS "jemalloc disabled -- required for compute-sanitizer, ncu and nsys")
endif ()

# `argparse` -- the command-line grammar shared by `smallfield_batch` and `smallfield_descriptors`, and nothing else.
# Three costs, none of them fixable from this side:
#
# * It reports every error by throwing, with no macro to disable that, so both `main`s wrap `parse_args` in a `try` and
#   convert the exception into an exit code.
# * It copies `argv`, which the OS already owns and which outlives `main`, and parks the parsed values in `std::any`,
#   against the caller-owned-scratch rule in `CONTRIBUTING.md`.
# * It adds about 3.4 s to every consuming translation unit: 8.3 s to 11.9 s for `smallfield_batch.cpp` and 8.9 s to
#   12.2 s for `smallfield_descriptors.cpp`. Only 0.4 s of that is the front end -- the header parses in 1.0 s alone,
#   but both files already included most of what it drags in, `<any>`, `<variant>`, `<sstream>`, `<iostream>`,
#   `<filesystem>`, `<list>` and `<map>`. The rest is `-O2` on the instantiated parser.
#
# Resolved from `pixi` as `cpp-argparse`, like every other dependency here, so a fresh clone needs no network fetch
# beyond the environment it already installs.
if (SMALLFIELD_BUILD_CLI)
    find_package(argparse CONFIG REQUIRED)
endif ()

# Build settings are interface targets rather than mutations of `CMAKE_<LANG>_FLAGS`. Global flag variables leak into
# every subdirectory including future dependencies, and the config-conditional forms are invisible to multi-config
# generators, where `CMAKE_BUILD_TYPE` is empty and an `if(CMAKE_BUILD_TYPE STREQUAL "Release")` guard silently drops
# every optimization flag. Generator expressions are evaluated per configuration and do the right thing under both.

add_library(smallfield_warnings INTERFACE)
if (NOT MSVC)
    # No `-Werror`: warnings should surface without blocking a build. CUDA device code gets the host warnings forwarded
    # through `-Xcompiler`.
    target_compile_options(
        smallfield_warnings
        INTERFACE $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wconversion;-Wsign-conversion>
                  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wconversion,-Wsign-conversion>
    )
endif ()

add_library(smallfield_settings INTERFACE)
target_link_libraries(smallfield_settings INTERFACE smallfield_warnings)
target_include_directories(
    smallfield_settings INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}> $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
                                  $<INSTALL_INTERFACE:include>
)
# C++20. As a compile feature rather than `CMAKE_CXX_STANDARD` this travels with whatever links the settings, including
# RDKit's headers, which use `constexpr` destructors and fail to parse under anything older.
#
# The CUDA standard is deliberately NOT an interface feature: `cuda_std_20` demands a CUDA compiler from everything that
# links these settings, so a consumer that only wants the headers cannot even configure without `enable_language(CUDA)`.
# It is a property of our own CUDA targets instead.
target_compile_features(smallfield_settings INTERFACE cxx_std_20)
target_compile_definitions(smallfield_settings INTERFACE $<$<BOOL:${SMALLFIELD_USE_NVTX}>:SMALLFIELD_ENABLE_NVTX>)
if (NOT MSVC)
    # `-march=native` uses the host's widest ISA. It also makes the resulting archive and `.so` non-redistributable,
    # which is a shipping question rather than a build-speed one.
    target_compile_options(
        smallfield_settings INTERFACE $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release,RelWithDebInfo>>:-march=native>
    )
endif ()
# `--expt-relaxed-constexpr` lets device code call `constexpr` functions not themselves annotated `__host__ __device__`.
# It does not rescue libstdc++'s <algorithm>: `std::lower_bound` silently miscompiles on device, which is why the MMFF
# device lookups use the hand-rolled `lower_bound_` in `mmff94_tables.hpp` instead. `--extended-lambda` permits the
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
# to emitting NVVM LTO IR, which defers every architecture's `ptxas` codegen to the device-link step. That step then
# runs once per executable, each regenerating the whole library's SASS. Measured at 101.9 s per device
# link against 0.25 s for a plain `nvlink`, it was roughly 84 % of all CPU work in the build, and `ccache` cannot cache
# a device link at all.
#
# Device LTO optimizes across translation units, and every target here has at most one CUDA translation unit, so it had
# no input to work on. Host LTO is separately unattractive: the host link measures 0.64 s, and `-fno-fat-lto-objects`
# leaves GIMPLE rather than machine code in the static archive, which breaks any consumer on a different GCC.

# Every executable links the same interface settings, so this absorbs what was repeated at each site. CUDA is the linker
# language because these targets pull in a CUDA static library.
function (define_smallfield_target target_name)
    cmake_parse_arguments(ARG "" "" "SOURCES;LINK;DEFINES" ${ARGN})
    add_executable(${target_name} ${ARG_SOURCES})
    target_link_libraries(
        ${target_name} PRIVATE smallfield_settings OpenMP::OpenMP_CXX ${ARG_LINK}
                               $<$<BOOL:${SMALLFIELD_JEMALLOC_LINK}>:${SMALLFIELD_JEMALLOC_LINK}>
    )
    target_compile_definitions(${target_name} PRIVATE ${ARG_DEFINES})
    set_target_properties(
        ${target_name}
        PROPERTIES LINKER_LANGUAGE CUDA
                   CUDA_SEPARABLE_COMPILATION OFF
                   CUDA_STANDARD 20
                   CUDA_STANDARD_REQUIRED ON
    )
endfunction ()

# The library is header-only and funnels through one translation unit, so it is compiled once into an object library and
# assembled into both the archive the C++ tools link and the shared object `setup.py` hands to pybind11. Compiling it
# twice produced byte-identical objects.
#
# `CUDA_SEPARABLE_COMPILATION` is off: relocatable device code is only required when a `__device__` function or
# `__device__`/`__constant__` variable defined in one translation unit is referenced from another. Every `__constant__`
# here is `static`, and there is no `extern __device__` anywhere, so each target's device code is whole-program within
# its own translation unit.
# `csd_torsion_table.cpp` is host-only on purpose: it carries the 470-row torsion corpus that used to arrive as escaped
# string literals inside a header, which every CUDA translation unit then re-tokenized once per architecture for data no
# kernel reads.
add_library(smallfield_objects OBJECT src/smallfield/api/smallfield.cu src/smallfield/preprocess/csd_torsion_table.cpp)
target_link_libraries(
    smallfield_objects PUBLIC smallfield_settings smallfield_rdkit CUDA::curand CUDA::cudart OpenMP::OpenMP_CXX
                              fmt::fmt
)
set_target_properties(
    smallfield_objects
    PROPERTIES CUDA_SEPARABLE_COMPILATION OFF
               POSITION_INDEPENDENT_CODE ON
               CUDA_STANDARD 20
               CUDA_STANDARD_REQUIRED ON
)

add_library(smallfield_lib STATIC)
target_link_libraries(smallfield_lib PUBLIC smallfield_objects)
# `EXPORT_NAME` is what the installed package offers; the ALIAS is what the vendored tree offers. Both are needed for
# `smallfield::smallfield` to mean the same thing either way -- an ALIAS alone is build-tree only, and the export would
# otherwise publish `smallfield::smallfield_lib`.
set_target_properties(smallfield_lib PROPERTIES EXPORT_NAME smallfield)
add_library(smallfield::smallfield ALIAS smallfield_lib)

# Only `setup.py` consumes the shared object, so a plain `cmake --build` does not pay for it.
add_library(smallfield_python_lib SHARED EXCLUDE_FROM_ALL )
target_link_libraries(smallfield_python_lib PUBLIC smallfield_objects)

# Installation. The headers are the library -- everything but one translation unit is header-only -- so the archive
# alone would be useless to a consumer.
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(
    TARGETS smallfield_lib smallfield_objects smallfield_settings smallfield_warnings
    EXPORT smallfieldTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(
    DIRECTORY src/smallfield
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING
    PATTERN "*.hpp"
    PATTERN "*.cuh"
)
install(FILES "${PROJECT_SOURCE_DIR}/cmake/rdkit.cmake" DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/smallfield)
install(
    EXPORT smallfieldTargets
    NAMESPACE smallfield::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/smallfield
)

configure_package_config_file(
    "${PROJECT_SOURCE_DIR}/cmake/smallfieldConfig.cmake.in" "${PROJECT_BINARY_DIR}/smallfieldConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/smallfield
)
write_basic_package_version_file(
    "${PROJECT_BINARY_DIR}/smallfieldConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMinorVersion
)
install(FILES "${PROJECT_BINARY_DIR}/smallfieldConfig.cmake" "${PROJECT_BINARY_DIR}/smallfieldConfigVersion.cmake"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/smallfield
)

# `enable_testing()` has to be called from the top-level directory, not from `test/`, or `ctest` finds no tests at all.
if (SMALLFIELD_BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif ()
if (SMALLFIELD_BUILD_BENCHMARKS)
    add_subdirectory(bench)
endif ()
if (SMALLFIELD_BUILD_CLI)
    add_subdirectory(cli)
endif ()
