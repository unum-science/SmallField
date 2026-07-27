# SmallField batch binary: restartable Parquet folder → conformer Parquet folder.
# Builds a CLI executable that processes a folder of Parquet files (one `smiles`
# column) and emits 3D conformer coordinates + energies + status for K conformers
# per molecule.  Three-tier fallback: GPU → RDKit → NULL.

option(SMALLFIELD_BUILD_BATCH_BINARY "Build smallfield_batch CLI (requires Apache Arrow)" OFF)
# Opt-in host-side NVTX ranges for `nsys` timelines.  Header-only (NVTX3 ships with
# the CUDA toolkit); compiles to nothing when OFF, so shipping builds are unaffected.
option(SMALLFIELD_ENABLE_NVTX "Annotate batch pipeline roles with NVTX ranges" OFF)

if(NOT SMALLFIELD_BUILD_BATCH_BINARY)
    return()
endif()

# Locate Arrow / Parquet.  conda-forge libraries ship cmake config files under
# ${RDKIT_PREFIX}/lib/cmake/Arrow and Parquet.
if(DEFINED RDKIT_PREFIX)
    list(PREPEND CMAKE_PREFIX_PATH "${RDKIT_PREFIX}")
endif()

find_package(Arrow CONFIG REQUIRED)
find_package(Parquet CONFIG REQUIRED)

add_executable(smallfield_batch
    smallfield_batch.cpp
    parquet_reader.cpp
    parquet_writer.cpp
    shard_inventory.cpp
    rdkit_fallback.cpp)

set_source_files_properties(
    smallfield_batch.cpp parquet_reader.cpp parquet_writer.cpp
    shard_inventory.cpp rdkit_fallback.cpp
    PROPERTIES LANGUAGE CXX)

target_link_libraries(smallfield_batch PRIVATE
    smallfield_lib
    OpenMP::OpenMP_CXX
    Arrow::arrow_shared
    Parquet::parquet_shared
    $<$<BOOL:${JEMALLOC_LIB}>:${JEMALLOC_LIB}>)

target_include_directories(smallfield_batch PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

set_target_properties(smallfield_batch PROPERTIES
    LINKER_LANGUAGE CUDA)

# Embed the producer git SHA so output shards are self-describing in a content-addressed
# store (see the provenance key-value metadata in parquet_writer.cpp).
execute_process(
    COMMAND git -C ${CMAKE_SOURCE_DIR} rev-parse --short=12 HEAD
    OUTPUT_VARIABLE SMALLFIELD_GIT_SHA_VALUE
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
if(NOT SMALLFIELD_GIT_SHA_VALUE)
    set(SMALLFIELD_GIT_SHA_VALUE "unknown")
endif()
target_compile_definitions(smallfield_batch PRIVATE SMALLFIELD_GIT_SHA="${SMALLFIELD_GIT_SHA_VALUE}")

message(STATUS "SmallField batch binary: ENABLED (git ${SMALLFIELD_GIT_SHA_VALUE})")

if(SMALLFIELD_ENABLE_NVTX)
    target_compile_definitions(smallfield_batch PRIVATE SMALLFIELD_ENABLE_NVTX)
endif()

# Descriptor-augmentation binary: rewrites existing *.3D.parquet shards into the
# canonical 14-column schema by computing USR/USRCAT from the stored fp16 coordinates
# and re-emitting through the shared writer (parquet_writer.cpp) — byte-identical to the
# batch producer's output.  CPU-only; reuses the producer's writer + descriptor code.
add_executable(smallfield_descriptors
    smallfield_descriptors.cpp
    parquet_writer.cpp)

set_source_files_properties(
    smallfield_descriptors.cpp parquet_writer.cpp
    PROPERTIES LANGUAGE CXX)

target_link_libraries(smallfield_descriptors PRIVATE
    smallfield_lib
    OpenMP::OpenMP_CXX
    Arrow::arrow_shared
    Parquet::parquet_shared
    $<$<BOOL:${JEMALLOC_LIB}>:${JEMALLOC_LIB}>)

target_include_directories(smallfield_descriptors PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

set_target_properties(smallfield_descriptors PROPERTIES
    LINKER_LANGUAGE CUDA)

# Same provenance SHA as the batch producer so augmented shards report an identical
# producer_git_sha in their key-value metadata.
target_compile_definitions(smallfield_descriptors PRIVATE SMALLFIELD_GIT_SHA="${SMALLFIELD_GIT_SHA_VALUE}")
