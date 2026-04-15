# SmallField batch binary: restartable Parquet folder → conformer Parquet folder.
# Builds a CLI executable that processes a folder of Parquet files (one `smiles`
# column) and emits 3D conformer coordinates + energies + status for K conformers
# per molecule.  Three-tier fallback: GPU → RDKit → NULL.

option(SMALLFIELD_BUILD_BATCH_BINARY "Build smallfield_batch CLI (requires Apache Arrow)" OFF)

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
    Parquet::parquet_shared)

target_include_directories(smallfield_batch PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

set_target_properties(smallfield_batch PROPERTIES
    LINKER_LANGUAGE CUDA)

message(STATUS "SmallField batch binary: ENABLED")
