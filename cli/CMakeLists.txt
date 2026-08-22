# SmallField batch binary: a restartable Parquet folder of SMILES in, a Parquet folder of 3D conformer coordinates,
# energies and per-conformer status out.  Three-tier fallback: GPU, then RDKit, then NULL.

# conda-forge ships the Arrow and Parquet config files under the RDKit prefix, which the root already prepended to
# CMAKE_PREFIX_PATH -- subdirectories inherit it.
find_package(Arrow CONFIG REQUIRED)
find_package(Parquet CONFIG REQUIRED)

# The git SHA stamped into every shard's metadata is read at configure time, so HEAD is registered as a configure
# dependency and moving it re-runs cmake; without that a shard would name a revision that did not produce it.
set_property(
    DIRECTORY "${PROJECT_SOURCE_DIR}"
    APPEND
    PROPERTY CMAKE_CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/.git/HEAD"
)
execute_process(
    COMMAND git -C "${PROJECT_SOURCE_DIR}" rev-parse --short=12 HEAD
    OUTPUT_VARIABLE SMALLFIELD_GIT_SHA_VALUE
    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET
)
if (NOT SMALLFIELD_GIT_SHA_VALUE)
    set(SMALLFIELD_GIT_SHA_VALUE "unknown")
endif ()

# Both binaries emit through the same writer, so it is compiled once rather than once per binary.
add_library(smallfield_parquet_writer OBJECT parquet_writer.cpp)
target_link_libraries(
    smallfield_parquet_writer PUBLIC smallfield_settings smallfield_lib Arrow::arrow_shared Parquet::parquet_shared
)
target_compile_definitions(smallfield_parquet_writer PRIVATE SMALLFIELD_GIT_SHA="${SMALLFIELD_GIT_SHA_VALUE}")

define_smallfield_target(
    smallfield_generate
    SOURCES generate.cpp parquet_reader.cpp shard_inventory.cpp rdkit_fallback.cpp
    LINK smallfield_lib smallfield_parquet_writer Arrow::arrow_shared Parquet::parquet_shared argparse::argparse
    DEFINES SMALLFIELD_GIT_SHA="${SMALLFIELD_GIT_SHA_VALUE}"
)

# Descriptor augmentation rewrites existing *.3D.parquet shards through the same writer.  CPU-only.
define_smallfield_target(
    smallfield_describe
    SOURCES describe.cpp
    LINK smallfield_lib smallfield_parquet_writer Arrow::arrow_shared Parquet::parquet_shared argparse::argparse
    DEFINES SMALLFIELD_GIT_SHA="${SMALLFIELD_GIT_SHA_VALUE}"
)

message(STATUS "SmallField batch binary: ENABLED (git ${SMALLFIELD_GIT_SHA_VALUE})")
