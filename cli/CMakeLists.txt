# SmallField batch binary: a restartable Parquet folder of SMILES in, a Parquet folder of 3D conformer coordinates,
# energies and per-conformer status out.  Three-tier fallback: GPU, then RDKit, then NULL.

# conda-forge ships the Arrow and Parquet config files under the RDKit prefix, which the root already prepended to
# CMAKE_PREFIX_PATH -- subdirectories inherit it.
find_package(Arrow CONFIG REQUIRED)
find_package(Parquet CONFIG REQUIRED)

# The producer git SHA is stamped into every shard's key-value metadata, so a shard stays self-describing in a
# content-addressed store.  Reading it at configure time alone would go stale on the next commit and report a revision
# that did not produce the file, so HEAD is registered as a configure dependency and moving it re-runs cmake.
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
    smallfield_batch
    SOURCES smallfield_batch.cpp parquet_reader.cpp shard_inventory.cpp rdkit_fallback.cpp
    LINK smallfield_lib smallfield_parquet_writer Arrow::arrow_shared Parquet::parquet_shared argparse::argparse
    DEFINES SMALLFIELD_GIT_SHA="${SMALLFIELD_GIT_SHA_VALUE}"
)

# Descriptor augmentation rewrites existing *.3D.parquet shards into the canonical 13-column schema by computing
# USR/USRCAT from the stored fp16 coordinates and re-emitting through the same writer, so its output is byte-identical
# to the batch producer's.  CPU-only.
define_smallfield_target(
    smallfield_descriptors
    SOURCES smallfield_descriptors.cpp
    LINK smallfield_lib smallfield_parquet_writer Arrow::arrow_shared Parquet::parquet_shared argparse::argparse
    DEFINES SMALLFIELD_GIT_SHA="${SMALLFIELD_GIT_SHA_VALUE}"
)

message(STATUS "SmallField batch binary: ENABLED (git ${SMALLFIELD_GIT_SHA_VALUE})")
