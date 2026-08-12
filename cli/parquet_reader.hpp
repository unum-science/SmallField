/**
 *  @file cli/parquet_reader.hpp
 *  @brief Declares `read_shard`, which loads a Parquet shard into zero-copy SMILES views.
 *
 *  Keeps the Arrow table alive so SMILES can be accessed as `string_view` directly into Arrow's contiguous
 *  buffer, with no per-string heap allocation.
 */

#pragma once

#include <filesystem>  // `std::filesystem::path`
#include <string>      // `std::string`
#include <string_view> // `std::string_view`
#include <vector>      // `std::vector`

#include <arrow/table.h>

#include "smallfield/log.hpp"

namespace smallfield::cli {

/**
 *  @brief Zero-copy SMILES views backed by an Arrow table; the table must outlive the views.
 *
 *  The row count is the extent of `smiles` rather than a field beside it, so a reader that appended a view
 *  and forgot the counter cannot exist.
 */
struct shard_data_t {
    std::shared_ptr<arrow::Table> table;
    std::vector<std::string_view> smiles;
};

/**
 *  @brief Why an input shard was read, or which of five distinguishable refusals stopped it.
 *
 *  A `bool` folded a misspelled `--smiles-column` into the same answer as an unreadable file, so an operator
 *  whose column is named `canonical_smiles` and one whose disk is failing both read `read failed, skipping`.
 */
enum class input_read_t {
    /** The table materialised and every SMILES was sliced out of it. */
    read_k,
    /** The path did not open. */
    cannot_open_k,
    /** The file opened but no Parquet reader could be built over it. */
    not_parquet_k,
    /** No column carries the name the caller asked for. */
    column_missing_k,
    /** The Parquet reader could not materialise the table. */
    unreadable_table_k,
    /** The named column exists but is not UTF-8, so its values are not SMILES. */
    column_not_string_k,
};

/** Sentence fragment naming the refusal, meant to follow the shard's path in a diagnostic. */
char const *describe_input_read(input_read_t status);

/**
 *  @brief Read @p path 's @p smiles_column into @p out as views over the table it keeps alive.
 *
 *  @param[out] out Caller-owned destination, cleared first; holds the Arrow table behind its views.
 *  @param[in] log Logger read failures are reported through; null falls back to a default one.
 *  @return `read_k`, or the refusal that stopped the read.
 */
[[nodiscard]] input_read_t read_shard(std::filesystem::path const &path, std::string const &smiles_column,
                                      shard_data_t &out, logger_t const *log = nullptr);

} // namespace smallfield::cli
