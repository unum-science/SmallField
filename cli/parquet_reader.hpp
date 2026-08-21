/**
 *  @file cli/parquet_reader.hpp
 *  @brief Declares `read_shard`, which loads a Parquet shard into zero-copy SMILES views.
 */

#pragma once

#include <filesystem>  // `std::filesystem::path`
#include <string>      // `std::string`
#include <string_view> // `std::string_view`
#include <vector>      // `std::vector`

#include <arrow/table.h>

#include "smallfield/log.hpp"

namespace smallfield::cli {

/** Zero-copy SMILES views into `table`'s own buffers, so the table must outlive the views. */
struct shard_data_t {
    std::shared_ptr<arrow::Table> table;
    std::vector<std::string_view> smiles;
};

/** Why an input shard was read, or which of five distinguishable refusals stopped it. `column_not_string_k`
 *  means the named column exists but is not UTF-8, so its values are not SMILES. */
enum class input_read_t {
    read_k,
    cannot_open_k,
    not_parquet_k,
    column_missing_k,
    unreadable_table_k,
    column_not_string_k,
};

/** Sentence fragment naming the refusal, meant to follow the shard's path in a diagnostic. */
char const *describe_input_read(input_read_t status);

/** Read @p path 's @p smiles_column into @p out as views over the table it keeps alive. @p out is cleared
 *  first, and a null @p log falls back to a default one. */
[[nodiscard]] input_read_t read_shard(std::filesystem::path const &path, std::string const &smiles_column,
                                      shard_data_t &out, logger_t const *log = nullptr);

} // namespace smallfield::cli
