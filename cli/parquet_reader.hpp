/**
 *  @file cli/parquet_reader.hpp
 *  @brief Declares `read_shard`, which loads a Parquet shard into zero-copy SMILES views.
 *
 *  Keeps the Arrow table alive so SMILES can be accessed as `string_view` directly into Arrow's contiguous
 *  buffer, with no per-string heap allocation.
 */

#pragma once

#include <cstdint> // `std::uint64_t`

#include <filesystem>  // `std::filesystem::path`
#include <string>      // `std::string`
#include <string_view> // `std::string_view`
#include <vector>      // `std::vector`

#include <arrow/table.h>

namespace smallfield::cli {

/** Zero-copy SMILES views backed by an Arrow table; the table must outlive the views. */
struct shard_data_t {
    std::shared_ptr<arrow::Table> table;
    std::vector<std::string_view> smiles;
    std::uint64_t base_row_index = 0;

    std::uint64_t num_rows() const noexcept { return smiles.size(); }
};

[[nodiscard]] bool read_shard(std::filesystem::path const &path, std::string const &smiles_column, shard_data_t &out);

} // namespace smallfield::cli
