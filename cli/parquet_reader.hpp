/**
 *  @file parquet_reader.hpp
 *  @brief Read a single Parquet shard into a vector of SMILES strings.
 *
 *  Shard schema: any Parquet with a string column named by the user
 *  (default `smiles`).  The reader materialises the full column to
 *  std::string to decouple from Arrow chunk lifetime.
 */

#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace smallfield::cli {

struct shard_data_t {
    std::vector<std::string> smiles;
    std::vector<std::uint64_t> input_row_indices; ///< 0..num_rows-1 of the source shard
};

bool read_shard(std::filesystem::path const &path, std::string const &smiles_column, shard_data_t &out);

} // namespace smallfield::cli
