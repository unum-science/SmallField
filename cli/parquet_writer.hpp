/**
 *  @file parquet_writer.hpp
 *  @brief Write a single conformer Parquet output shard.
 */

#pragma once

#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <cuda_fp16.h>

namespace smallfield::cli {

struct writer_inputs_t {
    std::string input_shard_stem;
    std::uint64_t base_row_index = 0;
    std::uint64_t num_molecules = 0;
    std::uint32_t num_conformers;
    std::uint32_t max_atoms;
    std::span<std::string_view const> smiles;
    std::span<std::uint8_t const> num_atoms_per_molecule;
    std::span<std::uint8_t const> status;
    std::span<std::uint8_t const> solver;
    std::span<__half const> coordinates;
    std::span<float const> energies;
};

[[nodiscard]] bool write_shard(std::filesystem::path const &path, writer_inputs_t const &inputs);

} // namespace smallfield::cli
