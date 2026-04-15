/**
 *  @file parquet_writer.hpp
 *  @brief Write a single conformer Parquet output shard.
 *
 *  Schema (one row per (input_row, conformer_index)):
 *      input_shard         string      (file stem of the input Parquet)
 *      input_row           uint64      (0-indexed row in the input shard)
 *      conformer_index     uint8
 *      status              uint8       (conformer_status_t)
 *      solver              uint8       (0 = gpu, 1 = rdkit, 2 = failed)
 *      num_atoms           uint8       (nullable; unknown on invalid SMILES)
 *      energy_kcal_per_mol float32     (nullable when solver == 2)
 *      coordinates_xyz     fixed_size_list<float16, 384>  (128·3)
 *                                       NOT nullable; zero-filled when
 *                                       solver == 2.  Consumers detect
 *                                       invalid rows via `solver == 2` or
 *                                       `energy_kcal_per_mol IS NULL`.
 */

#pragma once

#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

#include <cuda_fp16.h>

namespace smallfield::batch {

struct writer_inputs_t {
    std::string input_shard_stem;            ///< e.g. "usearch_part_00042"
    std::span<std::uint64_t const> input_rows; ///< length N
    std::uint32_t num_conformers;
    std::uint32_t max_atoms;                 ///< must be 128 for the current schema
    std::span<std::uint8_t const> num_atoms_per_molecule;    ///< length N (or empty)
    std::span<std::uint8_t const> status;    ///< length N × num_conformers
    std::span<std::uint8_t const> solver;    ///< length N × num_conformers (0/1/2)
    std::span<__half const> coordinates;     ///< length N × num_conformers × max_atoms × 3
    std::span<float const> energies;         ///< length N × num_conformers
};

bool write_shard(std::filesystem::path const &path, writer_inputs_t const &inputs);

} // namespace smallfield::batch
