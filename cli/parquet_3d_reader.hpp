/**
 *  @file parquet_3d_reader.hpp
 *  @brief Read a SmallField conformer-output Parquet shard into device-ready buffers.
 *
 *  Companion to `parquet_reader.hpp`, which only handles the input-side
 *  SMILES column.  The output of `smallfield_batch` (see `parquet_writer.cpp`)
 *  is a parquet with these columns:
 *
 *    input_shard       : utf8
 *    input_row         : uint64
 *    smiles            : utf8                         <-- needed (atomic number lookup)
 *    conformer_index   : uint8
 *    status            : uint8                        <-- needed (skip non-success rows)
 *    solver            : uint8
 *    num_atoms         : uint8                        <-- needed (variable per row)
 *    energy_kcal_per_mol : float32                    <-- carried over
 *    coordinates_xyz   : list<float16>  (num_atoms × 3 halves) <-- needed (input + output)
 *
 *  This reader pulls those columns out and gives the caller flat host-side
 *  vectors plus per-row offsets so the standalone refiner can pack them
 *  into a `smallfield::mace::batch_view_t`.  It deliberately skips the
 *  zero-copy path used by `parquet_reader.cpp` for SMILES because the
 *  coordinate column is already a list-of-half (no flat slab to wrap).
 */

#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <arrow/table.h>
#include <cuda_fp16.h>

namespace smallfield::cli {

struct conformer_shard_data_t {
    /** @brief Backing Arrow table — keep alive while string_views are in use. */
    std::shared_ptr<arrow::Table> table;

    /** @brief Per-row SMILES (one entry per conformer; molecules with K conformers
     *         have K rows with the same SMILES). */
    std::vector<std::string> smiles;
    std::vector<std::uint8_t> num_atoms_per_row;
    std::vector<std::uint8_t> status_per_row;
    std::vector<std::uint8_t> solver_per_row;
    std::vector<std::uint8_t> conformer_index_per_row;

    /** @brief Per-row source-shard stem & input row (echoed through to writer). */
    std::vector<std::string> input_shard_per_row;
    std::vector<std::uint64_t> input_row_per_row;

    /** @brief Per-row baseline energy (kcal/mol from MMFF) — preserved alongside
     *         the new MACE energy in eV when we write back. */
    std::vector<float> baseline_energy_per_row;
    std::vector<bool> baseline_energy_null_per_row;

    /** @brief Flat list of coordinate halves with `[N_total × 3]` layout where
     *         `N_total = sum(num_atoms_per_row)`.  Use `row_atom_offsets[i]` to
     *         find row `i`'s slice. */
    std::vector<__half> coordinates_flat;
    std::vector<std::uint64_t> row_atom_offsets; ///< [num_rows + 1]; offsets in atoms (not halves)

    std::uint64_t num_rows() const noexcept { return smiles.size(); }
    std::uint64_t total_atoms() const noexcept { return row_atom_offsets.empty() ? 0 : row_atom_offsets.back(); }
};

[[nodiscard]] bool read_conformer_shard(std::filesystem::path const &path, conformer_shard_data_t &out);

} // namespace smallfield::cli
