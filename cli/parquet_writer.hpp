/**
 *  @file parquet_writer.hpp
 *  @brief Write a single per-conformer-row 3D Parquet shard ("*.3D.parquet").
 *
 *  Output schema (canonical, 13 fields):
 *
 *  | Field                | Type             | Nullable | Per   | Notes                                      |
 *  |----------------------|------------------|----------|-------|--------------------------------------------|
 *  | input_shard          | utf8             | no       | row   | source parquet stem                        |
 *  | input_row            | uint64           | no       | row   | 0-based row in source                      |
 *  | smiles               | utf8             | no       | row   | the input SMILES, salt form included       |
 *  | conformer_index      | uint8            | no       | row   | 0..M-1, energy rank (0 = lowest energy)    |
 *  | status               | uint8            | no       | row   | conformer_status_t enum                    |
 *  | n_heavy_atoms        | uint16           | yes      | mol   | null if mol failed preprocess              |
 *  | n_atoms              | uint16           | yes      | mol   | atoms incl. H; null on preprocess failure  |
 *  | n_bonds              | uint16           | yes      | mol   | bonds incl. H; null on preprocess failure  |
 *  | molecular_weight     | float32          | yes      | mol   | exact MW in Daltons; null on failure       |
 *  | n_conformers         | uint8            | yes      | mol   | K (no dedup); null on preprocess failure   |
 *  | conformer_coords     | List<float16>    | yes      | row   | 3*n_atoms halves; null if status != success|
 *  | conformer_energy     | float32          | yes      | row   | this row's MMFF94 energy; null on failure  |
 *  | usrcat               | List<float16>    | yes      | row   | 60-dim USRCAT shape vector; null if !success|
 *
 *  USRCAT's first 12 values ARE the USR descriptor, so USR is not stored separately. USRCAT is
 *  fp16 (its value range is small and it is derived from fp16 coordinates, so higher precision
 *  would be illusory).
 *
 *  Preprocessing keeps only a molecule's largest fragment, so every column derived from the
 *  geometry — n_atoms, n_heavy_atoms, n_bonds, molecular_weight, conformer_coords, usrcat —
 *  describes that parent fragment, while `smiles` is the input as given and may name a salt.
 *  Re-parsing `smiles` through `smallfield::parse_smiles` reproduces the parent, which is what
 *  keeps the two consistent; computing a property from `smiles` without that step will not agree.
 *
 *  Rows within a molecule are emitted energy-sorted (lowest first); conformer_index
 *  is the energy rank. Schema-level key-value metadata records provenance (producer
 *  git sha, num_conformers, max_atoms, dataset, coordinate frame, schema version).
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
    std::string dataset; ///< dataset name for provenance (e.g. "pubchem", "gdb13"); "" = omit
    std::uint64_t base_row_index = 0;
    std::uint64_t num_molecules = 0;
    std::uint32_t max_conformers = 0; ///< K, the slot stride per molecule in the input arrays
    std::uint32_t max_atoms = 0;      ///< per-conformer coord stride = max_atoms * 3
    std::span<std::string_view const> smiles;

    // Per-molecule scalar readbacks (length == num_molecules). Zero values mean "preprocess failed";
    // the writer translates those into nulls on the output column. n_atoms is uint8 in the SmallField
    // public API (max 128) but widens to uint16 in the canonical schema for forward-compat.
    std::span<std::uint8_t const> num_atoms_per_molecule;
    std::span<std::uint16_t const> num_heavy_atoms_per_molecule;
    std::span<std::uint16_t const> num_bonds_per_molecule;
    std::span<float const> molecular_weight_per_molecule;
    std::span<std::uint8_t const> n_conformers_per_molecule; ///< M, post-Kabsch (0 if preprocess failed)

    // Per-conformer slot arrays (length == num_molecules * max_conformers). Writer only reads the
    // first `n_conformers_per_molecule[m]` slots per molecule — slots [M..K) are unused.
    std::span<std::uint8_t const> status;
    /// Slot-addressed via `smallfield::api::conformer_slot_layout_t{max_atoms, max_conformers}`;
    /// structure-of-arrays within each slot, filled by either the GPU or the RDKit fallback.
    std::span<__half const> coordinates;
    std::span<float const> energies;

    // Per-conformer USRCAT shape descriptor, slot-addressed like `energies` but with a stride of 60:
    // slot s = m*max_conformers + c occupies usrcat[s*60 .. s*60+60). Supplied as float64 (computed
    // precision) and downcast to float16 on write. Empty span opts the column out (all-null); non-
    // success slots are written null. The 12-dim USR descriptor is not stored separately — it is
    // exactly the first 12 values of USRCAT.
    std::span<double const> usrcat; ///< 60 doubles per slot; stored as list<float16>
};

[[nodiscard]] bool write_shard(std::filesystem::path const &path, writer_inputs_t const &inputs);

} // namespace smallfield::cli
