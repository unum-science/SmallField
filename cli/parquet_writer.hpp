/**
 *  @file cli/parquet_writer.hpp
 *  @brief Declares `write_shard` and the canonical schema of a per-conformer-row 3D Parquet
 *  shard ("*.3D.parquet").
 *
 *  Output schema (canonical, 13 fields):
 *
 *  | Field                | Type             | Nullable | Per   | Notes                                       |
 *  |----------------------|------------------|----------|-------|---------------------------------------------|
 *  | input_shard          | utf8             | no       | row   | source parquet stem                         |
 *  | input_row            | uint64           | no       | row   | 0-based row in source                       |
 *  | smiles               | utf8             | no       | row   | the input SMILES, salt form included        |
 *  | conformer_index      | uint8            | no       | row   | 0..M-1, energy rank (0 = lowest energy)     |
 *  | status               | uint8            | no       | row   | conformer_status_t enum                     |
 *  | n_heavy_atoms        | uint16           | yes      | mol   | null if mol failed preprocess               |
 *  | n_atoms              | uint16           | yes      | mol   | atoms incl. H; null on preprocess failure   |
 *  | n_bonds              | uint16           | yes      | mol   | bonds incl. H; null on preprocess failure   |
 *  | molecular_weight     | float32          | yes      | mol   | exact MW in Daltons; null on failure        |
 *  | n_conformers         | uint8            | yes      | mol   | K (no dedup); null on preprocess failure    |
 *  | conformer_coords     | List<float16>    | yes      | row   | 3*n_atoms halves; null if status != success |
 *  | conformer_energy     | float32          | yes      | row   | this row's MMFF94 energy; null on failure   |
 *  | usrcat               | List<float16>    | yes      | row   | 60-dim USRCAT shape; null if !success       |
 *
 *  USRCAT's first 12 values @b are the USR descriptor, so USR is not stored separately. USRCAT is fp16 (its
 *  value range is small and it is derived from fp16 coordinates, so higher precision would be illusory).
 *
 *  Preprocessing keeps only a molecule's largest fragment, so every column derived from the geometry —
 *  n_atoms, n_heavy_atoms, n_bonds, molecular_weight, conformer_coords, usrcat — describes that parent
 *  fragment, while `smiles` is the input as given and may name a salt. Re-parsing `smiles` through
 *  `smallfield::parse_smiles` reproduces the parent, which is what keeps the two consistent; computing a
 *  property from `smiles` without that step will not agree.
 *
 *  Rows within a molecule are emitted energy-sorted (lowest first); conformer_index is the energy rank.
 *  Schema-level key-value metadata records provenance (producer git sha, num_conformers, max_atoms, dataset,
 *  coordinate frame, schema version).
 */

#pragma once

#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint64_t`, `std::uint32_t`
#include <cstring> // `std::memcpy`

#include <filesystem>  // `std::filesystem::path`
#include <span>        // `std::span`
#include <string>      // `std::string`
#include <string_view> // `std::string_view`

#include <cuda_fp16.h>

#include "smallfield/api/conformer_layout.hpp"
#include "smallfield/log.hpp"

namespace smallfield::cli {

#pragma region Coordinate Slot Transposition

/* Move one coordinate component across the boundary the two layouts meet at: a numeric conversion
 * where one side is a host `float`, and a bit copy where both sides already hold half-precision
 * patterns. The overload set is what lets one transposition body serve every caller. */
inline void assign_coordinate_(__half &destination, float source) noexcept { destination = __float2half(source); }
inline void assign_coordinate_(float &destination, __half source) noexcept { destination = __half2float(source); }
inline void assign_coordinate_(__half &destination, std::uint16_t source) noexcept {
    std::memcpy(&destination, &source, sizeof(destination));
}
inline void assign_coordinate_(std::uint16_t &destination, std::uint16_t source) noexcept { destination = source; }

/**
 *  @brief Read one conformer slot's structure-of-arrays components into a tight interleaved `(atom, 3)` block.
 *
 *  @param[in] slot Base of the slot inside the coordinate pool, `pool.data() + layout.slot_offset(...)`.
 *  @param[in] layout Addressing contract the producers and the writer all agree on.
 *  @param[in] num_atoms Atoms the molecule actually has; the slot's padding atoms are not read.
 *  @param[out] interleaved Destination for `3 * num_atoms` components, atom-major.
 */
template <typename slot_type_, typename interleaved_type_>
inline void gather_slot_into_interleaved(slot_type_ const *slot, api::conformer_slot_layout_t const &layout,
                                         std::uint32_t num_atoms, interleaved_type_ *interleaved) noexcept {
    for (std::uint32_t atom = 0; atom < num_atoms; ++atom)
        for (std::uint32_t axis = 0; axis < 3u; ++axis)
            assign_coordinate_(interleaved[atom * 3u + axis], slot[layout.component_offset(axis, atom)]);
}

/**
 *  @brief Write a tight interleaved `(atom, 3)` block back into one conformer slot.
 *
 *  @param[in] interleaved Source of `3 * num_atoms` components, atom-major.
 *  @param[in] layout Addressing contract the producers and the writer all agree on.
 *  @param[in] num_atoms Atoms the molecule actually has; the slot's padding atoms are left alone.
 *  @param[out] slot Base of the slot inside the coordinate pool, `pool.data() + layout.slot_offset(...)`.
 */
template <typename interleaved_type_, typename slot_type_>
inline void scatter_interleaved_into_slot(interleaved_type_ const *interleaved,
                                          api::conformer_slot_layout_t const &layout, std::uint32_t num_atoms,
                                          slot_type_ *slot) noexcept {
    for (std::uint32_t atom = 0; atom < num_atoms; ++atom)
        for (std::uint32_t axis = 0; axis < 3u; ++axis)
            assign_coordinate_(slot[layout.component_offset(axis, atom)], interleaved[atom * 3u + axis]);
}

#pragma endregion Coordinate Slot Transposition

struct writer_inputs_t {
    std::string input_shard_stem;
    /** Dataset name for provenance (e.g. "pubchem", "gdb13"); "" = omit. */
    std::string dataset;
    std::uint64_t base_row_index = 0;
    std::uint64_t num_molecules = 0;
    /** K, the slot stride per molecule in the input arrays. */
    std::uint32_t max_conformers = 0;
    /** Per-conformer coord stride = max_atoms * 3. */
    std::uint32_t max_atoms = 0;
    std::span<std::string_view const> smiles;

    /**
     *  @brief Per-molecule scalar readbacks, each of length `num_molecules`.
     *
     *  A zero means preprocessing failed, which the writer translates into a null on the output column.
     *  `n_atoms` is a `uint8` in the public API, capped at 128, and widens to `uint16` in the canonical
     *  schema so the format stays forward-compatible.
     */
    std::span<std::uint8_t const> num_atoms_per_molecule;
    std::span<std::uint16_t const> num_heavy_atoms_per_molecule;
    std::span<std::uint16_t const> num_bonds_per_molecule;
    std::span<float const> molecular_weight_per_molecule;
    /** M, post-Kabsch (0 if preprocess failed). */
    std::span<std::uint8_t const> n_conformers_per_molecule;

    /**
     *  @brief Per-conformer slot arrays, each of length `num_molecules * max_conformers`.
     *
     *  The writer reads only the first `n_conformers_per_molecule[molecule_index]` slots of each molecule, so
     *  the trailing slots are unused.
     */
    std::span<std::uint8_t const> status;
    /**
     *  @brief Per-conformer coordinates.
     *
     *  Slot-addressed via `smallfield::api::conformer_slot_layout_t{max_atoms, max_conformers}`;
     *  structure-of-arrays within each slot, filled by either the GPU or the RDKit fallback.
     */
    std::span<__half const> coordinates;
    std::span<float const> energies;

    /**
     *  @brief Per-conformer USRCAT shape descriptor, 60 doubles per slot, stored as `list<float16>`.
     *
     *  With a stride of 60 rather than 1, this is slot-addressed exactly like `energies`: slot
     *  `flat = molecule_index * max_conformers + conformer_slot` occupies `usrcat[flat*60 .. flat*60+60)`.
     *  Supplied as float64, the precision it is computed at, and downcast to float16 on write. An empty span
     *  opts the column out, marking it all-null; non-success slots are written null. The 12-dim USR descriptor
     *  is not stored separately — it is exactly the first 12 values of USRCAT.
     */
    std::span<double const> usrcat;
};

#pragma region Refusals

/**
 *  @brief Why a shard was written, or which class of refusal stopped it.
 *
 *  The writer refuses in twenty-nine places, and a caller holding only a `bool` cannot tell a short input
 *  span from an allocation failure from a read-only output directory — all three read as `write failed`.
 *  Each enumerator names one class of refusal and `shard_write_report_t` carries the column and the sizes
 *  that go with it, so the operator-facing sentence names the thing that actually went wrong.
 */
enum class shard_write_t {
    /** The table was assembled, written, and the file closed. */
    written_k,
    /** `num_atoms_per_molecule` or `n_conformers_per_molecule` is empty, so no row can be described. */
    missing_per_molecule_arrays_k,
    /** An input span is shorter than `num_molecules` or `num_molecules * max_conformers` requires. */
    input_span_too_short_k,
    /** A per-row sidecar is shorter than the row count the writer derived from the inputs. */
    row_sidecar_too_short_k,
    /** A row names a molecule beyond the end of the per-molecule arrays. */
    molecule_index_out_of_range_k,
    /** A conformer slot's coordinates fall outside the `coordinates` span. */
    coordinate_slot_out_of_range_k,
    /** Arrow could not allocate one of the coordinate column's buffers. */
    buffer_allocation_failed_k,
    /** A column builder could not reserve room for every row. */
    column_reserve_failed_k,
    /** A column builder refused a value or a null, which would leave the columns unequal in length. */
    column_append_failed_k,
    /** A column builder could not hand back its finished array. */
    column_finish_failed_k,
    /** The output path could not be opened for writing. */
    output_open_failed_k,
    /** Parquet refused the assembled table. */
    parquet_write_failed_k,
    /** The output stream did not close cleanly, so the file on disk may be short. */
    output_close_failed_k,
};

/** Sentence fragment naming the refusal, meant to follow the shard path in a diagnostic. */
char const *describe_shard_write(shard_write_t status);

/**
 *  @brief One outcome of `write_shard`, with the column and the sizes the refusal names.
 *
 *  `column` is the schema column or input span the refusal is about and is empty where it names none.
 *  `held` and `required` carry the size contract `input_span_too_short_k` broke and stay zero elsewhere.
 */
struct shard_write_report_t {
    shard_write_t status = shard_write_t::written_k;
    std::string_view column {};
    std::size_t held = 0;
    std::size_t required = 0;
};

/** Full sentence for @p report, naming the column and the sizes where the refusal carries them. */
std::string describe_shard_write(shard_write_report_t const &report);

#pragma endregion Refusals

/**
 *  @brief Write @p inputs to @p path as one ZSTD-compressed Parquet shard in the canonical schema.
 *
 *  Every refusal is logged through @p log as it happens, so a caller that only tests the status still
 *  leaves a named reason in the operator's log.
 *
 *  @param[in] path Output shard, opened for writing and closed before this returns.
 *  @param[in] inputs Per-molecule and per-conformer arrays, and the provenance recorded in the metadata.
 *  @param[in] log Logger refusals are reported through; null falls back to a default one.
 *  @return `written_k` and no column on success, otherwise the refusal that stopped the write.
 */
[[nodiscard]] shard_write_report_t write_shard(std::filesystem::path const &path, writer_inputs_t const &inputs,
                                               logger_t const *log = nullptr);

} // namespace smallfield::cli
