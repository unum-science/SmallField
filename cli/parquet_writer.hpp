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
 *  The column names keep the `n_` spelling because they are what is already on disk in every shard this
 *  project has produced; the C++ members feeding them are spelled out like everything else here.
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

#include <filesystem> // `std::filesystem::path`
#include <string>      // `std::string`
#include <string_view> // `std::string_view`

#include <cuda_fp16.h>

#include "smallfield/api/conformer_layout.hpp"
#include "smallfield/api/conformer_status.hpp"
#include "smallfield/log.hpp"
#include "smallfield/types/views.hpp"

namespace smallfield::cli {

/** Where a shard's rows came from, recorded per row and in the schema metadata. */
struct shard_provenance_t {
    /** Stem of the input shard, written verbatim into the `input_shard` column. */
    std::string input_shard_stem;
    /** Dataset name for provenance (e.g. "pubchem", "gdb13"); "" = omit. */
    std::string dataset;
    /** Row the first molecule occupied in the input shard; `input_row` counts up from here. */
    std::uint64_t base_row_index = 0;
};

/**
 *  @brief The molecule axis: one element per molecule, in one order, across every non-empty column.
 *
 *  `count()` is that shared extent, read from `num_atoms` rather than carried beside the spans, so no
 *  caller states the molecule count a second time and no writer has to keep the two agreeing. `write_shard`
 *  refuses any sibling column shorter than `count()`.
 *
 *  A zero in `num_atoms` means preprocessing failed, which the writer translates into a null on the output
 *  column. `num_atoms` is a `uint8` in the public API, capped at `max_atoms_per_molecule_k`, and widens to
 *  `uint16` in the canonical schema so the format stays forward-compatible.
 *
 *  `num_heavy_atoms`, `num_bonds` and `molecular_weight` are optional: an empty span opts the column out and
 *  writes it as all-null, and any other length must cover every molecule.
 */
struct molecule_columns_t {
    custd::span<std::string_view const> smiles;
    custd::span<std::uint8_t const> num_atoms;
    /** M, post-Kabsch (0 if preprocess failed); also the rows this molecule contributes. */
    custd::span<std::uint8_t const> num_conformers;
    custd::span<std::uint16_t const> num_heavy_atoms;
    custd::span<std::uint16_t const> num_bonds;
    custd::span<float const> molecular_weight;

    /** Molecules the shard describes, which is the extent every column above shares. */
    [[nodiscard]] std::uint64_t count() const noexcept { return num_atoms.size(); }
};

/** Components a USRCAT vector carries; its first 12 are the USR descriptor, not stored separately. */
inline constexpr std::uint32_t usrcat_components_k = 60;

/**
 *  @brief The conformer-slot axis: one slot per molecule per conformer, addressed through @p layout.
 *
 *  The layout carries both the slot stride and the conformer count, so the writer never re-derives one from
 *  the other — dividing a coordinate stride by three to recover an atom cap is exactly the agreement this
 *  type exists to remove. The writer reads only the first `num_conformers[molecule]` slots of each molecule,
 *  so the trailing slots are unused.
 */
struct conformer_slot_columns_t {
    api::conformer_slot_layout_t layout {};
    /** The producers' own status type, narrowed to the schema's `uint8` at the one place it is written. */
    custd::span<api::conformer_status_t const> status;
    custd::span<float const> energies;
    /**
     *  @brief Structure-of-arrays coordinates, filled by either the GPU or the RDKit fallback.
     *
     *  `layout.slot_offset(molecule, slot)` names a slot's base and `layout.component_offset` a component
     *  inside it; a contiguous `3 * num_atoms` read from the slot base is not a valid gather.
     */
    custd::span<__half const> coordinates;
    /**
     *  @brief `usrcat_components_k` doubles per slot, stored as `list<float16>`.
     *
     *  Slot-addressed exactly like `energies`, at a stride of `usrcat_components_k` rather than one. Supplied
     *  as float64, the precision it is computed at, and downcast to float16 on write. An empty span opts the
     *  column out, marking it all-null; non-success slots are written null either way.
     */
    custd::span<double const> usrcat;

    /** Slots a molecule count of @p num_molecules spans, which is what every column above is sized against. */
    [[nodiscard]] std::size_t slot_count(std::uint64_t num_molecules) const noexcept {
        return static_cast<std::size_t>(num_molecules) * layout.num_conformers;
    }
};

/** Everything one shard is written from: where its rows came from, and its two axes. */
struct writer_inputs_t {
    shard_provenance_t provenance;
    molecule_columns_t molecules;
    conformer_slot_columns_t conformers;
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
    /** `num_atoms` or `num_conformers` is empty, or the layout carries no slots, so no row is describable. */
    missing_per_molecule_arrays_k,
    /** An input span is shorter than the molecule count or the slot count requires. */
    input_span_too_short_k,
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
