/**
 *  @file cli/parquet_writer.hpp
 *  @brief Declares `write_shard`, which emits one `*.3D.parquet` shard: one row per conformer, with the
 *  per-molecule columns repeated across a molecule's rows.
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
 *  | usrcat               | List<float16>    | yes      | row   | 60-dim USRCAT shape; null if undescribed    |
 *
 *  The on-disk names keep the `n_` spelling every shard this project has produced already uses; the C++ members
 *  feeding them are spelled out. Beside the columns, the schema's key-value metadata carries `producer`,
 *  `producer_git_sha`, `num_conformers`, `max_atoms`, `dataset`, `coordinate_frame`, `schema_version` and
 *  `dedup`. Preprocessing keeps only a molecule's largest fragment, so every geometry-derived column describes
 *  that parent while `smiles` is the input as given and may name a salt: re-parsing it through
 *  `smallfield::parse_smiles` reproduces the parent, and computing a property without that step will not agree.
 */

#pragma once

#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint64_t`, `std::uint32_t`

#include <filesystem> // `std::filesystem::path`
#include <string>      // `std::string`
#include <string_view> // `std::string_view`

#include <cuda_fp16.h>

#include "smallfield/api/conformer_layout.hpp"
#include "smallfield/log.hpp"
#include "smallfield/types/conformer_status.hpp"
#include "smallfield/types/views.hpp"

namespace smallfield::cli {

/**
 *  Where a shard's rows came from, recorded per row and in the schema metadata. An empty `dataset` omits the
 *  key rather than writing a blank one, and `base_row_index` is the row the first molecule occupied in the
 *  input shard, which `input_row` counts up from.
 */
struct shard_provenance_t {
    std::string input_shard_stem;
    std::string dataset;
    std::uint64_t base_row_index = 0;
};

/**
 *  @brief Whether a molecule carries a shape descriptor at all, which no value of one can say for itself.
 *
 *  A producer that cannot rebuild a molecule's shape context leaves its slots' `usrcat` blocks unwritten, and an
 *  unwritten block is zero-filled — a descriptor a single-atom molecule genuinely has. So the absence travels
 *  beside the numbers rather than inside them.
 */
enum class shape_described_t : std::uint8_t {
    described_k,
    undescribed_k,
};

/**
 *  @brief The molecule axis: one element per molecule, in one order, across every non-empty column.
 *
 *  `num_heavy_atoms`, `num_bonds`, `molecular_weight` and `shape_described` are optional: an empty span opts the
 *  column out, and any other length must cover every molecule. An axis empty in `smiles`, `num_atoms` and
 *  `num_conformers` alike is a shard of no molecules, written with zero rows rather than refused. A zero in
 *  `num_atoms` means preprocessing failed, which nulls every per-molecule column; `num_conformers` is also the
 *  rows a molecule contributes, and one at zero still contributes one; `undescribed_k` writes `usrcat` null on
 *  every row a molecule owns.
 */
struct molecule_columns_t {
    custd::span<std::string_view const> smiles;
    custd::span<std::uint8_t const> num_atoms;
    custd::span<std::uint8_t const> num_conformers;
    custd::span<std::uint16_t const> num_heavy_atoms;
    custd::span<std::uint16_t const> num_bonds;
    custd::span<float const> molecular_weight;
    custd::span<shape_described_t const> shape_described;

    /** The extent every column above shares; `write_shard` refuses a sibling shorter than it. */
    [[nodiscard]] std::uint64_t count() const noexcept { return num_atoms.size(); }
};

/** Components a USRCAT vector carries; its first 12 are the USR descriptor, not stored separately. A literal
 *  rather than `postprocess::usrcat_components_k`, so the on-disk schema costs no RDKit include; the two are
 *  held equal by a `static_assert` in `parquet_writer.cpp`, which sees both. */
inline constexpr std::uint32_t usrcat_components_k = 60;

/**
 *  @brief The conformer-slot axis: one slot per molecule per conformer, addressed through @p layout.
 *
 *  The writer reads only the first `num_conformers[molecule]` slots of each molecule, so trailing slots go
 *  unused. `coordinates` is structure-of-arrays, filled by either the GPU or the RDKit fallback:
 *  `layout.slot_offset(molecule, slot)` names a slot's base and `gather_slot_into_interleaved` reads it into an
 *  atom-major row, while a contiguous `3 * num_atoms` read from that base is not a valid gather. `usrcat` holds
 *  `usrcat_components_k` doubles per slot, downcast on write to the float16 its coordinates already carry, so a
 *  wider column would be illusory precision; an empty span opts it out as all-null, and a non-success slot or
 *  an undescribed molecule is written null either way.
 */
struct conformer_slot_columns_t {
    api::conformer_slot_layout_t layout {};
    custd::span<types::conformer_status_t const> status;
    custd::span<float const> energies;
    custd::span<__half const> coordinates;
    custd::span<double const> usrcat;

    /** Slots @p num_molecules molecules span, which is what every column above is sized against. */
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
 *  `missing_per_molecule_arrays_k` is the one whose name does not carry it: the layout holds no slots, or
 *  `num_atoms` is empty while a sibling molecule column is not, so the molecule axis is missing rather than
 *  empty, and an axis empty throughout is written out with zero rows. `list_offsets_overflowed_k` is more
 *  packed fp16 halves in one list column than an Arrow `List` 32-bit offset addresses.
 */
enum class shard_write_t {
    written_k,
    missing_per_molecule_arrays_k,
    input_span_too_short_k,
    list_offsets_overflowed_k,
    buffer_allocation_failed_k,
    column_reserve_failed_k,
    column_append_failed_k,
    column_finish_failed_k,
    output_open_failed_k,
    parquet_write_failed_k,
    output_close_failed_k,
};

/** Sentence fragment naming the refusal, meant to follow the shard path in a diagnostic. */
char const *describe_shard_write(shard_write_t status);

/**
 *  One outcome of `write_shard`. `column` names the schema column or input span the refusal is about, empty
 *  where none; `held` and `required` are the size contract `input_span_too_short_k` or
 *  `list_offsets_overflowed_k` broke -- a span against its axis, or packed halves against an offset's reach --
 *  and stay zero under every other refusal.
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
 *  @brief Write @p inputs to @p path as one ZSTD-compressed Parquet shard, closed before this returns.
 *
 *  Every refusal is logged as it happens, so a caller testing only the status still leaves a named reason in
 *  the operator's log. A null @p log falls back to a default one.
 */
[[nodiscard]] shard_write_report_t write_shard(std::filesystem::path const &path, writer_inputs_t const &inputs,
                                               logger_t const *log = nullptr);

} // namespace smallfield::cli
