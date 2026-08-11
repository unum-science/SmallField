/**
 *  @file cli/parquet_writer.cpp
 *  @brief Packs per-molecule and per-conformer output arrays into Arrow builders and writes them
 *  as a single ZSTD-compressed Parquet shard matching the schema declared in `parquet_writer.hpp`.
 */

#include <cstddef> // `std::size_t`
#include <cstring> // `std::memset`, `std::memcpy`

#include <algorithm>   // `std::min`, `std::stable_sort`
#include <array>       // `std::array`
#include <memory>      // `std::shared_ptr`, `std::make_shared`
#include <string>      // `std::string`, `std::to_string`
#include <string_view> // `std::string_view`
#include <utility>     // `std::move`, `std::pair`
#include <vector>      // `std::vector`

#include <arrow/array/array_nested.h>
#include <arrow/array/builder_binary.h>
#include <arrow/array/builder_nested.h>
#include <arrow/array/builder_primitive.h>
#include <arrow/array/data.h>
#include <arrow/array/util.h>
#include <arrow/buffer.h>
#include <arrow/io/file.h>
#include <arrow/table.h>
#include <arrow/type.h>
#include <arrow/util/float16.h>
#include <arrow/util/key_value_metadata.h>
#include <fmt/std.h> // formats `std::filesystem::path`
#include <parquet/arrow/writer.h>

#include "smallfield/api/conformer_layout.hpp"
#include "smallfield/types/views.hpp"

#include "cli/parquet_writer.hpp"

namespace fs = std::filesystem;

namespace smallfield::cli {

#pragma region Refusals

char const *describe_shard_write(shard_write_t status) {
    switch (status) {
    case shard_write_t::written_k: return "written";
    case shard_write_t::missing_per_molecule_arrays_k: return "was handed no per-molecule arrays";
    case shard_write_t::input_span_too_short_k: return "was handed a short input span";
    case shard_write_t::row_sidecar_too_short_k: return "derived more rows than its per-row sidecars describe";
    case shard_write_t::molecule_index_out_of_range_k: return "reached past the end of the per-molecule arrays";
    case shard_write_t::coordinate_slot_out_of_range_k: return "reached a conformer slot past the coordinate span";
    case shard_write_t::buffer_allocation_failed_k: return "could not allocate a column buffer";
    case shard_write_t::column_reserve_failed_k: return "could not reserve a column for every row";
    case shard_write_t::column_append_failed_k: return "could not append to a column";
    case shard_write_t::column_finish_failed_k: return "could not finish a column";
    case shard_write_t::output_open_failed_k: return "could not open the output file";
    case shard_write_t::parquet_write_failed_k: return "was refused by the Parquet writer";
    case shard_write_t::output_close_failed_k: return "could not close the output file";
    }
    return "was refused";
}

std::string describe_shard_write(shard_write_report_t const &report) {
    char const *const reason = describe_shard_write(report.status);
    if (report.status == shard_write_t::input_span_too_short_k)
        return fmt::format("{}: {} holds {} of the {} elements required", reason, report.column,
                           fmt::group_digits(report.held), fmt::group_digits(report.required));
    if (!report.column.empty()) return fmt::format("{}: {}", reason, report.column);
    return std::string {reason};
}

/** The refusal naming an input span whose length broke the writer's size contract. */
static shard_write_report_t short_input_span_(std::string_view column, std::size_t held, std::size_t required) {
    return {.status = shard_write_t::input_span_too_short_k, .column = column, .held = held, .required = required};
}

/**
 *  @brief What one output row describes, which is what decides whether its per-conformer columns carry data.
 *
 *  The two failures null the same columns today and are still kept apart, because they are not the same fact:
 *  a molecule failure is the synthetic row a molecule with no surviving conformers is still given, so its
 *  slot and its `conformer_index` are zero rather than an energy rank.
 */
enum class row_outcome_t : std::uint8_t {
    /** The conformer survived, so its coordinates, energy and descriptors are written. */
    conformer_success_k,
    /** This conformer slot failed, so its coordinate, energy and descriptor columns are null. */
    conformer_failure_k,
    /** Preprocessing rejected the molecule, so it contributes one row with every per-conformer column null. */
    molecule_failure_k,
};

#pragma endregion Refusals

/**
 *  @brief Pack fp16 coordinates from a strided GPU output buffer into a tight Arrow `List<float16>` array,
 *  one element per output row.
 *
 *  Input layout: `coordinates` is a strided fp16 buffer of `num_molecules * max_conformers * stride_halves`
 *  halves, where `stride_halves = max_atoms * 3`. For a `molecule_index` and a `conformer_slot`, the coordinate
 *  block starts at offset `(molecule_index * max_conformers + conformer_slot) * stride_halves`.
 *
 *  Output layout: one row per surviving conformer, walked in molecule-major then conformer-index order,
 *  copying only the first `num_atoms_per_molecule[molecule_index] * 3` halves per row. Post-Kabsch,
 *  `n_conformers_per_molecule[molecule_index]` gives the number of slots [0..surviving_conformers) to emit
 *  for that molecule. Failed molecules (surviving_conformers = 0) emit one null row; conformers whose
 *  status is non-success emit null lists.
 *
 *  @param[in] row_outcomes Per-output-row outcome; only a success row carries three halves per atom, and
 *      every other row is null. Length = num_output_rows.
 *  @param[in] molecule_index_per_row Maps output row index back to its source molecule.
 *  @param[in] conformer_slot_per_row Maps output row index to which K-slot it came from.
 *  @param[out] array A nullable `List<float16>` array with one element per output row, untouched on refusal.
 *  @return `written_k`, or the refusal that stopped the pack.
 */
static shard_write_report_t pack_coordinates_as_list(custd::span<__half const> coordinates,
                                                     custd::span<std::uint8_t const> num_atoms_per_molecule,
                                                     custd::span<row_outcome_t const> row_outcomes,
                                                     custd::span<std::uint64_t const> molecule_index_per_row,
                                                     custd::span<std::uint32_t const> conformer_slot_per_row,
                                                     std::uint32_t max_conformers, std::uint32_t stride_halves,
                                                     std::shared_ptr<arrow::Array> &array) {

    std::int64_t const num_rows = static_cast<std::int64_t>(row_outcomes.size());
    smallfield::api::conformer_slot_layout_t const slot_layout {stride_halves / 3u, max_conformers};

    // Caller (write_shard) is responsible for the size invariants on
    // num_atoms_per_molecule and coordinates; the parallel index sidecars
    // (molecule_index_per_row, conformer_slot_per_row) are populated locally
    // so their lengths match num_rows by construction. The two range checks
    // below catch the case where the caller passed mismatched per-row sidecars
    // (e.g. a future refactor) and bail before OOB.
    if (molecule_index_per_row.size() < static_cast<std::size_t>(num_rows) ||
        conformer_slot_per_row.size() < static_cast<std::size_t>(num_rows))
        return {.status = shard_write_t::row_sidecar_too_short_k, .column = "conformer_coords"};
    std::int64_t total_halves = 0;
    for (std::int64_t row = 0; row < num_rows; ++row) {
        if (row_outcomes[row] != row_outcome_t::conformer_success_k) continue;
        std::uint64_t const molecule_index = molecule_index_per_row[row];
        if (molecule_index >= num_atoms_per_molecule.size()) // caller-contract violation
            return {.status = shard_write_t::molecule_index_out_of_range_k, .column = "conformer_coords"};
        // Clamp exactly as the emit loop below does, so the allocation and the
        // offsets stay in agreement when a caller reports more atoms than fit.
        std::uint32_t atom_count = static_cast<std::uint32_t>(num_atoms_per_molecule[molecule_index]);
        if (atom_count > slot_layout.max_atoms) atom_count = slot_layout.max_atoms;
        total_halves += static_cast<std::int64_t>(atom_count) * 3;
    }

    shard_write_report_t const allocation_failed {.status = shard_write_t::buffer_allocation_failed_k,
                                                  .column = "conformer_coords"};
    auto offsets_buffer_result = arrow::AllocateBuffer((num_rows + 1) * sizeof(std::int32_t));
    if (!offsets_buffer_result.ok()) return allocation_failed;
    auto offsets_buffer = std::move(offsets_buffer_result).ValueOrDie();
    std::int32_t *offsets = reinterpret_cast<std::int32_t *>(offsets_buffer->mutable_data());

    auto values_buffer_result = arrow::AllocateBuffer(total_halves * sizeof(std::uint16_t));
    if (!values_buffer_result.ok()) return allocation_failed;
    auto values_buffer = std::move(values_buffer_result).ValueOrDie();
    std::uint16_t *values = reinterpret_cast<std::uint16_t *>(values_buffer->mutable_data());

    // Validity bitmap: bit `row` = 1 if the row has data, 0 if null.
    std::int64_t const bitmap_bytes = (num_rows + 7) / 8;
    auto validity_buffer_result = arrow::AllocateBuffer(bitmap_bytes);
    if (!validity_buffer_result.ok()) return allocation_failed;
    auto validity_buffer = std::move(validity_buffer_result).ValueOrDie();
    std::memset(validity_buffer->mutable_data(), 0, bitmap_bytes);
    std::uint8_t *validity = validity_buffer->mutable_data();

    std::int32_t running = 0;
    std::int64_t null_count = 0;
    for (std::int64_t row = 0; row < num_rows; ++row) {
        offsets[row] = running;
        if (row_outcomes[row] == row_outcome_t::conformer_success_k) {
            std::uint64_t const molecule_index = molecule_index_per_row[row];
            std::uint32_t const conformer_slot = conformer_slot_per_row[row];
            std::uint32_t const max_atoms = slot_layout.max_atoms;
            // Defence against a contract-violating caller: per-mol num_atoms must
            // fit in the stride. Clamp rather than OOB; the slot's coords beyond
            // stride_halves don't exist in `coordinates` regardless.
            std::uint32_t atom_count = static_cast<std::uint32_t>(num_atoms_per_molecule[molecule_index]);
            if (atom_count > max_atoms) atom_count = max_atoms;
            if (atom_count > 0) {
                std::size_t const source_offset = slot_layout.slot_offset(molecule_index, conformer_slot);
                if (source_offset + stride_halves > coordinates.size())
                    return {.status = shard_write_t::coordinate_slot_out_of_range_k, .column = "conformer_coords"};
                std::uint16_t const *source = reinterpret_cast<std::uint16_t const *>(coordinates.data()) +
                                              source_offset;
                // Transpose the structure-of-arrays slot into the tight interleaved
                // `(atom_count, 3)` block a Parquet row exposes. See
                // `conformer_slot_layout_t` for why a contiguous copy is not valid here.
                api::gather_slot_into_interleaved(source, slot_layout, atom_count, values + running);
                running += static_cast<std::int32_t>(atom_count * 3u);
            }
            validity[row >> 3] |= static_cast<std::uint8_t>(1u << (row & 7));
        }
        else { ++null_count; }
    }
    offsets[num_rows] = running;

    auto leaf_data = arrow::ArrayData::Make(arrow::float16(), total_halves, {nullptr, std::move(values_buffer)},
                                            /*null_count=*/0);
    auto leaf_array = arrow::MakeArray(std::move(leaf_data));

    auto list_data = arrow::ArrayData::Make(arrow::list(arrow::float16()), num_rows,
                                            {std::move(validity_buffer), std::move(offsets_buffer)},
                                            {leaf_array->data()}, null_count);
    array = arrow::MakeArray(std::move(list_data));
    return {.status = shard_write_t::written_k};
}

/**
 *  @brief Pack a per-slot USRCAT descriptor (60 doubles per slot) into a nullable `List<float16>`, one
 *  element per output row, downcasting each value to half precision on write. A row emits its `stride`
 *  values only when it is a success row and `values` is non-empty; otherwise it is null. Slot addressing
 *  matches the coord packer: `flat = molecule_index * max_conformers + conformer_slot`, values live at
 *  `[flat*stride, flat*stride+stride)`. fp16 is honest precision here: USRCAT's value range is small and
 *  it is derived from fp16 coordinates, so storing float64 would be illusory precision.
 *
 *  @param[out] array A nullable `List<float16>` array with one element per output row, untouched on refusal.
 *  @return `written_k`, or the refusal that stopped the pack.
 */
static shard_write_report_t pack_usrcat_as_fp16_list(custd::span<double const> values, std::uint32_t stride,
                                                     custd::span<row_outcome_t const> row_outcomes,
                                                     custd::span<std::uint64_t const> molecule_index_per_row,
                                                     custd::span<std::uint32_t const> conformer_slot_per_row,
                                                     std::uint32_t max_conformers,
                                                     std::shared_ptr<arrow::Array> &array) {

    auto value_builder = std::make_shared<arrow::HalfFloatBuilder>();
    arrow::ListBuilder list_builder(arrow::default_memory_pool(), value_builder);
    std::int64_t const num_rows = static_cast<std::int64_t>(row_outcomes.size());
    bool const have_values = !values.empty();
    shard_write_report_t const append_failed {.status = shard_write_t::column_append_failed_k, .column = "usrcat"};

    for (std::int64_t row = 0; row < num_rows; ++row) {
        if (have_values && row_outcomes[row] == row_outcome_t::conformer_success_k) {
            std::uint64_t const molecule_index = molecule_index_per_row[row];
            std::uint32_t const conformer_slot = conformer_slot_per_row[row];
            std::size_t const base = (static_cast<std::size_t>(molecule_index) * max_conformers + conformer_slot) *
                                     stride;
            if (base + stride > values.size()) {
                if (!list_builder.AppendNull().ok()) return append_failed;
                continue;
            }
            if (!list_builder.Append().ok()) return append_failed;
            for (std::uint32_t component = 0; component < stride; ++component) {
                __half const half_value = __float2half(static_cast<float>(values[base + component]));
                std::uint16_t bits;
                std::memcpy(&bits, &half_value, sizeof(bits));
                if (!value_builder->Append(bits).ok()) return append_failed;
            }
        }
        else {
            if (!list_builder.AppendNull().ok()) return append_failed;
        }
    }

    std::shared_ptr<arrow::Array> finished;
    if (!list_builder.Finish(&finished).ok())
        return {.status = shard_write_t::column_finish_failed_k, .column = "usrcat"};
    array = std::move(finished);
    return {.status = shard_write_t::written_k};
}

shard_write_report_t write_shard(fs::path const &path, writer_inputs_t const &inputs, logger_t const *log) {
    std::uint64_t const num_molecules = inputs.num_molecules;
    std::uint32_t const max_conformers = inputs.max_conformers;
    std::uint32_t const coord_stride_halves = inputs.max_atoms * 3u;
    logger_t const fallback;
    logger_t const &logger = log != nullptr ? *log : fallback;

    /* Every refusal below leaves through here, so the reason reaches the operator's log whatever the caller
     * does with the report it gets back. @p detail carries whatever the failing layer said for itself —
     * Arrow's own message, or the shape the size contract was checked against. */
    auto refuse_ = [&](shard_write_report_t const &report, std::string_view detail = {}) {
        if (detail.empty()) logger.error("{}: {}", path, describe_shard_write(report));
        else logger.error("{}: {}: {}", path, describe_shard_write(report), detail);
        return report;
    };

    if (inputs.num_atoms_per_molecule.empty() || inputs.n_conformers_per_molecule.empty())
        return refuse_({.status = shard_write_t::missing_per_molecule_arrays_k});

    // Every code path below indexes these spans
    // unguarded; if a caller violates the contract, a single early failure
    // here is preferable to OOB memory reads inside Arrow builders or the
    // coordinate packer.
    std::size_t const total_slots = static_cast<std::size_t>(num_molecules) * max_conformers;
    auto fail_size_ = [&](std::string_view column, std::size_t held, std::size_t required) {
        return refuse_(short_input_span_(column, held, required),
                       fmt::format("num_molecules={}, max_conformers={}, max_atoms={}",
                                   fmt::group_digits(num_molecules), max_conformers, inputs.max_atoms));
    };
    if (inputs.num_atoms_per_molecule.size() < num_molecules)
        return fail_size_("num_atoms_per_molecule", inputs.num_atoms_per_molecule.size(), num_molecules);
    if (inputs.n_conformers_per_molecule.size() < num_molecules)
        return fail_size_("n_conformers_per_molecule", inputs.n_conformers_per_molecule.size(), num_molecules);
    if (inputs.smiles.size() < num_molecules) return fail_size_("smiles", inputs.smiles.size(), num_molecules);
    if (inputs.status.size() < total_slots) return fail_size_("status", inputs.status.size(), total_slots);
    if (inputs.energies.size() < total_slots) return fail_size_("energies", inputs.energies.size(), total_slots);
    if (inputs.coordinates.size() < total_slots * coord_stride_halves)
        return fail_size_("coordinates", inputs.coordinates.size(), total_slots * coord_stride_halves);
    // Optional per-molecule columns: callers may pass an empty span to opt out;
    // otherwise the size must cover every molecule.
    if (!inputs.num_heavy_atoms_per_molecule.empty() && inputs.num_heavy_atoms_per_molecule.size() < num_molecules)
        return fail_size_("num_heavy_atoms_per_molecule", inputs.num_heavy_atoms_per_molecule.size(), num_molecules);
    if (!inputs.num_bonds_per_molecule.empty() && inputs.num_bonds_per_molecule.size() < num_molecules)
        return fail_size_("num_bonds_per_molecule", inputs.num_bonds_per_molecule.size(), num_molecules);
    if (!inputs.molecular_weight_per_molecule.empty() && inputs.molecular_weight_per_molecule.size() < num_molecules)
        return fail_size_("molecular_weight_per_molecule", inputs.molecular_weight_per_molecule.size(), num_molecules);

    // Determine total output rows. For each molecule, surviving_conformers =
    // n_conformers_per_molecule[molecule_index] surviving conformers (post-Kabsch). surviving_conformers = 0
    // indicates the molecule failed preprocess entirely; the writer still emits one row with all-null per-mol
    // fields so every input row has at least one output row.
    std::uint64_t total_rows = 0;
    for (std::uint64_t molecule_index = 0; molecule_index < num_molecules; ++molecule_index) {
        std::uint8_t const surviving_conformers = inputs.n_conformers_per_molecule[molecule_index];
        total_rows += (surviving_conformers == 0) ? 1u : static_cast<std::uint64_t>(surviving_conformers);
    }

    // Build per-output-row index sidecars used by the coord packer + the row-loop.
    std::vector<row_outcome_t> row_outcomes;
    row_outcomes.reserve(total_rows);
    std::vector<std::uint64_t> molecule_index_per_row;
    molecule_index_per_row.reserve(total_rows);
    std::vector<std::uint32_t> conformer_slot_per_row;
    conformer_slot_per_row.reserve(total_rows);
    // conformer_index emitted per row is the energy rank within the molecule (0 = lowest energy),
    // not the original K-slot; rows within a molecule are energy-sorted below.
    std::vector<std::uint8_t> conformer_index_per_row;
    conformer_index_per_row.reserve(total_rows);

    // K is tiny (3 in production); a fixed stack buffer avoids a per-molecule heap allocation on
    // the hot path. Molecules with more slots than this fall back to unsorted emission.
    constexpr std::uint32_t max_sortable_conformers_k = 64;

    for (std::uint64_t molecule_index = 0; molecule_index < num_molecules; ++molecule_index) {
        std::uint8_t const surviving_conformers = inputs.n_conformers_per_molecule[molecule_index];
        if (surviving_conformers == 0) {
            row_outcomes.push_back(row_outcome_t::molecule_failure_k);
            molecule_index_per_row.push_back(molecule_index);
            conformer_slot_per_row.push_back(0);
            conformer_index_per_row.push_back(0);
            continue;
        }

        // Energy-sort this molecule's slots: successful conformers first (ascending energy), failed
        // slots (null energy) trail in slot order. conformer_index then reads as an energy rank.
        auto slot_key = [&](std::uint32_t conformer_slot) -> std::pair<int, float> {
            std::size_t const flat = static_cast<std::size_t>(molecule_index) * max_conformers + conformer_slot;
            std::uint8_t const status_code = (flat < inputs.status.size()) ? inputs.status[flat] : 1;
            float const energy = (flat < inputs.energies.size()) ? inputs.energies[flat] : 0.0f;
            return {status_code == 0 ? 0 : 1, energy};
        };
        std::array<std::uint32_t, max_sortable_conformers_k> order {};
        std::uint32_t const sortable = std::min<std::uint32_t>(surviving_conformers, max_sortable_conformers_k);
        for (std::uint32_t conformer_slot = 0; conformer_slot < sortable; ++conformer_slot)
            order[conformer_slot] = conformer_slot;
        std::stable_sort(order.begin(), order.begin() + sortable,
                         [&](std::uint32_t left, std::uint32_t right) { return slot_key(left) < slot_key(right); });

        for (std::uint32_t rank = 0; rank < surviving_conformers; ++rank) {
            std::uint32_t const conformer_slot = rank < sortable ? order[rank] : rank; // overflow: unsorted tail
            std::size_t const flat = static_cast<std::size_t>(molecule_index) * max_conformers + conformer_slot;
            std::uint8_t const status_code = (flat < inputs.status.size()) ? inputs.status[flat] : 0;
            row_outcomes.push_back(status_code == 0 ? row_outcome_t::conformer_success_k
                                                    : row_outcome_t::conformer_failure_k);
            molecule_index_per_row.push_back(molecule_index);
            conformer_slot_per_row.push_back(conformer_slot);
            conformer_index_per_row.push_back(static_cast<std::uint8_t>(rank));
        }
    }

    // Allocate Arrow builders.
    arrow::StringBuilder input_shard_builder;
    arrow::UInt64Builder input_row_builder;
    arrow::StringBuilder smiles_builder;
    arrow::UInt8Builder conformer_index_builder;
    arrow::UInt8Builder status_builder;
    arrow::UInt16Builder n_heavy_atoms_builder;
    arrow::UInt16Builder n_atoms_builder;
    arrow::UInt16Builder n_bonds_builder;
    arrow::FloatBuilder molecular_weight_builder;
    arrow::UInt8Builder n_conformers_builder;
    arrow::FloatBuilder conformer_energy_builder;

    // Keep the first column that cannot reserve and check once after the pass, so the refusal names the
    // column rather than the eleven call sites collapsing into one anonymous answer.
    shard_write_report_t reserve_report;
    auto reserve_column_ = [&](std::string_view column, auto &builder) {
        arrow::Status const status = builder.Reserve(static_cast<std::int64_t>(total_rows));
        if (status.ok() || reserve_report.status != shard_write_t::written_k) return;
        reserve_report = refuse_({.status = shard_write_t::column_reserve_failed_k, .column = column},
                                 status.message());
    };
    reserve_column_("input_shard", input_shard_builder);
    reserve_column_("input_row", input_row_builder);
    reserve_column_("smiles", smiles_builder);
    reserve_column_("conformer_index", conformer_index_builder);
    reserve_column_("status", status_builder);
    reserve_column_("n_heavy_atoms", n_heavy_atoms_builder);
    reserve_column_("n_atoms", n_atoms_builder);
    reserve_column_("n_bonds", n_bonds_builder);
    reserve_column_("molecular_weight", molecular_weight_builder);
    reserve_column_("n_conformers", n_conformers_builder);
    reserve_column_("conformer_energy", conformer_energy_builder);
    if (reserve_report.status != shard_write_t::written_k) return reserve_report;

    // Helper: bool indicating whether a molecule's preprocess succeeded (n_atoms > 0).
    auto molecule_ok = [&](std::uint64_t molecule_index) -> bool {
        return inputs.num_atoms_per_molecule[molecule_index] != 0;
    };

    // Every builder above is reserved for `total_rows`, so the fixed-width appends below cannot
    // fail on capacity. `smiles_builder` is the exception: `Reserve` sizes the offset array, not
    // the character data, so its append can still hit an allocation failure and truncate the shard
    // silently. Keep the first failure and check once after the loop rather than branching per row.
    arrow::Status append_status;
    std::string_view append_column;
    auto record_append_ = [&](std::string_view column, arrow::Status status) {
        if (!append_status.ok() || status.ok()) return;
        append_status = std::move(status);
        append_column = column;
    };

    for (std::uint64_t row = 0; row < total_rows; ++row) {
        std::uint64_t const molecule_index = molecule_index_per_row[row];
        std::uint32_t const conformer_slot = conformer_slot_per_row[row];
        row_outcome_t const outcome = row_outcomes[row];

        std::uint64_t const input_row = inputs.base_row_index + molecule_index;
        std::string_view const smiles = molecule_index < inputs.smiles.size() ? inputs.smiles[molecule_index] : "";
        std::size_t const flat = static_cast<std::size_t>(molecule_index) * max_conformers + conformer_slot;
        std::uint8_t const status_code = (flat < inputs.status.size()) ? inputs.status[flat] : 0;

        record_append_("input_shard", input_shard_builder.Append(inputs.input_shard_stem));
        record_append_("input_row", input_row_builder.Append(input_row));
        record_append_("smiles", smiles_builder.Append(smiles.data(), static_cast<std::int32_t>(smiles.size())));
        record_append_("conformer_index", conformer_index_builder.Append(conformer_index_per_row[row]));
        record_append_("status", status_builder.Append(status_code));

        if (molecule_ok(molecule_index)) {
            record_append_("n_heavy_atoms",
                           n_heavy_atoms_builder.Append(inputs.num_heavy_atoms_per_molecule.empty()
                                                            ? 0u
                                                            : inputs.num_heavy_atoms_per_molecule[molecule_index]));
            record_append_("n_atoms", n_atoms_builder.Append(
                                          static_cast<std::uint16_t>(inputs.num_atoms_per_molecule[molecule_index])));
            record_append_("n_bonds", n_bonds_builder.Append(inputs.num_bonds_per_molecule.empty()
                                                                 ? 0u
                                                                 : inputs.num_bonds_per_molecule[molecule_index]));
            record_append_("molecular_weight",
                           molecular_weight_builder.Append(inputs.molecular_weight_per_molecule.empty()
                                                               ? 0.0f
                                                               : inputs.molecular_weight_per_molecule[molecule_index]));
            record_append_("n_conformers",
                           n_conformers_builder.Append(inputs.n_conformers_per_molecule[molecule_index]));
        }
        else {
            record_append_("n_heavy_atoms", n_heavy_atoms_builder.AppendNull());
            record_append_("n_atoms", n_atoms_builder.AppendNull());
            record_append_("n_bonds", n_bonds_builder.AppendNull());
            record_append_("molecular_weight", molecular_weight_builder.AppendNull());
            record_append_("n_conformers", n_conformers_builder.AppendNull());
        }

        if (outcome == row_outcome_t::conformer_success_k) {
            record_append_("conformer_energy", conformer_energy_builder.Append(inputs.energies[flat]));
        }
        else { record_append_("conformer_energy", conformer_energy_builder.AppendNull()); }
    }

    // A failed append leaves the column arrays inconsistent in length, which Parquet would happily
    // write as a truncated shard. Refuse instead.
    if (!append_status.ok())
        return refuse_({.status = shard_write_t::column_append_failed_k, .column = append_column},
                       append_status.message());

    std::shared_ptr<arrow::Array> input_shard_array, input_row_array, smiles_array, conformer_index_array;
    std::shared_ptr<arrow::Array> status_array, n_heavy_atoms_array, n_atoms_array, n_bonds_array;
    std::shared_ptr<arrow::Array> molecular_weight_array, n_conformers_array, conformer_energy_array;
    // Same shape as the reserve pass: the first column that cannot finish names itself, and one check
    // after the pass carries it out.
    shard_write_report_t finish_report;
    auto finish_column_ = [&](std::string_view column, auto &builder, std::shared_ptr<arrow::Array> &array) {
        arrow::Status const status = builder.Finish(&array);
        if (status.ok() || finish_report.status != shard_write_t::written_k) return;
        finish_report = refuse_({.status = shard_write_t::column_finish_failed_k, .column = column}, status.message());
    };
    finish_column_("input_shard", input_shard_builder, input_shard_array);
    finish_column_("input_row", input_row_builder, input_row_array);
    finish_column_("smiles", smiles_builder, smiles_array);
    finish_column_("conformer_index", conformer_index_builder, conformer_index_array);
    finish_column_("status", status_builder, status_array);
    finish_column_("n_heavy_atoms", n_heavy_atoms_builder, n_heavy_atoms_array);
    finish_column_("n_atoms", n_atoms_builder, n_atoms_array);
    finish_column_("n_bonds", n_bonds_builder, n_bonds_array);
    finish_column_("molecular_weight", molecular_weight_builder, molecular_weight_array);
    finish_column_("n_conformers", n_conformers_builder, n_conformers_array);
    finish_column_("conformer_energy", conformer_energy_builder, conformer_energy_array);
    if (finish_report.status != shard_write_t::written_k) return finish_report;

    std::shared_ptr<arrow::Array> coord_array;
    if (shard_write_report_t const report = pack_coordinates_as_list(
            inputs.coordinates, inputs.num_atoms_per_molecule, row_outcomes, molecule_index_per_row,
            conformer_slot_per_row, max_conformers, coord_stride_halves, coord_array);
        report.status != shard_write_t::written_k)
        return refuse_(report);

    std::shared_ptr<arrow::Array> usrcat_array;
    if (shard_write_report_t const report = pack_usrcat_as_fp16_list(inputs.usrcat, 60u, row_outcomes,
                                                                     molecule_index_per_row, conformer_slot_per_row,
                                                                     max_conformers, usrcat_array);
        report.status != shard_write_t::written_k)
        return refuse_(report);

    // Provenance: schema-level key-value metadata so the bytes are self-describing (a
    // content-addressed store keeps no path context). SMALLFIELD_GIT_SHA is injected by CMake.
    auto provenance = std::make_shared<arrow::KeyValueMetadata>();
    provenance->Append("producer", "smallfield_batch");
#ifdef SMALLFIELD_GIT_SHA
    provenance->Append("producer_git_sha", SMALLFIELD_GIT_SHA);
#endif
    provenance->Append("num_conformers", std::to_string(max_conformers));
    provenance->Append("max_atoms", std::to_string(inputs.max_atoms));
    if (!inputs.dataset.empty()) provenance->Append("dataset", inputs.dataset);
    // Coordinates are centroid-centered but orientation is not canonicalized: the metric embedding aligns to
    // principal axes, but the subsequent FIRE/MMFF relaxation drifts the pose off that frame. Report the
    // frame actually shipped. A true canonical principal-axis pose, a final reframe with a fixed sign
    // convention, is a possible future enhancement.
    provenance->Append("coordinate_frame", "centered");
    provenance->Append("schema_version", "2");
    provenance->Append("dedup", "none");

    auto schema = arrow::schema(
        {
            arrow::field("input_shard", arrow::utf8(), false),
            arrow::field("input_row", arrow::uint64(), false),
            arrow::field("smiles", arrow::utf8(), false),
            arrow::field("conformer_index", arrow::uint8(), false),
            arrow::field("status", arrow::uint8(), false),
            arrow::field("n_heavy_atoms", arrow::uint16(), true),
            arrow::field("n_atoms", arrow::uint16(), true),
            arrow::field("n_bonds", arrow::uint16(), true),
            arrow::field("molecular_weight", arrow::float32(), true),
            arrow::field("n_conformers", arrow::uint8(), true),
            arrow::field("conformer_coords", arrow::list(arrow::float16()), true),
            arrow::field("conformer_energy", arrow::float32(), true),
            arrow::field("usrcat", arrow::list(arrow::float16()), true),
        },
        provenance);

    auto table = arrow::Table::Make(schema, {
                                                input_shard_array,
                                                input_row_array,
                                                smiles_array,
                                                conformer_index_array,
                                                status_array,
                                                n_heavy_atoms_array,
                                                n_atoms_array,
                                                n_bonds_array,
                                                molecular_weight_array,
                                                n_conformers_array,
                                                coord_array,
                                                conformer_energy_array,
                                                usrcat_array,
                                            });

    auto outfile_result = arrow::io::FileOutputStream::Open(path.string());
    if (!outfile_result.ok())
        return refuse_({.status = shard_write_t::output_open_failed_k}, outfile_result.status().message());
    auto outfile = outfile_result.ValueOrDie();

    auto writer_props =
        parquet::WriterProperties::Builder().compression(parquet::Compression::ZSTD)->compression_level(3)->build();
    auto arrow_props = parquet::ArrowWriterProperties::Builder().store_schema()->build();
    auto write_status = parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), outfile, 65536, writer_props,
                                                   arrow_props);
    if (!write_status.ok()) return refuse_({.status = shard_write_t::parquet_write_failed_k}, write_status.message());

    auto close_status = outfile->Close();
    if (!close_status.ok()) return refuse_({.status = shard_write_t::output_close_failed_k}, close_status.message());
    return {.status = shard_write_t::written_k};
}

} // namespace smallfield::cli
