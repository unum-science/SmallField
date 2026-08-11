/**
 *  @file cli/parquet_writer.cpp
 *  @brief Packs per-molecule and per-conformer output arrays into Arrow builders and writes them
 *  as a single ZSTD-compressed Parquet shard matching the schema declared in `parquet_writer.hpp`.
 */

#include <cstdio>  // `fprintf`, `stderr`
#include <cstring> // `std::memset`, `std::memcpy`

#include <algorithm> // `std::min`, `std::stable_sort`
#include <array>     // `std::array`
#include <memory>    // `std::shared_ptr`, `std::make_shared`
#include <string>    // `std::string`, `std::to_string`
#include <utility>   // `std::move`, `std::pair`

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
#include <parquet/arrow/writer.h>

#include "smallfield/api/conformer_layout.hpp"

#include "cli/parquet_writer.hpp"

namespace fs = std::filesystem;

namespace smallfield::cli {

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
 *  @param[in] row_outcomes Per-output-row indicator: 0 = null row (no coords), 1 = success row
 *      (copy num_atoms × 3 halves), -1 = full-mol-failure null row (also empty list, but
 *      the column is null rather than empty). Length = num_output_rows.
 *  @param[in] molecule_index_per_row Maps output row index back to its source molecule.
 *  @param[in] conformer_slot_per_row Maps output row index to which K-slot it came from.
 *  @return A nullable `List<float16>` array with one element per output row.
 */
static std::shared_ptr<arrow::Array> pack_coordinates_as_list(
    std::span<__half const> coordinates, std::span<std::uint8_t const> num_atoms_per_molecule,
    std::span<std::int8_t const> row_outcomes, std::span<std::uint64_t const> molecule_index_per_row,
    std::span<std::uint32_t const> conformer_slot_per_row, std::uint32_t max_conformers, std::uint32_t stride_halves) {

    std::int64_t const num_rows = static_cast<std::int64_t>(row_outcomes.size());
    smallfield::api::conformer_slot_layout_t const slot_layout {stride_halves / 3u, max_conformers};

    // Caller (write_shard) is responsible for the size invariants on
    // num_atoms_per_molecule and coordinates; the parallel index sidecars
    // (mol_per_row, slot_per_row) are populated locally so their lengths
    // match num_rows by construction. The two range checks below catch
    // the case where the caller passed mismatched per-row sidecars (e.g. a
    // future refactor) and bail before OOB.
    if (molecule_index_per_row.size() < static_cast<std::size_t>(num_rows) ||
        conformer_slot_per_row.size() < static_cast<std::size_t>(num_rows))
        return nullptr;
    std::int64_t total_halves = 0;
    for (std::int64_t row = 0; row < num_rows; ++row) {
        if (row_outcomes[row] != 1) continue;
        std::uint64_t const molecule_index = molecule_index_per_row[row];
        if (molecule_index >= num_atoms_per_molecule.size()) return nullptr; // caller-contract violation
        // Clamp exactly as the emit loop below does, so the allocation and the
        // offsets stay in agreement when a caller reports more atoms than fit.
        std::uint32_t atom_count = static_cast<std::uint32_t>(num_atoms_per_molecule[molecule_index]);
        if (atom_count > slot_layout.max_atoms) atom_count = slot_layout.max_atoms;
        total_halves += static_cast<std::int64_t>(atom_count) * 3;
    }

    auto offsets_buffer_result = arrow::AllocateBuffer((num_rows + 1) * sizeof(std::int32_t));
    if (!offsets_buffer_result.ok()) return nullptr;
    auto offsets_buffer = std::move(offsets_buffer_result).ValueOrDie();
    std::int32_t *offsets = reinterpret_cast<std::int32_t *>(offsets_buffer->mutable_data());

    auto values_buffer_result = arrow::AllocateBuffer(total_halves * sizeof(std::uint16_t));
    if (!values_buffer_result.ok()) return nullptr;
    auto values_buffer = std::move(values_buffer_result).ValueOrDie();
    std::uint16_t *values = reinterpret_cast<std::uint16_t *>(values_buffer->mutable_data());

    // Validity bitmap: bit `row` = 1 if the row has data, 0 if null.
    std::int64_t const bitmap_bytes = (num_rows + 7) / 8;
    auto validity_buffer_result = arrow::AllocateBuffer(bitmap_bytes);
    if (!validity_buffer_result.ok()) return nullptr;
    auto validity_buffer = std::move(validity_buffer_result).ValueOrDie();
    std::memset(validity_buffer->mutable_data(), 0, bitmap_bytes);
    std::uint8_t *validity = validity_buffer->mutable_data();

    std::int32_t running = 0;
    std::int64_t null_count = 0;
    for (std::int64_t row = 0; row < num_rows; ++row) {
        offsets[row] = running;
        if (row_outcomes[row] == 1) {
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
                if (source_offset + stride_halves > coordinates.size()) return nullptr;
                std::uint16_t const *source = reinterpret_cast<std::uint16_t const *>(coordinates.data()) +
                                              source_offset;
                // Transpose the structure-of-arrays slot into the tight interleaved
                // `(atom_count, 3)` block a Parquet row exposes. See
                // `conformer_slot_layout_t` for why a contiguous copy is not valid here.
                for (std::uint32_t atom = 0; atom < atom_count; ++atom)
                    for (std::uint32_t axis = 0; axis < 3u; ++axis)
                        values[running + atom * 3u + axis] = source[slot_layout.component_offset(axis, atom)];
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
    return arrow::MakeArray(std::move(list_data));
}

/**
 *  @brief Pack a per-slot USRCAT descriptor (60 doubles per slot) into a nullable `List<float16>`, one
 *  element per output row, downcasting each value to half precision on write. A row emits its `stride`
 *  values only when it is a success row and `values` is non-empty; otherwise it is null. Slot addressing
 *  matches the coord packer: `flat = molecule_index * max_conformers + conformer_slot`, values live at
 *  `[flat*stride, flat*stride+stride)`. fp16 is honest precision here: USRCAT's value range is small and
 *  it is derived from fp16 coordinates, so storing float64 would be illusory precision.
 */
static std::shared_ptr<arrow::Array> pack_usrcat_as_fp16_list(std::span<double const> values, std::uint32_t stride,
                                                              std::span<std::int8_t const> row_outcomes,
                                                              std::span<std::uint64_t const> molecule_index_per_row,
                                                              std::span<std::uint32_t const> conformer_slot_per_row,
                                                              std::uint32_t max_conformers) {

    auto value_builder = std::make_shared<arrow::HalfFloatBuilder>();
    arrow::ListBuilder list_builder(arrow::default_memory_pool(), value_builder);
    std::int64_t const num_rows = static_cast<std::int64_t>(row_outcomes.size());
    bool const have_values = !values.empty();

    for (std::int64_t row = 0; row < num_rows; ++row) {
        if (have_values && row_outcomes[row] == 1) {
            std::uint64_t const molecule_index = molecule_index_per_row[row];
            std::uint32_t const conformer_slot = conformer_slot_per_row[row];
            std::size_t const base = (static_cast<std::size_t>(molecule_index) * max_conformers + conformer_slot) *
                                     stride;
            if (base + stride > values.size()) {
                if (!list_builder.AppendNull().ok()) return nullptr;
                continue;
            }
            if (!list_builder.Append().ok()) return nullptr;
            for (std::uint32_t component = 0; component < stride; ++component) {
                __half const half_value = __float2half(static_cast<float>(values[base + component]));
                std::uint16_t bits;
                std::memcpy(&bits, &half_value, sizeof(bits));
                if (!value_builder->Append(bits).ok()) return nullptr;
            }
        }
        else {
            if (!list_builder.AppendNull().ok()) return nullptr;
        }
    }

    std::shared_ptr<arrow::Array> out;
    if (!list_builder.Finish(&out).ok()) return nullptr;
    return out;
}

bool write_shard(fs::path const &path, writer_inputs_t const &inputs) {
    std::uint64_t const num_molecules = inputs.num_molecules;
    std::uint32_t const max_conformers = inputs.max_conformers;
    std::uint32_t const coord_stride_halves = inputs.max_atoms * 3u;

    if (inputs.num_atoms_per_molecule.empty() || inputs.n_conformers_per_molecule.empty()) {
        std::fprintf(stderr, "[smallfield_batch] write_shard: per-molecule arrays missing\n");
        return false;
    }

    // Every code path below indexes these spans
    // unguarded; if a caller violates the contract, a single early failure
    // here is preferable to OOB memory reads inside Arrow builders or the
    // coordinate packer.
    std::size_t const total_slots = static_cast<std::size_t>(num_molecules) * max_conformers;
    auto fail_size_ = [&](char const *what, std::size_t got, std::size_t want) {
        std::fprintf(stderr,                                                                            //
                     "[smallfield_batch] write_shard: %s size %zu < expected %zu (num_molecules=%llu, " //
                     "max_conformers=%u, max_atoms=%u)\n",                                              //
                     what, got, want, static_cast<unsigned long long>(num_molecules), max_conformers, inputs.max_atoms);
        return false;
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
    // row_outcomes[row] ∈ {1: success row, 0: per-conformer failure (status != success), -1: full
    // molecule failure (surviving_conformers == 0)}.
    std::vector<std::int8_t> row_outcomes;
    row_outcomes.reserve(total_rows);
    std::vector<std::uint64_t> mol_per_row;
    mol_per_row.reserve(total_rows);
    std::vector<std::uint32_t> slot_per_row;
    slot_per_row.reserve(total_rows);
    // conformer_index emitted per row is the energy rank within the molecule (0 = lowest energy),
    // not the original K-slot; rows within a molecule are energy-sorted below.
    std::vector<std::uint8_t> conf_index_per_row;
    conf_index_per_row.reserve(total_rows);

    // K is tiny (3 in production); a fixed stack buffer avoids a per-molecule heap allocation on
    // the hot path. Molecules with more slots than this fall back to unsorted emission.
    constexpr std::uint32_t max_sortable_conformers_k = 64;

    for (std::uint64_t molecule_index = 0; molecule_index < num_molecules; ++molecule_index) {
        std::uint8_t const surviving_conformers = inputs.n_conformers_per_molecule[molecule_index];
        if (surviving_conformers == 0) {
            row_outcomes.push_back(-1);
            mol_per_row.push_back(molecule_index);
            slot_per_row.push_back(0);
            conf_index_per_row.push_back(0);
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
            row_outcomes.push_back(status_code == 0 ? 1 : 0);
            mol_per_row.push_back(molecule_index);
            slot_per_row.push_back(conformer_slot);
            conf_index_per_row.push_back(static_cast<std::uint8_t>(rank));
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

    auto reserve_or_fail = [&](auto &builder) { return builder.Reserve(static_cast<std::int64_t>(total_rows)).ok(); };
    if (!reserve_or_fail(input_shard_builder)) return false;
    if (!reserve_or_fail(input_row_builder)) return false;
    if (!reserve_or_fail(smiles_builder)) return false;
    if (!reserve_or_fail(conformer_index_builder)) return false;
    if (!reserve_or_fail(status_builder)) return false;
    if (!reserve_or_fail(n_heavy_atoms_builder)) return false;
    if (!reserve_or_fail(n_atoms_builder)) return false;
    if (!reserve_or_fail(n_bonds_builder)) return false;
    if (!reserve_or_fail(molecular_weight_builder)) return false;
    if (!reserve_or_fail(n_conformers_builder)) return false;
    if (!reserve_or_fail(conformer_energy_builder)) return false;

    // Helper: bool indicating whether a molecule's preprocess succeeded (n_atoms > 0).
    auto molecule_ok = [&](std::uint64_t molecule_index) -> bool {
        return inputs.num_atoms_per_molecule[molecule_index] != 0;
    };

    // Every builder above is reserved for `total_rows`, so the fixed-width appends below cannot
    // fail on capacity. `smiles_builder` is the exception: `Reserve` sizes the offset array, not
    // the character data, so its append can still hit an allocation failure and truncate the shard
    // silently. Keep the first failure and check once after the loop rather than branching per row.
    arrow::Status append_status;
    auto record_append = [&append_status](arrow::Status status) {
        if (append_status.ok() && !status.ok()) append_status = std::move(status);
    };

    for (std::uint64_t row = 0; row < total_rows; ++row) {
        std::uint64_t const molecule_index = mol_per_row[row];
        std::uint32_t const conformer_slot = slot_per_row[row];
        std::int8_t const outcome = row_outcomes[row];

        std::uint64_t const input_row = inputs.base_row_index + molecule_index;
        std::string_view const smiles = molecule_index < inputs.smiles.size() ? inputs.smiles[molecule_index] : "";
        std::size_t const flat = static_cast<std::size_t>(molecule_index) * max_conformers + conformer_slot;
        std::uint8_t const status_code = (flat < inputs.status.size()) ? inputs.status[flat] : 0;

        record_append(input_shard_builder.Append(inputs.input_shard_stem));
        record_append(input_row_builder.Append(input_row));
        record_append(smiles_builder.Append(smiles.data(), static_cast<std::int32_t>(smiles.size())));
        record_append(conformer_index_builder.Append(conf_index_per_row[row]));
        record_append(status_builder.Append(status_code));

        if (molecule_ok(molecule_index)) {
            record_append(n_heavy_atoms_builder.Append(inputs.num_heavy_atoms_per_molecule.empty()
                                                           ? 0u
                                                           : inputs.num_heavy_atoms_per_molecule[molecule_index]));
            record_append(
                n_atoms_builder.Append(static_cast<std::uint16_t>(inputs.num_atoms_per_molecule[molecule_index])));
            record_append(n_bonds_builder.Append(
                inputs.num_bonds_per_molecule.empty() ? 0u : inputs.num_bonds_per_molecule[molecule_index]));
            record_append(molecular_weight_builder.Append(inputs.molecular_weight_per_molecule.empty()
                                                              ? 0.0f
                                                              : inputs.molecular_weight_per_molecule[molecule_index]));
            record_append(n_conformers_builder.Append(inputs.n_conformers_per_molecule[molecule_index]));
        }
        else {
            record_append(n_heavy_atoms_builder.AppendNull());
            record_append(n_atoms_builder.AppendNull());
            record_append(n_bonds_builder.AppendNull());
            record_append(molecular_weight_builder.AppendNull());
            record_append(n_conformers_builder.AppendNull());
        }

        if (outcome == 1) { record_append(conformer_energy_builder.Append(inputs.energies[flat])); }
        else { record_append(conformer_energy_builder.AppendNull()); }
    }

    // A failed append leaves the column arrays inconsistent in length, which Parquet would happily
    // write as a truncated shard. Refuse instead.
    if (!append_status.ok()) return false;

    std::shared_ptr<arrow::Array> input_shard_array, input_row_array, smiles_array, conformer_index_array;
    std::shared_ptr<arrow::Array> status_array, n_heavy_atoms_array, n_atoms_array, n_bonds_array;
    std::shared_ptr<arrow::Array> molecular_weight_array, n_conformers_array, conformer_energy_array;
    if (!input_shard_builder.Finish(&input_shard_array).ok()) return false;
    if (!input_row_builder.Finish(&input_row_array).ok()) return false;
    if (!smiles_builder.Finish(&smiles_array).ok()) return false;
    if (!conformer_index_builder.Finish(&conformer_index_array).ok()) return false;
    if (!status_builder.Finish(&status_array).ok()) return false;
    if (!n_heavy_atoms_builder.Finish(&n_heavy_atoms_array).ok()) return false;
    if (!n_atoms_builder.Finish(&n_atoms_array).ok()) return false;
    if (!n_bonds_builder.Finish(&n_bonds_array).ok()) return false;
    if (!molecular_weight_builder.Finish(&molecular_weight_array).ok()) return false;
    if (!n_conformers_builder.Finish(&n_conformers_array).ok()) return false;
    if (!conformer_energy_builder.Finish(&conformer_energy_array).ok()) return false;

    std::shared_ptr<arrow::Array> coord_array = pack_coordinates_as_list(
        inputs.coordinates, inputs.num_atoms_per_molecule, std::span<std::int8_t const>(row_outcomes),
        std::span<std::uint64_t const>(mol_per_row), std::span<std::uint32_t const>(slot_per_row), max_conformers,
        coord_stride_halves);
    if (!coord_array) {
        std::fprintf(stderr, "[smallfield_batch] coordinate buffer allocation failed\n");
        return false;
    }

    std::shared_ptr<arrow::Array> usrcat_array = pack_usrcat_as_fp16_list(
        inputs.usrcat, 60u, std::span<std::int8_t const>(row_outcomes), std::span<std::uint64_t const>(mol_per_row),
        std::span<std::uint32_t const>(slot_per_row), max_conformers);
    if (!usrcat_array) {
        std::fprintf(stderr, "[smallfield_batch] descriptor buffer build failed\n");
        return false;
    }

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
    if (!outfile_result.ok()) {
        std::fprintf(stderr, "[smallfield_batch] output open failed: %s: %s\n", path.string().c_str(),
                     outfile_result.status().message().c_str());
        return false;
    }
    auto outfile = outfile_result.ValueOrDie();

    auto writer_props =
        parquet::WriterProperties::Builder().compression(parquet::Compression::ZSTD)->compression_level(3)->build();
    auto arrow_props = parquet::ArrowWriterProperties::Builder().store_schema()->build();
    auto write_status = parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), outfile, 65536, writer_props,
                                                   arrow_props);
    if (!write_status.ok()) {
        std::fprintf(stderr, "[smallfield_batch] write failed: %s: %s\n", path.string().c_str(),
                     write_status.message().c_str());
        return false;
    }

    auto close_status = outfile->Close();
    return close_status.ok();
}

} // namespace smallfield::cli
