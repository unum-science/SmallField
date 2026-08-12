/**
 *  @file cli/parquet_writer.cpp
 *  @brief Packs per-molecule and per-conformer output arrays into Arrow builders and writes them
 *  as a single ZSTD-compressed Parquet shard matching the schema declared in `parquet_writer.hpp`.
 */

#include <cstddef> // `std::size_t`
#include <cstring> // `std::memset`, `std::memcpy`

#include <algorithm>   // `std::min`, `std::stable_sort`
#include <limits>      // `std::numeric_limits`
#include <memory>      // `std::shared_ptr`, `std::make_shared`
#include <string>      // `std::string`, `std::to_string`
#include <string_view> // `std::string_view`
#include <utility>     // `std::move`
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
#include "smallfield/api/conformer_status.hpp"
#include "smallfield/types/views.hpp"

#include "cli/parquet_writer.hpp"

namespace fs = std::filesystem;

namespace smallfield::cli {

#pragma region Refusals

char const *describe_shard_write(shard_write_t status) {
    switch (status) {
    case shard_write_t::written_k: return "written";
    case shard_write_t::missing_per_molecule_arrays_k: return "was handed nothing a row could describe";
    case shard_write_t::input_span_too_short_k: return "was handed a short input span";
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

/**
 *  @brief Log @p report against @p path and hand it back, so every refusal leaves a named reason behind.
 *
 *  @param[in] detail Whatever the failing layer said for itself, omitted when it carries nothing.
 */
static shard_write_report_t refuse_(logger_t const &logger, fs::path const &path, shard_write_report_t const &report,
                                    std::string_view detail = {}) {
    if (detail.empty()) logger.error("{}: {}", path, describe_shard_write(report));
    else logger.error("{}: {}: {}", path, describe_shard_write(report), detail);
    return report;
}

#pragma endregion Refusals

#pragma region Input Extents

/** One input span measured against the extent its axis requires; `required` of zero opts a column out. */
struct column_extent_t {
    std::string_view column;
    std::size_t held;
    std::size_t required;
};

/** The first entry of @p extents that is short, or a `written_k` report when every one covers its axis. */
static shard_write_report_t first_short_column_(std::initializer_list<column_extent_t> extents) {
    for (column_extent_t const &extent : extents)
        if (extent.held < extent.required)
            return {.status = shard_write_t::input_span_too_short_k,
                    .column = extent.column,
                    .held = extent.held,
                    .required = extent.required};
    return {.status = shard_write_t::written_k};
}

/** An optional column is either opted out with an empty span, or long enough to cover @p required. */
static std::size_t optional_required_(std::size_t held, std::size_t required) { return held == 0 ? 0 : required; }

/**
 *  @brief Check every input span against the extent its axis requires, before anything indexes one.
 *
 *  Once this passes no read below needs a bound of its own: every row names a molecule under
 *  `molecules.count()` and a slot under `layout.num_conformers`, so the widest offset any row reaches is
 *  inside the coordinate span checked here.
 */
static shard_write_report_t check_input_extents_(writer_inputs_t const &inputs) {
    molecule_columns_t const &molecules = inputs.molecules;
    conformer_slot_columns_t const &conformers = inputs.conformers;
    if (molecules.num_atoms.empty() || molecules.num_conformers.empty() || conformers.layout.num_conformers == 0)
        return {.status = shard_write_t::missing_per_molecule_arrays_k};

    std::uint64_t const num_molecules = molecules.count();
    std::size_t const num_slots = conformers.slot_count(num_molecules);
    return first_short_column_({
        {"num_conformers", molecules.num_conformers.size(), num_molecules},
        {"smiles", molecules.smiles.size(), num_molecules},
        {"num_heavy_atoms", molecules.num_heavy_atoms.size(),
         optional_required_(molecules.num_heavy_atoms.size(), num_molecules)},
        {"num_bonds", molecules.num_bonds.size(), optional_required_(molecules.num_bonds.size(), num_molecules)},
        {"molecular_weight", molecules.molecular_weight.size(),
         optional_required_(molecules.molecular_weight.size(), num_molecules)},
        {"status", conformers.status.size(), num_slots},
        {"energies", conformers.energies.size(), num_slots},
        {"coordinates", conformers.coordinates.size(), num_slots * conformers.layout.slot_stride()},
        {"usrcat", conformers.usrcat.size(),
         optional_required_(conformers.usrcat.size(), num_slots * usrcat_components_k)},
    });
}

/** The shape the size contract was checked against, quoted beside a short-span refusal. */
static std::string describe_extents_(writer_inputs_t const &inputs) {
    return fmt::format("num_molecules={}, max_conformers={}, max_atoms={}", fmt::group_digits(inputs.molecules.count()),
                       inputs.conformers.layout.num_conformers, inputs.conformers.layout.max_atoms);
}

#pragma endregion Input Extents

#pragma region Output Rows

/**
 *  @brief What one output row describes, which is what decides whether its per-conformer columns carry data.
 *
 *  The two failures null the same columns and stay distinct: a molecule failure is the synthetic row a
 *  molecule with no surviving conformers is still given, so its slot and `conformer_index` are zero rather
 *  than an energy rank.
 */
enum class row_outcome_t : std::uint8_t {
    /** The conformer survived, so its coordinates, energy and descriptors are written. */
    conformer_success_k,
    /** This conformer slot failed, so its coordinate, energy and descriptor columns are null. */
    conformer_failure_k,
    /** Preprocessing rejected the molecule, so it contributes one row with every per-conformer column null. */
    molecule_failure_k,
};

/**
 *  @brief One output row of the shard.
 *
 *  Every field is inside its axis by construction: `molecule_index` below `molecules.count()` because the
 *  builder loops it, and `conformer_slot` below `layout.num_conformers` because the builder clamps it.
 */
struct output_row_t {
    std::uint64_t molecule_index = 0;
    std::uint32_t conformer_slot = 0;
    /** Rank by ascending energy inside the molecule, which is what the `conformer_index` column publishes. */
    std::uint8_t energy_rank = 0;
    row_outcome_t outcome = row_outcome_t::molecule_failure_k;
};

/**
 *  @brief Sort key of one conformer slot: its energy, or infinity when the slot did not succeed.
 *
 *  Every failed slot compares equal, so a stable sort leaves the failures behind the successes @b in slot
 *  order — the key `python/smallfield/_binding.cpp` sorts by, which is what makes a shard and the Python
 *  API rank one molecule's conformers identically.
 */
static float slot_ordering_energy_(conformer_slot_columns_t const &conformers, std::size_t flat) {
    return conformers.status[flat] == api::conformer_status_t::success_k ? conformers.energies[flat]
                                                                         : std::numeric_limits<float>::infinity();
}

/**
 *  @brief Conformers of a molecule the shard can address, which is what the `n_conformers` column publishes.
 *
 *  An `n_conformers` above the slot stride is clamped rather than read, because slot `num_conformers` of
 *  one molecule is slot zero of the next. The column and the rows emitted are the same number by
 *  construction, because both come from here.
 */
static std::uint8_t addressable_conformers_(writer_inputs_t const &inputs, std::uint64_t molecule_index) {
    return static_cast<std::uint8_t>(std::min<std::uint32_t>(inputs.molecules.num_conformers[molecule_index],
                                                             inputs.conformers.layout.num_conformers));
}

/** Rows molecule @p molecule_index contributes; a molecule with no surviving conformer still gets one. */
static std::uint32_t rows_of_molecule_(writer_inputs_t const &inputs, std::uint64_t molecule_index) {
    std::uint8_t const addressable = addressable_conformers_(inputs, molecule_index);
    return addressable == 0 ? 1u : addressable;
}

/**
 *  @brief Describe every output row of the shard, energy-sorted inside each molecule.
 *
 *  @param[out] rows Caller-owned destination, cleared first.
 */
static void fill_output_rows_into(writer_inputs_t const &inputs, std::vector<output_row_t> &rows) {
    conformer_slot_columns_t const &conformers = inputs.conformers;
    std::uint64_t const num_molecules = inputs.molecules.count();
    rows.clear();

    std::vector<std::uint32_t> order;
    for (std::uint64_t molecule_index = 0; molecule_index < num_molecules; ++molecule_index) {
        if (inputs.molecules.num_conformers[molecule_index] == 0) {
            rows.push_back({.molecule_index = molecule_index, .outcome = row_outcome_t::molecule_failure_k});
            continue;
        }

        std::uint32_t const num_rows = rows_of_molecule_(inputs, molecule_index);
        order.resize(num_rows);
        for (std::uint32_t conformer_slot = 0; conformer_slot < num_rows; ++conformer_slot)
            order[conformer_slot] = conformer_slot;
        std::stable_sort(order.begin(), order.end(), [&](std::uint32_t left, std::uint32_t right) {
            return slot_ordering_energy_(conformers, conformers.layout.flat_slot(molecule_index, left)) <
                   slot_ordering_energy_(conformers, conformers.layout.flat_slot(molecule_index, right));
        });

        for (std::uint32_t rank = 0; rank < num_rows; ++rank) {
            std::uint32_t const conformer_slot = order[rank];
            bool const survived = conformers.status[conformers.layout.flat_slot(molecule_index, conformer_slot)] ==
                                  api::conformer_status_t::success_k;
            rows.push_back(
                {.molecule_index = molecule_index,
                 .conformer_slot = conformer_slot,
                 .energy_rank = static_cast<std::uint8_t>(rank),
                 .outcome = survived ? row_outcome_t::conformer_success_k : row_outcome_t::conformer_failure_k});
        }
    }
}

#pragma endregion Output Rows

#pragma region List Columns

/**
 *  @brief Pack fp16 coordinates from the strided slot pool into a tight Arrow `List<float16>`, one element
 *  per output row.
 *
 *  A success row carries the first `num_atoms[molecule] * 3` halves of its slot, transposed out of the
 *  structure-of-arrays layout into the interleaved `(atom, 3)` block a Parquet row exposes; every other row
 *  is null. Atom counts are clamped to the slot's own stride so the allocation pass and the emit pass agree
 *  even when a caller reports more atoms than fit.
 *
 *  @param[out] array Untouched on refusal.
 *  @return `written_k`, or the refusal that stopped the pack.
 */
static shard_write_report_t pack_coordinates_as_list(writer_inputs_t const &inputs,
                                                     custd::span<output_row_t const> rows,
                                                     std::shared_ptr<arrow::Array> &array) {
    api::conformer_slot_layout_t const &layout = inputs.conformers.layout;
    custd::span<std::uint8_t const> const num_atoms_per_molecule = inputs.molecules.num_atoms;
    std::size_t const num_rows = rows.size();

    auto atoms_of_ = [&](output_row_t const &row) {
        return std::min<std::uint32_t>(num_atoms_per_molecule[row.molecule_index], layout.max_atoms);
    };

    std::size_t total_halves = 0;
    for (output_row_t const &row : rows)
        if (row.outcome == row_outcome_t::conformer_success_k)
            total_halves += static_cast<std::size_t>(atoms_of_(row)) * 3u;

    shard_write_report_t const allocation_failed {.status = shard_write_t::buffer_allocation_failed_k,
                                                  .column = "conformer_coords"};
    auto offsets_buffer_result = arrow::AllocateBuffer(
        static_cast<std::int64_t>((num_rows + 1) * sizeof(std::int32_t)));
    if (!offsets_buffer_result.ok()) return allocation_failed;
    auto offsets_buffer = std::move(offsets_buffer_result).ValueOrDie();
    std::int32_t *offsets = reinterpret_cast<std::int32_t *>(offsets_buffer->mutable_data());

    auto values_buffer_result = arrow::AllocateBuffer(static_cast<std::int64_t>(total_halves * sizeof(std::uint16_t)));
    if (!values_buffer_result.ok()) return allocation_failed;
    auto values_buffer = std::move(values_buffer_result).ValueOrDie();
    std::uint16_t *values = reinterpret_cast<std::uint16_t *>(values_buffer->mutable_data());

    // Arrow's validity bitmap: bit `row` is set when the row carries data and clear when it is null.
    std::size_t const bitmap_bytes = (num_rows + 7) / 8;
    auto validity_buffer_result = arrow::AllocateBuffer(static_cast<std::int64_t>(bitmap_bytes));
    if (!validity_buffer_result.ok()) return allocation_failed;
    auto validity_buffer = std::move(validity_buffer_result).ValueOrDie();
    std::memset(validity_buffer->mutable_data(), 0, bitmap_bytes);
    std::uint8_t *validity = validity_buffer->mutable_data();

    std::int32_t running = 0;
    std::int64_t null_count = 0;
    for (std::size_t row_index = 0; row_index < num_rows; ++row_index) {
        output_row_t const &row = rows[row_index];
        offsets[row_index] = running;
        if (row.outcome != row_outcome_t::conformer_success_k) {
            ++null_count;
            continue;
        }
        std::uint32_t const atom_count = atoms_of_(row);
        if (atom_count > 0) {
            std::uint16_t const *source = reinterpret_cast<std::uint16_t const *>(
                                              inputs.conformers.coordinates.data()) +
                                          layout.slot_offset(row.molecule_index, row.conformer_slot);
            api::gather_slot_into_interleaved(source, layout, atom_count, values + running);
            running += static_cast<std::int32_t>(atom_count * 3u);
        }
        validity[row_index >> 3] |= static_cast<std::uint8_t>(1u << (row_index & 7u));
    }
    offsets[num_rows] = running;

    auto leaf_data = arrow::ArrayData::Make(arrow::float16(), static_cast<std::int64_t>(total_halves),
                                            {nullptr, std::move(values_buffer)}, /*null_count=*/0);
    auto leaf_array = arrow::MakeArray(std::move(leaf_data));

    auto list_data = arrow::ArrayData::Make(arrow::list(arrow::float16()), static_cast<std::int64_t>(num_rows),
                                            {std::move(validity_buffer), std::move(offsets_buffer)},
                                            {leaf_array->data()}, null_count);
    array = arrow::MakeArray(std::move(list_data));
    return {.status = shard_write_t::written_k};
}

/**
 *  @brief Pack the per-slot USRCAT descriptor into a nullable `List<float16>`, one element per output row.
 *
 *  A row emits its `usrcat_components_k` values only when it is a success row and the column was supplied;
 *  otherwise it is null.
 *
 *  @param[out] array Untouched on refusal.
 *  @return `written_k`, or the refusal that stopped the pack.
 */
static shard_write_report_t pack_usrcat_as_fp16_list(writer_inputs_t const &inputs,
                                                     custd::span<output_row_t const> rows,
                                                     std::shared_ptr<arrow::Array> &array) {
    auto value_builder = std::make_shared<arrow::HalfFloatBuilder>();
    arrow::ListBuilder list_builder(arrow::default_memory_pool(), value_builder);
    custd::span<double const> const values = inputs.conformers.usrcat;
    shard_write_report_t const append_failed {.status = shard_write_t::column_append_failed_k, .column = "usrcat"};

    for (output_row_t const &row : rows) {
        if (values.empty() || row.outcome != row_outcome_t::conformer_success_k) {
            if (!list_builder.AppendNull().ok()) return append_failed;
            continue;
        }
        std::size_t const base = inputs.conformers.layout.flat_slot(row.molecule_index, row.conformer_slot) *
                                 usrcat_components_k;
        if (!list_builder.Append().ok()) return append_failed;
        for (std::uint32_t component = 0; component < usrcat_components_k; ++component) {
            __half const half_value = __float2half(static_cast<float>(values[base + component]));
            std::uint16_t bits;
            std::memcpy(&bits, &half_value, sizeof(bits));
            if (!value_builder->Append(bits).ok()) return append_failed;
        }
    }

    std::shared_ptr<arrow::Array> finished;
    if (!list_builder.Finish(&finished).ok())
        return {.status = shard_write_t::column_finish_failed_k, .column = "usrcat"};
    array = std::move(finished);
    return {.status = shard_write_t::written_k};
}

#pragma endregion List Columns

#pragma region Assembled Table

/** Whether a schema column admits nulls, which the writer decides and the array cannot say for itself. */
enum class column_nullability_t : bool { required_k, nullable_k };

/**
 *  @brief The schema's fields and the arrays behind them, appended together so the two cannot drift.
 *
 *  A field's type comes from the array it names rather than from a second list beside it.
 */
struct column_table_t {
    std::vector<std::shared_ptr<arrow::Field>> fields;
    std::vector<std::shared_ptr<arrow::Array>> arrays;

    void append(std::string name, column_nullability_t nullability, std::shared_ptr<arrow::Array> array) {
        fields.push_back(arrow::field(std::move(name), array->type(), nullability == column_nullability_t::nullable_k));
        arrays.push_back(std::move(array));
    }
};

/**
 *  @brief The eleven scalar row columns, reserved once, appended per row, and finished together.
 *
 *  Each phase keeps the first column that refused and reports it once the phase ends, rather than costing
 *  eleven branches a row for a condition that fires at most once.
 */
class row_columns_t {
    arrow::StringBuilder input_shard_;
    arrow::UInt64Builder input_row_;
    arrow::StringBuilder smiles_;
    arrow::UInt8Builder conformer_index_;
    arrow::UInt8Builder status_;
    arrow::UInt16Builder num_heavy_atoms_;
    arrow::UInt16Builder num_atoms_;
    arrow::UInt16Builder num_bonds_;
    arrow::FloatBuilder molecular_weight_;
    arrow::UInt8Builder num_conformers_;
    arrow::FloatBuilder conformer_energy_;

    shard_write_report_t report_ {};
    std::string detail_;

    /** Keep the first refusal of the current phase, with the column that produced it. */
    void record_(shard_write_t status, std::string_view column, arrow::Status const &arrow_status) {
        if (arrow_status.ok() || report_.status != shard_write_t::written_k) return;
        report_ = {.status = status, .column = column};
        detail_ = arrow_status.message();
    }

  public:
    /** Whatever Arrow said about the refusal, empty when nothing refused. */
    [[nodiscard]] std::string const &detail() const noexcept { return detail_; }

    /** Size every column for @p num_rows, so the fixed-width appends below cannot fail on capacity. */
    [[nodiscard]] shard_write_report_t reserve(std::size_t num_rows) {
        std::int64_t const rows = static_cast<std::int64_t>(num_rows);
        auto reserve_ = [&](std::string_view column, auto &builder) {
            record_(shard_write_t::column_reserve_failed_k, column, builder.Reserve(rows));
        };
        reserve_("input_shard", input_shard_);
        reserve_("input_row", input_row_);
        reserve_("smiles", smiles_);
        reserve_("conformer_index", conformer_index_);
        reserve_("status", status_);
        reserve_("n_heavy_atoms", num_heavy_atoms_);
        reserve_("n_atoms", num_atoms_);
        reserve_("n_bonds", num_bonds_);
        reserve_("molecular_weight", molecular_weight_);
        reserve_("n_conformers", num_conformers_);
        reserve_("conformer_energy", conformer_energy_);
        return report_;
    }

    /**
     *  @brief Append one row to every column.
     *
     *  `Reserve` sizes `smiles_`'s offset array rather than its character data, so that one append can still
     *  hit an allocation failure; the refusal is kept and read once after the pass.
     */
    void append(writer_inputs_t const &inputs, output_row_t const &row) {
        molecule_columns_t const &molecules = inputs.molecules;
        std::uint64_t const molecule_index = row.molecule_index;
        std::size_t const flat = inputs.conformers.layout.flat_slot(molecule_index, row.conformer_slot);
        std::string_view const smiles = molecules.smiles[molecule_index];
        auto record_append_ = [&](std::string_view column, arrow::Status status) {
            record_(shard_write_t::column_append_failed_k, column, status);
        };

        record_append_("input_shard", input_shard_.Append(inputs.provenance.input_shard_stem));
        record_append_("input_row", input_row_.Append(inputs.provenance.base_row_index + molecule_index));
        record_append_("smiles", smiles_.Append(smiles.data(), static_cast<std::int32_t>(smiles.size())));
        record_append_("conformer_index", conformer_index_.Append(row.energy_rank));
        record_append_("status", status_.Append(static_cast<std::uint8_t>(inputs.conformers.status[flat])));

        if (molecules.num_atoms[molecule_index] == 0) { // preprocess failed, so every per-molecule column is null
            record_append_("n_heavy_atoms", num_heavy_atoms_.AppendNull());
            record_append_("n_atoms", num_atoms_.AppendNull());
            record_append_("n_bonds", num_bonds_.AppendNull());
            record_append_("molecular_weight", molecular_weight_.AppendNull());
            record_append_("n_conformers", num_conformers_.AppendNull());
        }
        else {
            record_append_("n_heavy_atoms", num_heavy_atoms_.Append(molecules.num_heavy_atoms.empty()
                                                                        ? std::uint16_t {0}
                                                                        : molecules.num_heavy_atoms[molecule_index]));
            record_append_("n_atoms", num_atoms_.Append(molecules.num_atoms[molecule_index]));
            record_append_("n_bonds",
                           num_bonds_.Append(molecules.num_bonds.empty() ? std::uint16_t {0}
                                                                         : molecules.num_bonds[molecule_index]));
            record_append_("molecular_weight",
                           molecular_weight_.Append(
                               molecules.molecular_weight.empty() ? 0.0f : molecules.molecular_weight[molecule_index]));
            record_append_("n_conformers", num_conformers_.Append(addressable_conformers_(inputs, molecule_index)));
        }

        if (row.outcome == row_outcome_t::conformer_success_k)
            record_append_("conformer_energy", conformer_energy_.Append(inputs.conformers.energies[flat]));
        else record_append_("conformer_energy", conformer_energy_.AppendNull());
    }

    /** The first append that refused; a short column would otherwise be written as a truncated shard. */
    [[nodiscard]] shard_write_report_t append_report() const noexcept { return report_; }

    /** Hand every finished column to @p table in schema order. */
    [[nodiscard]] shard_write_report_t finish_into(column_table_t &table) {
        auto finish_ = [&](std::string_view column, column_nullability_t nullability, auto &builder) {
            std::shared_ptr<arrow::Array> array;
            record_(shard_write_t::column_finish_failed_k, column, builder.Finish(&array));
            if (array) table.append(std::string {column}, nullability, std::move(array));
        };
        finish_("input_shard", column_nullability_t::required_k, input_shard_);
        finish_("input_row", column_nullability_t::required_k, input_row_);
        finish_("smiles", column_nullability_t::required_k, smiles_);
        finish_("conformer_index", column_nullability_t::required_k, conformer_index_);
        finish_("status", column_nullability_t::required_k, status_);
        finish_("n_heavy_atoms", column_nullability_t::nullable_k, num_heavy_atoms_);
        finish_("n_atoms", column_nullability_t::nullable_k, num_atoms_);
        finish_("n_bonds", column_nullability_t::nullable_k, num_bonds_);
        finish_("molecular_weight", column_nullability_t::nullable_k, molecular_weight_);
        finish_("n_conformers", column_nullability_t::nullable_k, num_conformers_);
        return report_;
    }

    /** The energy column trails the two list columns in the schema, so it is finished after them. */
    [[nodiscard]] shard_write_report_t finish_energy_into(column_table_t &table) {
        std::shared_ptr<arrow::Array> array;
        record_(shard_write_t::column_finish_failed_k, "conformer_energy", conformer_energy_.Finish(&array));
        if (array) table.append("conformer_energy", column_nullability_t::nullable_k, std::move(array));
        return report_;
    }
};

/**
 *  @brief Schema-level key-value metadata, so the bytes stay self-describing in a content-addressed store.
 *
 *  `SMALLFIELD_GIT_SHA` is injected by CMake. The frame is reported as `centered` rather than canonical:
 *  the metric embedding aligns to principal axes, but the FIRE/MMFF relaxation after it drifts the pose off
 *  that frame.
 */
static std::shared_ptr<arrow::KeyValueMetadata> build_provenance_(writer_inputs_t const &inputs) {
    auto provenance = std::make_shared<arrow::KeyValueMetadata>();
    provenance->Append("producer", "smallfield_batch");
#ifdef SMALLFIELD_GIT_SHA
    provenance->Append("producer_git_sha", SMALLFIELD_GIT_SHA);
#endif
    provenance->Append("num_conformers", std::to_string(inputs.conformers.layout.num_conformers));
    provenance->Append("max_atoms", std::to_string(inputs.conformers.layout.max_atoms));
    if (!inputs.provenance.dataset.empty()) provenance->Append("dataset", inputs.provenance.dataset);
    provenance->Append("coordinate_frame", "centered");
    provenance->Append("schema_version", "2");
    provenance->Append("dedup", "none");
    return provenance;
}

/** Write @p table to @p path as one ZSTD-compressed Parquet file, closed before this returns. */
static shard_write_report_t write_parquet_(fs::path const &path, arrow::Table const &table, logger_t const &logger) {
    auto outfile_result = arrow::io::FileOutputStream::Open(path.string());
    if (!outfile_result.ok())
        return refuse_(logger, path, {.status = shard_write_t::output_open_failed_k},
                       outfile_result.status().message());
    auto outfile = outfile_result.ValueOrDie();

    auto writer_props =
        parquet::WriterProperties::Builder().compression(parquet::Compression::ZSTD)->compression_level(3)->build();
    auto arrow_props = parquet::ArrowWriterProperties::Builder().store_schema()->build();
    auto write_status = parquet::arrow::WriteTable(table, arrow::default_memory_pool(), outfile, 65536, writer_props,
                                                   arrow_props);
    if (!write_status.ok())
        return refuse_(logger, path, {.status = shard_write_t::parquet_write_failed_k}, write_status.message());

    auto close_status = outfile->Close();
    if (!close_status.ok())
        return refuse_(logger, path, {.status = shard_write_t::output_close_failed_k}, close_status.message());
    return {.status = shard_write_t::written_k};
}

#pragma endregion Assembled Table

shard_write_report_t write_shard(fs::path const &path, writer_inputs_t const &inputs, logger_t const *log) {
    logger_t const fallback;
    logger_t const &logger = log != nullptr ? *log : fallback;

    if (shard_write_report_t const report = check_input_extents_(inputs); report.status != shard_write_t::written_k)
        return refuse_(logger, path, report, describe_extents_(inputs));

    std::vector<output_row_t> rows;
    fill_output_rows_into(inputs, rows);

    row_columns_t columns;
    if (shard_write_report_t const report = columns.reserve(rows.size()); report.status != shard_write_t::written_k)
        return refuse_(logger, path, report, columns.detail());

    for (output_row_t const &row : rows) columns.append(inputs, row);
    if (shard_write_report_t const report = columns.append_report(); report.status != shard_write_t::written_k)
        return refuse_(logger, path, report, columns.detail());

    column_table_t table;
    if (shard_write_report_t const report = columns.finish_into(table); report.status != shard_write_t::written_k)
        return refuse_(logger, path, report, columns.detail());

    std::shared_ptr<arrow::Array> coordinates_array;
    if (shard_write_report_t const report = pack_coordinates_as_list(inputs, rows, coordinates_array);
        report.status != shard_write_t::written_k)
        return refuse_(logger, path, report);
    table.append("conformer_coords", column_nullability_t::nullable_k, std::move(coordinates_array));

    if (shard_write_report_t const report = columns.finish_energy_into(table);
        report.status != shard_write_t::written_k)
        return refuse_(logger, path, report, columns.detail());

    std::shared_ptr<arrow::Array> usrcat_array;
    if (shard_write_report_t const report = pack_usrcat_as_fp16_list(inputs, rows, usrcat_array);
        report.status != shard_write_t::written_k)
        return refuse_(logger, path, report);
    table.append("usrcat", column_nullability_t::nullable_k, std::move(usrcat_array));

    auto schema = arrow::schema(table.fields, build_provenance_(inputs));
    return write_parquet_(path, *arrow::Table::Make(schema, table.arrays), logger);
}

} // namespace smallfield::cli
