/**
 *  @file parquet_3d_reader.cpp
 *  @brief Implementation of `read_conformer_shard` against the SmallField
 *         output parquet schema (see `parquet_writer.cpp::write_shard`).
 */

#include "parquet_3d_reader.hpp"

#include <cstdio>

#include <arrow/array/array_binary.h>
#include <arrow/array/array_nested.h>
#include <arrow/array/array_primitive.h>
#include <arrow/io/file.h>
#include <arrow/util/float16.h>
#include <parquet/arrow/reader.h>

namespace smallfield::cli {

namespace {

template <typename ArrowArrayT>
std::shared_ptr<ArrowArrayT> resolve_column_(arrow::Table const &table, std::string const &name) {
    auto column = table.GetColumnByName(name);
    if (!column || column->num_chunks() == 0) return nullptr;
    auto chunk = column->chunk(0);
    return std::dynamic_pointer_cast<ArrowArrayT>(chunk);
}

} // namespace

bool read_conformer_shard(std::filesystem::path const &path, conformer_shard_data_t &out) {
    auto open_result = arrow::io::ReadableFile::Open(path.string());
    if (!open_result.ok()) {
        std::fprintf(stderr, "[smallfield_mace_refine] open failed: %s: %s\n", path.string().c_str(),
                     open_result.status().message().c_str());
        return false;
    }

    std::unique_ptr<parquet::arrow::FileReader> reader;
    auto pq_result = parquet::arrow::OpenFile(open_result.ValueUnsafe(), arrow::default_memory_pool());
    if (!pq_result.ok()) {
        std::fprintf(stderr, "[smallfield_mace_refine] parquet open failed: %s: %s\n", path.string().c_str(),
                     pq_result.status().message().c_str());
        return false;
    }
    reader = std::move(pq_result).ValueOrDie();

    std::shared_ptr<arrow::Table> table;
    auto read_status = reader->ReadTable(&table);
    if (!read_status.ok() || !table) {
        std::fprintf(stderr, "[smallfield_mace_refine] read failed: %s: %s\n", path.string().c_str(),
                     read_status.message().c_str());
        return false;
    }

    // Combine chunks so we can reach into one contiguous Array per column.
    auto combined = table->CombineChunks();
    if (!combined.ok()) return false;
    table = combined.ValueUnsafe();
    out.table = table;

    auto const num_rows = static_cast<std::uint64_t>(table->num_rows());
    out.smiles.reserve(num_rows);
    out.num_atoms_per_row.reserve(num_rows);
    out.status_per_row.reserve(num_rows);
    out.solver_per_row.reserve(num_rows);
    out.conformer_index_per_row.reserve(num_rows);
    out.input_shard_per_row.reserve(num_rows);
    out.input_row_per_row.reserve(num_rows);
    out.baseline_energy_per_row.reserve(num_rows);
    out.baseline_energy_null_per_row.reserve(num_rows);
    out.row_atom_offsets.reserve(num_rows + 1);
    out.row_atom_offsets.push_back(0);

    auto smiles_arr = resolve_column_<arrow::StringArray>(*table, "smiles");
    auto num_atoms_arr = resolve_column_<arrow::UInt8Array>(*table, "num_atoms");
    auto status_arr = resolve_column_<arrow::UInt8Array>(*table, "status");
    auto solver_arr = resolve_column_<arrow::UInt8Array>(*table, "solver");
    auto conf_idx_arr = resolve_column_<arrow::UInt8Array>(*table, "conformer_index");
    auto input_shard_arr = resolve_column_<arrow::StringArray>(*table, "input_shard");
    auto input_row_arr = resolve_column_<arrow::UInt64Array>(*table, "input_row");
    auto energy_arr = resolve_column_<arrow::FloatArray>(*table, "energy_kcal_per_mol");
    auto coord_list_arr = resolve_column_<arrow::ListArray>(*table, "coordinates_xyz");

    bool const minimum_columns_present = smiles_arr && num_atoms_arr && status_arr && coord_list_arr;
    if (!minimum_columns_present) {
        std::fprintf(
            stderr,
            "[smallfield_mace_refine] %s missing required columns " "(smiles / num_atoms / status / coordinates_xyz)\n",
            path.string().c_str());
        return false;
    }

    // The list's inner Arrow array is `arrow::HalfFloatArray` (uint16 storage
    // bit-cast to fp16).  We bit-cast back to `__half` for direct indexing.
    auto coord_half_arr = std::dynamic_pointer_cast<arrow::HalfFloatArray>(coord_list_arr->values());
    if (!coord_half_arr) {
        std::fprintf(stderr, "[smallfield_mace_refine] coordinates_xyz inner type is not float16\n");
        return false;
    }
    std::uint16_t const *flat_halves = coord_half_arr->raw_values();

    // First pass: validate and reserve flat capacity.
    std::uint64_t total_atoms = 0;
    for (std::uint64_t i = 0; i < num_rows; ++i) {
        std::uint8_t const n_atoms = num_atoms_arr->Value(static_cast<std::int64_t>(i));
        total_atoms += n_atoms;
    }
    out.coordinates_flat.resize(static_cast<std::size_t>(total_atoms) * 3u);

    // Second pass: copy per-row data into the flat layouts.
    std::uint64_t atom_cursor = 0;
    for (std::uint64_t i = 0; i < num_rows; ++i) {
        out.smiles.emplace_back(smiles_arr->Value(static_cast<std::int64_t>(i)));
        std::uint8_t const n_atoms = num_atoms_arr->Value(static_cast<std::int64_t>(i));
        out.num_atoms_per_row.push_back(n_atoms);
        out.status_per_row.push_back(status_arr->Value(static_cast<std::int64_t>(i)));
        out.solver_per_row.push_back(solver_arr ? solver_arr->Value(static_cast<std::int64_t>(i)) : 0u);
        out.conformer_index_per_row.push_back(conf_idx_arr ? conf_idx_arr->Value(static_cast<std::int64_t>(i)) : 0u);
        out.input_shard_per_row.emplace_back(
            input_shard_arr ? std::string(input_shard_arr->Value(static_cast<std::int64_t>(i))) : "");
        out.input_row_per_row.push_back(input_row_arr ? input_row_arr->Value(static_cast<std::int64_t>(i)) : 0ull);
        if (energy_arr && !energy_arr->IsNull(static_cast<std::int64_t>(i))) {
            out.baseline_energy_per_row.push_back(energy_arr->Value(static_cast<std::int64_t>(i)));
            out.baseline_energy_null_per_row.push_back(false);
        }
        else {
            out.baseline_energy_per_row.push_back(0.0f);
            out.baseline_energy_null_per_row.push_back(true);
        }

        // Coordinates: list<float16> with num_atoms*3 entries when the row is a
        // success; ListArray gives us the offset into the inner half-float array.
        auto const list_start = coord_list_arr->value_offset(static_cast<std::int64_t>(i));
        std::uint32_t const expected = static_cast<std::uint32_t>(n_atoms) * 3u;
        // We trust `num_atoms`; the list length should equal `expected` for non-failures.
        for (std::uint32_t k = 0; k < expected; ++k) {
            std::uint16_t const bits = flat_halves[static_cast<std::int64_t>(list_start) + k];
            __half h;
            std::memcpy(&h, &bits, sizeof(__half));
            out.coordinates_flat[(atom_cursor * 3u) + k] = h;
        }
        atom_cursor += n_atoms;
        out.row_atom_offsets.push_back(atom_cursor);
    }
    return true;
}

} // namespace smallfield::cli
