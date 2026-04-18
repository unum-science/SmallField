#include "parquet_writer.hpp"

#include <cstdio>
#include <memory>

#include <arrow/array/array_nested.h>
#include <arrow/array/builder_binary.h>
#include <arrow/array/builder_primitive.h>
#include <arrow/array/data.h>
#include <arrow/array/util.h>
#include <arrow/buffer.h>
#include <arrow/io/file.h>
#include <arrow/table.h>
#include <arrow/type.h>
#include <arrow/util/float16.h>
#include <parquet/arrow/writer.h>

namespace smallfield::cli {

namespace fs = std::filesystem;

/**
 *  @brief Wrap a contiguous fp16 coordinate buffer as an Arrow FixedSizeList array without copying.
 *  @param coordinates Contiguous fp16 buffer of @p num_rows × @p coordinate_count halves.
 *  @param num_rows Number of output conformers (rows of the parent list).
 *  @param coordinate_count Halves per row — typically @c max_atoms × 3.
 *  @return A FixedSizeList<float16> array whose leaf data points into the caller's buffer.
 *  The returned array holds a shared_ptr to a non-owning Arrow Buffer; the caller must keep
 *  the backing storage alive until the Parquet writer finishes flushing. */
static std::shared_ptr<arrow::Array> wrap_coordinates_as_fixed_size_list(
    std::span<__half const> coordinates, std::uint64_t num_rows, std::uint32_t coordinate_count) {
    std::int64_t const num_halves = static_cast<std::int64_t>(num_rows) * static_cast<std::int64_t>(coordinate_count);

    // Zero-copy wrap of the caller's fp16 buffer as an Arrow Buffer.
    // Buffer::Wrap accepts a const pointer and stores it as uint8_t*; the Buffer is non-owning.
    auto coord_buffer = arrow::Buffer::Wrap(coordinates.data(), static_cast<std::size_t>(num_halves));

    // Build the leaf HalfFloat array from the wrapped buffer.
    // Buffers vector slot 0 is the null bitmap (nullptr == all valid); slot 1 is the values.
    auto leaf_data = arrow::ArrayData::Make(arrow::float16(), num_halves,
                                            {nullptr, std::move(coord_buffer)},
                                            /*null_count=*/0);
    auto leaf_array = arrow::MakeArray(std::move(leaf_data));

    // Wrap the leaf as a FixedSizeList — one row of @p coordinate_count halves per conformer.
    auto list_type = arrow::fixed_size_list(arrow::float16(), static_cast<std::int32_t>(coordinate_count));
    return std::make_shared<arrow::FixedSizeListArray>(list_type, static_cast<std::int64_t>(num_rows),
                                                       std::move(leaf_array));
}

bool write_shard(fs::path const &path, writer_inputs_t const &inputs) {
    std::uint64_t const num_molecules = inputs.num_molecules;
    std::uint32_t const num_conformers = inputs.num_conformers;
    std::uint32_t const coordinate_count = inputs.max_atoms * 3u;
    std::uint64_t const num_output_rows = num_molecules * static_cast<std::uint64_t>(num_conformers);

    arrow::StringBuilder input_shard_builder;
    arrow::UInt64Builder input_row_builder;
    arrow::StringBuilder smiles_builder;
    arrow::UInt8Builder conformer_index_builder;
    arrow::UInt8Builder status_builder;
    arrow::UInt8Builder solver_builder;
    arrow::UInt8Builder num_atoms_builder;
    arrow::FloatBuilder energy_builder;

    if (!input_shard_builder.Reserve(num_output_rows).ok()) return false;
    (void)input_row_builder.Reserve(num_output_rows);
    (void)smiles_builder.Reserve(num_output_rows);
    (void)conformer_index_builder.Reserve(num_output_rows);
    (void)status_builder.Reserve(num_output_rows);
    (void)solver_builder.Reserve(num_output_rows);
    (void)num_atoms_builder.Reserve(num_output_rows);
    (void)energy_builder.Reserve(num_output_rows);

    bool const have_num_atoms = !inputs.num_atoms_per_molecule.empty();

    // Populate the scalar metadata columns.
    // The coordinate column is built separately via zero-copy wrap after this loop.
    // Per-molecule invariants (shard stem, input row, smiles, atom count) are appended once per
    // conformer; the inner loop writes the per-conformer status/solver/energy fields.
    for (std::uint64_t molecule_index = 0; molecule_index < num_molecules; ++molecule_index) {
        std::uint64_t const input_row = inputs.base_row_index + molecule_index;
        std::string_view const smi = molecule_index < inputs.smiles.size() ? inputs.smiles[molecule_index] : "";
        std::uint8_t const atom_count = have_num_atoms ? inputs.num_atoms_per_molecule[molecule_index] : 0;
        bool const atom_count_valid = have_num_atoms && atom_count != 0;

        for (std::uint32_t conformer_index = 0; conformer_index < num_conformers; ++conformer_index) {
            std::uint64_t const flat = molecule_index * num_conformers + conformer_index;

            (void)input_shard_builder.Append(inputs.input_shard_stem);
            (void)input_row_builder.Append(input_row);
            (void)smiles_builder.Append(smi.data(), static_cast<std::int32_t>(smi.size()));
            (void)conformer_index_builder.Append(static_cast<std::uint8_t>(conformer_index));
            (void)status_builder.Append(inputs.status[flat]);
            (void)solver_builder.Append(inputs.solver[flat]);

            if (atom_count_valid) { (void)num_atoms_builder.Append(atom_count); }
            else { (void)num_atoms_builder.AppendNull(); }

            bool const solver_failed = (inputs.solver[flat] == 2);
            if (solver_failed) { (void)energy_builder.AppendNull(); }
            else { (void)energy_builder.Append(inputs.energies[flat]); }
        }
    }

    std::shared_ptr<arrow::Array> input_shard_array, input_row_array, smiles_array, conformer_index_array;
    std::shared_ptr<arrow::Array> status_array, solver_array, num_atoms_array, energy_array;
    if (!input_shard_builder.Finish(&input_shard_array).ok()) return false;
    if (!input_row_builder.Finish(&input_row_array).ok()) return false;
    if (!smiles_builder.Finish(&smiles_array).ok()) return false;
    if (!conformer_index_builder.Finish(&conformer_index_array).ok()) return false;
    if (!status_builder.Finish(&status_array).ok()) return false;
    if (!solver_builder.Finish(&solver_array).ok()) return false;
    if (!num_atoms_builder.Finish(&num_atoms_array).ok()) return false;
    if (!energy_builder.Finish(&energy_array).ok()) return false;

    // Bulk-wrap the coordinate buffer as a FixedSizeList<float16> array with zero copy.
    // The caller guarantees the buffer covers all rows (failed rows are zero-filled upstream),
    // so there is no per-row work on the writer side.
    std::shared_ptr<arrow::Array> coord_array =
        wrap_coordinates_as_fixed_size_list(inputs.coordinates, num_output_rows, coordinate_count);

    auto coord_type = arrow::fixed_size_list(arrow::float16(), static_cast<std::int32_t>(coordinate_count));
    auto schema = arrow::schema({
        arrow::field("input_shard", arrow::utf8(), false),
        arrow::field("input_row", arrow::uint64(), false),
        arrow::field("smiles", arrow::utf8(), false),
        arrow::field("conformer_index", arrow::uint8(), false),
        arrow::field("status", arrow::uint8(), false),
        arrow::field("solver", arrow::uint8(), false),
        arrow::field("num_atoms", arrow::uint8(), true),
        arrow::field("energy_kcal_per_mol", arrow::float32(), true),
        arrow::field("coordinates_xyz", coord_type, false),
    });

    auto table = arrow::Table::Make(schema, {input_shard_array, input_row_array, smiles_array, conformer_index_array,
                                             status_array, solver_array, num_atoms_array, energy_array, coord_array});

    auto outfile_result = arrow::io::FileOutputStream::Open(path.string());
    if (!outfile_result.ok()) {
        std::fprintf(stderr, "[smallfield_batch] output open failed: %s: %s\n", path.string().c_str(),
                     outfile_result.status().message().c_str());
        return false;
    }
    auto outfile = outfile_result.ValueOrDie();

    auto write_status = parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), outfile, 65536);
    if (!write_status.ok()) {
        std::fprintf(stderr, "[smallfield_batch] write failed: %s: %s\n", path.string().c_str(),
                     write_status.message().c_str());
        return false;
    }

    auto close_status = outfile->Close();
    return close_status.ok();
}

} // namespace smallfield::cli
