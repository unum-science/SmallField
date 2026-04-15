#include "parquet_writer.hpp"

#include <cstdio>
#include <memory>

#include <arrow/array/builder_binary.h>
#include <arrow/array/builder_nested.h>
#include <arrow/array/builder_primitive.h>
#include <arrow/io/file.h>
#include <arrow/table.h>
#include <arrow/type.h>
#include <arrow/util/float16.h>
#include <parquet/arrow/writer.h>

namespace smallfield::cli {

namespace fs = std::filesystem;

bool write_shard(fs::path const &path, writer_inputs_t const &inputs) {
    std::uint64_t const num_input_rows = inputs.input_rows.size();
    std::uint32_t const num_conformers = inputs.num_conformers;
    std::uint32_t const coordinate_count = inputs.max_atoms * 3u;
    std::uint64_t const num_output_rows = num_input_rows * static_cast<std::uint64_t>(num_conformers);

    arrow::StringBuilder input_shard_builder;
    arrow::UInt64Builder input_row_builder;
    arrow::UInt8Builder conformer_index_builder;
    arrow::UInt8Builder status_builder;
    arrow::UInt8Builder solver_builder;
    arrow::UInt8Builder num_atoms_builder;
    arrow::FloatBuilder energy_builder;

    auto coord_value_builder = std::make_shared<arrow::HalfFloatBuilder>();
    auto coord_type = arrow::fixed_size_list(arrow::float16(), static_cast<int32_t>(coordinate_count));
    arrow::FixedSizeListBuilder coord_list_builder(arrow::default_memory_pool(), coord_value_builder, coord_type);

    auto reserve_result = input_shard_builder.Reserve(num_output_rows);
    if (!reserve_result.ok()) return false;
    (void)input_row_builder.Reserve(num_output_rows);
    (void)conformer_index_builder.Reserve(num_output_rows);
    (void)status_builder.Reserve(num_output_rows);
    (void)solver_builder.Reserve(num_output_rows);
    (void)num_atoms_builder.Reserve(num_output_rows);
    (void)energy_builder.Reserve(num_output_rows);
    (void)coord_list_builder.Reserve(num_output_rows);

    bool const have_num_atoms = !inputs.num_atoms_per_molecule.empty();

    for (std::uint64_t molecule_index = 0; molecule_index < num_input_rows; ++molecule_index) {
        std::uint64_t const input_row = inputs.input_rows[molecule_index];
        for (std::uint32_t conformer_index = 0; conformer_index < num_conformers; ++conformer_index) {
            std::uint64_t const flat = molecule_index * num_conformers + conformer_index;

            (void)input_shard_builder.Append(inputs.input_shard_stem);
            (void)input_row_builder.Append(input_row);
            (void)conformer_index_builder.Append(static_cast<std::uint8_t>(conformer_index));
            (void)status_builder.Append(inputs.status[flat]);
            (void)solver_builder.Append(inputs.solver[flat]);

            if (have_num_atoms) {
                std::uint8_t const n = inputs.num_atoms_per_molecule[molecule_index];
                if (n == 0) { (void)num_atoms_builder.AppendNull(); }
                else { (void)num_atoms_builder.Append(n); }
            }
            else { (void)num_atoms_builder.AppendNull(); }

            bool const solver_failed = (inputs.solver[flat] == 2);
            if (solver_failed) { (void)energy_builder.AppendNull(); }
            else { (void)energy_builder.Append(inputs.energies[flat]); }

            // FixedSizeList cannot have null list entries (Arrow limitation):
            // when the row failed, emit zero-filled coordinates. Readers must
            // check `solver == 2` OR `energy_kcal_per_mol IS NULL` to detect
            // invalid rows.
            (void)coord_list_builder.Append();
            auto *raw_builder = static_cast<arrow::HalfFloatBuilder *>(coord_list_builder.value_builder());
            if (solver_failed) {
                static std::vector<std::uint16_t> const zero_pad(coordinate_count, 0);
                (void)raw_builder->AppendValues(zero_pad.data(), static_cast<int64_t>(coordinate_count));
            }
            else {
                __half const *coord_base = inputs.coordinates.data() +
                                           flat * static_cast<std::size_t>(coordinate_count);
                (void)raw_builder->AppendValues(reinterpret_cast<std::uint16_t const *>(coord_base),
                                                static_cast<int64_t>(coordinate_count));
            }
        }
    }

    std::shared_ptr<arrow::Array> input_shard_array, input_row_array, conformer_index_array;
    std::shared_ptr<arrow::Array> status_array, solver_array, num_atoms_array, energy_array, coord_array;
    if (!input_shard_builder.Finish(&input_shard_array).ok()) return false;
    if (!input_row_builder.Finish(&input_row_array).ok()) return false;
    if (!conformer_index_builder.Finish(&conformer_index_array).ok()) return false;
    if (!status_builder.Finish(&status_array).ok()) return false;
    if (!solver_builder.Finish(&solver_array).ok()) return false;
    if (!num_atoms_builder.Finish(&num_atoms_array).ok()) return false;
    if (!energy_builder.Finish(&energy_array).ok()) return false;
    if (!coord_list_builder.Finish(&coord_array).ok()) return false;

    auto schema = arrow::schema({
        arrow::field("input_shard", arrow::utf8(), /*nullable=*/false),
        arrow::field("input_row", arrow::uint64(), false),
        arrow::field("conformer_index", arrow::uint8(), false),
        arrow::field("status", arrow::uint8(), false),
        arrow::field("solver", arrow::uint8(), false),
        arrow::field("num_atoms", arrow::uint8(), true),
        arrow::field("energy_kcal_per_mol", arrow::float32(), true),
        arrow::field("coordinates_xyz", coord_type, /*nullable=*/false),
    });

    auto table = arrow::Table::Make(schema, {input_shard_array, input_row_array, conformer_index_array, status_array,
                                             solver_array, num_atoms_array, energy_array, coord_array});

    auto outfile_result = arrow::io::FileOutputStream::Open(path.string());
    if (!outfile_result.ok()) {
        std::fprintf(stderr, "[smallfield_batch] output open failed: %s: %s\n", path.string().c_str(),
                     outfile_result.status().message().c_str());
        return false;
    }
    auto outfile = outfile_result.ValueOrDie();

    auto write_status = parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), outfile,
                                                   /*chunk_size=*/65536);
    if (!write_status.ok()) {
        std::fprintf(stderr, "[smallfield_batch] write failed: %s: %s\n", path.string().c_str(),
                     write_status.message().c_str());
        return false;
    }

    auto close_status = outfile->Close();
    return close_status.ok();
}

} // namespace smallfield::cli
