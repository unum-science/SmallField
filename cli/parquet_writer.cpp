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
 *  @brief Pack fp16 coordinates from a strided GPU output buffer into a tight
 *  Arrow `List<float16>` array.
 *
 *  The GPU writes coordinates at a fixed stride of @p stride_halves per conformer
 *  slot (equal to `max_atoms_per_molecule × 3`), so most rows are 50-90 % trailing
 *  zero padding.  This helper copies only the first `num_atoms × 3` halves of each
 *  row into a compacted leaf buffer and emits per-row offsets.  The resulting
 *  column is a variable-length `List<float16>` — roughly 40-60 % smaller on
 *  drug-like inputs (Enamine, PubChem) than the old `FixedSizeList<float16, N>`
 *  representation, and readers slice out exactly the valid coordinates without
 *  needing the max-atoms stride.
 *
 *  @param coordinates Strided fp16 buffer; total size @p num_rows × @p stride_halves.
 *  @param num_atoms_per_molecule Per-molecule atom count (after addHs). 0 means
 *      "failed preprocess / unknown" — those rows emit an empty list.
 *  @param num_molecules Number of molecules; rows = molecules × conformers.
 *  @param num_conformers Conformers per molecule.
 *  @param stride_halves Row stride in halves — equal to max_atoms_per_molecule × 3.
 *  @return A List<float16> array with one row per conformer.
 */
static std::shared_ptr<arrow::Array> pack_coordinates_as_list(std::span<__half const> coordinates,
                                                              std::span<std::uint8_t const> num_atoms_per_molecule,
                                                              std::uint64_t num_molecules, std::uint32_t num_conformers,
                                                              std::uint32_t stride_halves) {
    std::uint64_t const num_rows = num_molecules * static_cast<std::uint64_t>(num_conformers);

    // Prefix-scan offsets (int32 is sufficient: even at 10 M rows × 128 atoms × 3 =
    // 3.8 G halves = 7.6 GB — still under INT32_MAX for row counts up to ~22 M).
    std::int64_t total_halves = 0;
    for (std::uint64_t m = 0; m < num_molecules; ++m) {
        std::uint32_t const row_halves = static_cast<std::uint32_t>(num_atoms_per_molecule[m]) * 3u;
        total_halves += static_cast<std::int64_t>(row_halves) * num_conformers;
    }

    auto offsets_builder_result = arrow::AllocateBuffer((num_rows + 1) * sizeof(std::int32_t));
    if (!offsets_builder_result.ok()) return nullptr;
    auto offsets_buffer = std::move(offsets_builder_result).ValueOrDie();
    std::int32_t *offsets = reinterpret_cast<std::int32_t *>(offsets_buffer->mutable_data());

    auto values_buffer_result = arrow::AllocateBuffer(total_halves * sizeof(std::uint16_t));
    if (!values_buffer_result.ok()) return nullptr;
    auto values_buffer = std::move(values_buffer_result).ValueOrDie();
    std::uint16_t *values = reinterpret_cast<std::uint16_t *>(values_buffer->mutable_data());

    std::int32_t running = 0;
    std::uint64_t row = 0;
    for (std::uint64_t m = 0; m < num_molecules; ++m) {
        std::uint32_t const row_halves = static_cast<std::uint32_t>(num_atoms_per_molecule[m]) * 3u;
        for (std::uint32_t c = 0; c < num_conformers; ++c, ++row) {
            offsets[row] = running;
            if (row_halves > 0) {
                std::uint16_t const *src = reinterpret_cast<std::uint16_t const *>(coordinates.data()) +
                                           row * stride_halves;
                std::memcpy(values + running, src, row_halves * sizeof(std::uint16_t));
                running += static_cast<std::int32_t>(row_halves);
            }
        }
    }
    offsets[num_rows] = running;

    auto leaf_data = arrow::ArrayData::Make(arrow::float16(), total_halves, {nullptr, std::move(values_buffer)},
                                            /*null_count=*/0);
    auto leaf_array = arrow::MakeArray(std::move(leaf_data));

    auto list_data = arrow::ArrayData::Make(arrow::list(arrow::float16()), num_rows,
                                            {nullptr, std::move(offsets_buffer)}, {leaf_array->data()},
                                            /*null_count=*/0);
    return arrow::MakeArray(std::move(list_data));
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

    // The library now populates num_atoms for every molecule that survived preprocess,
    // so `inputs.num_atoms_per_molecule` is always present and always valid for
    // successful rows.  Failed rows (invalid_smiles / too_many / preprocess_timeout)
    // carry num_atoms = 0, which the writer represents as an empty coord list.
    bool const have_num_atoms = !inputs.num_atoms_per_molecule.empty();
    if (!have_num_atoms) {
        std::fprintf(stderr, "[smallfield_batch] write_shard: num_atoms_per_molecule is required\n");
        return false;
    }

    // Populate the scalar metadata columns.
    // The coordinate column is built separately via zero-copy wrap after this loop.
    // Per-molecule invariants (shard stem, input row, smiles, atom count) are appended once per
    // conformer; the inner loop writes the per-conformer status/solver/energy fields.
    for (std::uint64_t molecule_index = 0; molecule_index < num_molecules; ++molecule_index) {
        std::uint64_t const input_row = inputs.base_row_index + molecule_index;
        std::string_view const smi = molecule_index < inputs.smiles.size() ? inputs.smiles[molecule_index] : "";
        std::uint8_t const atom_count = inputs.num_atoms_per_molecule[molecule_index];

        for (std::uint32_t conformer_index = 0; conformer_index < num_conformers; ++conformer_index) {
            std::uint64_t const flat = molecule_index * num_conformers + conformer_index;

            (void)input_shard_builder.Append(inputs.input_shard_stem);
            (void)input_row_builder.Append(input_row);
            (void)smiles_builder.Append(smi.data(), static_cast<std::int32_t>(smi.size()));
            (void)conformer_index_builder.Append(static_cast<std::uint8_t>(conformer_index));
            (void)status_builder.Append(inputs.status[flat]);
            (void)solver_builder.Append(inputs.solver[flat]);
            // Always non-null: 0 ↔ preprocess failure / too_many / timeout.
            (void)num_atoms_builder.Append(atom_count);

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

    // Pack coordinates into a tight List<float16> (num_atoms × 3 halves per row).
    // Replaces the old FixedSizeList<float16, max_atoms × 3> which padded every row
    // with ~50-60 % trailing zeros on drug-like inputs.
    std::shared_ptr<arrow::Array> coord_array = pack_coordinates_as_list(
        inputs.coordinates, inputs.num_atoms_per_molecule, num_molecules, num_conformers, coordinate_count);
    if (!coord_array) {
        std::fprintf(stderr, "[smallfield_batch] coordinate buffer allocation failed\n");
        return false;
    }

    auto coord_type = arrow::list(arrow::float16());
    auto schema = arrow::schema({
        arrow::field("input_shard", arrow::utf8(), false),
        arrow::field("input_row", arrow::uint64(), false),
        arrow::field("smiles", arrow::utf8(), false),
        arrow::field("conformer_index", arrow::uint8(), false),
        arrow::field("status", arrow::uint8(), false),
        arrow::field("solver", arrow::uint8(), false),
        arrow::field("num_atoms", arrow::uint8(), false),
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

    // Zstd level 3: 3-5× smaller output on drug-like coordinate buffers
    // (which are 90%+ trailing zeros from the fixed-128-atom coord stride),
    // <5 % writer-CPU overhead, and zstd is universally supported in Parquet
    // readers (pyarrow, polars, duckdb).  Without explicit compression the
    // writer defaults to UNCOMPRESSED, inflating drug-like output ~3×.
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
