#include "parquet_reader.hpp"

#include <cstdio>
#include <memory>

#include <arrow/array.h>
#include <arrow/io/file.h>
#include <arrow/table.h>
#include <parquet/arrow/reader.h>

namespace smallfield::cli {

namespace fs = std::filesystem;

bool read_shard(fs::path const &path, std::string const &smiles_column, shard_data_t &out) {
    out.smiles.clear();
    out.input_row_indices.clear();

    std::shared_ptr<arrow::io::ReadableFile> infile;
    auto open_result = arrow::io::ReadableFile::Open(path.string());
    if (!open_result.ok()) {
        std::fprintf(stderr, "[smallfield_batch] open failed: %s: %s\n", path.string().c_str(),
                     open_result.status().message().c_str());
        return false;
    }
    infile = open_result.ValueOrDie();

    auto reader_result = parquet::arrow::OpenFile(infile, arrow::default_memory_pool());
    if (!reader_result.ok()) {
        std::fprintf(stderr, "[smallfield_batch] parquet open failed: %s: %s\n", path.string().c_str(),
                     reader_result.status().message().c_str());
        return false;
    }
    std::unique_ptr<parquet::arrow::FileReader> reader = std::move(reader_result).ValueOrDie();

    std::shared_ptr<arrow::Table> table;
    auto read_status = reader->ReadTable(&table);
    if (!read_status.ok() || !table) {
        std::fprintf(stderr, "[smallfield_batch] read failed: %s: %s\n", path.string().c_str(),
                     read_status.message().c_str());
        return false;
    }

    auto column = table->GetColumnByName(smiles_column);
    if (!column) {
        std::fprintf(stderr, "[smallfield_batch] column %s not found in %s\n", smiles_column.c_str(),
                     path.string().c_str());
        return false;
    }

    std::uint64_t const num_rows = static_cast<std::uint64_t>(table->num_rows());
    out.smiles.reserve(num_rows);
    out.input_row_indices.reserve(num_rows);

    std::uint64_t row_index = 0;
    for (int chunk_index = 0; chunk_index < column->num_chunks(); ++chunk_index) {
        auto chunk = std::static_pointer_cast<arrow::StringArray>(column->chunk(chunk_index));
        if (!chunk) {
            std::fprintf(stderr, "[smallfield_batch] column %s is not string in %s\n", smiles_column.c_str(),
                         path.string().c_str());
            return false;
        }
        for (int64_t local_index = 0; local_index < chunk->length(); ++local_index) {
            if (chunk->IsNull(local_index)) {
                out.smiles.emplace_back(); // empty → downstream rejects
            }
            else {
                auto value = chunk->GetView(local_index);
                out.smiles.emplace_back(value.data(), value.size());
            }
            out.input_row_indices.push_back(row_index++);
        }
    }

    return true;
}

} // namespace smallfield::cli
