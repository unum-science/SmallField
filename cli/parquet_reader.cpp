#include "parquet_reader.hpp"

#include <cstdio>
#include <memory>

#include <arrow/array.h>
#include <arrow/io/file.h>
#include <arrow/table.h>
#include <parquet/arrow/reader.h>
#include <parquet/file_reader.h>

namespace smallfield::cli {

namespace fs = std::filesystem;

bool read_shard(fs::path const &path, std::string const &smiles_column, shard_data_t &out) {
    out.table.reset();
    out.smiles.clear();

    auto open_result = arrow::io::ReadableFile::Open(path.string(), arrow::default_memory_pool());
    if (!open_result.ok()) {
        std::fprintf(stderr, "  read failed (open): %s: %s\n", path.string().c_str(),
                     open_result.status().message().c_str());
        return false;
    }
    auto infile = open_result.ValueOrDie();

    // Build the Arrow reader through the Builder pattern so we can attach ArrowReaderProperties.
    // Leaving pre_buffer at its default value keeps behavior unchanged; enabling it is deferred until the async-IO path
    // is validated.
    parquet::arrow::FileReaderBuilder builder;
    auto builder_open_status = builder.Open(infile);
    if (!builder_open_status.ok()) {
        std::fprintf(stderr, "  read failed (parquet): %s: %s\n", path.string().c_str(),
                     builder_open_status.message().c_str());
        return false;
    }
    parquet::ArrowReaderProperties arrow_props;
    builder.memory_pool(arrow::default_memory_pool());
    builder.properties(arrow_props);
    auto reader_result = builder.Build();
    if (!reader_result.ok()) {
        std::fprintf(stderr, "  read failed (parquet): %s: %s\n", path.string().c_str(),
                     reader_result.status().message().c_str());
        return false;
    }
    auto reader = std::move(reader_result).ValueOrDie();

    int const num_parquet_columns = reader->parquet_reader()->metadata()->schema()->num_columns();
    int smiles_col_idx = -1;
    for (int i = 0; i < num_parquet_columns; ++i) {
        if (reader->parquet_reader()->metadata()->schema()->Column(i)->name() == smiles_column) {
            smiles_col_idx = i;
            break;
        }
    }
    if (smiles_col_idx < 0) {
        std::fprintf(stderr, "  column %s not found in %s\n", smiles_column.c_str(), path.string().c_str());
        return false;
    }

    std::shared_ptr<arrow::Table> table;
    auto read_status = reader->ReadTable({smiles_col_idx}, &table);
    if (!read_status.ok() || !table) {
        std::fprintf(stderr, "  read failed: %s: %s\n", path.string().c_str(), read_status.message().c_str());
        return false;
    }

    auto column = table->column(0);
    std::uint64_t const num_rows = static_cast<std::uint64_t>(table->num_rows());
    out.smiles.reserve(num_rows);

    for (int chunk_index = 0; chunk_index < column->num_chunks(); ++chunk_index) {
        auto chunk = std::static_pointer_cast<arrow::StringArray>(column->chunk(chunk_index));
        if (!chunk) {
            std::fprintf(stderr, "  column %s is not string in %s\n", smiles_column.c_str(), path.string().c_str());
            return false;
        }
        for (int64_t local_index = 0; local_index < chunk->length(); ++local_index) {
            if (chunk->IsNull(local_index)) { out.smiles.emplace_back(); }
            else {
                auto value = chunk->GetView(local_index);
                out.smiles.emplace_back(value.data(), value.size());
            }
        }
    }

    out.table = std::move(table);
    return true;
}

} // namespace smallfield::cli
