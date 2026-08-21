/**
 *  @file cli/parquet_reader.cpp
 *  @brief Opens a Parquet shard through Arrow and slices its SMILES column into views over the retained table.
 */

#include <memory> // `std::shared_ptr`, `std::move`
#include <vector> // `std::vector`

#include <arrow/array.h>
#include <arrow/io/file.h>
#include <arrow/table.h>
#include <fmt/std.h> // formats `std::filesystem::path`
#include <parquet/arrow/reader.h>
#include <parquet/file_reader.h>

#include "cli/parquet_reader.hpp"

namespace fs = std::filesystem;

namespace smallfield::cli {

char const *describe_input_read(input_read_t status) {
    switch (status) {
    case input_read_t::read_k: return "read";
    case input_read_t::cannot_open_k: return "cannot be opened";
    case input_read_t::not_parquet_k: return "is not a readable parquet file";
    case input_read_t::column_missing_k: return "carries no column of that name";
    case input_read_t::unreadable_table_k: return "holds a table Arrow could not materialise";
    case input_read_t::column_not_string_k: return "types that column as something other than UTF-8";
    }
    return "was refused";
}

/** Index of the column named @p smiles_column, or `-1` when the file carries no such name. */
static int smiles_column_index_(parquet::arrow::FileReader const &reader, std::string const &smiles_column) {
    auto const *schema = reader.parquet_reader()->metadata()->schema();
    for (int column_index = 0; column_index < schema->num_columns(); ++column_index)
        if (schema->Column(column_index)->name() == smiles_column) return column_index;
    return -1;
}

/**
 *  @brief Slice every chunk of @p column into views over its own buffers.
 *
 *  A null row becomes an empty view rather than being dropped, so a view's position stays the row it came
 *  from — the writer maps molecule index back onto `input_row` by that position alone.
 *
 *  @param[out] smiles Caller-owned destination, cleared and reserved before the call.
 */
static input_read_t fill_smiles_views_into_(arrow::ChunkedArray const &column, std::vector<std::string_view> &smiles) {
    for (int chunk_index = 0; chunk_index < column.num_chunks(); ++chunk_index) {
        // Checked: `static_pointer_cast` never yields null, so a non-UTF-8 column would read as UTF-8 unrefused.
        auto const *chunk = dynamic_cast<arrow::StringArray const *>(column.chunk(chunk_index).get());
        if (chunk == nullptr) return input_read_t::column_not_string_k;
        for (std::int64_t row = 0; row < chunk->length(); ++row) {
            if (chunk->IsNull(row)) { smiles.emplace_back(); }
            else {
                auto const value = chunk->GetView(row);
                smiles.emplace_back(value.data(), value.size());
            }
        }
    }
    return input_read_t::read_k;
}

input_read_t read_shard(fs::path const &path, std::string const &smiles_column, shard_data_t &out,
                        logger_t const *log) {
    out.table.reset();
    out.smiles.clear();
    logger_t const fallback;
    logger_t const &logger = log != nullptr ? *log : fallback;

    // Every refusal leaves through here, so the reason reaches the log whatever the caller does with the status.
    auto refuse_ = [&](input_read_t status, std::string_view detail = {}) {
        if (detail.empty()) logger.error("{} {}", path, describe_input_read(status));
        else logger.error("{} {}: {}", path, describe_input_read(status), detail);
        return status;
    };

    auto open_result = arrow::io::ReadableFile::Open(path.string(), arrow::default_memory_pool());
    if (!open_result.ok()) return refuse_(input_read_t::cannot_open_k, open_result.status().message());

    // `pre_buffer` keeps its default, leaving Arrow's async-IO read-ahead path out of this reader.
    parquet::arrow::FileReaderBuilder builder;
    if (arrow::Status const status = builder.Open(open_result.ValueOrDie()); !status.ok())
        return refuse_(input_read_t::not_parquet_k, status.message());
    parquet::ArrowReaderProperties arrow_properties;
    builder.memory_pool(arrow::default_memory_pool());
    builder.properties(arrow_properties);
    auto reader_result = builder.Build();
    if (!reader_result.ok()) return refuse_(input_read_t::not_parquet_k, reader_result.status().message());
    auto reader = std::move(reader_result).ValueOrDie();

    int const column_index = smiles_column_index_(*reader, smiles_column);
    if (column_index < 0) return refuse_(input_read_t::column_missing_k, smiles_column);

    auto table_result = reader->ReadTable(std::vector<int> {column_index});
    if (!table_result.ok() || !table_result.ValueOrDie())
        return refuse_(input_read_t::unreadable_table_k, table_result.status().message());
    std::shared_ptr<arrow::Table> table = std::move(table_result).ValueOrDie();

    auto const column = table->column(0);
    out.smiles.reserve(static_cast<std::size_t>(table->num_rows()));
    if (input_read_t const status = fill_smiles_views_into_(*column, out.smiles); status != input_read_t::read_k) {
        out.smiles.clear();
        return refuse_(status, smiles_column);
    }

    out.table = std::move(table);
    return input_read_t::read_k;
}

} // namespace smallfield::cli
