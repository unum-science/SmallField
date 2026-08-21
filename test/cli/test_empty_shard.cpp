/**
 *  @file test/cli/test_empty_shard.cpp
 *  @brief Pin that a shard of no molecules is written, and written under the same schema as a populated one.
 *
 *  An input shard with no rows is a corpus fact, not a defect — `s3 sync` writes each shard directly to its
 *  final name, so a truncated fetch produces one — and downstream completeness keys on the `.encoded` sentinel
 *  alone. A producer that refuses the empty case writes no shard, seals nothing, and hands the runner one it
 *  will offer again on every pass, forever. So a zero-row shard is first-class: readable, carrying all thirteen
 *  fields, and differing from a populated shard in nothing but row count. The refusal guarding the same call
 *  stays live for its own case, a caller that dropped the molecule axis while its molecules exist.
 */

#include <cstdint> // `std::uint8_t`, `std::uint16_t`

#include <algorithm>    // `std::min`
#include <array>        // `std::array`
#include <filesystem>   // `std::filesystem::path`
#include <memory>       // `std::shared_ptr`
#include <string_view>  // `std::string_view`
#include <system_error> // `std::error_code`

#include <arrow/io/file.h>
#include <arrow/table.h>
#include <arrow/type.h>
#include <cuda_fp16.h>
#include <fmt/format.h> // `fmt::println`
#include <parquet/arrow/reader.h>

#include "cli/parquet_writer.hpp"
#include "smallfield/api/conformer_layout.hpp"
#include "smallfield/types/conformer_status.hpp"

#include "test/test_helpers.hpp"

namespace fs = std::filesystem;
namespace sfcli = smallfield::cli;
namespace sftest = smallfield::tests;

static constexpr std::string_view test_name_k = "test_empty_shard";

/** The canonical output schema, in the order `write_shard` appends its columns. */
static constexpr std::array<std::string_view, 13> schema_field_names_k {
    "input_shard",      "input_row",        "smiles",  "conformer_index",  "status",
    "n_heavy_atoms",    "n_atoms",          "n_bonds", "molecular_weight", "n_conformers",
    "conformer_coords", "conformer_energy", "usrcat"};

namespace smallfield::tests {

/** One molecule of one atom and one conformer, which is the smallest populated shard the writer accepts. */
struct populated_molecule_t {
    std::string_view smiles = "C";
    std::uint8_t num_atoms = 1;
    std::uint8_t num_conformers = 1;
    std::uint16_t num_heavy_atoms = 1;
    std::uint16_t num_bonds = 0;
    float molecular_weight = 16.043f;
    types::conformer_status_t status = types::conformer_status_t::success_k;
    float energy = 0.0f;
    std::array<__half, 3> coordinates {};
};

/** Inputs describing @p molecule as a one-row shard. */
static cli::writer_inputs_t populated_inputs_(populated_molecule_t const &molecule) {
    return {
        .provenance = {.input_shard_stem = "probe", .dataset = "test", .base_row_index = 0},
        .molecules = {.smiles = {&molecule.smiles, 1},
                      .num_atoms = {&molecule.num_atoms, 1},
                      .num_conformers = {&molecule.num_conformers, 1},
                      .num_heavy_atoms = {&molecule.num_heavy_atoms, 1},
                      .num_bonds = {&molecule.num_bonds, 1},
                      .molecular_weight = {&molecule.molecular_weight, 1},
                      .shape_described = {}},
        .conformers = {.layout = {.max_atoms = 1, .num_conformers = 1},
                       .status = {&molecule.status, 1},
                       .energies = {&molecule.energy, 1},
                       .coordinates = {molecule.coordinates.data(), molecule.coordinates.size()},
                       .usrcat = {}},
    };
}

/** Inputs describing a shard whose molecule axis is empty on every column. */
static cli::writer_inputs_t empty_inputs_() {
    return {
        .provenance = {.input_shard_stem = "probe", .dataset = "test", .base_row_index = 0},
        .molecules = {},
        .conformers = {.layout = {.max_atoms = 1, .num_conformers = 1}},
    };
}

/** Read @p path back through Arrow, so the assertions below run against the bytes rather than the builders. */
static std::shared_ptr<arrow::Table> read_table_(fs::path const &path) {
    auto open_result = arrow::io::ReadableFile::Open(path.string());
    sf_verify_(open_result.ok(), fmt::format("the shard just written could not be reopened: {}", path.string()));

    parquet::arrow::FileReaderBuilder builder;
    sf_verify_(builder.Open(open_result.ValueOrDie()).ok(),
               fmt::format("the shard just written is not a readable parquet file: {}", path.string()));
    auto reader_result = builder.Build();
    sf_verify_(reader_result.ok(), fmt::format("no parquet reader could be built for {}", path.string()));

    std::shared_ptr<arrow::Table> table;
    sf_verify_(reader_result.ValueOrDie()->ReadTable(&table).ok() && table,
               fmt::format("the shard just written holds no materialisable table: {}", path.string()));
    return table;
}

/**
 *  @brief Compare the two tables column by column against each other and against the canonical roster.
 *  A column name is asserted rather than reported: the reporter's delta is arithmetic, so names belong in
 *  the sentence `sf_verify_` prints and numbers in the `MISMATCH` lines beside it.
 */
static void check_zero_row_shard_(arrow::Table const &populated, arrow::Table const &empty,
                                  diagnostic_reporter_t &reporter) {
    reporter.mismatch("schema", "shard=empty")
        .field("num_columns", populated.num_columns(), empty.num_columns(), exactly_equal);
    reporter.mismatch("schema", "roster=canonical")
        .field("num_columns", schema_field_names_k.size(), static_cast<std::size_t>(empty.num_columns()),
               exactly_equal);
    reporter.mismatch("rows", "shard=populated").field("num_rows", 1, populated.num_rows(), exactly_equal);
    reporter.mismatch("rows", "shard=empty").field("num_rows", 0, empty.num_rows(), exactly_equal);

    int const columns = std::min(populated.num_columns(), empty.num_columns());
    sf_verify_(static_cast<std::size_t>(columns) == schema_field_names_k.size(),
               "a shard was written with a column count the canonical schema does not have");
    for (int column = 0; column < columns; ++column) {
        auto const &expected = *populated.schema()->field(column);
        auto const &observed = *empty.schema()->field(column);
        std::string_view const canonical = schema_field_names_k[static_cast<std::size_t>(column)];
        sf_verify_(expected.name() == canonical && observed.name() == canonical,
                   fmt::format("column {} is canonically {}, written as {} and {}", column, canonical, expected.name(),
                               observed.name()));
        reporter.mismatch("field", "column={} name={}", column, canonical)
            .field("type_id", static_cast<int>(expected.type()->id()), static_cast<int>(observed.type()->id()),
                   exactly_equal)
            .field("nullable", expected.nullable(), observed.nullable(), exactly_equal);
    }
}

} // namespace smallfield::tests

int main() {
    fmt::println("{}: a shard of no molecules writes zero rows under the canonical thirteen-field schema", test_name_k);

    std::error_code error_code;
    fs::path const directory = fs::temp_directory_path(error_code) / "smallfield_test_empty_shard";
    sf_verify_(!error_code, "no temporary directory is available to write a probe shard into");
    fs::create_directories(directory, error_code);
    sf_verify_(!error_code, fmt::format("probe directory could not be created: {}", directory.string()));
    fs::path const populated_path = directory / "populated.3D.parquet";
    fs::path const empty_path = directory / "empty.3D.parquet";

    sftest::populated_molecule_t const molecule;
    sfcli::shard_write_report_t const populated_report = sfcli::write_shard(populated_path,
                                                                            sftest::populated_inputs_(molecule));
    sf_verify_(
        populated_report.status == sfcli::shard_write_t::written_k,
        fmt::format("the one-molecule control shard was refused: {}", sfcli::describe_shard_write(populated_report)));

    sfcli::shard_write_report_t const empty_report = sfcli::write_shard(empty_path, sftest::empty_inputs_());
    sf_verify_(empty_report.status == sfcli::shard_write_t::written_k,
               fmt::format("a shard of no molecules was refused: {}", sfcli::describe_shard_write(empty_report)));

    // Axis dropped, not empty — molecules exist, undescribed — or a file with rows seals a shard holding none.
    sfcli::writer_inputs_t dropped_axis = sftest::empty_inputs_();
    dropped_axis.molecules.num_conformers = {&molecule.num_conformers, 1};
    sfcli::shard_write_report_t const dropped_report = sfcli::write_shard(directory / "dropped.3D.parquet",
                                                                          dropped_axis);
    sf_verify_(dropped_report.status == sfcli::shard_write_t::missing_per_molecule_arrays_k,
               "a molecule axis dropped from a populated shard was not refused");

    sftest::diagnostic_reporter_t reporter("expected", "observed");
    std::shared_ptr<arrow::Table> const populated_table = sftest::read_table_(populated_path);
    std::shared_ptr<arrow::Table> const empty_table = sftest::read_table_(empty_path);
    sftest::check_zero_row_shard_(*populated_table, *empty_table, reporter);
    reporter.summarize("zero_row_schema");

    fs::remove_all(directory, error_code);
    if (!reporter.passed())
        return sftest::report_fail(test_name_k, "{} mismatches over {} comparisons", reporter.mismatch_count(),
                                   reporter.comparison_count());
    return sftest::report_pass(test_name_k);
}
