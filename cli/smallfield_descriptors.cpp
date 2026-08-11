/**
 *  @file cli/smallfield_descriptors.cpp
 *  @brief Augment existing 3D conformer parquet-shards in place with shape descriptors.
 *
 *  The batch producer `smallfield_batch` emits the canonical `*.3D.parquet` schema: per-conformer USRCAT
 *  shape vectors, energy-sorted conformers, and provenance metadata. Older shards lack this schema entirely,
 *  with no `usrcat` column and conformers not energy-sorted. Re-running the GPU to regenerate them would be
 *  wasteful and would perturb coordinates; instead this tool reads each old shard, computes the missing
 *  descriptors from the @b already-stored (fp16-rounded) coordinates using the exact same native descriptor
 *  code the producer uses (`smallfield::postprocess`), and re-emits the full canonical schema through the
 *  producer's own writer (`smallfield::cli::write_shard`). Output is therefore identical to what the batch
 *  producer would have written for the same geometry.
 *
 *  Because the input coordinates in a 3D shard are already a tight interleaved `(num_atoms, 3)` fp16 list,
 *  one list per conformer row, the descriptor path here matches the producer's byte for byte: the producer
 *  also reads its fp16 buffer back before computing descriptors, so both round through fp16 identically.
 *
 *  Usage:
 *  @code
 *    smallfield_descriptors --input DIR --output DIR [--reembed DIR] [--dataset NAME] [--threads N]
 *  @endcode
 *
 *  Resumable exactly like `smallfield_batch`: a `<output>.encoded` sentinel is written after each shard
 *  closes, and a resume run skips shards whose sentinel already exists. SIGINT and SIGTERM stop the walk
 *  at the next shard boundary for the same reason — the shard in flight keeps its sentinel, so a resume
 *  run picks up exactly where this one left off.
 */

#include <csignal> // `std::signal`
#include <cstdint> // `std::uint32_t`
#include <cstdio>  // `std::fopen`
#include <cstring> // `std::memcpy`

#include <algorithm>     // `std::max`
#include <array>         // `std::array`
#include <atomic>        // `std::atomic`
#include <charconv>      // `std::from_chars`
#include <chrono>        // `std::chrono::steady_clock`
#include <filesystem>    // `std::filesystem::path`
#include <memory>        // `std::shared_ptr`
#include <span>          // `std::span`
#include <stdexcept>     // `std::runtime_error`
#include <string>        // `std::string`
#include <string_view>   // `std::string_view`
#include <system_error>  // `std::errc`
#include <unordered_map> // `std::unordered_map`
#include <vector>        // `std::vector`

#include <argparse/argparse.hpp>
#include <omp.h>
#include <cuda_fp16.h>
#include <arrow/array.h>
#include <arrow/io/file.h>
#include <arrow/table.h>
#include <parquet/arrow/reader.h>

#include "smallfield/api/conformer_layout.hpp"
#include "smallfield/log.hpp"
#include "smallfield/postprocess/fragment_separation.hpp"
#include "smallfield/postprocess/shape_descriptors.hpp"

#include "cli/parquet_writer.hpp"

namespace fs = std::filesystem;
namespace sfapi = smallfield::api;
namespace sfcli = smallfield::cli;
namespace sfpost = smallfield::postprocess;
using smallfield::logger_t;

namespace {

/** The one name this binary prints, so the announcement and the usage cannot drift from the executable. */
constexpr char const *executable_name_k = "smallfield_descriptors";

#pragma region Interruption

/** Set by SIGINT/SIGTERM. The shard walk stops at the next shard boundary, so the shard in flight still
 *  writes its parquet and its sentinel and a resume run continues from there. Async-signal-safe: a
 *  lock-free atomic flag, nothing else touched in the handler.
 *
 *  `handle_stop_signal_` keeps C language linkage for `std::signal`, which an ordinary `static` function
 *  cannot combine with, so this unnamed namespace supplies the internal linkage. */
std::atomic<bool> stop_requested_ {false};
extern "C" void handle_stop_signal_(int) { stop_requested_.store(true, std::memory_order_relaxed); }

#pragma endregion Interruption

#pragma region Configuration

/** Ceiling on `--threads`: an OpenMP team wider than this is a mistyped token, not a machine. */
constexpr int max_threads_k = 4096;

/** All CLI flags. Immutable after parse. */
struct cli_config_t {
    fs::path input_dir;
    fs::path output_dir;
    /** Optional folder of corrected shards; "" = disabled. */
    fs::path reembed_dir;
    /** Provenance dataset name (e.g. "pubchem"); "" = omit from metadata. */
    std::string dataset;
    /** OpenMP threads across molecules within a shard; resolved to the core count when the flag says 0. */
    int threads = 0;
    /** Skip shards whose `<output>.encoded` sentinel exists. */
    bool resume = true;
    /** Coord stride / provenance `max_atoms`. */
    std::uint32_t max_atoms = sfapi::max_atoms_per_molecule_k;
};

void print_usage_() {
    fmt::print(stderr, "Usage: {} --input DIR --output DIR [options]\n", executable_name_k);
    fmt::print(stderr, R"(  --input DIR       folder of 3D parquet-shards to augment (*.3D.parquet)
  --output DIR      folder for augmented output shards (*.3D.parquet)
  --reembed DIR     folder of corrected 13-column shards; molecules present there are
                    copied verbatim instead of augmented (e.g. re-embedded multi-fragment)
  --dataset NAME    provenance dataset name embedded in output metadata (e.g. pubchem)
  --threads N       OpenMP threads across molecules within a shard (0 = all cores, the default)
  --max-atoms N     per-molecule atom cap / coord stride (default: 192)
  --restart         overwrite already-complete output shards
)");
}

/** Outcome of a command line: a usable configuration, a help request, or a rejected command line. */
enum class cli_parse_t {
    parsed_k,
    help_requested_k,
    invalid_k,
};

/** Why a command-line token was accepted or refused as a number. */
enum class number_parse_t {
    /** The whole token is a number inside the accepted range. */
    parsed_k,
    /** The token is not a number, or carries trailing characters. */
    malformed_k,
    /** The token is a number the destination type cannot hold. */
    out_of_range_k,
    /** The token parses but falls under the caller's floor. */
    below_minimum_k,
    /** The token parses but exceeds the caller's ceiling. */
    above_maximum_k,
};

/**
 *  @brief Parse @p text whole as a number in `[minimum, maximum]`, leaving @p number untouched on refusal.
 *
 *  Partial parses are refusals: `12x` is malformed, not twelve. The destination keeps its prior value unless
 *  the token is accepted, so a caller may seed it with the default and read it back either way.
 *
 *  @param[in] text Token as it appeared on the command line.
 *  @param[in] minimum Smallest value the flag accepts.
 *  @param[in] maximum Largest value the flag accepts.
 *  @param[out] number Value the token denotes; untouched unless the whole token parses inside the bounds.
 */
template <typename number_type_>
number_parse_t parse_number_(std::string_view text, number_type_ minimum, number_type_ maximum, number_type_ &number) {
    number_type_ parsed {};
    auto const [unparsed, error] = std::from_chars(text.data(), text.data() + text.size(), parsed);
    if (error == std::errc::result_out_of_range) return number_parse_t::out_of_range_k;
    if (error != std::errc {} || unparsed != text.data() + text.size()) return number_parse_t::malformed_k;
    if (parsed < minimum) return number_parse_t::below_minimum_k;
    if (parsed > maximum) return number_parse_t::above_maximum_k;
    number = parsed;
    return number_parse_t::parsed_k;
}

/** Sentence fragment naming the refusal, meant to follow the flag name in a diagnostic. */
char const *describe_number_parse_(number_parse_t status) {
    switch (status) {
    case number_parse_t::parsed_k: return "parsed";
    case number_parse_t::malformed_k: return "is not a whole number";
    case number_parse_t::out_of_range_k: return "does not fit the destination type";
    case number_parse_t::below_minimum_k: return "is below the accepted minimum";
    case number_parse_t::above_maximum_k: return "is above the accepted maximum";
    }
    return "was refused";
}

/**
 *  @brief Bind a string flag straight to its destination, so the parsed value never sits anywhere but the config.
 *
 *  `append` marks the flag repeatable, which is what keeps the last occurrence winning: without it `argparse` rejects
 *  a second occurrence outright, and a wrapper that appends an override to a fixed prefix stops working.
 */
template <typename string_type_>
void add_string_(argparse::ArgumentParser &parser, char const *flag, string_type_ &destination) {
    parser.add_argument(flag).append().action([&destination](std::string const &text) { destination = text; });
}

/**
 *  @brief Bind a numeric flag bounded by `[minimum, maximum]`, naming both the flag and the reason on refusal.
 *
 *  The conversion stays ours rather than `argparse`'s `scan`, whose diagnostic is `pattern 'abc' does not match to
 *  the end` — it never names the flag the operator actually typed, and it has no bounds to enforce.
 */
template <typename number_type_>
void add_number_(argparse::ArgumentParser &parser, char const *flag, number_type_ minimum, number_type_ maximum,
                 number_type_ &number) {
    parser.add_argument(flag).append().action([flag, minimum, maximum, &number](std::string const &text) {
        number_parse_t const status = parse_number_(text, minimum, maximum, number);
        if (status != number_parse_t::parsed_k)
            throw std::runtime_error(fmt::format("{} {}, got: {}", flag, describe_number_parse_(status), text));
    });
}

cli_parse_t parse_cli_(int argc, char **argv, cli_config_t &config) {
    // `default_arguments::none` suppresses the `-h` and `--version` that `argparse` would otherwise supply, so
    // `print_usage_` stays the one help text and `-h` is bound by hand below.
    argparse::ArgumentParser parser {executable_name_k, "", argparse::default_arguments::none};
    bool help_requested = false;

    add_string_(parser, "--input", config.input_dir);
    add_string_(parser, "--output", config.output_dir);
    add_string_(parser, "--reembed", config.reembed_dir);
    add_string_(parser, "--dataset", config.dataset);
    // Zero threads is the documented "all cores", resolved below; the atom cap cannot exceed what the
    // kernels and the interleaved scratch here are dimensioned for.
    add_number_(parser, "--threads", 0, max_threads_k, config.threads);
    add_number_(parser, "--max-atoms", 1u, sfapi::max_atoms_per_molecule_k, config.max_atoms);
    parser.add_argument("--restart").flag().append().action([&config](std::string const &) { config.resume = false; });
    parser.add_argument("-h", "--help").flag().append().action([&](std::string const &) {
        print_usage_();
        help_requested = true;
    });

    try {
        parser.parse_args(argc, argv);
    }
    catch (std::exception const &error) {
        // `argparse` reports every rejection by throwing, so this is the only place a bad command line surfaces.
        if (help_requested) return cli_parse_t::help_requested_k;
        fmt::print(stderr, "{}\n", error.what());
        return cli_parse_t::invalid_k;
    }
    if (help_requested) return cli_parse_t::help_requested_k;

    if (config.input_dir.empty() || config.output_dir.empty()) {
        print_usage_();
        return cli_parse_t::invalid_k;
    }
    // Resolve the thread count once, here, so no consumer has to re-derive what zero meant.
    if (config.threads == 0) config.threads = std::max(1, omp_get_max_threads());
    return cli_parse_t::parsed_k;
}

#pragma endregion Configuration

#pragma region fp16 Conversion

/** Decode one fp16 bit pattern (as stored by Arrow `HalfFloat`) to a host float. */
inline float half_bits_to_float_(std::uint16_t bits) {
    __half value;
    std::memcpy(&value, &bits, sizeof(value));
    return __half2float(value);
}

#pragma endregion fp16 Conversion

#pragma region Shard Reading

/**
 *  @brief A single input 3D shard read into typed columns, keeping the Arrow table alive.
 *
 *  The shard is single-chunked via `CombineChunks` so every column is one contiguous
 *  array and per-row access is a plain index. SMILES and the input-shard stem are held
 *  as `string_view`s into the table's buffers, so the table must outlive this struct.
 */
struct input_shard_t {
    std::shared_ptr<arrow::Table> table;
    arrow::StringArray const *input_shard = nullptr;
    arrow::UInt64Array const *input_row = nullptr;
    arrow::StringArray const *smiles = nullptr;
    arrow::UInt8Array const *conformer_index = nullptr;
    arrow::UInt8Array const *status = nullptr;
    arrow::UInt16Array const *n_heavy_atoms = nullptr;
    arrow::UInt16Array const *n_atoms = nullptr;
    arrow::UInt16Array const *n_bonds = nullptr;
    arrow::FloatArray const *molecular_weight = nullptr;
    arrow::UInt8Array const *n_conformers = nullptr;
    arrow::ListArray const *conformer_coords = nullptr;
    arrow::HalfFloatArray const *coords_values = nullptr;
    arrow::FloatArray const *conformer_energy = nullptr;
    std::int64_t num_rows = 0;
};

template <typename array_type_>
array_type_ const *bind_column_(arrow::Table const &table, char const *name) {
    auto column = table.GetColumnByName(name);
    if (!column || column->num_chunks() != 1) return nullptr;
    return dynamic_cast<array_type_ const *>(column->chunk(0).get());
}

/** Why one shard was accepted or refused, so a failing run names which of the four went wrong. */
enum class shard_read_t {
    /** Every column this tool reads bound to an array of the expected type. */
    read_k,
    /** The path did not open. */
    cannot_open_k,
    /** The file opened but no Parquet reader could be built over it. */
    not_parquet_k,
    /** The Parquet reader could not materialise the table. */
    unreadable_table_k,
    /** The table is missing a column this tool needs, or a column has an unexpected type. */
    schema_mismatch_k,
};

/**
 *  @brief Read @p path into @p out, binding every column the augmentation walks.
 *
 *  @param[out] out Caller-owned shard, holding the Arrow table alive behind the bound column pointers.
 */
shard_read_t read_input_shard_(fs::path const &path, input_shard_t &out) {
    auto open_result = arrow::io::ReadableFile::Open(path.string(), arrow::default_memory_pool());
    if (!open_result.ok()) return shard_read_t::cannot_open_k;
    parquet::arrow::FileReaderBuilder builder;
    if (!builder.Open(open_result.ValueOrDie()).ok()) return shard_read_t::not_parquet_k;
    auto reader_result = builder.Build();
    if (!reader_result.ok()) return shard_read_t::not_parquet_k;
    auto reader = std::move(reader_result).ValueOrDie();

    std::shared_ptr<arrow::Table> table;
    if (!reader->ReadTable(&table).ok() || !table) return shard_read_t::unreadable_table_k;
    auto combined = table->CombineChunks(arrow::default_memory_pool());
    if (!combined.ok()) return shard_read_t::unreadable_table_k;
    out.table = combined.ValueOrDie();

    out.input_shard = bind_column_<arrow::StringArray>(*out.table, "input_shard");
    out.input_row = bind_column_<arrow::UInt64Array>(*out.table, "input_row");
    out.smiles = bind_column_<arrow::StringArray>(*out.table, "smiles");
    out.conformer_index = bind_column_<arrow::UInt8Array>(*out.table, "conformer_index");
    out.status = bind_column_<arrow::UInt8Array>(*out.table, "status");
    out.n_heavy_atoms = bind_column_<arrow::UInt16Array>(*out.table, "n_heavy_atoms");
    out.n_atoms = bind_column_<arrow::UInt16Array>(*out.table, "n_atoms");
    out.n_bonds = bind_column_<arrow::UInt16Array>(*out.table, "n_bonds");
    out.molecular_weight = bind_column_<arrow::FloatArray>(*out.table, "molecular_weight");
    out.n_conformers = bind_column_<arrow::UInt8Array>(*out.table, "n_conformers");
    out.conformer_coords = bind_column_<arrow::ListArray>(*out.table, "conformer_coords");
    out.conformer_energy = bind_column_<arrow::FloatArray>(*out.table, "conformer_energy");
    out.num_rows = out.table->num_rows();

    if (!out.input_shard || !out.input_row || !out.smiles || !out.conformer_index || !out.status || !out.n_atoms ||
        !out.n_conformers || !out.conformer_coords || !out.conformer_energy)
        return shard_read_t::schema_mismatch_k;
    out.coords_values = dynamic_cast<arrow::HalfFloatArray const *>(out.conformer_coords->values().get());
    if (!out.coords_values) return shard_read_t::schema_mismatch_k;
    return shard_read_t::read_k;
}

/** Sentence fragment naming the refusal, meant to follow the shard's path in a diagnostic. */
char const *describe_shard_read_(shard_read_t status) {
    switch (status) {
    case shard_read_t::read_k: return "read";
    case shard_read_t::cannot_open_k: return "cannot be opened";
    case shard_read_t::not_parquet_k: return "is not a readable parquet file";
    case shard_read_t::unreadable_table_k: return "holds a table Arrow could not materialise";
    case shard_read_t::schema_mismatch_k: return "is missing a column of the 3D schema, or types one differently";
    }
    return "was refused";
}

/** Why a folder named on the command line was accepted or refused as a source of shards. */
enum class shard_listing_t {
    /** The folder opened and held at least one `*.3D.parquet`. */
    listed_k,
    /** The folder does not exist, or could not be walked. */
    cannot_open_k,
    /** The folder opened and held no `*.3D.parquet`. */
    empty_k,
};

/**
 *  @brief Collect every `*.3D.parquet` in @p directory into @p out, sorted lexicographically.
 *
 *  Here a `*.3D.parquet` is an @b input, the opposite of `smallfield_batch` where the same name marks
 *  an output. The `.encoded` sentinels sitting beside them are skipped by the suffix test. The walk
 *  reports its refusal rather than throwing, because a folder the operator named and misspelled has to
 *  reach the exit code, not `std::terminate`.
 *
 *  @param[in] directory Folder to walk, as the operator spelled it.
 *  @param[out] out Caller-owned destination, cleared first.
 */
shard_listing_t fill_shard_paths_into_(fs::path const &directory, std::vector<fs::path> &out) {
    out.clear();
    std::error_code error;
    fs::directory_iterator entry {directory, error};
    if (error) return shard_listing_t::cannot_open_k;
    for (fs::directory_iterator const end; entry != end;) {
        std::string const name = entry->path().filename().string();
        if (name.ends_with(".3D.parquet")) out.push_back(entry->path());
        entry.increment(error);
        if (error) return shard_listing_t::cannot_open_k;
    }
    std::sort(out.begin(), out.end());
    return out.empty() ? shard_listing_t::empty_k : shard_listing_t::listed_k;
}

/** Sentence fragment naming the refusal, meant to follow the folder's path in a diagnostic. */
char const *describe_shard_listing_(shard_listing_t status) {
    switch (status) {
    case shard_listing_t::listed_k: return "listed";
    case shard_listing_t::cannot_open_k: return "cannot be opened";
    case shard_listing_t::empty_k: return "holds no *.3D.parquet shards";
    }
    return "was refused";
}

#pragma endregion Shard Reading

#pragma region Reembed Store

/** One conformer row copied verbatim from a `--reembed` shard. */
struct reembed_conformer_t {
    std::uint8_t status = 0;
    float energy = 0.0f;
    bool has_coords = false;
    /** Tight interleaved (n_atoms, 3) fp16 bit patterns. */
    std::vector<std::uint16_t> coord_bits;
    /** USR is the first 12 of USRCAT, not stored separately. */
    std::array<double, 60> usrcat {};
};

/** One molecule copied verbatim from a `--reembed` shard (its already-corrected geometry). */
struct reembed_molecule_t {
    std::uint8_t num_atoms = 0;
    std::uint16_t n_heavy_atoms = 0;
    std::uint16_t n_bonds = 0;
    float molecular_weight = 0.0f;
    std::uint8_t n_conformers = 0;
    std::string smiles;
    std::vector<reembed_conformer_t> conformers;
};

/**
 *  @brief Every molecule a `--reembed` run corrected, keyed by `(input_shard, input_row)`.
 *
 *  `widest_molecule_conformers` travels with the map because the slot stride of every output shard has to
 *  cover the widest reembed molecule: derived once here, it would otherwise be a full walk of the map per
 *  shard, which is the whole index re-scanned once per bucket.
 */
struct reembed_store_t {
    std::unordered_map<std::string, reembed_molecule_t> molecules;
    std::uint32_t widest_molecule_conformers = 0;
};

/**
 *  @brief Compose the `(input_shard, input_row)` identity key reembed molecules are matched on.
 *
 *  Filled into caller-owned storage rather than returned: the augmentation builds one key per molecule
 *  inside its parallel region, so a returned string would be a heap allocation per molecule, freed on
 *  the next line once the lookup is done. Reusing one buffer per worker keeps the capacity instead.
 *
 *  @param[in] input_shard Stem of the shard the molecule came from.
 *  @param[in] input_row Row the molecule occupied in that shard.
 *  @param[out] out Caller-owned destination, cleared first; its capacity survives the call.
 */
inline void fill_reembed_key_into_(std::string_view input_shard, std::uint64_t input_row, std::string &out) {
    // 20 digits is the widest `std::uint64_t`, so the conversion never runs out of room.
    std::array<char, 20> digits {};
    [[maybe_unused]] auto const [digits_end, digits_error] = std::to_chars(digits.data(), digits.data() + digits.size(),
                                                                           input_row);
    out.clear();
    out.append(input_shard);
    out.push_back('\0');
    out.append(digits.data(), static_cast<std::size_t>(digits_end - digits.data()));
}

/**
 *  @brief Copy one reembed shard's molecules into @p store, keyed by `(input_shard, input_row)`.
 *
 *  @param[in] max_atoms Coordinate stride an output slot has; a molecule claiming more atoms than this
 *  cannot be laid into one, so it is carried with its geometry dropped rather than narrowed into
 *  another molecule's atoms.
 */
void load_reembed_shard_(input_shard_t const &shard, std::uint32_t max_atoms, reembed_store_t &store) {
    arrow::HalfFloatArray const &coords_values = *shard.coords_values;
    arrow::ListArray const &coords_list = *shard.conformer_coords;
    // USRCAT is optional on a reembed shard; bind once so the verbatim path stays truly verbatim
    // when present, and leaves zeros when the shard predates it. Stored as fp16 (list<float16>).
    auto const *usrcat_column = bind_column_<arrow::ListArray>(*shard.table, "usrcat");
    auto const *usrcat_values = usrcat_column
                                    ? dynamic_cast<arrow::HalfFloatArray const *>(usrcat_column->values().get())
                                    : nullptr;
    std::string key;
    for (std::int64_t row = 0; row < shard.num_rows; ++row) {
        std::string_view const stem = shard.input_shard->GetView(row);
        std::uint64_t const input_row = shard.input_row->Value(row);
        fill_reembed_key_into_(stem, input_row, key);

        reembed_molecule_t &molecule = store.molecules[key];
        if (molecule.conformers.empty()) {
            molecule.smiles = shard.smiles->GetString(row);
            std::uint16_t const stored_atoms = shard.n_atoms->IsValid(row) ? shard.n_atoms->Value(row) : 0;
            molecule.num_atoms = stored_atoms <= max_atoms ? static_cast<std::uint8_t>(stored_atoms) : 0;
            molecule.n_heavy_atoms = shard.n_heavy_atoms && shard.n_heavy_atoms->IsValid(row)
                                         ? shard.n_heavy_atoms->Value(row)
                                         : 0;
            molecule.n_bonds = shard.n_bonds && shard.n_bonds->IsValid(row) ? shard.n_bonds->Value(row) : 0;
            molecule.molecular_weight = shard.molecular_weight && shard.molecular_weight->IsValid(row)
                                            ? shard.molecular_weight->Value(row)
                                            : 0.0f;
            molecule.n_conformers = shard.n_conformers->IsValid(row) ? shard.n_conformers->Value(row) : 0;
        }

        reembed_conformer_t conformer;
        conformer.status = shard.status->Value(row);
        conformer.has_coords = coords_list.IsValid(row);
        if (conformer.has_coords) {
            std::int64_t const begin = coords_list.value_offset(row);
            std::int64_t const end = coords_list.value_offset(row + 1);
            conformer.coord_bits.reserve(static_cast<std::size_t>(end - begin));
            for (std::int64_t value_index = begin; value_index < end; ++value_index)
                conformer.coord_bits.push_back(coords_values.Value(value_index));
        }
        if (shard.conformer_energy->IsValid(row)) conformer.energy = shard.conformer_energy->Value(row);

        if (usrcat_values && usrcat_column->IsValid(row)) {
            std::int64_t const begin = usrcat_column->value_offset(row);
            if (usrcat_column->value_offset(row + 1) - begin >= 60)
                for (std::size_t component = 0; component < 60; ++component)
                    conformer.usrcat[component] = half_bits_to_float_(
                        usrcat_values->Value(begin + static_cast<std::int64_t>(component)));
        }

        molecule.conformers.push_back(std::move(conformer));
        store.widest_molecule_conformers = std::max(store.widest_molecule_conformers,
                                                    static_cast<std::uint32_t>(molecule.conformers.size()));
    }
}

/**
 *  @brief Read every shard under @p reembed_dir into @p store.
 *
 *  @param[out] store Caller-owned index, filled in place; a shard that fails to read is reported and skipped.
 *  @return Whether the folder itself could be listed, so a misspelled `--reembed` reaches the exit code.
 */
shard_listing_t fill_reembed_store_(fs::path const &reembed_dir, std::uint32_t max_atoms, logger_t const &logger,
                                    reembed_store_t &store) {
    std::vector<fs::path> shards;
    shard_listing_t const listing = fill_shard_paths_into_(reembed_dir, shards);
    if (listing == shard_listing_t::cannot_open_k) return listing;
    for (auto const &path : shards) {
        input_shard_t shard;
        shard_read_t const status = read_input_shard_(path, shard);
        if (status != shard_read_t::read_k) {
            logger.error("reembed shard {} {}, skipping", path.string(), describe_shard_read_(status));
            continue;
        }
        load_reembed_shard_(shard, max_atoms, store);
    }
    logger.info("reembed index: {} molecules from {} shards", fmt::group_digits(store.molecules.size()),
                fmt::group_digits(shards.size()));
    return listing;
}

#pragma endregion Reembed Store

#pragma region Shard Augmentation

/** Per-shard counters folded into the progress log. */
struct shard_stats_t {
    /** Molecules whose SMILES parse did not match stored n_atoms. */
    std::atomic<std::uint64_t> context_mismatch {0};
    std::atomic<std::uint64_t> fragments_separated {0};
};

/** Grow @p buffer to hold @p count elements, never shrinking, so a run allocates its peak once. */
template <typename element_type_>
void grow_if_needed_(std::vector<element_type_> &buffer, std::size_t count) {
    if (buffer.size() < count) buffer.resize(count);
}

/**
 *  @brief Every array one shard's augmentation writes, owned by the shard walk and reused across shards.
 *
 *  The slot arrays mirror the layout the GPU producer hands to `write_shard`: structure-of-arrays
 *  coordinates addressed through `conformer_slot_layout_t`, and one entry per conformer slot for status,
 *  energy and the 60-component USRCAT vector. `molecule_rows` is the shard's rows bucketed by molecule as
 *  a flat arena with `molecule_row_offsets` delimiting each bucket — a vector per molecule would be an
 *  allocation per molecule, a million of them on a production shard.
 */
struct shard_buffers_t {
    std::vector<__half> coordinates;
    std::vector<float> energies;
    std::vector<std::uint8_t> status_codes;
    std::vector<double> usrcat;

    std::vector<std::string_view> smiles;
    std::vector<std::uint8_t> num_atoms_per_molecule;
    std::vector<std::uint16_t> num_heavy_atoms_per_molecule;
    std::vector<std::uint16_t> num_bonds_per_molecule;
    std::vector<float> molecular_weight_per_molecule;
    std::vector<std::uint8_t> n_conformers_per_molecule;

    /** `num_molecules + 1` bucket bounds into `molecule_rows`. */
    std::vector<std::uint64_t> molecule_row_offsets;
    /** Write cursors used while bucketing, one per molecule. */
    std::vector<std::uint64_t> molecule_row_cursors;
    /** Every row of the shard, grouped by molecule and in file order within a molecule. */
    std::vector<std::int64_t> molecule_rows;

    /**
     *  @brief Grow to fit one shard and put the used prefix back into its start-of-shard state.
     *
     *  Capacity is never handed back, so the shard after a wide one reuses the wide allocation. Every
     *  element the writer can read is written here, because a stale `smiles` view would otherwise point
     *  into the previous shard's released Arrow table.
     */
    void reset_for(std::uint64_t num_molecules, std::uint32_t max_conformers, std::uint32_t max_atoms) {
        std::size_t const total_slots = static_cast<std::size_t>(num_molecules) * max_conformers;
        std::size_t const coordinate_total = total_slots * static_cast<std::size_t>(max_atoms) * 3u;
        grow_if_needed_(coordinates, coordinate_total);
        grow_if_needed_(energies, total_slots);
        grow_if_needed_(status_codes, total_slots);
        grow_if_needed_(usrcat, total_slots * 60u);
        grow_if_needed_(smiles, num_molecules);
        grow_if_needed_(num_atoms_per_molecule, num_molecules);
        grow_if_needed_(num_heavy_atoms_per_molecule, num_molecules);
        grow_if_needed_(num_bonds_per_molecule, num_molecules);
        grow_if_needed_(molecular_weight_per_molecule, num_molecules);
        grow_if_needed_(n_conformers_per_molecule, num_molecules);

        std::fill_n(coordinates.begin(), coordinate_total, __float2half(0.0f));
        std::fill_n(energies.begin(), total_slots, 0.0f);
        // A slot no molecule claims must not read as a success, so the unused tail defaults non-success.
        std::fill_n(status_codes.begin(), total_slots, static_cast<std::uint8_t>(1));
        std::fill_n(usrcat.begin(), total_slots * 60u, 0.0);
        std::fill_n(smiles.begin(), num_molecules, std::string_view {});
        std::fill_n(num_atoms_per_molecule.begin(), num_molecules, static_cast<std::uint8_t>(0));
        std::fill_n(num_heavy_atoms_per_molecule.begin(), num_molecules, static_cast<std::uint16_t>(0));
        std::fill_n(num_bonds_per_molecule.begin(), num_molecules, static_cast<std::uint16_t>(0));
        std::fill_n(molecular_weight_per_molecule.begin(), num_molecules, 0.0f);
        std::fill_n(n_conformers_per_molecule.begin(), num_molecules, static_cast<std::uint8_t>(0));
    }
};

/**
 *  @brief Bucket the shard's rows by molecule into @p buffers, keeping file order inside each bucket.
 *
 *  File order is conformer_index order within a molecule, which is what makes slot `conformer_slot` of the
 *  output the same conformer the input row described.
 *
 *  @return Row count of the widest molecule, never below one.
 */
std::uint32_t group_rows_by_molecule_(input_shard_t const &shard, std::uint64_t base_row, std::uint64_t num_molecules,
                                      shard_buffers_t &buffers) {
    grow_if_needed_(buffers.molecule_row_offsets, num_molecules + 1);
    grow_if_needed_(buffers.molecule_row_cursors, num_molecules);
    grow_if_needed_(buffers.molecule_rows, static_cast<std::size_t>(shard.num_rows));
    std::fill_n(buffers.molecule_row_offsets.begin(), num_molecules + 1, 0ull);

    for (std::int64_t row = 0; row < shard.num_rows; ++row)
        ++buffers.molecule_row_offsets[shard.input_row->Value(row) - base_row + 1];

    std::uint32_t widest = 1;
    for (std::uint64_t molecule = 0; molecule < num_molecules; ++molecule) {
        widest = std::max(widest, static_cast<std::uint32_t>(buffers.molecule_row_offsets[molecule + 1]));
        buffers.molecule_row_offsets[molecule + 1] += buffers.molecule_row_offsets[molecule];
    }

    std::copy_n(buffers.molecule_row_offsets.begin(), num_molecules, buffers.molecule_row_cursors.begin());
    for (std::int64_t row = 0; row < shard.num_rows; ++row)
        buffers.molecule_rows[buffers.molecule_row_cursors[shard.input_row->Value(row) - base_row]++] = row;
    return widest;
}

/** How one shard's augmentation ended, so an empty input and a refused write stay distinguishable. */
enum class augment_t {
    written_k,
    empty_input_k,
    write_failed_k,
};

/**
 *  @brief Augment one input shard and write the canonical output shard.
 *
 *  Rows are grouped by `input_row`; molecules are contiguous, so `input_row = base_row_index + molecule_index`.
 *  Every molecule's conformer rows are laid into structure-of-arrays conformer slots exactly as the GPU
 *  producer would, so the producer's own `sfcli::write_shard` can energy-sort, rank `conformer_index`, couple
 *  nulls, and pack descriptors — guaranteeing identical output bytes. Descriptors are computed from the stored
 *  fp16 coordinates via the native `sfpost` path; multi-fragment molecules are `separate_fragments`'d and their
 *  corrected coordinates written back. Molecules found in @p reembed are copied verbatim from there instead.
 *
 *  @param[inout] buffers Slot arrays for this shard, grown and reset in place; owned by the shard walk.
 */
augment_t augment_shard_(input_shard_t const &shard, cli_config_t const &config, reembed_store_t const &reembed,
                         shard_buffers_t &buffers, fs::path const &output_path, logger_t const &logger,
                         shard_stats_t &stats) {
    if (shard.num_rows == 0) {
        logger.warn("empty shard, skipping: {}", output_path.string());
        return augment_t::empty_input_k;
    }

    // Contiguous molecule span: input_row = base_row + molecule_index. The 3D schema guarantees rows are
    // grouped and input_row is contiguous, so every molecule owns at least one row.
    std::uint64_t base_row = shard.input_row->Value(0);
    std::uint64_t max_row = base_row;
    for (std::int64_t row = 0; row < shard.num_rows; ++row) {
        std::uint64_t const value = shard.input_row->Value(row);
        base_row = std::min(base_row, value);
        max_row = std::max(max_row, value);
    }
    std::uint64_t const num_molecules = max_row - base_row + 1;

    // The slot stride K must cover the widest molecule (original or reembed) and match the
    // producer's `num_conformers` provenance value. Reembed molecules can only add rows.
    std::uint32_t const observed_conformers = group_rows_by_molecule_(shard, base_row, num_molecules, buffers);
    std::uint32_t const max_conformers = std::max(observed_conformers, reembed.widest_molecule_conformers);

    std::uint32_t const max_atoms = config.max_atoms;
    sfapi::conformer_slot_layout_t const slot_layout {max_atoms, max_conformers};
    std::size_t const total_slots = static_cast<std::size_t>(num_molecules) * max_conformers;
    buffers.reset_for(num_molecules, max_conformers, max_atoms);

    arrow::HalfFloatArray const &coords_values = *shard.coords_values;
    arrow::ListArray const &coords_list = *shard.conformer_coords;

    // Per-worker scratch for the reembed lookup key: `private` hands each thread its own string, which then
    // keeps its capacity across every molecule that thread walks. Declaring it inside the body instead would
    // put a heap allocation and a free on the descriptor path once per molecule.
    std::string reembed_key;

#pragma omp parallel for schedule(dynamic, 256) num_threads(config.threads) private(reembed_key) if (config.threads > 1)
    for (std::uint64_t molecule_index = 0; molecule_index < num_molecules; ++molecule_index) {
        std::uint64_t const rows_begin = buffers.molecule_row_offsets[molecule_index];
        std::uint64_t const rows_end = buffers.molecule_row_offsets[molecule_index + 1];
        if (rows_begin == rows_end) continue; // contiguous input guarantees this never fires; defensive only
        std::span<std::int64_t const> const rows {buffers.molecule_rows.data() + rows_begin,
                                                  static_cast<std::size_t>(rows_end - rows_begin)};
        std::int64_t const first_row = rows.front();
        std::string_view const stem = shard.input_shard->GetView(first_row);
        std::uint64_t const input_row = shard.input_row->Value(first_row);

        // Reembed override: copy the corrected molecule verbatim into this molecule's slots.
        fill_reembed_key_into_(stem, input_row, reembed_key);
        if (auto found = reembed.molecules.find(reembed_key); found != reembed.molecules.end()) {
            reembed_molecule_t const &molecule = found->second;
            // string_view into the reembed store's persistent storage, which outlives every shard
            buffers.smiles[molecule_index] = molecule.smiles;
            buffers.num_atoms_per_molecule[molecule_index] = molecule.num_atoms;
            buffers.num_heavy_atoms_per_molecule[molecule_index] = molecule.n_heavy_atoms;
            buffers.num_bonds_per_molecule[molecule_index] = molecule.n_bonds;
            buffers.molecular_weight_per_molecule[molecule_index] = molecule.molecular_weight;
            buffers.n_conformers_per_molecule[molecule_index] = molecule.n_conformers;
            std::uint32_t const num_atoms = molecule.num_atoms;
            for (std::size_t conformer_slot = 0;
                 conformer_slot < molecule.conformers.size() && conformer_slot < max_conformers; ++conformer_slot) {
                reembed_conformer_t const &conformer = molecule.conformers[conformer_slot];
                std::size_t const flat = static_cast<std::size_t>(molecule_index) * max_conformers + conformer_slot;
                buffers.status_codes[flat] = conformer.status;
                buffers.energies[flat] = conformer.energy;
                if (!conformer.has_coords) continue; // the reembed row carried no geometry
                if (conformer.status != 0) continue; // a failure row stays null in the output
                if (num_atoms == 0) continue;        // preprocess failed: no coordinates to lay out
                // the row's own geometry is shorter than the atom count it claims
                if (conformer.coord_bits.size() < static_cast<std::size_t>(num_atoms) * 3u) continue;
                std::size_t const slot_base = slot_layout.slot_offset(molecule_index,
                                                                      static_cast<std::uint32_t>(conformer_slot));
                sfcli::scatter_interleaved_into_slot(conformer.coord_bits.data(), slot_layout, num_atoms,
                                                     buffers.coordinates.data() + slot_base);
                std::copy(conformer.usrcat.begin(), conformer.usrcat.end(), buffers.usrcat.data() + flat * 60u);
            }
            continue;
        }

        // Augment path: per-molecule scalars come straight from the original shard.
        buffers.smiles[molecule_index] = shard.smiles->GetView(first_row);
        std::uint16_t const stored_atoms = shard.n_atoms->IsValid(first_row) ? shard.n_atoms->Value(first_row) : 0;
        // A molecule wider than the coordinate stride cannot be laid into a slot, and narrowing its count
        // to the stride would describe another molecule's atoms; treat it exactly like a preprocess failure.
        std::uint8_t const num_atoms_u8 = stored_atoms <= max_atoms ? static_cast<std::uint8_t>(stored_atoms) : 0;
        buffers.num_atoms_per_molecule[molecule_index] = num_atoms_u8;
        buffers.num_heavy_atoms_per_molecule[molecule_index] =
            shard.n_heavy_atoms && shard.n_heavy_atoms->IsValid(first_row) ? shard.n_heavy_atoms->Value(first_row) : 0;
        buffers.num_bonds_per_molecule[molecule_index] = shard.n_bonds && shard.n_bonds->IsValid(first_row)
                                                             ? shard.n_bonds->Value(first_row)
                                                             : 0;
        buffers.molecular_weight_per_molecule[molecule_index] = shard.molecular_weight &&
                                                                        shard.molecular_weight->IsValid(first_row)
                                                                    ? shard.molecular_weight->Value(first_row)
                                                                    : 0.0f;
        buffers.n_conformers_per_molecule[molecule_index] = shard.n_conformers->IsValid(first_row)
                                                                ? shard.n_conformers->Value(first_row)
                                                                : 0;

        if (num_atoms_u8 == 0) continue; // preprocess-failed molecule: single null row, no coords

        // Type the molecule once (same parse contract the producer uses). A mismatch between
        // the reparsed atom count and the stored n_atoms means the coordinate order can't be
        // trusted for descriptors — keep the coordinates, emit zero-filled descriptors, and count.
        sfpost::molecule_shape_context_t const context = sfpost::build_shape_context(buffers.smiles[molecule_index]);
        bool const context_usable = context.ok && context.num_atoms == num_atoms_u8;
        if (!context_usable) stats.context_mismatch.fetch_add(1, std::memory_order_relaxed);
        std::uint32_t const num_atoms = num_atoms_u8;

        // Every conformer slot of this molecule is written below, so the loop runs to completion even
        // when an early row is a failure: a `break` would leave the later slots holding the reset value.
        std::array<float, sfapi::max_atoms_per_molecule_k * 3u> interleaved {};
        for (std::size_t conformer_slot = 0; conformer_slot < rows.size() && conformer_slot < max_conformers;
             ++conformer_slot) {
            std::int64_t const row = rows[conformer_slot];
            std::uint8_t const row_status = shard.status->Value(row);
            std::size_t const flat = static_cast<std::size_t>(molecule_index) * max_conformers + conformer_slot;
            buffers.status_codes[flat] = row_status;
            if (shard.conformer_energy->IsValid(row)) buffers.energies[flat] = shard.conformer_energy->Value(row);
            if (row_status != 0 || !coords_list.IsValid(row)) continue; // failure row stays null

            std::size_t const begin = static_cast<std::size_t>(coords_list.value_offset(row));
            std::size_t const stored = static_cast<std::size_t>(coords_list.value_offset(row + 1)) - begin;
            std::size_t const wanted = static_cast<std::size_t>(num_atoms) * 3u;
            std::size_t const copy = std::min(stored, wanted);
            for (std::size_t value_index = 0; value_index < copy; ++value_index)
                interleaved[value_index] = half_bits_to_float_(
                    coords_values.Value(static_cast<std::int64_t>(begin + value_index)));

            std::size_t const slot_base = slot_layout.slot_offset(molecule_index,
                                                                  static_cast<std::uint32_t>(conformer_slot));
            std::span<float> coords_view {interleaved.data(), wanted};

            // Multi-fragment safety net: separate disconnected components and write the corrected
            // geometry back, mirroring the producer. Only when the context is usable (needs labels).
            if (context_usable && context.num_components > 1) {
                sfpost::separate_fragments(coords_view, context);
                stats.fragments_separated.fetch_add(1, std::memory_order_relaxed);
            }

            sfcli::scatter_interleaved_into_slot(interleaved.data(), slot_layout, num_atoms,
                                                 buffers.coordinates.data() + slot_base);

            if (context_usable) {
                std::array<double, 60> usrcat_row {};
                sfpost::usrcat_descriptor(coords_view, context, usrcat_row);
                std::copy(usrcat_row.begin(), usrcat_row.end(), buffers.usrcat.data() + flat * 60u);
            }
            // else: descriptors stay zero-filled (not null), coordinates preserved.
        }
    }

    // Hand the reconstructed slot layout to the producer's writer for byte-identical output. Each span
    // carries the extent this shard actually filled, which is below the buffers' own size whenever an
    // earlier shard was the wider one.
    sfcli::writer_inputs_t const writer_inputs {
        .input_shard_stem = std::string(shard.input_shard->GetView(0)),
        .dataset = config.dataset,
        .base_row_index = base_row,
        .num_molecules = num_molecules,
        .max_conformers = max_conformers,
        .max_atoms = max_atoms,
        .smiles = std::span<std::string_view const>(buffers.smiles.data(), num_molecules),
        .num_atoms_per_molecule = std::span<std::uint8_t const>(buffers.num_atoms_per_molecule.data(), num_molecules),
        .num_heavy_atoms_per_molecule = std::span<std::uint16_t const>(buffers.num_heavy_atoms_per_molecule.data(),
                                                                       num_molecules),
        .num_bonds_per_molecule = std::span<std::uint16_t const>(buffers.num_bonds_per_molecule.data(), num_molecules),
        .molecular_weight_per_molecule = std::span<float const>(buffers.molecular_weight_per_molecule.data(),
                                                                num_molecules),
        .n_conformers_per_molecule = std::span<std::uint8_t const>(buffers.n_conformers_per_molecule.data(),
                                                                   num_molecules),
        .status = std::span<std::uint8_t const>(buffers.status_codes.data(), total_slots),
        .coordinates = std::span<__half const>(buffers.coordinates.data(), total_slots * slot_layout.slot_stride()),
        .energies = std::span<float const>(buffers.energies.data(), total_slots),
        .usrcat = std::span<double const>(buffers.usrcat.data(), total_slots * 60u),
    };
    // `write_shard` names its own refusal in the log, so the outcome only has to survive the return.
    sfcli::shard_write_report_t const report = sfcli::write_shard(output_path, writer_inputs, &logger);
    return report.status == sfcli::shard_write_t::written_k ? augment_t::written_k : augment_t::write_failed_k;
}

#pragma endregion Shard Augmentation

#pragma region Driver

/** Running totals over one invocation's shards, threaded through `process_shard_`. */
struct run_tally_t {
    std::uint64_t completed = 0;
    std::uint64_t skipped = 0;
    std::uint64_t failed = 0;
    std::uint64_t context_mismatch = 0;
    std::uint64_t fragments_separated = 0;
};

/** Write the 3-byte `<output>.encoded` sentinel a `--resume` run keys on. */
void write_encoded_sentinel_(fs::path const &sentinel, logger_t const &logger) {
    FILE *sentinel_file = std::fopen(sentinel.string().c_str(), "wb");
    if (!sentinel_file) {
        logger.error("sentinel write failed: {}", sentinel.string());
        return;
    }
    // `--resume` skips any shard whose sentinel exists, so a short write would leave a
    // file that makes the next run skip an unprocessed shard.
    std::size_t const written = std::fwrite("v1\n", 1, 3, sentinel_file);
    std::fclose(sentinel_file);
    if (written != 3) {
        std::error_code remove_error;
        fs::remove(sentinel, remove_error);
        logger.error("sentinel write truncated, removed: {}", sentinel.string());
    }
}

/** Read one input shard, augment it, seal it with its sentinel, and fold the outcome into @p tally. */
void process_shard_(fs::path const &input_path, cli_config_t const &config, reembed_store_t const &reembed,
                    shard_buffers_t &buffers, std::uint64_t total_shards, logger_t const &logger, run_tally_t &tally) {
    fs::path const output_path = config.output_dir / input_path.filename();
    fs::path sentinel = output_path;
    sentinel += ".encoded";
    if (config.resume && fs::exists(sentinel)) {
        ++tally.skipped;
        return;
    }

    auto const shard_start = std::chrono::steady_clock::now();
    input_shard_t shard;
    shard_read_t const read_status = read_input_shard_(input_path, shard);
    if (read_status != shard_read_t::read_k) {
        logger.error("shard {} {}, skipping", input_path.string(), describe_shard_read_(read_status));
        ++tally.failed;
        return;
    }

    shard_stats_t stats;
    switch (augment_shard_(shard, config, reembed, buffers, output_path, logger, stats)) {
    case augment_t::written_k: break;
    // Both refusals already name themselves — the empty input in `augment_shard_`, the refused write in
    // `write_shard`, which reports the column or the file that stopped it. A second "write failed" line
    // here would name a failure that never happened, or bury the one that did.
    case augment_t::empty_input_k:
    case augment_t::write_failed_k: ++tally.failed; return;
    }

    write_encoded_sentinel_(sentinel, logger);

    double const shard_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - shard_start).count();
    double const rows = static_cast<double>(shard.num_rows);
    std::uint64_t const context_mismatch = stats.context_mismatch.load();
    std::uint64_t const separated = stats.fragments_separated.load();
    tally.context_mismatch += context_mismatch;
    tally.fragments_separated += separated;
    ++tally.completed;

    double const rows_per_second = rows / std::max(shard_seconds, 1e-3);
    logger.info("{:>4}/{:<4} {}  {:.1f}s  {} rows/s ({} rows/s/core)  ctx-mismatch {}  separated {}",
                fmt::group_digits(tally.completed + tally.skipped), fmt::group_digits(total_shards),
                input_path.filename().string(), shard_seconds,
                fmt::group_digits(static_cast<std::uint64_t>(rows_per_second)),
                fmt::group_digits(static_cast<std::uint64_t>(rows_per_second / std::max(config.threads, 1))),
                fmt::group_digits(context_mismatch), fmt::group_digits(separated));
}

/**
 *  @brief Walk @p inputs in order, augmenting each shard into `--output`.
 *
 *  The slot buffers are declared above the loop and reused, so the run's widest shard is the only one that
 *  allocates. A stop signal is honoured between shards, where the last output and its sentinel agree.
 */
void augment_shards_(std::span<fs::path const> inputs, cli_config_t const &config, reembed_store_t const &reembed,
                     logger_t const &logger, run_tally_t &tally) {
    shard_buffers_t buffers;
    for (auto const &input_path : inputs) {
        if (stop_requested_.load(std::memory_order_relaxed)) {
            logger.info("stop requested — {} shards left untouched",
                        fmt::group_digits(inputs.size() - tally.completed - tally.skipped - tally.failed));
            break;
        }
        process_shard_(input_path, config, reembed, buffers, inputs.size(), logger, tally);
    }
}

#pragma endregion Driver

} // namespace

int main(int argc, char **argv) {
    cli_config_t config;
    switch (parse_cli_(argc, argv, config)) {
    case cli_parse_t::parsed_k: break;
    case cli_parse_t::help_requested_k: return 0;
    case cli_parse_t::invalid_k: return 2;
    }

    // Graceful pre-emption: SIGINT/SIGTERM stop the walk at the next shard boundary so the shard in
    // flight still writes its parquet and its sentinel, and a resume run continues from there.
    std::signal(SIGINT, handle_stop_signal_);
    std::signal(SIGTERM, handle_stop_signal_);

    logger_t logger;
    std::error_code error_code;
    fs::create_directories(config.output_dir, error_code);

    std::vector<fs::path> inputs;
    shard_listing_t const listing = fill_shard_paths_into_(config.input_dir, inputs);
    if (listing != shard_listing_t::listed_k) {
        logger.error("--input {} {}", config.input_dir.string(), describe_shard_listing_(listing));
        // A folder the operator named that does not open is a refused command line; one that opens and
        // holds nothing is a run with no work, and the two must not share an exit code.
        return listing == shard_listing_t::cannot_open_k ? 2 : 1;
    }

    logger.info("{}: {} input shards, dataset='{}', reembed={}", executable_name_k, fmt::group_digits(inputs.size()),
                config.dataset, config.reembed_dir.empty() ? "off" : config.reembed_dir.string());

    reembed_store_t reembed;
    if (!config.reembed_dir.empty()) {
        shard_listing_t const reembed_listing = fill_reembed_store_(config.reembed_dir, config.max_atoms, logger,
                                                                    reembed);
        if (reembed_listing == shard_listing_t::cannot_open_k) {
            logger.error("--reembed {} {}", config.reembed_dir.string(), describe_shard_listing_(reembed_listing));
            return 2;
        }
    }

    run_tally_t tally;
    auto const start = std::chrono::steady_clock::now();
    augment_shards_(std::span<fs::path const>(inputs), config, reembed, logger, tally);

    double const total = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    logger.info("complete  {} augmented, {} skipped, {} failed  in {:.0f}s  (ctx-mismatch {}, separated {})",
                fmt::group_digits(tally.completed), fmt::group_digits(tally.skipped), fmt::group_digits(tally.failed),
                total, fmt::group_digits(tally.context_mismatch), fmt::group_digits(tally.fragments_separated));
    return tally.failed > 0 ? 1 : 0;
}
