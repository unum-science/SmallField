/**
 *  @file cli/smallfield_descriptors.cpp
 *  @brief Augment `*.3D.parquet` shards that predate the canonical schema, in place.
 *
 *  A shard carrying no `usrcat` column and unsorted conformers is re-emitted through the producer's own
 *  `write_shard`, with the descriptors computed from the stored coordinates rather than from a fresh GPU
 *  run, which would perturb the geometry. Both paths round through fp16 identically — the producer also
 *  reads its own fp16 buffer back before describing it — so the output is byte-identical to what
 *  `smallfield_batch` would have written for the same geometry.
 */

#include <csignal> // `std::signal`
#include <cstdint> // `std::uint32_t`
#include <cstring> // `std::memcpy`

#include <algorithm>     // `std::max`
#include <array>         // `std::array`
#include <atomic>        // `std::atomic`
#include <charconv>      // `std::from_chars`
#include <chrono>        // `std::chrono::steady_clock`
#include <filesystem>    // `std::filesystem::path`
#include <memory>        // `std::shared_ptr`
#include <string>        // `std::string`
#include <string_view>   // `std::string_view`
#include <system_error>  // `std::error_code`
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
#include "smallfield/molecule_parse.hpp"
#include "smallfield/postprocess/fragment_separation.hpp"
#include "smallfield/postprocess/shape_descriptors.hpp"
#include "smallfield/types/views.hpp"

#include "cli/cli_arguments.hpp"
#include "cli/parquet_writer.hpp"
#include "cli/shard_inventory.hpp"

namespace fs = std::filesystem;
namespace sfapi = smallfield::api;
namespace sfcli = smallfield::cli;
namespace sfpost = smallfield::postprocess;
namespace sft = smallfield::types;
using smallfield::logger_t;

namespace {

/** The one name this binary prints, so the announcement and the usage cannot drift from the executable. */
constexpr char const *executable_name_k = "smallfield_descriptors";

#pragma region Interruption

/* Set by SIGINT/SIGTERM. The shard walk stops at the next shard boundary, so the shard in flight still
 * writes its parquet and its sentinel and a resume run continues from there.
 *
 * `handle_stop_signal_` keeps C language linkage for `std::signal`, which an ordinary `static` function
 * cannot combine with, so this unnamed namespace supplies the internal linkage. */
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
    /** Empty disables the reembed override. */
    fs::path reembed_dir;
    /** Empty omits the key from the shard's metadata. */
    std::string dataset;
    /** Zero resolves to the core count in `parse_cli_`, so no consumer re-derives what it meant. */
    int threads = 0;
    sfcli::restart_policy_t restart_policy = sfcli::restart_policy_t::resume_k;
    /** Both the coordinate slot stride and the `max_atoms` written into the provenance metadata. */
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

using sfcli::cli_parse_t;

cli_parse_t parse_cli_(int argc, char **argv, cli_config_t &config) {
    // `default_arguments::none` suppresses `argparse`'s own `-h` and `--version`; `-h` is bound by hand below.
    argparse::ArgumentParser parser {executable_name_k, "", argparse::default_arguments::none};
    bool help_requested = false;

    sfcli::add_string(parser, "--input", config.input_dir);
    sfcli::add_string(parser, "--output", config.output_dir);
    sfcli::add_string(parser, "--reembed", config.reembed_dir);
    sfcli::add_string(parser, "--dataset", config.dataset);
    // The atom cap cannot exceed what the interleaved scratch here is dimensioned for.
    sfcli::add_number(parser, "--threads", 0, max_threads_k, config.threads);
    sfcli::add_number(parser, "--max-atoms", 1u, sfapi::max_atoms_per_molecule_k, config.max_atoms);
    parser.add_argument("--restart").flag().append().action([&config](std::string const &) {
        config.restart_policy = sfcli::restart_policy_t::restart_k;
    });
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
 *  @brief A single input 3D shard read into typed columns.
 *
 *  `CombineChunks` leaves every column one contiguous array, so per-row access is a plain index. The bound
 *  pointers and the SMILES views are into `table`'s own buffers, so the table must outlive them.
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
    read_k,
    cannot_open_k,
    not_parquet_k,
    unreadable_table_k,
    schema_mismatch_k,
};

/** Read @p path into @p out, binding every column the augmentation walks. */
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
    listed_k,
    cannot_open_k,
    empty_k,
};

/**
 *  @brief Collect every `*.3D.parquet` in @p directory into a cleared @p out, sorted lexicographically.
 *
 *  Here a `*.3D.parquet` is an @b input, the opposite of `smallfield_batch` where the same name marks an
 *  output. A refusal is returned rather than thrown, so a misspelled folder reaches the exit code.
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
    sft::conformer_status_t status = sft::conformer_status_t::success_k;
    float energy = 0.0f;
    /** Tight interleaved `(num_atoms, 3)` fp16 bit patterns, empty when the row carried no geometry. */
    std::vector<std::uint16_t> coordinate_bits;
    std::array<double, sfcli::usrcat_components_k> usrcat {};
};

/** One molecule copied verbatim from a `--reembed` shard, with its already-corrected geometry. */
struct reembed_molecule_t {
    std::uint8_t num_atoms = 0;
    std::uint16_t num_heavy_atoms = 0;
    std::uint16_t num_bonds = 0;
    float molecular_weight = 0.0f;
    /** K as the source shard recorded it, which is not the same fact as the rows actually carried. */
    std::uint8_t num_conformers = 0;
    std::string smiles;
    std::vector<reembed_conformer_t> conformers;
};

/**
 *  @brief Every molecule a `--reembed` run corrected, keyed by `(input_shard, input_row)`.
 *
 *  Every output shard's slot stride has to cover the widest reembed molecule, so that width is derived
 *  once as the map is filled rather than by re-walking the map per shard.
 */
struct reembed_store_t {
    std::unordered_map<std::string, reembed_molecule_t> molecules;
    std::uint32_t widest_molecule_conformers = 0;
};

/** Compose the `(input_shard, input_row)` identity key reembed molecules are matched on, into a cleared @p out. */
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
 *  @param[in] max_atoms Coordinate stride of an output slot. A molecule claiming more atoms than this is
 *      carried with its geometry dropped, because narrowing the count would describe another molecule's
 *      atoms.
 */
void load_reembed_shard_(input_shard_t const &shard, std::uint32_t max_atoms, reembed_store_t &store) {
    arrow::HalfFloatArray const &coords_values = *shard.coords_values;
    arrow::ListArray const &coords_list = *shard.conformer_coords;
    // USRCAT is optional on a reembed shard, so the verbatim path leaves zeros where the column is absent.
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
            molecule.num_heavy_atoms = shard.n_heavy_atoms && shard.n_heavy_atoms->IsValid(row)
                                           ? shard.n_heavy_atoms->Value(row)
                                           : 0;
            molecule.num_bonds = shard.n_bonds && shard.n_bonds->IsValid(row) ? shard.n_bonds->Value(row) : 0;
            molecule.molecular_weight = shard.molecular_weight && shard.molecular_weight->IsValid(row)
                                            ? shard.molecular_weight->Value(row)
                                            : 0.0f;
            molecule.num_conformers = shard.n_conformers->IsValid(row) ? shard.n_conformers->Value(row) : 0;
        }

        reembed_conformer_t conformer;
        conformer.status = static_cast<sft::conformer_status_t>(shard.status->Value(row));
        if (coords_list.IsValid(row)) {
            std::int64_t const begin = coords_list.value_offset(row);
            std::int64_t const end = coords_list.value_offset(row + 1);
            conformer.coordinate_bits.reserve(static_cast<std::size_t>(end - begin));
            for (std::int64_t value_index = begin; value_index < end; ++value_index)
                conformer.coordinate_bits.push_back(coords_values.Value(value_index));
        }
        if (shard.conformer_energy->IsValid(row)) conformer.energy = shard.conformer_energy->Value(row);

        if (usrcat_values && usrcat_column->IsValid(row)) {
            std::int64_t const begin = usrcat_column->value_offset(row);
            if (usrcat_column->value_offset(row + 1) - begin >= sfcli::usrcat_components_k)
                for (std::size_t component = 0; component < sfcli::usrcat_components_k; ++component)
                    conformer.usrcat[component] = half_bits_to_float_(
                        usrcat_values->Value(begin + static_cast<std::int64_t>(component)));
        }

        molecule.conformers.push_back(std::move(conformer));
        store.widest_molecule_conformers = std::max(store.widest_molecule_conformers,
                                                    static_cast<std::uint32_t>(molecule.conformers.size()));
    }
}

/**
 *  @brief Read every shard under @p reembed_dir into @p store, reporting and skipping any that will not read.
 *
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
    /** Molecules refused by `build_shape_context_into`, on any of the ways a reparse can miss the stored one. */
    std::atomic<std::uint64_t> context_unusable {0};
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
 *  The slot arrays mirror the layout the GPU producer hands to `write_shard`, so `write_shard` cannot tell
 *  the two producers apart.
 */
struct shard_buffers_t {
    std::vector<__half> coordinates;
    std::vector<float> energies;
    std::vector<sft::conformer_status_t> status;
    std::vector<double> usrcat;

    std::vector<std::string_view> smiles;
    std::vector<std::uint8_t> num_atoms_per_molecule;
    std::vector<std::uint16_t> num_heavy_atoms_per_molecule;
    std::vector<std::uint16_t> num_bonds_per_molecule;
    std::vector<float> molecular_weight_per_molecule;
    std::vector<std::uint8_t> num_conformers_per_molecule;

    /** `num_molecules + 1` bucket bounds into `molecule_rows`. */
    std::vector<std::uint64_t> molecule_row_offsets;
    /** Every row of the shard, grouped by molecule and in file order within a molecule. */
    std::vector<std::int64_t> molecule_rows;

    /**
     *  @brief Grow to fit one shard and put the used prefix back into its start-of-shard state.
     *
     *  Every element the writer can read is rewritten here, because a stale `smiles` view would otherwise
     *  point into the previous shard's released Arrow table.
     */
    void reset_for(std::uint64_t num_molecules, sfapi::conformer_slot_layout_t const &layout) {
        std::size_t const total_slots = static_cast<std::size_t>(num_molecules) * layout.num_conformers;
        std::size_t const coordinate_total = total_slots * layout.slot_stride();
        auto reset_ = [](auto &buffer, std::size_t count, auto value) {
            if (buffer.size() < count) buffer.resize(count);
            std::fill_n(buffer.begin(), count, value);
        };
        reset_(coordinates, coordinate_total, __float2half(0.0f));
        reset_(energies, total_slots, 0.0f);
        // A slot no molecule claims must not read as a success, so the unused tail defaults non-success.
        reset_(status, total_slots, sft::conformer_status_t::mmff_unconverged_k);
        reset_(usrcat, total_slots * sfcli::usrcat_components_k, 0.0);
        reset_(smiles, num_molecules, std::string_view {});
        reset_(num_atoms_per_molecule, num_molecules, std::uint8_t {0});
        reset_(num_heavy_atoms_per_molecule, num_molecules, std::uint16_t {0});
        reset_(num_bonds_per_molecule, num_molecules, std::uint16_t {0});
        reset_(molecular_weight_per_molecule, num_molecules, 0.0f);
        reset_(num_conformers_per_molecule, num_molecules, std::uint8_t {0});
    }

    /** The rows molecule @p molecule_index owns, in the file order the bucketing preserved. */
    [[nodiscard]] custd::span<std::int64_t const> rows_of(std::uint64_t molecule_index) const {
        std::uint64_t const begin = molecule_row_offsets[molecule_index];
        std::uint64_t const end = molecule_row_offsets[molecule_index + 1];
        return {molecule_rows.data() + begin, static_cast<std::size_t>(end - begin)};
    }
};

/** What the bucketing measured about the shard, beyond the buckets themselves. */
struct molecule_grouping_t {
    /** Row count of the widest molecule, never below one. */
    std::uint32_t widest_conformers = 1;
    /** Molecules inside the `input_row` span owning no row, which is the gap a contiguous shard never has. */
    std::uint64_t unclaimed_molecules = 0;
    /** `input_row` of the first such molecule, so a diagnostic points at the hole rather than counting it;
     *  meaningless while `unclaimed_molecules` is zero. */
    std::uint64_t first_unclaimed_row = 0;
};

/**
 *  @brief Bucket the shard's rows by molecule into @p buffers, keeping file order inside each bucket.
 *
 *  File order is `conformer_index` order within a molecule, which is what makes slot `conformer_slot` of
 *  the output the same conformer the input row described. The histogram this walks is also the only place
 *  a gap in `input_row` is visible, so the unclaimed molecules are counted in the same pass.
 */
molecule_grouping_t group_rows_by_molecule_(input_shard_t const &shard, std::uint64_t base_row,
                                            std::uint64_t num_molecules, shard_buffers_t &buffers) {
    std::vector<std::uint64_t> &offsets = buffers.molecule_row_offsets;
    grow_if_needed_(offsets, num_molecules + 1);
    grow_if_needed_(buffers.molecule_rows, static_cast<std::size_t>(shard.num_rows));
    std::fill_n(offsets.begin(), num_molecules + 1, 0ull);

    for (std::int64_t row = 0; row < shard.num_rows; ++row) ++offsets[shard.input_row->Value(row) - base_row + 1];

    molecule_grouping_t grouping;
    for (std::uint64_t molecule = 0; molecule < num_molecules; ++molecule) {
        std::uint64_t const owned = offsets[molecule + 1];
        grouping.widest_conformers = std::max(grouping.widest_conformers, static_cast<std::uint32_t>(owned));
        if (owned == 0) {
            if (grouping.unclaimed_molecules == 0) grouping.first_unclaimed_row = base_row + molecule;
            ++grouping.unclaimed_molecules;
        }
        offsets[molecule + 1] += offsets[molecule];
    }

    // Scatter with `offsets[molecule]` as the cursor, which leaves every entry holding the end of its own
    // bucket; shifting the array right by one restores the bounds `rows_of` reads.
    for (std::int64_t row = 0; row < shard.num_rows; ++row)
        buffers.molecule_rows[offsets[shard.input_row->Value(row) - base_row]++] = row;
    for (std::uint64_t molecule = num_molecules; molecule > 0; --molecule) offsets[molecule] = offsets[molecule - 1];
    offsets[0] = 0;
    return grouping;
}

/** How one shard's augmentation ended, so each refusal stays distinguishable from the others. */
enum class augment_t {
    written_k,
    empty_input_k,
    /** A molecule inside the shard's `input_row` span owns no row, so the shard is not what the schema promises. */
    gapped_input_k,
    write_failed_k,
};

/** The contiguous `input_row` span a shard covers, from which `input_row = first_row + molecule_index`. */
struct molecule_span_t {
    std::uint64_t first_row = 0;
    std::uint64_t num_molecules = 0;
};

/**
 *  @brief The `input_row` range the shard's rows cover.
 *
 *  Only the ends are read here. Whether the range is genuinely contiguous is what
 *  `molecule_grouping_t::unclaimed_molecules` answers, because only the bucket histogram can see a hole.
 */
molecule_span_t molecule_span_of_(input_shard_t const &shard) {
    std::uint64_t first_row = shard.input_row->Value(0);
    std::uint64_t last_row = first_row;
    for (std::int64_t row = 1; row < shard.num_rows; ++row) {
        std::uint64_t const value = shard.input_row->Value(row);
        first_row = std::min(first_row, value);
        last_row = std::max(last_row, value);
    }
    return {.first_row = first_row, .num_molecules = last_row - first_row + 1};
}

/**
 *  @brief Copy one already-corrected `--reembed` molecule into its slots, verbatim.
 *
 *  Nothing here is recomputed, so a corrected molecule survives the augmentation unchanged.
 */
void copy_reembed_molecule_into_(reembed_molecule_t const &molecule, std::uint64_t molecule_index,
                                 sfapi::conformer_slot_layout_t const &layout, shard_buffers_t &buffers) {
    // A view into the reembed store's storage, which outlives every shard.
    buffers.smiles[molecule_index] = molecule.smiles;
    buffers.num_atoms_per_molecule[molecule_index] = molecule.num_atoms;
    buffers.num_heavy_atoms_per_molecule[molecule_index] = molecule.num_heavy_atoms;
    buffers.num_bonds_per_molecule[molecule_index] = molecule.num_bonds;
    buffers.molecular_weight_per_molecule[molecule_index] = molecule.molecular_weight;
    buffers.num_conformers_per_molecule[molecule_index] = molecule.num_conformers;

    std::uint32_t const num_atoms = molecule.num_atoms;
    std::size_t const num_slots = std::min<std::size_t>(molecule.conformers.size(), layout.num_conformers);
    for (std::size_t conformer_slot = 0; conformer_slot < num_slots; ++conformer_slot) {
        reembed_conformer_t const &conformer = molecule.conformers[conformer_slot];
        std::size_t const flat = layout.flat_slot(molecule_index, static_cast<std::uint32_t>(conformer_slot));
        buffers.status[flat] = conformer.status;
        buffers.energies[flat] = conformer.energy;
        if (conformer.status != sft::conformer_status_t::success_k) continue; // a failure row stays null
        if (num_atoms == 0) continue;                                         // preprocess failed: no geometry
        // The row's own geometry is shorter than the atom count it claims, so it cannot fill the slot.
        if (conformer.coordinate_bits.size() < static_cast<std::size_t>(num_atoms) * 3u) continue;

        // `coordinate_bits` holds Arrow's float16 patterns, which is the pool's own storage, so naming them
        // `__half` keeps the transposition a verbatim copy rather than a conversion of the bit pattern.
        auto const *coordinate_halves = reinterpret_cast<__half const *>(conformer.coordinate_bits.data());
        sfapi::scatter_interleaved_into_slot(
            coordinate_halves, layout, num_atoms,
            buffers.coordinates.data() +
                layout.slot_offset(molecule_index, static_cast<std::uint32_t>(conformer_slot)));
        std::copy(conformer.usrcat.begin(), conformer.usrcat.end(),
                  buffers.usrcat.data() + flat * sfcli::usrcat_components_k);
    }
}

/** What this run can do with a molecule's stored atom count, which is what its slots' status has to say. */
enum class molecule_geometry_t {
    /** The atom count fits the coordinate stride, so the stored conformers can be laid into slots. */
    describable_k,
    /** The source shard recorded no atoms, so the molecule was refused before any geometry existed. */
    preprocess_failed_k,
    /** More atoms than `--max-atoms` addresses; narrowing the count would describe another molecule's atoms. */
    wider_than_stride_k,
};

/**
 *  @brief Copy one molecule's per-molecule scalars out of the original shard.
 *
 *  @return What the stored atom count leaves this run able to do, which the atom count alone cannot say: a
 *      molecule wider than the stride and a molecule that never preprocessed both land at zero atoms.
 */
molecule_geometry_t copy_molecule_scalars_into_(input_shard_t const &shard, std::int64_t first_row,
                                                std::uint64_t molecule_index, std::uint32_t max_atoms,
                                                shard_buffers_t &buffers) {
    buffers.smiles[molecule_index] = shard.smiles->GetView(first_row);
    std::uint16_t const stored_atoms = shard.n_atoms->IsValid(first_row) ? shard.n_atoms->Value(first_row) : 0;
    std::uint8_t const num_atoms = stored_atoms <= max_atoms ? static_cast<std::uint8_t>(stored_atoms) : 0;
    buffers.num_atoms_per_molecule[molecule_index] = num_atoms;
    buffers.num_heavy_atoms_per_molecule[molecule_index] =
        shard.n_heavy_atoms && shard.n_heavy_atoms->IsValid(first_row) ? shard.n_heavy_atoms->Value(first_row) : 0;
    buffers.num_bonds_per_molecule[molecule_index] = shard.n_bonds && shard.n_bonds->IsValid(first_row)
                                                         ? shard.n_bonds->Value(first_row)
                                                         : 0;
    buffers.molecular_weight_per_molecule[molecule_index] = shard.molecular_weight &&
                                                                    shard.molecular_weight->IsValid(first_row)
                                                                ? shard.molecular_weight->Value(first_row)
                                                                : 0.0f;
    buffers.num_conformers_per_molecule[molecule_index] = shard.n_conformers->IsValid(first_row)
                                                              ? shard.n_conformers->Value(first_row)
                                                              : 0;
    if (num_atoms > 0) return molecule_geometry_t::describable_k;
    return stored_atoms == 0 ? molecule_geometry_t::preprocess_failed_k : molecule_geometry_t::wider_than_stride_k;
}

/**
 *  @brief Copy every stored row's status and energy into its conformer slot.
 *
 *  This runs for every molecule, geometry or not, because the stored status is the only record of why a
 *  molecule was refused and a slot left at `reset_for`'s fill would publish the fill instead — a molecule
 *  turned away at the SMILES parser reported as one whose force field did not converge.
 */
void copy_row_outcomes_into_(input_shard_t const &shard, custd::span<std::int64_t const> rows,
                             std::uint64_t molecule_index, molecule_geometry_t geometry,
                             sfapi::conformer_slot_layout_t const &layout, shard_buffers_t &buffers) {
    std::size_t const num_slots = std::min<std::size_t>(rows.size(), layout.num_conformers);
    for (std::size_t conformer_slot = 0; conformer_slot < num_slots; ++conformer_slot) {
        std::int64_t const row = rows[conformer_slot];
        std::size_t const flat = layout.flat_slot(molecule_index, static_cast<std::uint32_t>(conformer_slot));
        auto const stored = static_cast<sft::conformer_status_t>(shard.status->Value(row));
        // A success this run has no slot wide enough to carry is restated as the refusal it is, so it is not
        // published as a success with an empty coordinate list. Every other stored value already is the
        // truest thing known about the row.
        buffers.status[flat] = geometry == molecule_geometry_t::wider_than_stride_k &&
                                       stored == sft::conformer_status_t::success_k
                                   ? sft::conformer_status_t::too_many_atoms_k
                                   : stored;
        if (shard.conformer_energy->IsValid(row)) buffers.energies[flat] = shard.conformer_energy->Value(row);
    }
}

/**
 *  @brief Read one stored conformer row into @p interleaved, in the atom-major order the descriptors want.
 *
 *  @return The components the row actually held, which is short of `3 * num_atoms` on a truncated row.
 */
std::size_t read_row_coordinates_into_(input_shard_t const &shard, std::int64_t row, std::uint32_t num_atoms,
                                       custd::span<float> interleaved) {
    std::size_t const begin = static_cast<std::size_t>(shard.conformer_coords->value_offset(row));
    std::size_t const stored = static_cast<std::size_t>(shard.conformer_coords->value_offset(row + 1)) - begin;
    std::size_t const wanted = static_cast<std::size_t>(num_atoms) * 3u;
    std::size_t const copied = std::min(stored, wanted);
    for (std::size_t value_index = 0; value_index < copied; ++value_index)
        interleaved[value_index] = half_bits_to_float_(
            shard.coords_values->Value(static_cast<std::int64_t>(begin + value_index)));
    return copied;
}

/**
 *  @brief Lay one molecule's stored conformers into its slots and compute their descriptors.
 *
 *  The status each slot carries is already there, from `copy_row_outcomes_into_`, and is what decides which
 *  slots get geometry. The loop runs to completion deliberately, even past a failed row: a `break` would
 *  leave the later slots holding no coordinates. Where the reparsed atom order cannot be trusted the
 *  coordinates are still preserved and only the descriptors stay zero-filled.
 */
void augment_molecule_(input_shard_t const &shard, custd::span<std::int64_t const> rows, std::uint64_t molecule_index,
                       std::uint32_t num_atoms, sfpost::molecule_shape_context_t const &context,
                       sfpost::shape_typing_t typing, sfapi::conformer_slot_layout_t const &layout,
                       shard_buffers_t &buffers, shard_stats_t &stats) {
    std::array<float, sfapi::max_atoms_per_molecule_k * 3u> interleaved {};
    std::size_t const num_slots = std::min<std::size_t>(rows.size(), layout.num_conformers);
    for (std::size_t conformer_slot = 0; conformer_slot < num_slots; ++conformer_slot) {
        std::int64_t const row = rows[conformer_slot];
        std::size_t const flat = layout.flat_slot(molecule_index, static_cast<std::uint32_t>(conformer_slot));
        if (buffers.status[flat] != sft::conformer_status_t::success_k) continue; // a failure row stays null
        if (!shard.conformer_coords->IsValid(row)) continue;                      // the row carried no geometry

        // A row holding fewer components than its atom count claims cannot fill the slot, and copying the
        // prefix would leave the previous conformer's coordinates in the tail of the shared scratch.
        std::size_t const wanted = static_cast<std::size_t>(num_atoms) * 3u;
        if (read_row_coordinates_into_(shard, row, num_atoms, interleaved) < wanted) continue;
        custd::span<float> const coordinates {interleaved.data(), wanted};

        // Separating disconnected components needs the context's per-atom labels, so it is skipped along
        // with the descriptors on every typing the reparse could not be trusted for.
        if (typing == sfpost::shape_typing_t::typed_k && context.num_components > 1) {
            sfpost::separate_fragments(coordinates, context);
            stats.fragments_separated.fetch_add(1, std::memory_order_relaxed);
        }
        sfapi::scatter_interleaved_into_slot(
            interleaved.data(), layout, num_atoms,
            buffers.coordinates.data() +
                layout.slot_offset(molecule_index, static_cast<std::uint32_t>(conformer_slot)));

        // Descriptors stay zero-filled on a refusal, while the coordinates written above are preserved.
        if (typing != sfpost::shape_typing_t::typed_k) continue;
        std::array<double, sfcli::usrcat_components_k> usrcat_row {};
        sfpost::usrcat_descriptor(coordinates, context, usrcat_row);
        std::copy(usrcat_row.begin(), usrcat_row.end(), buffers.usrcat.data() + flat * sfcli::usrcat_components_k);
    }
}

/**
 *  @brief Hand the reconstructed slot layout to the producer's own writer.
 *
 *  Each span carries the extent this shard filled, which is below the buffers' own size whenever an
 *  earlier shard was the wider one.
 */
sfcli::writer_inputs_t shard_writer_inputs_(input_shard_t const &shard, cli_config_t const &config,
                                            molecule_span_t const &span, sfapi::conformer_slot_layout_t const &layout,
                                            shard_buffers_t const &buffers) {
    std::uint64_t const num_molecules = span.num_molecules;
    std::size_t const num_slots = static_cast<std::size_t>(num_molecules) * layout.num_conformers;
    return {
        .provenance = {.input_shard_stem = std::string {shard.input_shard->GetView(0)},
                       .dataset = config.dataset,
                       .base_row_index = span.first_row},
        .molecules = {.smiles = {buffers.smiles.data(), num_molecules},
                      .num_atoms = {buffers.num_atoms_per_molecule.data(), num_molecules},
                      .num_conformers = {buffers.num_conformers_per_molecule.data(), num_molecules},
                      .num_heavy_atoms = {buffers.num_heavy_atoms_per_molecule.data(), num_molecules},
                      .num_bonds = {buffers.num_bonds_per_molecule.data(), num_molecules},
                      .molecular_weight = {buffers.molecular_weight_per_molecule.data(), num_molecules}},
        .conformers = {.layout = layout,
                       .status = {buffers.status.data(), num_slots},
                       .energies = {buffers.energies.data(), num_slots},
                       .coordinates = {buffers.coordinates.data(), num_slots * layout.slot_stride()},
                       .usrcat = {buffers.usrcat.data(), num_slots * sfcli::usrcat_components_k}},
    };
}

/**
 *  @brief Augment one input shard and write the canonical output shard.
 *
 *  Every molecule's rows are laid into conformer slots exactly as the GPU producer would lay them, so it is
 *  `sfcli::write_shard` that energy-sorts, ranks `conformer_index`, nulls the failures and packs the
 *  descriptors — which is what makes the output bytes identical to the producer's. A molecule found in
 *  @p reembed is copied verbatim from there instead of being described again.
 *
 *  @param[inout] buffers Grown and reset in place; owned by the shard walk.
 */
augment_t augment_shard_(input_shard_t const &shard, cli_config_t const &config, reembed_store_t const &reembed,
                         shard_buffers_t &buffers, fs::path const &input_path, fs::path const &output_path,
                         logger_t const &logger, shard_stats_t &stats) {
    if (shard.num_rows == 0) {
        logger.warn("empty shard, skipping: {}", input_path.string());
        return augment_t::empty_input_k;
    }

    molecule_span_t const span = molecule_span_of_(shard);
    molecule_grouping_t const grouping = group_rows_by_molecule_(shard, span.first_row, span.num_molecules, buffers);

    // A molecule with no rows would be written out as an all-null row carrying an `input_row` the input
    // shard never held, so a hole is refused rather than filled.
    if (grouping.unclaimed_molecules > 0) {
        logger.error("shard {} is gapped over input_row {}..{}, skipping — {} of {} molecules own no row, " //
                     "the first at input_row {}",                                                           //
                     input_path.string(), fmt::group_digits(span.first_row),
                     fmt::group_digits(span.first_row + span.num_molecules - 1),
                     fmt::group_digits(grouping.unclaimed_molecules), fmt::group_digits(span.num_molecules),
                     fmt::group_digits(grouping.first_unclaimed_row));
        return augment_t::gapped_input_k;
    }

    // The slot stride K must cover the widest molecule, original or reembed, and match the producer's
    // `num_conformers` provenance value. Reembed molecules can only add rows.
    sfapi::conformer_slot_layout_t const layout {
        config.max_atoms, std::max(grouping.widest_conformers, reembed.widest_molecule_conformers)};
    buffers.reset_for(span.num_molecules, layout);

    // `private` hands each thread its own key buffer and its own typing context, both of which keep their
    // capacity across the molecules that thread walks; declaring either inside the body would allocate and
    // free once per molecule — six heap blocks between them.
    std::string reembed_key;
    sfpost::molecule_shape_context_t context;

#pragma omp parallel for schedule(dynamic, 256) \
    num_threads(config.threads) private(reembed_key, context) if (config.threads > 1)
    for (std::uint64_t molecule_index = 0; molecule_index < span.num_molecules; ++molecule_index) {
        custd::span<std::int64_t const> const rows = buffers.rows_of(molecule_index);
        std::int64_t const first_row = rows.front();

        fill_reembed_key_into_(shard.input_shard->GetView(first_row), shard.input_row->Value(first_row), reembed_key);
        if (auto found = reembed.molecules.find(reembed_key); found != reembed.molecules.end()) {
            copy_reembed_molecule_into_(found->second, molecule_index, layout, buffers);
            continue;
        }

        molecule_geometry_t const geometry = copy_molecule_scalars_into_(shard, first_row, molecule_index,
                                                                         config.max_atoms, buffers);
        copy_row_outcomes_into_(shard, rows, molecule_index, geometry, layout, buffers);
        // A molecule this run cannot describe carries the refusal just written and nothing else: there are no
        // coordinates to lay out, and no atom order for a reparse to agree with.
        if (geometry != molecule_geometry_t::describable_k) continue;

        // Typed once, under the same parse contract the producer uses. Every refusal names a way the reparse
        // failed to reproduce the atom order the stored coordinates were written in.
        std::uint8_t const num_atoms = buffers.num_atoms_per_molecule[molecule_index];
        sfpost::shape_typing_t const typing = sfpost::build_shape_context_into(
            buffers.smiles[molecule_index], num_atoms, smallfield::keep_largest_fragment_k, context);
        if (typing != sfpost::shape_typing_t::typed_k) stats.context_unusable.fetch_add(1, std::memory_order_relaxed);
        augment_molecule_(shard, rows, molecule_index, num_atoms, context, typing, layout, buffers, stats);
    }

    // `write_shard` already logged its own refusal, so the outcome only has to survive the return.
    sfcli::shard_write_report_t const report = sfcli::write_shard(
        output_path, shard_writer_inputs_(shard, config, span, layout, buffers), &logger);
    return report.status == sfcli::shard_write_t::written_k ? augment_t::written_k : augment_t::write_failed_k;
}

#pragma endregion Shard Augmentation

#pragma region Driver

/** Running totals over one invocation's shards, threaded through `process_shard_`. */
struct run_tally_t {
    std::uint64_t completed = 0;
    std::uint64_t skipped = 0;
    std::uint64_t failed = 0;
    std::uint64_t context_unusable = 0;
    std::uint64_t fragments_separated = 0;
};

/** Read one input shard, augment it, seal it with its sentinel, and fold the outcome into @p tally. */
void process_shard_(fs::path const &input_path, cli_config_t const &config, reembed_store_t const &reembed,
                    shard_buffers_t &buffers, std::uint64_t total_shards, logger_t const &logger, run_tally_t &tally) {
    fs::path const output_path = config.output_dir / input_path.filename();
    fs::path sentinel = output_path;
    sentinel += sfcli::encoded_sentinel_suffix_k;
    if (config.restart_policy == sfcli::restart_policy_t::resume_k && fs::exists(sentinel)) {
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
    switch (augment_shard_(shard, config, reembed, buffers, input_path, output_path, logger, stats)) {
    case augment_t::written_k: break;
    // Every refusal already logged itself, in `augment_shard_` or in `write_shard`.
    case augment_t::empty_input_k:
    case augment_t::gapped_input_k:
    case augment_t::write_failed_k: ++tally.failed; return;
    }

    sfcli::write_encoded_sentinel(output_path, &logger);

    double const shard_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - shard_start).count();
    double const rows = static_cast<double>(shard.num_rows);
    std::uint64_t const context_unusable = stats.context_unusable.load();
    std::uint64_t const separated = stats.fragments_separated.load();
    tally.context_unusable += context_unusable;
    tally.fragments_separated += separated;
    ++tally.completed;

    double const rows_per_second = rows / std::max(shard_seconds, 1e-3);
    logger.info("{:>4}/{:<4} {}  {:.1f}s  {} rows/s ({} rows/s/core)  ctx-unusable {}  separated {}",
                fmt::group_digits(tally.completed + tally.skipped), fmt::group_digits(total_shards),
                input_path.filename().string(), shard_seconds,
                fmt::group_digits(static_cast<std::uint64_t>(rows_per_second)),
                fmt::group_digits(static_cast<std::uint64_t>(rows_per_second / std::max(config.threads, 1))),
                fmt::group_digits(context_unusable), fmt::group_digits(separated));
}

/**
 *  @brief Walk @p inputs in order, augmenting each shard into `--output`.
 *
 *  A stop signal is honoured between shards, where the last output and its sentinel agree.
 */
void augment_shards_(custd::span<fs::path const> inputs, cli_config_t const &config, reembed_store_t const &reembed,
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

    std::signal(SIGINT, handle_stop_signal_);
    std::signal(SIGTERM, handle_stop_signal_);

    logger_t logger;
    std::error_code error_code;
    fs::create_directories(config.output_dir, error_code);

    std::vector<fs::path> inputs;
    shard_listing_t const listing = fill_shard_paths_into_(config.input_dir, inputs);
    if (listing != shard_listing_t::listed_k) {
        logger.error("--input {} {}", config.input_dir.string(), describe_shard_listing_(listing));
        // A folder that does not open is a refused command line; one that opens empty is a run with no work.
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
    augment_shards_(inputs, config, reembed, logger, tally);

    double const total = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    logger.info("complete  {} augmented, {} skipped, {} failed  in {:.0f}s  (ctx-unusable {}, separated {})",
                fmt::group_digits(tally.completed), fmt::group_digits(tally.skipped), fmt::group_digits(tally.failed),
                total, fmt::group_digits(tally.context_unusable), fmt::group_digits(tally.fragments_separated));
    return tally.failed > 0 ? 1 : 0;
}
