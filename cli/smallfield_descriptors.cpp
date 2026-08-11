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
 *  closes, and a resume run skips shards whose sentinel already exists.
 */

#include <cstdint> // `std::uint32_t`
#include <cstdio>  // `std::fopen`
#include <cstdlib> // `std::strtoul`
#include <cstring> // `std::memcpy`

#include <algorithm>     // `std::max`
#include <array>         // `std::array`
#include <atomic>        // `std::atomic`
#include <chrono>        // `std::chrono::steady_clock`
#include <filesystem>    // `std::filesystem::path`
#include <memory>        // `std::shared_ptr`
#include <span>          // `std::span`
#include <string>        // `std::string`
#include <string_view>   // `std::string_view`
#include <unordered_map> // `std::unordered_map`
#include <vector>        // `std::vector`

#include <omp.h>
#include <cuda_fp16.h>
#include <arrow/array.h>
#include <arrow/io/file.h>
#include <arrow/table.h>
#include <parquet/arrow/reader.h>

#include "smallfield/api/conformer_layout.hpp"
#include "smallfield/log.hpp"
#include "smallfield/postprocess.hpp"

#include "parquet_writer.hpp"

namespace fs = std::filesystem;
namespace sfapi = smallfield::api;
namespace sfcli = smallfield::cli;
namespace sfpost = smallfield::postprocess;
using smallfield::logger_t;
using smallfield::thousands;

namespace {

#pragma region Configuration

/** All CLI flags. Immutable after parse. */
struct cli_config_t {
    fs::path input_dir;
    fs::path output_dir;
    /** Optional folder of corrected shards; "" = disabled. */
    fs::path reembed_dir;
    /** Provenance dataset name (e.g. "pubchem"); "" = omit from metadata. */
    std::string dataset;
    /** OpenMP threads across molecules within a shard; 0 = all cores. */
    int threads = 0;
    /** Skip shards whose `<output>.encoded` sentinel exists. */
    bool resume = true;
    /** Coord stride / provenance `max_atoms`. */
    std::uint32_t max_atoms = sfapi::max_atoms_per_molecule_k;
};

void print_usage() {
    fmt::print(stderr, R"(Usage: smallfield_descriptors --input DIR --output DIR [options]
  --input DIR       folder of 3D parquet-shards to augment (*.3D.parquet)
  --output DIR      folder for augmented output shards (*.3D.parquet)
  --reembed DIR     folder of corrected 13-column shards; molecules present there are
                    copied verbatim instead of augmented (e.g. re-embedded multi-fragment)
  --dataset NAME    provenance dataset name embedded in output metadata (e.g. pubchem)
  --threads N       OpenMP threads across molecules within a shard (default: all cores)
  --max-atoms N     per-molecule atom cap / coord stride (default: 128)
  --restart         overwrite already-complete output shards
)");
}

bool parse_cli(int argc, char **argv, cli_config_t &config) {
    for (int argument_index = 1; argument_index < argc; ++argument_index) {
        std::string_view flag = argv[argument_index];
        auto require_value = [&](char const *flag_name) -> char const * {
            if (argument_index + 1 >= argc) {
                fmt::print(stderr, "--{} requires a value\n", flag_name);
                return nullptr;
            }
            return argv[++argument_index];
        };
        if (flag == "--input") {
            auto value = require_value("input");
            if (!value) return false;
            config.input_dir = value;
        }
        else if (flag == "--output") {
            auto value = require_value("output");
            if (!value) return false;
            config.output_dir = value;
        }
        else if (flag == "--reembed") {
            auto value = require_value("reembed");
            if (!value) return false;
            config.reembed_dir = value;
        }
        else if (flag == "--dataset") {
            auto value = require_value("dataset");
            if (!value) return false;
            config.dataset = value;
        }
        else if (flag == "--threads") {
            auto value = require_value("threads");
            if (!value) return false;
            config.threads = static_cast<int>(std::strtol(value, nullptr, 10));
        }
        else if (flag == "--max-atoms") {
            auto value = require_value("max-atoms");
            if (!value) return false;
            config.max_atoms = static_cast<std::uint32_t>(std::strtoul(value, nullptr, 10));
        }
        else if (flag == "--restart") { config.resume = false; }
        else if (flag == "--help" || flag == "-h") {
            print_usage();
            return false;
        }
        else {
            fmt::print(stderr, "unknown flag: {}\n", flag);
            return false;
        }
    }
    if (config.input_dir.empty() || config.output_dir.empty()) {
        print_usage();
        return false;
    }
    return true;
}

#pragma endregion Configuration

#pragma region fp16 Conversion

/** Decode one fp16 bit pattern (as stored by Arrow `HalfFloat`) to a host float. */
inline float half_bits_to_float(std::uint16_t bits) {
    __half value;
    std::memcpy(&value, &bits, sizeof(value));
    return __half2float(value);
}

/** Encode a host float to an fp16 bit pattern, packed into an `__half` slot. */
inline __half float_to_half(float value) { return __float2half(value); }

/** Reinterpret an fp16 bit pattern as an `__half` (no numeric conversion). */
inline __half half_from_bits(std::uint16_t bits) {
    __half value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
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
array_type_ const *bind_column(arrow::Table const &table, char const *name) {
    auto column = table.GetColumnByName(name);
    if (!column || column->num_chunks() != 1) return nullptr;
    return dynamic_cast<array_type_ const *>(column->chunk(0).get());
}

bool read_input_shard(fs::path const &path, input_shard_t &out) {
    auto open_result = arrow::io::ReadableFile::Open(path.string(), arrow::default_memory_pool());
    if (!open_result.ok()) return false;
    parquet::arrow::FileReaderBuilder builder;
    if (!builder.Open(open_result.ValueOrDie()).ok()) return false;
    auto reader_result = builder.Build();
    if (!reader_result.ok()) return false;
    auto reader = std::move(reader_result).ValueOrDie();

    std::shared_ptr<arrow::Table> table;
    if (!reader->ReadTable(&table).ok() || !table) return false;
    auto combined = table->CombineChunks(arrow::default_memory_pool());
    if (!combined.ok()) return false;
    out.table = combined.ValueOrDie();

    out.input_shard = bind_column<arrow::StringArray>(*out.table, "input_shard");
    out.input_row = bind_column<arrow::UInt64Array>(*out.table, "input_row");
    out.smiles = bind_column<arrow::StringArray>(*out.table, "smiles");
    out.conformer_index = bind_column<arrow::UInt8Array>(*out.table, "conformer_index");
    out.status = bind_column<arrow::UInt8Array>(*out.table, "status");
    out.n_heavy_atoms = bind_column<arrow::UInt16Array>(*out.table, "n_heavy_atoms");
    out.n_atoms = bind_column<arrow::UInt16Array>(*out.table, "n_atoms");
    out.n_bonds = bind_column<arrow::UInt16Array>(*out.table, "n_bonds");
    out.molecular_weight = bind_column<arrow::FloatArray>(*out.table, "molecular_weight");
    out.n_conformers = bind_column<arrow::UInt8Array>(*out.table, "n_conformers");
    out.conformer_coords = bind_column<arrow::ListArray>(*out.table, "conformer_coords");
    out.conformer_energy = bind_column<arrow::FloatArray>(*out.table, "conformer_energy");
    out.num_rows = out.table->num_rows();

    if (!out.input_shard || !out.input_row || !out.smiles || !out.conformer_index || !out.status || !out.n_atoms ||
        !out.n_conformers || !out.conformer_coords || !out.conformer_energy)
        return false;
    out.coords_values = dynamic_cast<arrow::HalfFloatArray const *>(out.conformer_coords->values().get());
    if (!out.coords_values) return false;
    return true;
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

using reembed_index_t = std::unordered_map<std::string, reembed_molecule_t>;

/** Compose the `(input_shard, input_row)` identity key used to match reembed molecules. */
inline std::string reembed_key(std::string_view input_shard, std::uint64_t input_row) {
    std::string key;
    key.reserve(input_shard.size() + 21);
    key.append(input_shard);
    key.push_back('\0');
    key.append(std::to_string(input_row));
    return key;
}

/** Copy one reembed shard's molecules into @p index, keyed by `(input_shard, input_row)`. */
void load_reembed_shard(input_shard_t const &shard, reembed_index_t &index) {
    arrow::HalfFloatArray const &coords_values = *shard.coords_values;
    arrow::ListArray const &coords_list = *shard.conformer_coords;
    // USRCAT is optional on a reembed shard; bind once so the verbatim path stays truly verbatim
    // when present, and leaves zeros when the shard predates it. Stored as fp16 (list<float16>).
    auto const *usrcat_column = bind_column<arrow::ListArray>(*shard.table, "usrcat");
    auto const *usrcat_values = usrcat_column
                                    ? dynamic_cast<arrow::HalfFloatArray const *>(usrcat_column->values().get())
                                    : nullptr;
    for (std::int64_t row = 0; row < shard.num_rows; ++row) {
        std::string_view const stem = shard.input_shard->GetView(row);
        std::uint64_t const input_row = shard.input_row->Value(row);
        std::string key = reembed_key(stem, input_row);

        reembed_molecule_t &molecule = index[key];
        if (molecule.conformers.empty()) {
            molecule.smiles = shard.smiles->GetString(row);
            molecule.num_atoms = shard.n_atoms->IsValid(row) ? static_cast<std::uint8_t>(shard.n_atoms->Value(row)) : 0;
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
                    conformer.usrcat[component] = half_bits_to_float(
                        usrcat_values->Value(begin + static_cast<std::int64_t>(component)));
        }

        molecule.conformers.push_back(std::move(conformer));
    }
}

reembed_index_t build_reembed_index(fs::path const &reembed_dir, logger_t const &logger) {
    reembed_index_t index;
    std::vector<fs::path> shards;
    for (auto const &entry : fs::directory_iterator(reembed_dir)) {
        std::string const name = entry.path().filename().string();
        if (name.ends_with(".3D.parquet")) shards.push_back(entry.path());
    }
    std::sort(shards.begin(), shards.end());
    for (auto const &path : shards) {
        input_shard_t shard;
        if (!read_input_shard(path, shard)) {
            logger.error("reembed read failed, skipping: {}", path.string());
            continue;
        }
        load_reembed_shard(shard, index);
    }
    logger.info("reembed index: {} molecules from {} shards", thousands(index.size()), thousands(shards.size()));
    return index;
}

#pragma endregion Reembed Store

#pragma region Shard Augmentation

/** Per-shard counters folded into the progress log. */
struct shard_stats_t {
    /** Molecules whose SMILES parse did not match stored n_atoms. */
    std::atomic<std::uint64_t> context_mismatch {0};
    std::atomic<std::uint64_t> fragments_separated {0};
    std::uint64_t from_reembed = 0;
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
 */
bool augment_shard(input_shard_t const &shard, cli_config_t const &config, reembed_index_t const &reembed,
                   fs::path const &output_path, logger_t const &logger, shard_stats_t &stats) {
    if (shard.num_rows == 0) {
        logger.warn("empty shard, skipping: {}", output_path.string());
        return false;
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

    // Row lists per molecule (in file order == conformer_index order within a molecule).
    std::vector<std::vector<std::int64_t>> rows_for_molecule(num_molecules);
    std::uint32_t observed_conformers = 1;
    for (std::int64_t row = 0; row < shard.num_rows; ++row) {
        std::uint64_t const molecule_index = shard.input_row->Value(row) - base_row;
        rows_for_molecule[molecule_index].push_back(row);
        observed_conformers = std::max<std::uint32_t>(
            observed_conformers, static_cast<std::uint32_t>(rows_for_molecule[molecule_index].size()));
    }
    // The slot stride K must cover the widest molecule (original or reembed) and match the
    // producer's `num_conformers` provenance value. Reembed molecules can only add rows.
    std::uint32_t max_conformers = observed_conformers;
    for (auto const &[key, molecule] : reembed)
        max_conformers = std::max<std::uint32_t>(max_conformers,
                                                 static_cast<std::uint32_t>(molecule.conformers.size()));

    std::uint32_t const max_atoms = config.max_atoms;
    sfapi::conformer_slot_layout_t const slot_layout {max_atoms, max_conformers};
    std::size_t const total_slots = static_cast<std::size_t>(num_molecules) * max_conformers;

    // Structure-of-arrays coordinate buffer + per-slot descriptor/status/energy arrays,
    // matching the layout the producer hands to write_shard.
    std::vector<__half> coordinates(total_slots * slot_layout.slot_stride(), __float2half(0.0f));
    std::vector<float> energies(total_slots, 0.0f);
    std::vector<std::uint8_t> status_codes(total_slots, 1); // default non-success for unused slots
    std::vector<double> usrcat(total_slots * 60u, 0.0);     // float64 input; writer downcasts to fp16

    std::vector<std::string_view> smiles(num_molecules);
    std::vector<std::uint8_t> num_atoms_per_molecule(num_molecules, 0);
    std::vector<std::uint16_t> num_heavy_atoms_per_molecule(num_molecules, 0);
    std::vector<std::uint16_t> num_bonds_per_molecule(num_molecules, 0);
    std::vector<float> molecular_weight_per_molecule(num_molecules, 0.0f);
    std::vector<std::uint8_t> n_conformers_per_molecule(num_molecules, 0);

    arrow::HalfFloatArray const &coords_values = *shard.coords_values;
    arrow::ListArray const &coords_list = *shard.conformer_coords;

    int const threads = config.threads > 0 ? config.threads : omp_get_max_threads();

#pragma omp parallel for schedule(dynamic, 256) num_threads(threads) if (threads > 1)
    for (std::uint64_t molecule_index = 0; molecule_index < num_molecules; ++molecule_index) {
        auto const &rows = rows_for_molecule[molecule_index];
        if (rows.empty()) continue; // contiguous input guarantees this never fires; defensive only
        std::int64_t const first_row = rows.front();
        std::string_view const stem = shard.input_shard->GetView(first_row);
        std::uint64_t const input_row = shard.input_row->Value(first_row);

        // Reembed override: copy the corrected molecule verbatim into this molecule's slots.
        std::string const key = reembed_key(stem, input_row);
        if (auto found = reembed.find(key); found != reembed.end()) {
            reembed_molecule_t const &molecule = found->second;
            smiles[molecule_index] = molecule.smiles; // string_view into the reembed table's persistent storage
            num_atoms_per_molecule[molecule_index] = molecule.num_atoms;
            num_heavy_atoms_per_molecule[molecule_index] = molecule.n_heavy_atoms;
            num_bonds_per_molecule[molecule_index] = molecule.n_bonds;
            molecular_weight_per_molecule[molecule_index] = molecule.molecular_weight;
            n_conformers_per_molecule[molecule_index] = molecule.n_conformers;
            std::uint32_t const num_atoms = molecule.num_atoms;
            for (std::size_t conformer_slot = 0;
                 conformer_slot < molecule.conformers.size() && conformer_slot < max_conformers; ++conformer_slot) {
                reembed_conformer_t const &conformer = molecule.conformers[conformer_slot];
                std::size_t const flat = static_cast<std::size_t>(molecule_index) * max_conformers + conformer_slot;
                status_codes[flat] = conformer.status;
                energies[flat] = conformer.energy;
                if (conformer.has_coords && conformer.status == 0 && num_atoms > 0) {
                    std::size_t const slot_base = slot_layout.slot_offset(molecule_index,
                                                                          static_cast<std::uint32_t>(conformer_slot));
                    for (std::uint32_t atom = 0; atom < num_atoms; ++atom)
                        for (std::uint32_t axis = 0; axis < 3u; ++axis)
                            coordinates[slot_base + slot_layout.component_offset(axis, atom)] = half_from_bits(
                                conformer.coord_bits[atom * 3u + axis]);
                    std::copy(conformer.usrcat.begin(), conformer.usrcat.end(), usrcat.data() + flat * 60u);
                }
            }
            continue;
        }

        // Augment path: per-molecule scalars come straight from the original shard.
        smiles[molecule_index] = shard.smiles->GetView(first_row);
        std::uint8_t const num_atoms_u8 = shard.n_atoms->IsValid(first_row)
                                              ? static_cast<std::uint8_t>(shard.n_atoms->Value(first_row))
                                              : 0;
        num_atoms_per_molecule[molecule_index] = num_atoms_u8;
        num_heavy_atoms_per_molecule[molecule_index] = shard.n_heavy_atoms && shard.n_heavy_atoms->IsValid(first_row)
                                                           ? shard.n_heavy_atoms->Value(first_row)
                                                           : 0;
        num_bonds_per_molecule[molecule_index] = shard.n_bonds && shard.n_bonds->IsValid(first_row)
                                                     ? shard.n_bonds->Value(first_row)
                                                     : 0;
        molecular_weight_per_molecule[molecule_index] = shard.molecular_weight &&
                                                                shard.molecular_weight->IsValid(first_row)
                                                            ? shard.molecular_weight->Value(first_row)
                                                            : 0.0f;
        n_conformers_per_molecule[molecule_index] = shard.n_conformers->IsValid(first_row)
                                                        ? shard.n_conformers->Value(first_row)
                                                        : 0;

        if (num_atoms_u8 == 0) continue; // preprocess-failed molecule: single null row, no coords

        // Type the molecule once (same parse contract the producer uses). A mismatch between
        // the reparsed atom count and the stored n_atoms means the coordinate order can't be
        // trusted for descriptors — keep the coordinates, emit zero-filled descriptors, and count.
        sfpost::molecule_shape_context_t const context = sfpost::build_shape_context(smiles[molecule_index]);
        bool const context_usable = context.ok && context.num_atoms == num_atoms_u8;
        if (!context_usable) stats.context_mismatch.fetch_add(1, std::memory_order_relaxed);
        std::uint32_t const num_atoms = num_atoms_u8;

        std::array<float, sfapi::max_atoms_per_molecule_k * 3u> interleaved {};
        for (std::size_t conformer_slot = 0; conformer_slot < rows.size() && conformer_slot < max_conformers;
             ++conformer_slot) {
            std::int64_t const row = rows[conformer_slot];
            std::uint8_t const row_status = shard.status->Value(row);
            std::size_t const flat = static_cast<std::size_t>(molecule_index) * max_conformers + conformer_slot;
            status_codes[flat] = row_status;
            if (shard.conformer_energy->IsValid(row)) energies[flat] = shard.conformer_energy->Value(row);
            if (row_status != 0 || !coords_list.IsValid(row)) continue; // failure row stays null

            std::size_t const begin = static_cast<std::size_t>(coords_list.value_offset(row));
            std::size_t const stored = static_cast<std::size_t>(coords_list.value_offset(row + 1)) - begin;
            std::size_t const wanted = static_cast<std::size_t>(num_atoms) * 3u;
            std::size_t const copy = std::min(stored, wanted);
            for (std::size_t value_index = 0; value_index < copy; ++value_index)
                interleaved[value_index] = half_bits_to_float(
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

            for (std::uint32_t atom = 0; atom < num_atoms; ++atom)
                for (std::uint32_t axis = 0; axis < 3u; ++axis)
                    coordinates[slot_base + slot_layout.component_offset(axis, atom)] = float_to_half(
                        interleaved[atom * 3u + axis]);

            if (context_usable) {
                std::array<double, 60> usrcat_row {};
                sfpost::usrcat_descriptor(coords_view, context, usrcat_row);
                std::copy(usrcat_row.begin(), usrcat_row.end(), usrcat.data() + flat * 60u);
            }
            // else: descriptors stay zero-filled (not null), coordinates preserved.
        }
    }

    // Hand the reconstructed slot layout to the producer's writer for byte-identical output.
    sfcli::writer_inputs_t writer_inputs {
        std::string(shard.input_shard->GetView(0)),
        config.dataset,
        base_row,
        num_molecules,
        max_conformers,
        max_atoms,
        std::span<std::string_view const>(smiles.data(), num_molecules),
        std::span<std::uint8_t const>(num_atoms_per_molecule.data(), num_molecules),
        std::span<std::uint16_t const>(num_heavy_atoms_per_molecule.data(), num_molecules),
        std::span<std::uint16_t const>(num_bonds_per_molecule.data(), num_molecules),
        std::span<float const>(molecular_weight_per_molecule.data(), num_molecules),
        std::span<std::uint8_t const>(n_conformers_per_molecule.data(), num_molecules),
        std::span<std::uint8_t const>(status_codes.data(), total_slots),
        std::span<__half const>(coordinates.data(), coordinates.size()),
        std::span<float const>(energies.data(), total_slots),
        std::span<double const>(usrcat.data(), usrcat.size()),
    };
    return sfcli::write_shard(output_path, writer_inputs);
}

#pragma endregion Shard Augmentation

} // namespace

int main(int argc, char **argv) {
    cli_config_t config;
    if (!parse_cli(argc, argv, config)) return 2;

    logger_t logger;
    std::error_code error_code;
    fs::create_directories(config.output_dir, error_code);

    // Inventory: every *.3D.parquet in --input is an input here (unlike smallfield_batch,
    // where *.3D.parquet names are outputs). Skip the .encoded sentinels themselves.
    std::vector<fs::path> inputs;
    for (auto const &entry : fs::directory_iterator(config.input_dir)) {
        std::string const name = entry.path().filename().string();
        if (name.ends_with(".3D.parquet")) inputs.push_back(entry.path());
    }
    std::sort(inputs.begin(), inputs.end());
    if (inputs.empty()) {
        logger.error("no *.3D.parquet input shards found in {}", config.input_dir.string());
        return 1;
    }

    reembed_index_t reembed;
    if (!config.reembed_dir.empty()) reembed = build_reembed_index(config.reembed_dir, logger);

    logger.info("smallfield_descriptors: {} input shards, dataset='{}', reembed={}", thousands(inputs.size()),
                config.dataset, config.reembed_dir.empty() ? "off" : config.reembed_dir.string());

    std::uint64_t completed = 0, skipped = 0, failed = 0;
    std::uint64_t total_context_mismatch = 0, total_separated = 0;
    auto const start = std::chrono::steady_clock::now();

    for (auto const &input_path : inputs) {
        fs::path const output_path = config.output_dir / input_path.filename();
        fs::path sentinel = output_path;
        sentinel += ".encoded";
        if (config.resume && fs::exists(sentinel)) {
            ++skipped;
            continue;
        }

        auto const shard_start = std::chrono::steady_clock::now();
        input_shard_t shard;
        if (!read_input_shard(input_path, shard)) {
            logger.error("read failed, skipping: {}", input_path.string());
            ++failed;
            continue;
        }

        shard_stats_t stats;
        if (!augment_shard(shard, config, reembed, output_path, logger, stats)) {
            logger.error("write failed: {}", output_path.string());
            ++failed;
            continue;
        }

        FILE *sentinel_file = std::fopen(sentinel.string().c_str(), "wb");
        if (sentinel_file) {
            // `--resume` skips any shard whose sentinel exists, so a short write would leave a
            // file that makes the next run skip an unprocessed shard.
            std::size_t const written = std::fwrite("v1\n", 1, 3, sentinel_file);
            std::fclose(sentinel_file);
            if (written != 3) {
                std::error_code remove_error;
                std::filesystem::remove(sentinel, remove_error);
                logger.error("sentinel write truncated, removed: {}", sentinel.string());
            }
        }
        else { logger.error("sentinel write failed: {}", sentinel.string()); }

        double const shard_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - shard_start).count();
        std::uint64_t const rows = static_cast<std::uint64_t>(shard.num_rows);
        std::uint64_t const context_mismatch = stats.context_mismatch.load();
        std::uint64_t const separated = stats.fragments_separated.load();
        total_context_mismatch += context_mismatch;
        total_separated += separated;
        ++completed;

        int const threads = config.threads > 0 ? config.threads : omp_get_max_threads();
        double const rows_per_second = rows / std::max(shard_seconds, 1e-3);
        logger.info("{:>4}/{:<4} {}  {:.1f}s  {} rows/s ({} rows/s/core)  ctx-mismatch {}  separated {}",
                    thousands(completed + skipped), thousands(inputs.size()), input_path.filename().string(),
                    shard_seconds, thousands(static_cast<std::uint64_t>(rows_per_second)),
                    thousands(static_cast<std::uint64_t>(rows_per_second / std::max(threads, 1))),
                    thousands(context_mismatch), thousands(separated));
    }

    double const total = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    logger.info("complete  {} augmented, {} skipped, {} failed  in {:.0f}s  (ctx-mismatch {}, separated {})",
                thousands(completed), thousands(skipped), thousands(failed), total, thousands(total_context_mismatch),
                thousands(total_separated));
    return failed > 0 ? 1 : 0;
}
