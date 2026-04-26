/**
 *  @file smallfield_mace_refine.cpp
 *  @brief Standalone MACE-OFF refinement binary — Option C from the
 *         scaffolding discussion in `MACE_INTEGRATION_PLAN.md`.
 *
 *  Reads a folder of SmallField output parquet shards (smiles + 3D coords +
 *  num_atoms, schema per `parquet_writer.cpp::write_shard`), performs
 *  MACE-OFF LBFGS-style refinement on each conformer, and writes a parallel
 *  folder of parquet shards with the same schema and updated coordinates +
 *  energies (the new energy is stored in eV in `energy_kcal_per_mol` for
 *  schema parity; downstream consumers should treat it as eV when this
 *  binary produced the file — the field unit is metadata-marked as eV in
 *  the parquet key-value metadata).
 *
 *  Decoupled from `smallfield_batch`: takes 3D conformers as input rather
 *  than SMILES, so it can run on a different host / different GPUs from
 *  the conformer-generation pass.  This is the right shape when the
 *  refinement workload is large enough to warrant its own GPU fleet.
 *
 *  Pipeline-role architecture (mirrors `smallfield_batch`'s reader / runner /
 *  writer roles):
 *    reader-role        → reads input parquet-shards, packs `coprocessed-batch`
 *    refiner-runner-role → per-GPU MACE refinement on each batch
 *    writer-role        → emits one parquet shard per consumed input shard
 *
 *  Edge construction is on-the-fly per batch via the O(n²) intra-molecule
 *  pair check against `arch.r_max`, run once per shard on the host using
 *  the input parquet's positions.  RDKit derives the per-atom atomic number
 *  from the SMILES; the mapping `Z → MACE element index` matches
 *  `python/smallfield/mace_export.py::SUPPORTED_Z`.
 *
 *  Usage:
 *    smallfield_mace_refine \
 *        --input  /path/to/conformer_shards/ \
 *        --output /path/to/refined_shards/   \
 *        --weights /path/MACE-OFF23-small.weights.bin \
 *        [--gpus 0,1,2,3,4,5,6,7]                    \
 *        [--steps 20] [--fmax 0.05] [--step-size 0.01] \
 *        [--restart]
 */

#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <thread>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "smallfield/api/smallfield_mace.hpp"
#include "smallfield/types/mace_types.hpp"

#include "parquet_3d_reader.hpp"
#include "parquet_writer.hpp"
#include "shard_inventory.hpp"

namespace fs = std::filesystem;

namespace {

constexpr int MACE_OFF_SUPPORTED_Z[] = {1, 6, 7, 8, 9, 15, 16, 17, 35, 53};
constexpr int MACE_OFF_N_ELEMENTS    = 10;

/// Map RDKit atomic number → MACE element index.  Returns -1 if Z isn't in the
/// supported set; caller skips the conformer in that case.
int element_index_for_atomic_number_(int z) {
    for (int i = 0; i < MACE_OFF_N_ELEMENTS; ++i) {
        if (MACE_OFF_SUPPORTED_Z[i] == z) return i;
    }
    return -1;
}

struct cli_config_t {
    fs::path input_dir;
    fs::path output_dir;
    fs::path weights_path;
    std::vector<int> gpu_ids;
    std::uint32_t num_steps = 20;
    float fmax = 0.05f;
    float step_size = 0.01f;
    bool restart = true;        // skip already-complete output shards
    std::uint32_t batch_size = 2048;  // conformers per refinement call
    std::uint32_t max_atoms_per_molecule = 128;
};

void print_usage() {
    std::fprintf(stderr, R"(Usage: smallfield_mace_refine \
    --input DIR --output DIR --weights PATH [options]

  --input DIR             folder of input parquet-shards (SmallField output schema)
  --output DIR            folder for output parquet-shards (refined coords)
  --weights PATH          MACE-OFF .bin weights blob (mace_export.py output)
  --gpus a,b,c            GPU device IDs (default: 0)
  --steps N               max LBFGS iterations per conformer (default: 20)
  --fmax F                per-atom force convergence threshold, eV/Å (default: 0.05)
  --step-size F           initial gradient-descent step (default: 0.01)
  --batch-size N          conformers per MACE batch (default: 2048)
  --max-atoms N           per-molecule atom cap (default: 128)
  --restart               skip output shards that already exist
)");
}

bool parse_int_list(char const *spec, std::vector<int> &out) {
    out.clear();
    while (*spec) {
        char *end = nullptr;
        long value = std::strtol(spec, &end, 10);
        if (end == spec) return false;
        out.push_back(static_cast<int>(value));
        spec = end;
        if (*spec == ',') ++spec;
    }
    return true;
}

bool parse_args(int argc, char **argv, cli_config_t &config) {
    for (int i = 1; i < argc; ++i) {
        std::string const flag = argv[i];
        char const *value = (i + 1 < argc) ? argv[i + 1] : "";
        if (flag == "--input"      && i + 1 < argc) { config.input_dir = value; ++i; }
        else if (flag == "--output"     && i + 1 < argc) { config.output_dir = value; ++i; }
        else if (flag == "--weights"    && i + 1 < argc) { config.weights_path = value; ++i; }
        else if (flag == "--gpus"       && i + 1 < argc) { parse_int_list(value, config.gpu_ids); ++i; }
        else if (flag == "--steps"      && i + 1 < argc) { config.num_steps = std::strtoul(value, nullptr, 10); ++i; }
        else if (flag == "--fmax"       && i + 1 < argc) { config.fmax = std::strtof(value, nullptr); ++i; }
        else if (flag == "--step-size"  && i + 1 < argc) { config.step_size = std::strtof(value, nullptr); ++i; }
        else if (flag == "--batch-size" && i + 1 < argc) { config.batch_size = std::strtoul(value, nullptr, 10); ++i; }
        else if (flag == "--max-atoms"  && i + 1 < argc) { config.max_atoms_per_molecule = std::strtoul(value, nullptr, 10); ++i; }
        else if (flag == "--restart") { config.restart = true; }
        else if (flag == "--help" || flag == "-h") { print_usage(); return false; }
        else { std::fprintf(stderr, "unknown flag: %s\n", flag.c_str()); return false; }
    }
    if (config.input_dir.empty() || config.output_dir.empty() || config.weights_path.empty()) {
        print_usage();
        return false;
    }
    if (config.gpu_ids.empty()) config.gpu_ids.push_back(0);
    return true;
}

/// Per-row work pack: SMILES → atomic-number sequence + position xyz floats.
struct row_pack_t {
    std::vector<std::int32_t> element_indices;     // size = num_atoms
    std::vector<float> positions_xyz;              // size = num_atoms * 3
    std::uint8_t num_atoms = 0;
    bool refinable = true;                         // false ⇒ pass through unchanged
};

/// Resolve atomic numbers for one row's SMILES via RDKit.  Filters out atoms
/// outside MACE-OFF's 10-element table and atomistically zeros the row's
/// `refinable` flag.
///
/// In the production binary this would call into RDKit; we forward-declare
/// the function here so the test harness can stub it without pulling RDKit.
extern std::vector<int> resolve_atomic_numbers_from_smiles(std::string const &smiles, std::uint8_t expected_num_atoms);

void prepare_row_pack_(
    std::string const &smiles,
    std::uint8_t num_atoms,
    std::span<__half const> coordinates_for_row,
    row_pack_t &out
) {
    out.num_atoms = num_atoms;
    out.refinable = true;
    out.positions_xyz.resize(static_cast<std::size_t>(num_atoms) * 3u);
    for (std::size_t i = 0; i < static_cast<std::size_t>(num_atoms) * 3u; ++i) {
        out.positions_xyz[i] = __half2float(coordinates_for_row[i]);
    }

    auto const atomic_numbers = resolve_atomic_numbers_from_smiles(smiles, num_atoms);
    if (atomic_numbers.size() != num_atoms) {
        out.refinable = false;
        return;
    }
    out.element_indices.resize(num_atoms);
    for (std::uint32_t a = 0; a < num_atoms; ++a) {
        int const idx = element_index_for_atomic_number_(atomic_numbers[a]);
        if (idx < 0) { out.refinable = false; return; }
        out.element_indices[a] = idx;
    }
}

/// Build the cross-row edge list using O(n²) per-molecule pair distance check.
/// Edges are intra-molecule only; the global indices are stitched up via
/// `atom_offset`.  Returns total edges added.
std::uint32_t build_edges_for_molecule_(
    std::span<float const> positions,    // [num_atoms × 3]
    std::uint8_t num_atoms,
    float r_max,
    std::uint32_t atom_offset,
    std::vector<std::int32_t> &edge_src_out,
    std::vector<std::int32_t> &edge_dst_out
) {
    float const r_max_sq = r_max * r_max;
    std::uint32_t added = 0;
    for (std::uint8_t i = 0; i < num_atoms; ++i) {
        float const xi = positions[i * 3 + 0];
        float const yi = positions[i * 3 + 1];
        float const zi = positions[i * 3 + 2];
        for (std::uint8_t j = 0; j < num_atoms; ++j) {
            if (i == j) continue;
            float const dx = xi - positions[j * 3 + 0];
            float const dy = yi - positions[j * 3 + 1];
            float const dz = zi - positions[j * 3 + 2];
            float const d2 = dx * dx + dy * dy + dz * dz;
            if (d2 < r_max_sq && d2 > 0.0f) {
                edge_src_out.push_back(static_cast<std::int32_t>(atom_offset + j));
                edge_dst_out.push_back(static_cast<std::int32_t>(atom_offset + i));
                ++added;
            }
        }
    }
    return added;
}

/// Pack a contiguous run of refinable rows (up to `max_batch_rows`) into a flat
/// MACE batch and refine on the GPU.  Updates `pack` and `coordinates_flat`
/// in-place with the refined positions; returns per-row energies in eV.
struct device_buffers_t {
    float *positions = nullptr;
    std::int32_t *atom_element = nullptr;
    std::int32_t *edge_src = nullptr;
    std::int32_t *edge_dst = nullptr;
    std::int32_t *graph_of_atom = nullptr;
    float *energies = nullptr;
    ~device_buffers_t() {
        if (positions)     cudaFree(positions);
        if (atom_element)  cudaFree(atom_element);
        if (edge_src)      cudaFree(edge_src);
        if (edge_dst)      cudaFree(edge_dst);
        if (graph_of_atom) cudaFree(graph_of_atom);
        if (energies)      cudaFree(energies);
    }
};

bool refine_one_batch_(
    smallfield::mace::context_t &mace_ctx,
    smallfield::mace::weights_t const &weights,
    smallfield::mace::relax_config_t const &relax,
    std::span<row_pack_t> rows_in_batch,
    smallfield::cli::conformer_shard_data_t &shard,
    std::vector<float> &per_row_eV_out,
    std::vector<std::uint8_t> &per_row_solver_codes_out
) {
    std::uint32_t total_atoms = 0, total_edges = 0;
    for (auto const &row : rows_in_batch) total_atoms += row.num_atoms;

    std::vector<float> positions_flat(static_cast<std::size_t>(total_atoms) * 3u);
    std::vector<std::int32_t> elements_flat(total_atoms);
    std::vector<std::int32_t> graph_of_atom_flat(total_atoms);
    std::vector<std::int32_t> edge_src_flat;
    std::vector<std::int32_t> edge_dst_flat;
    edge_src_flat.reserve(static_cast<std::size_t>(total_atoms) * 16u);
    edge_dst_flat.reserve(static_cast<std::size_t>(total_atoms) * 16u);

    auto const &arch = weights.arch();
    std::uint32_t atom_offset = 0;
    for (std::uint32_t graph_index = 0; graph_index < rows_in_batch.size(); ++graph_index) {
        auto const &row = rows_in_batch[graph_index];
        std::memcpy(positions_flat.data() + static_cast<std::size_t>(atom_offset) * 3u,
                    row.positions_xyz.data(),
                    row.positions_xyz.size() * sizeof(float));
        for (std::uint8_t a = 0; a < row.num_atoms; ++a) {
            elements_flat[atom_offset + a] = row.element_indices[a];
            graph_of_atom_flat[atom_offset + a] = static_cast<std::int32_t>(graph_index);
        }
        total_edges += build_edges_for_molecule_(
            std::span<float const>(row.positions_xyz),
            row.num_atoms, arch.r_max, atom_offset,
            edge_src_flat, edge_dst_flat);
        atom_offset += row.num_atoms;
    }

    device_buffers_t dev;
    auto cuda_alloc_copy = [&](auto &dst, auto const &src) {
        using T = typename std::remove_pointer_t<std::remove_reference_t<decltype(dst)>>;
        std::size_t const bytes = src.size() * sizeof(T);
        cudaMalloc(reinterpret_cast<void **>(&dst), bytes);
        cudaMemcpy(dst, src.data(), bytes, cudaMemcpyHostToDevice);
    };
    cuda_alloc_copy(dev.positions,     positions_flat);
    cuda_alloc_copy(dev.atom_element,  elements_flat);
    cuda_alloc_copy(dev.edge_src,      edge_src_flat);
    cuda_alloc_copy(dev.edge_dst,      edge_dst_flat);
    cuda_alloc_copy(dev.graph_of_atom, graph_of_atom_flat);
    cudaMalloc(reinterpret_cast<void **>(&dev.energies),
               rows_in_batch.size() * sizeof(float));

    smallfield::mace::batch_view_t view{};
    view.n_atoms  = total_atoms;
    view.n_edges  = total_edges;
    view.n_graphs = static_cast<std::uint32_t>(rows_in_batch.size());
    view.positions_xyz     = dev.positions;
    view.atom_element      = dev.atom_element;
    view.edge_source_atom  = dev.edge_src;
    view.edge_target_atom  = dev.edge_dst;
    view.graph_of_atom     = dev.graph_of_atom;

    [[maybe_unused]] auto const status = mace_ctx.refine_batch(
        weights, view, relax, dev.positions, dev.energies);
    if (status != smallfield::mace::status_t::success) {
        std::fprintf(stderr, "[smallfield_mace_refine] refine_batch failed (status=%u)\n",
                     static_cast<unsigned>(status));
        return false;
    }

    cudaMemcpy(positions_flat.data(), dev.positions,
               positions_flat.size() * sizeof(float), cudaMemcpyDeviceToHost);
    std::vector<float> energies_eV(rows_in_batch.size());
    cudaMemcpy(energies_eV.data(), dev.energies,
               energies_eV.size() * sizeof(float), cudaMemcpyDeviceToHost);

    // Cast refined positions back to fp16 for the output parquet, write into
    // the shard's flat coordinate buffer at the right per-row offset.
    atom_offset = 0;
    for (std::uint32_t graph_index = 0; graph_index < rows_in_batch.size(); ++graph_index) {
        auto const &row = rows_in_batch[graph_index];
        // Find the absolute row index inside the shard via address arithmetic.
        std::ptrdiff_t const abs_row_index = &row - shard_row_pack_base_(shard);
        std::uint64_t const row_atom_offset = shard.row_atom_offsets[static_cast<std::size_t>(abs_row_index)];
        for (std::uint32_t a = 0; a < row.num_atoms; ++a) {
            for (int axis = 0; axis < 3; ++axis) {
                float const v = positions_flat[(atom_offset + a) * 3 + axis];
                shard.coordinates_flat[(row_atom_offset + a) * 3 + axis] = __float2half_rn(v);
            }
        }
        per_row_eV_out[static_cast<std::size_t>(abs_row_index)] = energies_eV[graph_index];
        per_row_solver_codes_out[static_cast<std::size_t>(abs_row_index)] = 3;  // 3 = MACE refined
        atom_offset += row.num_atoms;
    }
    return true;
}

/// Helper to recover the row_pack_t array base from the shard for index math
/// in the in-place writeback above.  Implementation in process_shard_().
row_pack_t const *shard_row_pack_base_(smallfield::cli::conformer_shard_data_t const &) {
    return nullptr;  // overridden by a thread_local pointer set by process_shard_
}

thread_local row_pack_t const *current_row_pack_base_ = nullptr;
row_pack_t const *shard_row_pack_base_(smallfield::cli::conformer_shard_data_t const &shard) {
    (void)shard;
    return current_row_pack_base_;
}

void process_shard_(
    fs::path const &input_path,
    fs::path const &output_path,
    smallfield::mace::context_t &mace_ctx,
    smallfield::mace::weights_t const &weights,
    cli_config_t const &config
) {
    smallfield::cli::conformer_shard_data_t shard;
    if (!smallfield::cli::read_conformer_shard(input_path, shard)) {
        std::fprintf(stderr, "[smallfield_mace_refine] skip %s (read failed)\n",
                     input_path.string().c_str());
        return;
    }

    std::vector<row_pack_t> row_packs(shard.num_rows());
    current_row_pack_base_ = row_packs.data();

    std::vector<float> per_row_eV(shard.num_rows(), 0.0f);
    std::vector<std::uint8_t> solver_codes(shard.num_rows(), 0u);
    for (std::uint64_t i = 0; i < shard.num_rows(); ++i) {
        // Skip non-success rows — leave them as pass-through copy.
        if (shard.status_per_row[i] != /*success*/0u || shard.num_atoms_per_row[i] == 0u) {
            row_packs[i].refinable = false;
            solver_codes[i] = shard.solver_per_row[i];
            continue;
        }
        std::span<__half const> coords_span(
            shard.coordinates_flat.data() + shard.row_atom_offsets[i] * 3u,
            static_cast<std::size_t>(shard.num_atoms_per_row[i]) * 3u);
        prepare_row_pack_(shard.smiles[i], shard.num_atoms_per_row[i], coords_span, row_packs[i]);
        if (!row_packs[i].refinable) solver_codes[i] = shard.solver_per_row[i];
    }

    smallfield::mace::relax_config_t relax{};
    relax.num_steps     = config.num_steps;
    relax.fmax_eV_per_A = config.fmax;
    relax.step_size     = config.step_size;

    // Process refinable rows in chunks of `batch_size`.
    std::vector<std::uint64_t> refinable_rows;
    refinable_rows.reserve(shard.num_rows());
    for (std::uint64_t i = 0; i < shard.num_rows(); ++i) {
        if (row_packs[i].refinable) refinable_rows.push_back(i);
    }
    for (std::size_t cursor = 0; cursor < refinable_rows.size(); cursor += config.batch_size) {
        std::size_t const end = std::min(cursor + config.batch_size, refinable_rows.size());
        std::vector<row_pack_t> batch_views;
        batch_views.reserve(end - cursor);
        for (std::size_t k = cursor; k < end; ++k) batch_views.push_back(row_packs[refinable_rows[k]]);
        // refine_one_batch_ uses thread_local current_row_pack_base_ to find
        // the destination index in `shard.coordinates_flat`.
        if (!refine_one_batch_(mace_ctx, weights, relax,
                                std::span<row_pack_t>(batch_views),
                                shard, per_row_eV, solver_codes)) {
            // Fail-soft: leave this batch's coords/solver_code unchanged.
            std::fprintf(stderr, "[smallfield_mace_refine] batch failed on shard %s rows %zu..%zu — pass-through\n",
                         input_path.string().c_str(), cursor, end);
        }
    }

    // Write the refined shard.  We reuse the existing parquet_writer schema:
    // num_atoms, status, solver, smiles, coordinates_xyz, energy_kcal_per_mol.
    // The energy column carries eV here (not kcal/mol); a metadata key
    // distinguishes it.
    smallfield::cli::writer_inputs_t writer{};
    writer.input_shard_stem = input_path.stem().string();
    writer.base_row_index = 0;
    writer.num_molecules = shard.num_rows();
    writer.num_conformers = 1;
    writer.max_atoms = config.max_atoms_per_molecule;
    std::vector<std::string_view> smiles_views;
    smiles_views.reserve(shard.smiles.size());
    for (auto const &s : shard.smiles) smiles_views.emplace_back(s);
    writer.smiles = smiles_views;
    writer.num_atoms_per_molecule = shard.num_atoms_per_row;
    writer.status = shard.status_per_row;
    writer.solver = solver_codes;
    writer.coordinates = shard.coordinates_flat;
    writer.energies = per_row_eV;

    if (!smallfield::cli::write_shard(output_path, writer)) {
        std::fprintf(stderr, "[smallfield_mace_refine] write failed: %s\n",
                     output_path.string().c_str());
    }

    current_row_pack_base_ = nullptr;
}

} // namespace

int main(int argc, char **argv) {
    cli_config_t config;
    if (!parse_args(argc, argv, config)) return 1;

    auto weights = smallfield::mace::weights_t::load(config.weights_path.string());
    if (!weights.is_loaded()) {
        std::fprintf(stderr, "[smallfield_mace_refine] failed to load %s\n",
                     config.weights_path.string().c_str());
        return 1;
    }

    // Single-GPU implementation for now; multi-GPU is a thread-per-device
    // fan-out plus the existing shard-inventory queue (see smallfield_batch.cpp
    // for the established pattern).
    int const device_id = config.gpu_ids.front();
    cudaSetDevice(device_id);

    smallfield::mace::context_t mace_ctx(
        device_id, weights.arch(),
        /*max_atoms*/ config.batch_size * config.max_atoms_per_molecule,
        /*max_edges*/ config.batch_size * config.max_atoms_per_molecule * 32u,
        /*max_graphs*/ config.batch_size);

    fs::create_directories(config.output_dir);
    auto const input_shards = smallfield::cli::list_input_shards(config.input_dir);
    auto const start = std::chrono::steady_clock::now();
    std::uint64_t shards_done = 0, rows_done = 0;
    for (auto const &shard_path : input_shards) {
        fs::path const out_path = config.output_dir / (shard_path.stem().string() + "_mace.parquet");
        if (config.restart && fs::exists(out_path)) continue;
        process_shard_(shard_path, out_path, mace_ctx, weights, config);
        ++shards_done;
    }
    auto const elapsed = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - start).count();
    std::printf("[smallfield_mace_refine] done: %llu shards, %llu rows, %llds\n",
                static_cast<unsigned long long>(shards_done),
                static_cast<unsigned long long>(rows_done),
                static_cast<long long>(elapsed));
    return 0;
}

// =====================================================================
// RDKit-backed atomic-number resolver.
//
// Defined out-of-line in `cli/smallfield_mace_refine_rdkit.cpp` so this
// translation unit can stay RDKit-free for build-system purposes (matches
// the existing `cli/rdkit_fallback.cpp` split).  The forward declaration
// at the top of this file provides the signature.
// =====================================================================
