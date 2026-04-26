/**
 *  @file bench_mace_inference.cu
 *  @brief Production-scale batched throughput benchmark for the SmallField
 *         MACE-OFF inference module on H100 SXM5.
 *
 *  Sweeps the cross-product of `(batch_size, n_atoms_per_mol)` over the
 *  range that actually saturates the GPU — the single-molecule wall-time
 *  number reported by `test_mace_loader` reflects the launch-overhead
 *  floor, not real throughput.  Reported metric is `ms_per_config`
 *  (total elapsed time / iters / batch_size), the directly-comparable
 *  number to the cueq Python baseline in `MACE_BENCH_REPORT.md`.
 *
 *  Both the fp32 reference and the ThunderKittens (TK) build are
 *  exercised by the same translation unit; the build flavour is
 *  selected by the presence of `SMALLFIELD_USE_THUNDERKITTENS` at
 *  compile time.  Output JSONL goes to
 *  `/tmp/mace_inference_bench_<build>.jsonl`.
 *
 *  Synthetic batch construction:
 *
 *    1. Replicate one drug-like molecule `batch_size` times.
 *    2. Per molecule: per-atom positions = Gaussian(σ=0.4 Å) plus a
 *       linear spread along the x-axis at 1.5 Å/atom — same kind of
 *       coarse, dense-cluster geometry the export tool's `make_drug_like`
 *       helper produces.  Atom elements are uniform random integers
 *       in `[0, n_elements)`.
 *    3. Edge list: O(n_atoms²) brute-force pair check against `r_max`
 *       per molecule.  Edges are intra-molecule only.
 *    4. Sort edges by `edge_target_atom` per molecule and build the
 *       per-receiver `receiver_edge_segment_{start,end}` arrays so
 *       the TK build's fused conv_tp+scatter takes the segmented path
 *       instead of the cudaMalloc-bearing atomic fallback.
 */

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>

#include <cuda_runtime.h>

#include "smallfield/parameters/mace_weights.hpp"
#include "smallfield/types/mace_types.hpp"
#include "smallfield/cuda/mace_inference.cuh"

namespace {

#define CUDA_CHECK(stmt)                                                                              \
    do {                                                                                              \
        cudaError_t const err__ = (stmt);                                                             \
        if (err__ != cudaSuccess) {                                                                   \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,                     \
                         cudaGetErrorString(err__));                                                  \
            std::exit(1);                                                                             \
        }                                                                                             \
    } while (0)

#if defined(SMALLFIELD_USE_THUNDERKITTENS)
constexpr char const *bench_build_label_k = "tk";
#else
constexpr char const *bench_build_label_k = "fp32";
#endif

/** One synthetic config in the sweep grid. */
struct bench_config_t {
    std::uint32_t batch_size;
    std::uint32_t n_atoms_per_mol;
};

/** Output of one timing run. */
struct bench_result_t {
    std::uint32_t batch_size;
    std::uint32_t n_atoms_per_mol;
    std::uint32_t n_total_atoms;
    std::uint32_t n_total_edges;
    double ms_per_iter;
    double ms_per_config;
    double mean_energy_eV;
    double min_energy_eV;
    double max_energy_eV;
    bool segmented_path_active;
};

/**
 *  @brief Build a synthetic batch of `batch_size` drug-like-ish molecules,
 *         each with `n_atoms_per_mol` atoms.
 *
 *  All host-side outputs are sized for the whole batch so the upload to
 *  device is one cudaMemcpy per array.
 */
void build_synthetic_batch(
    std::uint32_t batch_size,
    std::uint32_t n_atoms_per_mol,
    std::uint32_t n_elements,
    float r_max,
    std::uint64_t seed,
    std::vector<float> &positions_xyz,
    std::vector<std::int32_t> &atom_element,
    std::vector<std::int32_t> &graph_of_atom,
    std::vector<std::int32_t> &edge_source_atom,
    std::vector<std::int32_t> &edge_target_atom,
    std::vector<std::int32_t> &graph_atom_offset,
    std::vector<std::int32_t> &graph_edge_offset,
    std::vector<std::int32_t> &receiver_edge_segment_start,
    std::vector<std::int32_t> &receiver_edge_segment_end
) {
    std::mt19937_64 random_engine(seed);
    std::normal_distribution<float> gaussian_distribution(0.0f, 0.4f);
    std::uniform_int_distribution<std::int32_t> element_distribution(0, static_cast<std::int32_t>(n_elements) - 1);

    std::uint32_t const n_total_atoms = batch_size * n_atoms_per_mol;
    positions_xyz.assign(static_cast<std::size_t>(n_total_atoms) * 3u, 0.0f);
    atom_element.assign(n_total_atoms, 0);
    graph_of_atom.assign(n_total_atoms, 0);

    graph_atom_offset.assign(batch_size + 1u, 0);
    for (std::uint32_t graph_index = 0; graph_index <= batch_size; ++graph_index) {
        graph_atom_offset[graph_index] = static_cast<std::int32_t>(graph_index * n_atoms_per_mol);
    }

    // Per-molecule atom population: Gaussian jitter + a linear spread along x
    // at 1.5 A/atom.  This produces molecules with ~5-25 intra-r_max
    // neighbours per atom for r_max = 4.5 A and n_atoms ≤ 80, which is the
    // drug-like density regime the production pipeline targets.
    for (std::uint32_t graph_index = 0; graph_index < batch_size; ++graph_index) {
        std::uint32_t const base_atom_index = graph_index * n_atoms_per_mol;
        for (std::uint32_t atom_within_mol = 0; atom_within_mol < n_atoms_per_mol; ++atom_within_mol) {
            std::uint32_t const flat_atom_index = base_atom_index + atom_within_mol;
            float const linear_x = 1.5f * static_cast<float>(atom_within_mol);
            positions_xyz[static_cast<std::size_t>(flat_atom_index) * 3u + 0u] =
                linear_x + gaussian_distribution(random_engine);
            positions_xyz[static_cast<std::size_t>(flat_atom_index) * 3u + 1u] =
                gaussian_distribution(random_engine);
            positions_xyz[static_cast<std::size_t>(flat_atom_index) * 3u + 2u] =
                gaussian_distribution(random_engine);
            atom_element[flat_atom_index] = element_distribution(random_engine);
            graph_of_atom[flat_atom_index] = static_cast<std::int32_t>(graph_index);
        }
    }

    // Brute-force edge enumeration per molecule (intra-molecule only).
    edge_source_atom.clear();
    edge_target_atom.clear();
    edge_source_atom.reserve(static_cast<std::size_t>(n_total_atoms) * 12u);
    edge_target_atom.reserve(static_cast<std::size_t>(n_total_atoms) * 12u);

    graph_edge_offset.assign(batch_size + 1u, 0);
    float const r_max_squared = r_max * r_max;
    for (std::uint32_t graph_index = 0; graph_index < batch_size; ++graph_index) {
        std::uint32_t const base_atom_index = graph_index * n_atoms_per_mol;
        graph_edge_offset[graph_index] = static_cast<std::int32_t>(edge_source_atom.size());
        for (std::uint32_t source_local = 0; source_local < n_atoms_per_mol; ++source_local) {
            std::uint32_t const source_global = base_atom_index + source_local;
            float const source_x = positions_xyz[static_cast<std::size_t>(source_global) * 3u + 0u];
            float const source_y = positions_xyz[static_cast<std::size_t>(source_global) * 3u + 1u];
            float const source_z = positions_xyz[static_cast<std::size_t>(source_global) * 3u + 2u];
            for (std::uint32_t target_local = 0; target_local < n_atoms_per_mol; ++target_local) {
                if (source_local == target_local) continue;
                std::uint32_t const target_global = base_atom_index + target_local;
                float const dx = positions_xyz[static_cast<std::size_t>(target_global) * 3u + 0u] - source_x;
                float const dy = positions_xyz[static_cast<std::size_t>(target_global) * 3u + 1u] - source_y;
                float const dz = positions_xyz[static_cast<std::size_t>(target_global) * 3u + 2u] - source_z;
                float const dist_squared = dx * dx + dy * dy + dz * dz;
                if (dist_squared < r_max_squared && dist_squared > 1e-8f) {
                    edge_source_atom.push_back(static_cast<std::int32_t>(source_global));
                    edge_target_atom.push_back(static_cast<std::int32_t>(target_global));
                }
            }
        }
    }
    graph_edge_offset[batch_size] = static_cast<std::int32_t>(edge_source_atom.size());

    // Sort edges by edge_target_atom (ascending), per molecule, so the TK
    // segmented conv_tp+scatter path can index by receiver.  Single-thread
    // CPU sort is fine — the bench amortises over many forward iterations.
    std::uint32_t const total_edges = static_cast<std::uint32_t>(edge_source_atom.size());
    std::vector<std::int32_t> sorted_source(total_edges);
    std::vector<std::int32_t> sorted_target(total_edges);
    {
        std::vector<std::uint32_t> permutation_per_graph;
        for (std::uint32_t graph_index = 0; graph_index < batch_size; ++graph_index) {
            std::uint32_t const begin_edge = static_cast<std::uint32_t>(graph_edge_offset[graph_index]);
            std::uint32_t const end_edge   = static_cast<std::uint32_t>(graph_edge_offset[graph_index + 1u]);
            permutation_per_graph.resize(end_edge - begin_edge);
            for (std::uint32_t i = 0; i < end_edge - begin_edge; ++i) permutation_per_graph[i] = begin_edge + i;
            std::sort(permutation_per_graph.begin(), permutation_per_graph.end(),
                      [&](std::uint32_t a, std::uint32_t b) {
                          return edge_target_atom[a] < edge_target_atom[b];
                      });
            for (std::uint32_t i = 0; i < end_edge - begin_edge; ++i) {
                sorted_source[begin_edge + i] = edge_source_atom[permutation_per_graph[i]];
                sorted_target[begin_edge + i] = edge_target_atom[permutation_per_graph[i]];
            }
        }
    }
    edge_source_atom = std::move(sorted_source);
    edge_target_atom = std::move(sorted_target);

    // Per-receiver segment ranges over the global edge index, sized
    // [n_total_atoms].  An atom with no incoming edges gets start==end.
    receiver_edge_segment_start.assign(n_total_atoms, 0);
    receiver_edge_segment_end.assign(n_total_atoms, 0);
    {
        std::uint32_t edge_cursor = 0;
        for (std::uint32_t atom_index = 0; atom_index < n_total_atoms; ++atom_index) {
            std::uint32_t const segment_begin = edge_cursor;
            while (edge_cursor < total_edges &&
                   static_cast<std::uint32_t>(edge_target_atom[edge_cursor]) == atom_index) {
                ++edge_cursor;
            }
            receiver_edge_segment_start[atom_index] = static_cast<std::int32_t>(segment_begin);
            receiver_edge_segment_end[atom_index]   = static_cast<std::int32_t>(edge_cursor);
        }
        if (edge_cursor != total_edges) {
            std::fprintf(stderr,
                         "bench: edge sort produced unsorted output (cursor=%u total=%u) — sort key violated\n",
                         edge_cursor, total_edges);
            std::exit(1);
        }
    }
}

/** Run one config end-to-end: H2D upload, warmup, timed loop, D2H of energy. */
bench_result_t run_one_config(
    smallfield::parameters::mace_weights_t const &weights,
    bench_config_t const &config,
    std::uint64_t base_seed
) {
    auto const &arch = weights.arch();

    std::vector<float> host_positions_xyz;
    std::vector<std::int32_t> host_atom_element;
    std::vector<std::int32_t> host_graph_of_atom;
    std::vector<std::int32_t> host_edge_source_atom;
    std::vector<std::int32_t> host_edge_target_atom;
    std::vector<std::int32_t> host_graph_atom_offset;
    std::vector<std::int32_t> host_graph_edge_offset;
    std::vector<std::int32_t> host_receiver_segment_start;
    std::vector<std::int32_t> host_receiver_segment_end;

    build_synthetic_batch(
        config.batch_size, config.n_atoms_per_mol, arch.n_elements,
        arch.r_max, base_seed,
        host_positions_xyz, host_atom_element, host_graph_of_atom,
        host_edge_source_atom, host_edge_target_atom,
        host_graph_atom_offset, host_graph_edge_offset,
        host_receiver_segment_start, host_receiver_segment_end);

    std::uint32_t const n_total_atoms = config.batch_size * config.n_atoms_per_mol;
    std::uint32_t const n_total_edges = static_cast<std::uint32_t>(host_edge_source_atom.size());
    std::uint32_t const n_total_graphs = config.batch_size;

    // Device allocations for the batch.
    float *device_positions_xyz = nullptr;
    std::int32_t *device_atom_element = nullptr;
    std::int32_t *device_graph_of_atom = nullptr;
    std::int32_t *device_edge_source_atom = nullptr;
    std::int32_t *device_edge_target_atom = nullptr;
    std::int32_t *device_graph_atom_offset = nullptr;
    std::int32_t *device_graph_edge_offset = nullptr;
    std::int32_t *device_receiver_segment_start = nullptr;
    std::int32_t *device_receiver_segment_end = nullptr;

    CUDA_CHECK(cudaMalloc(&device_positions_xyz,         host_positions_xyz.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&device_atom_element,          host_atom_element.size() * sizeof(std::int32_t)));
    CUDA_CHECK(cudaMalloc(&device_graph_of_atom,         host_graph_of_atom.size() * sizeof(std::int32_t)));
    CUDA_CHECK(cudaMalloc(&device_edge_source_atom,      host_edge_source_atom.size() * sizeof(std::int32_t)));
    CUDA_CHECK(cudaMalloc(&device_edge_target_atom,      host_edge_target_atom.size() * sizeof(std::int32_t)));
    CUDA_CHECK(cudaMalloc(&device_graph_atom_offset,     host_graph_atom_offset.size() * sizeof(std::int32_t)));
    CUDA_CHECK(cudaMalloc(&device_graph_edge_offset,     host_graph_edge_offset.size() * sizeof(std::int32_t)));
    CUDA_CHECK(cudaMalloc(&device_receiver_segment_start, host_receiver_segment_start.size() * sizeof(std::int32_t)));
    CUDA_CHECK(cudaMalloc(&device_receiver_segment_end,   host_receiver_segment_end.size() * sizeof(std::int32_t)));

    CUDA_CHECK(cudaMemcpy(device_positions_xyz,         host_positions_xyz.data(),
                          host_positions_xyz.size() * sizeof(float),               cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_atom_element,          host_atom_element.data(),
                          host_atom_element.size() * sizeof(std::int32_t),         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_graph_of_atom,         host_graph_of_atom.data(),
                          host_graph_of_atom.size() * sizeof(std::int32_t),        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_edge_source_atom,      host_edge_source_atom.data(),
                          host_edge_source_atom.size() * sizeof(std::int32_t),     cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_edge_target_atom,      host_edge_target_atom.data(),
                          host_edge_target_atom.size() * sizeof(std::int32_t),     cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_graph_atom_offset,     host_graph_atom_offset.data(),
                          host_graph_atom_offset.size() * sizeof(std::int32_t),    cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_graph_edge_offset,     host_graph_edge_offset.data(),
                          host_graph_edge_offset.size() * sizeof(std::int32_t),    cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_receiver_segment_start, host_receiver_segment_start.data(),
                          host_receiver_segment_start.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_receiver_segment_end,   host_receiver_segment_end.data(),
                          host_receiver_segment_end.size() * sizeof(std::int32_t),   cudaMemcpyHostToDevice));

    smallfield::cuda::mace_scratch_t scratch;
    scratch.allocate(arch, n_total_atoms, n_total_edges, n_total_graphs);

    smallfield::types::mace_batch_t batch{};
    batch.n_atoms = n_total_atoms;
    batch.n_edges = n_total_edges;
    batch.n_graphs = n_total_graphs;
    batch.positions = device_positions_xyz;
    batch.atom_element = device_atom_element;
    batch.edge_source_atom = device_edge_source_atom;
    batch.edge_target_atom = device_edge_target_atom;
    batch.graph_of_atom = device_graph_of_atom;
    batch.graph_atom_offset = device_graph_atom_offset;
    batch.graph_edge_offset = device_graph_edge_offset;
    batch.receiver_edge_segment_start = device_receiver_segment_start;
    batch.receiver_edge_segment_end   = device_receiver_segment_end;

    // Sanity assertion: the segment pointers are non-null, so the TK build
    // should pick its segmented conv_tp+scatter (not the cudaMalloc-bearing
    // atomic fallback).  This only matters at compile time when TK is in,
    // but the assert is unconditional because the wiring is the same in
    // both builds.
    bool const segmented_path_active =
        (batch.receiver_edge_segment_start != nullptr) &&
        (batch.receiver_edge_segment_end   != nullptr);
    if (!segmented_path_active) {
        std::fprintf(stderr, "bench: segment pointers are null — TK fused path will fall back to atomic\n");
    }

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Warm up: 5 iterations.  Includes any one-shot precompute (W3_eff in TK build).
    constexpr int n_warmup_iters_k = 5;
    constexpr int n_timed_iters_k  = 20;
    for (int iter = 0; iter < n_warmup_iters_k; ++iter) {
        smallfield::cuda::mace_inference_reset_per_atom_energy_accumulator(scratch, n_total_atoms, stream);
        smallfield::cuda::mace_inference_forward(arch, weights.device(), batch, scratch, stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Timed loop.
    cudaEvent_t event_start = nullptr, event_end = nullptr;
    CUDA_CHECK(cudaEventCreate(&event_start));
    CUDA_CHECK(cudaEventCreate(&event_end));
    CUDA_CHECK(cudaEventRecord(event_start, stream));
    for (int iter = 0; iter < n_timed_iters_k; ++iter) {
        smallfield::cuda::mace_inference_reset_per_atom_energy_accumulator(scratch, n_total_atoms, stream);
        smallfield::cuda::mace_inference_forward(arch, weights.device(), batch, scratch, stream);
    }
    CUDA_CHECK(cudaEventRecord(event_end, stream));
    CUDA_CHECK(cudaEventSynchronize(event_end));
    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, event_start, event_end));

    // Pull energy back for the sanity check.
    std::vector<float> host_energy(n_total_graphs);
    CUDA_CHECK(cudaMemcpy(host_energy.data(), scratch.energy,
                          n_total_graphs * sizeof(float), cudaMemcpyDeviceToHost));

    double sum_energy = 0.0;
    double min_energy = static_cast<double>(host_energy.front());
    double max_energy = min_energy;
    for (float e : host_energy) {
        double const ed = static_cast<double>(e);
        sum_energy += ed;
        if (ed < min_energy) min_energy = ed;
        if (ed > max_energy) max_energy = ed;
    }
    double const mean_energy = sum_energy / static_cast<double>(n_total_graphs);

    bench_result_t result{};
    result.batch_size = config.batch_size;
    result.n_atoms_per_mol = config.n_atoms_per_mol;
    result.n_total_atoms = n_total_atoms;
    result.n_total_edges = n_total_edges;
    result.ms_per_iter = static_cast<double>(total_ms) / static_cast<double>(n_timed_iters_k);
    result.ms_per_config = result.ms_per_iter / static_cast<double>(config.batch_size);
    result.mean_energy_eV = mean_energy;
    result.min_energy_eV = min_energy;
    result.max_energy_eV = max_energy;
    result.segmented_path_active = segmented_path_active;

    cudaEventDestroy(event_start);
    cudaEventDestroy(event_end);
    cudaStreamDestroy(stream);
    cudaFree(device_positions_xyz);
    cudaFree(device_atom_element);
    cudaFree(device_graph_of_atom);
    cudaFree(device_edge_source_atom);
    cudaFree(device_edge_target_atom);
    cudaFree(device_graph_atom_offset);
    cudaFree(device_graph_edge_offset);
    cudaFree(device_receiver_segment_start);
    cudaFree(device_receiver_segment_end);

    return result;
}

void emit_jsonl(std::vector<bench_result_t> const &results, std::string const &out_path) {
    std::ofstream out(out_path);
    if (!out) {
        std::fprintf(stderr, "bench: cannot open %s for writing\n", out_path.c_str());
        return;
    }
    for (auto const &r : results) {
        out << "{"
            << "\"build\":\"" << bench_build_label_k << "\","
            << "\"batch_size\":" << r.batch_size << ","
            << "\"n_atoms_per_mol\":" << r.n_atoms_per_mol << ","
            << "\"n_total_atoms\":" << r.n_total_atoms << ","
            << "\"n_total_edges\":" << r.n_total_edges << ","
            << "\"ms_per_iter\":" << r.ms_per_iter << ","
            << "\"ms_per_config\":" << r.ms_per_config << ","
            << "\"mean_energy_eV\":" << r.mean_energy_eV << ","
            << "\"segmented_path_active\":" << (r.segmented_path_active ? "true" : "false")
            << "}\n";
    }
}

void print_summary_table(std::vector<bench_result_t> const &results,
                         std::vector<std::uint32_t> const &batch_sizes,
                         std::vector<std::uint32_t> const &atom_counts) {
    std::printf("\n=== Summary: ms / config (build = %s) ===\n", bench_build_label_k);
    std::printf("\n");
    std::printf("| n_atoms \\ batch |");
    for (auto bs : batch_sizes) std::printf(" %8u |", bs);
    std::printf("\n");
    std::printf("| --------------: |");
    for (std::size_t i = 0; i < batch_sizes.size(); ++i) std::printf(" -------: |");
    std::printf("\n");
    for (auto na : atom_counts) {
        std::printf("| %15u |", na);
        for (auto bs : batch_sizes) {
            double ms = -1.0;
            for (auto const &r : results) {
                if (r.batch_size == bs && r.n_atoms_per_mol == na) {
                    ms = r.ms_per_config;
                    break;
                }
            }
            std::printf(" %8.4f |", ms);
        }
        std::printf("\n");
    }
    std::printf("\n");

    // Diagnostic table: ms per iteration (whole batch) and total edges.
    std::printf("=== Diagnostic: ms / iter and total edges (build = %s) ===\n", bench_build_label_k);
    std::printf("\n");
    std::printf("| batch | n_atoms | n_total_edges | ms/iter | ms/config | mean E (eV) | segmented |\n");
    std::printf("| ----: | ------: | ------------: | ------: | --------: | ----------: | :-------: |\n");
    for (auto const &r : results) {
        std::printf("| %5u | %7u | %13u | %7.3f | %9.4f | %11.2f | %9s |\n",
                    r.batch_size, r.n_atoms_per_mol, r.n_total_edges,
                    r.ms_per_iter, r.ms_per_config, r.mean_energy_eV,
                    r.segmented_path_active ? "yes" : "no");
    }
    std::printf("\n");
}

} // namespace

int main(int argc, char **argv) {
    std::string const export_dir = (argc > 1) ? argv[1] : "/tmp/mace_export";
    std::string const weights_path = export_dir + "/MACE-OFF23-small.weights.bin";

    std::printf("bench_mace_inference: build=%s\n", bench_build_label_k);
    std::printf("bench_mace_inference: weights = %s\n", weights_path.c_str());

    smallfield::parameters::mace_weights_t weights;
    weights.load(weights_path);
    auto const &arch = weights.arch();
    std::printf("bench_mace_inference: arch n_elements=%u n_radial=%u n_channels=%u r_max=%.3f avg_num_neighbors=%.3f\n",
                arch.n_elements, arch.n_radial, arch.n_channels, arch.r_max, arch.avg_num_neighbors);

    std::vector<std::uint32_t> const batch_sizes  = {64u, 128u, 256u, 512u};
    std::vector<std::uint32_t> const atom_counts  = {15u, 30u, 50u, 80u};

    std::vector<bench_result_t> results;
    results.reserve(batch_sizes.size() * atom_counts.size());
    std::uint64_t seed_counter = 0xC0FFEEull;
    for (auto na : atom_counts) {
        for (auto bs : batch_sizes) {
            std::printf("\n--- running batch=%u n_atoms=%u ---\n", bs, na);
            bench_config_t cfg{};
            cfg.batch_size = bs;
            cfg.n_atoms_per_mol = na;
            bench_result_t r = run_one_config(weights, cfg, seed_counter++);
            std::printf("    n_total_atoms=%u n_total_edges=%u  ms/iter=%.3f  ms/config=%.4f  mean_E=%.3f eV  segmented=%s\n",
                        r.n_total_atoms, r.n_total_edges, r.ms_per_iter, r.ms_per_config,
                        r.mean_energy_eV, r.segmented_path_active ? "yes" : "no");
            // Sanity: energy must be finite.
            if (!std::isfinite(r.mean_energy_eV)) {
                std::fprintf(stderr, "bench: mean energy is NaN / infinity at batch=%u n_atoms=%u\n", bs, na);
            }
            results.push_back(r);
        }
    }

    // Monotonicity check on ms/config (must drop with batch_size at each n_atoms).
    std::printf("\n=== Monotonicity check (ms/config should drop with batch_size) ===\n");
    for (auto na : atom_counts) {
        std::printf("  n_atoms=%u : ", na);
        double previous_ms = 1e30;
        bool monotone = true;
        for (auto bs : batch_sizes) {
            double ms = -1.0;
            for (auto const &r : results) if (r.batch_size == bs && r.n_atoms_per_mol == na) ms = r.ms_per_config;
            std::printf("bs=%u %.4f  ", bs, ms);
            if (ms > previous_ms * 1.05) monotone = false;  // 5% slack for noise
            previous_ms = ms;
        }
        std::printf(" %s\n", monotone ? "[monotone]" : "[NOT monotone]");
    }

    print_summary_table(results, batch_sizes, atom_counts);

    std::string const jsonl_path = std::string("/tmp/mace_inference_bench_") + bench_build_label_k + ".jsonl";
    emit_jsonl(results, jsonl_path);
    std::printf("bench_mace_inference: wrote %s\n", jsonl_path.c_str());

    return 0;
}
