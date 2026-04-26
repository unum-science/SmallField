/**
 *  @file test_mace_api.cu
 *  @brief Smoke test for the public smallfield::mace::context_t API.
 *
 *  Validates that:
 *    - weights_t::load reads the .bin format and exposes arch().
 *    - context_t::infer_energy returns finite energies for a synthetic batch.
 *    - context_t::infer_energy_and_forces returns finite energies and forces.
 *    - context_t::refine_batch runs N steps without diverging (energies stay
 *      finite; we don't check convergence — geometries are random).
 *
 *  Reuses the .bin produced by `python/smallfield/mace_export.py` at
 *  `/tmp/mace_export/MACE-OFF23-small.weights.bin` for weight loading.
 */

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "smallfield/api/smallfield_mace.hpp"
#include "smallfield/types/mace_types.hpp"

namespace {

#define CUDA_CHECK(stmt)                                                                                 \
    do {                                                                                                 \
        cudaError_t const e = (stmt);                                                                    \
        if (e != cudaSuccess) {                                                                          \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
            std::exit(1);                                                                                \
        }                                                                                                \
    } while (0)

struct synthetic_batch_t {
    std::uint32_t n_conformers;
    std::uint32_t n_atoms_per_conformer;

    std::vector<float> host_positions_xyz;        // [n_atoms × 3]
    std::vector<std::int32_t> host_atom_element;  // [n_atoms]
    std::vector<std::int32_t> host_graph_of_atom; // [n_atoms]
    std::vector<std::int32_t> host_edge_source;   // [n_edges]
    std::vector<std::int32_t> host_edge_target;   // [n_edges]
};

synthetic_batch_t build_synthetic_batch(std::uint32_t n_conformers, std::uint32_t n_atoms_per_conformer,
                                        std::uint32_t n_elements, float r_max, std::uint64_t seed) {
    synthetic_batch_t out;
    out.n_conformers = n_conformers;
    out.n_atoms_per_conformer = n_atoms_per_conformer;
    std::uint32_t const n_atoms_total = n_conformers * n_atoms_per_conformer;
    out.host_positions_xyz.resize(n_atoms_total * 3);
    out.host_atom_element.resize(n_atoms_total);
    out.host_graph_of_atom.resize(n_atoms_total);

    std::mt19937_64 rng(seed);
    std::normal_distribution<float> jitter(0.0f, 0.4f);
    std::uniform_int_distribution<std::int32_t> elem_dist(0, static_cast<std::int32_t>(n_elements) - 1);

    for (std::uint32_t conf = 0; conf < n_conformers; ++conf) {
        for (std::uint32_t atom = 0; atom < n_atoms_per_conformer; ++atom) {
            std::uint32_t const a = conf * n_atoms_per_conformer + atom;
            out.host_positions_xyz[3 * a + 0] = jitter(rng) + 1.5f * static_cast<float>(atom);
            out.host_positions_xyz[3 * a + 1] = jitter(rng);
            out.host_positions_xyz[3 * a + 2] = jitter(rng);
            out.host_atom_element[a] = elem_dist(rng);
            out.host_graph_of_atom[a] = static_cast<std::int32_t>(conf);
        }
    }

    // Brute-force edge list per conformer.
    for (std::uint32_t conf = 0; conf < n_conformers; ++conf) {
        for (std::uint32_t i = 0; i < n_atoms_per_conformer; ++i) {
            std::uint32_t const ai = conf * n_atoms_per_conformer + i;
            for (std::uint32_t j = 0; j < n_atoms_per_conformer; ++j) {
                if (i == j) continue;
                std::uint32_t const aj = conf * n_atoms_per_conformer + j;
                float dx = out.host_positions_xyz[3 * aj + 0] - out.host_positions_xyz[3 * ai + 0];
                float dy = out.host_positions_xyz[3 * aj + 1] - out.host_positions_xyz[3 * ai + 1];
                float dz = out.host_positions_xyz[3 * aj + 2] - out.host_positions_xyz[3 * ai + 2];
                float r2 = dx * dx + dy * dy + dz * dz;
                if (r2 < r_max * r_max && r2 > 1e-8f) {
                    out.host_edge_source.push_back(static_cast<std::int32_t>(ai));
                    out.host_edge_target.push_back(static_cast<std::int32_t>(aj));
                }
            }
        }
    }
    return out;
}

} // namespace

int main(int argc, char **argv) {
    std::string const export_dir = (argc > 1) ? argv[1] : "/tmp/mace_export";
    std::string const weights_path = export_dir + "/MACE-OFF23-small.weights.bin";

    std::printf("test_mace_api: weights = %s\n", weights_path.c_str());

    // 1.  Load weights via the public API.
    smallfield::mace::weights_t W;
    try {
        W = smallfield::mace::weights_t::load(weights_path);
    }
    catch (std::exception const &e) {
        std::fprintf(stderr, "weights load failed: %s\n", e.what());
        return 1;
    }
    auto const &arch = W.arch();
    std::printf("  arch: n_elements=%u n_radial=%u n_channels=%u r_max=%.2f num_interactions=%u\n", arch.n_elements,
                arch.n_radial, arch.n_channels, arch.r_max, arch.num_interactions);

    // 2.  Build a synthetic batch (4 conformers × 30 atoms each).
    constexpr std::uint32_t n_conformers = 4u;
    constexpr std::uint32_t n_atoms_per_conf = 30u;
    auto sb = build_synthetic_batch(n_conformers, n_atoms_per_conf, arch.n_elements, arch.r_max, 0xC0FFEEull);

    std::uint32_t const n_atoms_total = static_cast<std::uint32_t>(sb.host_positions_xyz.size() / 3);
    std::uint32_t const n_edges_total = static_cast<std::uint32_t>(sb.host_edge_source.size());
    std::printf("  synthetic batch: n_conformers=%u n_atoms=%u n_edges=%u\n", n_conformers, n_atoms_total,
                n_edges_total);

    // 3.  Upload to the GPU.
    float *device_positions = nullptr;
    std::int32_t *device_atom_element = nullptr;
    std::int32_t *device_graph_of_atom = nullptr;
    std::int32_t *device_edge_source = nullptr;
    std::int32_t *device_edge_target = nullptr;
    float *device_energies = nullptr;
    float *device_forces = nullptr;
    CUDA_CHECK(cudaMalloc(&device_positions, sb.host_positions_xyz.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&device_atom_element, sb.host_atom_element.size() * sizeof(std::int32_t)));
    CUDA_CHECK(cudaMalloc(&device_graph_of_atom, sb.host_graph_of_atom.size() * sizeof(std::int32_t)));
    CUDA_CHECK(cudaMalloc(&device_edge_source, sb.host_edge_source.size() * sizeof(std::int32_t)));
    CUDA_CHECK(cudaMalloc(&device_edge_target, sb.host_edge_target.size() * sizeof(std::int32_t)));
    CUDA_CHECK(cudaMalloc(&device_energies, n_conformers * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&device_forces, n_atoms_total * 3u * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(device_positions, sb.host_positions_xyz.data(), sb.host_positions_xyz.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_atom_element, sb.host_atom_element.data(),
                          sb.host_atom_element.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_graph_of_atom, sb.host_graph_of_atom.data(),
                          sb.host_graph_of_atom.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_edge_source, sb.host_edge_source.data(),
                          sb.host_edge_source.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_edge_target, sb.host_edge_target.data(),
                          sb.host_edge_target.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice));

    // 4.  Construct the inference context.
    smallfield::mace::context_t ctx(/*device_id=*/0, arch,
                                    /*max_atoms=*/n_atoms_total,
                                    /*max_edges=*/n_edges_total,
                                    /*max_graphs=*/n_conformers);

    smallfield::mace::batch_view_t batch {};
    batch.n_atoms = n_atoms_total;
    batch.n_edges = n_edges_total;
    batch.n_graphs = n_conformers;
    batch.positions_xyz = device_positions;
    batch.atom_element = device_atom_element;
    batch.edge_source_atom = device_edge_source;
    batch.edge_target_atom = device_edge_target;
    batch.graph_of_atom = device_graph_of_atom;

    // 5.  Test infer_energy.
    auto status = ctx.infer_energy(W, batch, device_energies);
    if (status != smallfield::mace::status_t::success) {
        std::fprintf(stderr, "infer_energy failed with status %u\n", static_cast<unsigned>(status));
        return 1;
    }
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));

    std::vector<float> host_energies(n_conformers);
    CUDA_CHECK(cudaMemcpy(host_energies.data(), device_energies, n_conformers * sizeof(float), cudaMemcpyDeviceToHost));
    std::printf("  infer_energy: energies = ");
    for (auto e : host_energies) std::printf("%.4f ", e);
    std::printf("\n");
    for (auto e : host_energies) {
        if (!std::isfinite(e)) {
            std::fprintf(stderr, "infer_energy returned non-finite value\n");
            return 1;
        }
    }

    // 6.  Test infer_energy_and_forces.
    status = ctx.infer_energy_and_forces(W, batch, device_energies, device_forces);
    if (status != smallfield::mace::status_t::success) {
        std::fprintf(stderr, "infer_energy_and_forces failed with status %u\n", static_cast<unsigned>(status));
        return 1;
    }
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));

    std::vector<float> host_forces(n_atoms_total * 3u);
    CUDA_CHECK(
        cudaMemcpy(host_forces.data(), device_forces, host_forces.size() * sizeof(float), cudaMemcpyDeviceToHost));

    double force_max = 0.0;
    bool forces_finite = true;
    for (auto f : host_forces) {
        if (!std::isfinite(f)) {
            forces_finite = false;
            break;
        }
        force_max = std::max<double>(force_max, std::abs(static_cast<double>(f)));
    }
    if (!forces_finite) {
        std::fprintf(stderr, "infer_energy_and_forces returned non-finite force\n");
        return 1;
    }
    std::printf("  infer_energy_and_forces: |F|_max = %.4f eV/Å\n", force_max);

    // 7.  Test refine_batch (5 GD-style steps; just check it doesn't blow up).
    smallfield::mace::relax_config_t config {};
    config.num_steps = 5;
    config.step_size = 0.001f;
    status = ctx.refine_batch(W, batch, config, device_positions, device_energies);
    if (status != smallfield::mace::status_t::success) {
        std::fprintf(stderr, "refine_batch failed with status %u\n", static_cast<unsigned>(status));
        return 1;
    }
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));

    CUDA_CHECK(cudaMemcpy(host_energies.data(), device_energies, n_conformers * sizeof(float), cudaMemcpyDeviceToHost));
    std::printf("  refine_batch (5 steps): energies = ");
    for (auto e : host_energies) std::printf("%.4f ", e);
    std::printf("\n");
    for (auto e : host_energies) {
        if (!std::isfinite(e)) {
            std::fprintf(stderr, "refine_batch returned non-finite energy\n");
            return 1;
        }
    }

    // Clean up.
    cudaFree(device_positions);
    cudaFree(device_atom_element);
    cudaFree(device_graph_of_atom);
    cudaFree(device_edge_source);
    cudaFree(device_edge_target);
    cudaFree(device_energies);
    cudaFree(device_forces);

    std::printf("test_mace_api: all checks passed\n");
    return 0;
}
