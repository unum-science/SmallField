/**
 *  @file test_mace_loader.cu
 *  @brief Phase 0 + Phase 1 validation: weight loader, edge geometry,
 *         radial basis, spherical harmonics, and end-to-end fp32 energy
 *         vs the Python golden trace.
 *
 *  Run after exporting the small model:
 *    python3 python/smallfield/mace_export.py --model small --out /tmp/mace_export
 *
 *  This test compiles all `mace_*.cuh` headers and exercises the forward
 *  driver in `mace_inference.cuh` end-to-end on the five reference goldens.
 *  Per-tensor diffs are reported; energy is compared atom-wise against the
 *  golden `node_energy`.
 */

#include <cassert>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>

#include <cuda_runtime.h>

#include "smallfield/parameters/mace_weights.hpp"
#include "smallfield/types/mace_types.hpp"
#include "smallfield/cuda/mace_inference.cuh"

namespace {

struct golden_trace_t {
    std::unordered_map<std::string, std::vector<double>> tensors;
    std::unordered_map<std::string, std::vector<std::int64_t>> shapes;

    void load(std::string const &path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) { std::cerr << "cannot open " << path << "\n"; std::exit(1); }
        char magic[8];
        std::uint32_t version;
        std::uint64_t mo, ml;
        f.read(magic, 8);
        f.read(reinterpret_cast<char *>(&version), 4);
        f.read(reinterpret_cast<char *>(&mo), 8);
        f.read(reinterpret_cast<char *>(&ml), 8);
        if (std::memcmp(magic, "SF_MACE\0", 8) != 0) { std::cerr << "bad magic\n"; std::exit(1); }
        std::string raw(ml, '\0');
        f.seekg(static_cast<std::streamoff>(mo), std::ios::beg);
        f.read(raw.data(), static_cast<std::streamsize>(ml));

        std::unordered_map<std::string, double> hnum;
        std::unordered_map<std::string, std::string> hstr;
        std::vector<smallfield::parameters::mace_detail_::manifest_entry_t> tensors_meta;
        smallfield::parameters::mace_detail_::parse_manifest_(raw, hnum, hstr, tensors_meta);

        for (auto const &t : tensors_meta) {
            std::uint64_t n = 1;
            for (auto d : t.shape) n *= static_cast<std::uint64_t>(d);
            f.seekg(static_cast<std::streamoff>(28 + t.offset), std::ios::beg);
            if (t.dtype == "float64") {
                std::vector<double> v(n);
                f.read(reinterpret_cast<char *>(v.data()), static_cast<std::streamsize>(n * 8));
                tensors[t.name] = std::move(v);
                shapes[t.name] = t.shape;
            } else if (t.dtype == "int64") {
                std::vector<std::int64_t> v(n);
                f.read(reinterpret_cast<char *>(v.data()), static_cast<std::streamsize>(n * 8));
                std::vector<double> dv(n);
                for (std::uint64_t i = 0; i < n; ++i) dv[i] = static_cast<double>(v[i]);
                tensors[t.name] = std::move(dv);
                shapes[t.name] = t.shape;
            }
        }
    }
};

void diff_report(char const *label, double const *ref, float const *got, std::size_t n) {
    double max_abs = 0.0, ss = 0.0, mean_abs = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double const d = std::abs(ref[i] - static_cast<double>(got[i]));
        max_abs = std::max(max_abs, d);
        ss += d * d;
        mean_abs += d;
    }
    double const rms = std::sqrt(ss / static_cast<double>(n));
    mean_abs /= static_cast<double>(n);
    std::printf("  %-32s  max_abs=%.3e  rms=%.3e  mean_abs=%.3e  (n=%zu)\n",
                label, max_abs, rms, mean_abs, n);
}

void run_one(smallfield::parameters::mace_weights_t const &W,
             std::string const &golden_path,
             char const *name) {
    auto const &arch = W.arch();

    golden_trace_t gt;
    gt.load(golden_path);
    auto const &positions = gt.tensors.at("positions");
    auto const &edge_index = gt.tensors.at("edge_index");
    auto const &node_attrs = gt.tensors.at("node_attrs");

    std::uint32_t const n_atoms = static_cast<std::uint32_t>(positions.size() / 3);
    std::uint32_t const n_edges = static_cast<std::uint32_t>(edge_index.size() / 2);
    std::printf("\n--- %s: %u atoms, %u edges ---\n", name, n_atoms, n_edges);

    // Pull atom_element from one-hot node_attrs (n_atoms × n_elements).
    std::vector<std::int32_t> h_atom_element(n_atoms);
    for (std::uint32_t a = 0; a < n_atoms; ++a) {
        for (std::uint32_t z = 0; z < arch.n_elements; ++z) {
            if (node_attrs[a * arch.n_elements + z] > 0.5) { h_atom_element[a] = static_cast<std::int32_t>(z); break; }
        }
    }
    std::vector<std::int32_t> h_graph_of_atom(n_atoms, 0);

    // Upload positions, edge index, atom element.
    float *d_pos = nullptr;
    std::int32_t *d_src = nullptr, *d_dst = nullptr, *d_atom_el = nullptr, *d_graph_of_atom = nullptr;
    cudaMalloc(reinterpret_cast<void **>(&d_pos), n_atoms * 3 * sizeof(float));
    cudaMalloc(reinterpret_cast<void **>(&d_src), n_edges * sizeof(std::int32_t));
    cudaMalloc(reinterpret_cast<void **>(&d_dst), n_edges * sizeof(std::int32_t));
    cudaMalloc(reinterpret_cast<void **>(&d_atom_el), n_atoms * sizeof(std::int32_t));
    cudaMalloc(reinterpret_cast<void **>(&d_graph_of_atom), n_atoms * sizeof(std::int32_t));
    {
        std::vector<float> h_pos(n_atoms * 3);
        for (std::size_t i = 0; i < positions.size(); ++i) h_pos[i] = static_cast<float>(positions[i]);
        std::vector<std::int32_t> h_src(n_edges), h_dst(n_edges);
        for (std::uint32_t e = 0; e < n_edges; ++e) {
            h_src[e] = static_cast<std::int32_t>(edge_index[0 * n_edges + e]);
            h_dst[e] = static_cast<std::int32_t>(edge_index[1 * n_edges + e]);
        }
        cudaMemcpy(d_pos,    h_pos.data(),            h_pos.size() * sizeof(float),  cudaMemcpyHostToDevice);
        cudaMemcpy(d_src,    h_src.data(),            h_src.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_dst,    h_dst.data(),            h_dst.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_atom_el,h_atom_element.data(),   n_atoms * sizeof(std::int32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_graph_of_atom, h_graph_of_atom.data(), n_atoms * sizeof(std::int32_t), cudaMemcpyHostToDevice);
    }

    // Allocate scratch.
    smallfield::cuda::mace_scratch_t S;
    S.allocate(arch, n_atoms, n_edges, /*max_graphs*/1u);

    smallfield::types::mace_batch_t batch{};
    batch.n_atoms = n_atoms;
    batch.n_edges = n_edges;
    batch.n_graphs = 1u;
    batch.positions = d_pos;
    batch.atom_element = d_atom_el;
    batch.edge_source_atom = d_src;
    batch.edge_target_atom = d_dst;
    batch.graph_of_atom = d_graph_of_atom;

    // Per-pass invariants — zero node_energy before driver.
    smallfield::cuda::mace_inference_reset_per_atom_energy_accumulator(S, n_atoms, /*stream*/0);
    smallfield::cuda::mace_inference_forward(arch, W.device(), batch, S, /*stream*/0);
    cudaDeviceSynchronize();

    // Wall-time timing: 5 warmup passes, then 50 timed passes.
    {
        for (int i = 0; i < 5; ++i) {
            smallfield::cuda::mace_inference_reset_per_atom_energy_accumulator(S, n_atoms, 0);
            smallfield::cuda::mace_inference_forward(arch, W.device(), batch, S, 0);
        }
        cudaDeviceSynchronize();
        cudaEvent_t t0, t1;
        cudaEventCreate(&t0); cudaEventCreate(&t1);
        cudaEventRecord(t0);
        constexpr int N_ITERS = 50;
        for (int i = 0; i < N_ITERS; ++i) {
            smallfield::cuda::mace_inference_reset_per_atom_energy_accumulator(S, n_atoms, 0);
            smallfield::cuda::mace_inference_forward(arch, W.device(), batch, S, 0);
        }
        cudaEventRecord(t1); cudaEventSynchronize(t1);
        float ms_total = 0; cudaEventElapsedTime(&ms_total, t0, t1);
        std::printf("  %-32s  %.4f ms / forward (mean over %d iters, single molecule)\n",
                    "wall time", ms_total / N_ITERS, N_ITERS);
        cudaEventDestroy(t0); cudaEventDestroy(t1);
    }

    // Pull intermediates for diff reporting.
    std::vector<float> h_radial(n_edges * arch.n_radial), h_sph(n_edges * 16);
    cudaMemcpy(h_radial.data(), S.radial_basis, h_radial.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_sph.data(), S.sph_harm, h_sph.size() * sizeof(float), cudaMemcpyDeviceToHost);
    if (gt.tensors.count("radial")) diff_report("radial_basis", gt.tensors.at("radial").data(), h_radial.data(), n_edges * arch.n_radial);
    if (gt.tensors.count("sph_harm")) diff_report("sph_harm", gt.tensors.at("sph_harm").data(), h_sph.data(), n_edges * 16);

    // Diagnostic: print the first few values of each intermediate tensor so we
    // can see where NaN / blowup originates.  Cheap to keep in for the next iteration.
    auto first_n = [](float const *p, std::size_t n) {
        for (std::size_t i = 0; i < std::min<std::size_t>(n, 5); ++i) std::printf(" %.4g", p[i]);
    };
    std::vector<float> h_node_features(n_atoms * arch.n_channels * 16u);
    cudaMemcpy(h_node_features.data(), S.node_features, h_node_features.size() * sizeof(float), cudaMemcpyDeviceToHost);
    std::printf("  node_feats[0..4] :"); first_n(h_node_features.data(), 5); std::printf("\n");
    std::vector<float> h_node_l0(n_atoms * arch.n_channels);
    cudaMemcpy(h_node_l0.data(), S.node_features_l0_only, h_node_l0.size() * sizeof(float), cudaMemcpyDeviceToHost);
    std::printf("  node_l0[0..4]    :"); first_n(h_node_l0.data(), 5); std::printf("\n");
    std::vector<float> h_mlp_h2(n_edges * 64u);
    cudaMemcpy(h_mlp_h2.data(), S.radial_mlp_hidden_2, h_mlp_h2.size() * sizeof(float), cudaMemcpyDeviceToHost);
    std::printf("  mlp_h2[0..4]     :"); first_n(h_mlp_h2.data(), 5); std::printf("\n");
    std::vector<float> h_tp(n_edges * 4u * arch.n_channels);
    cudaMemcpy(h_tp.data(), S.tensor_product_weights_per_edge, h_tp.size() * sizeof(float), cudaMemcpyDeviceToHost);
    std::printf("  tp_weights[0..4] :"); first_n(h_tp.data(), 5); std::printf("\n");
    std::vector<float> h_msg(n_edges * arch.n_channels * 16u);
    cudaMemcpy(h_msg.data(), S.edge_messages, h_msg.size() * sizeof(float), cudaMemcpyDeviceToHost);
    std::printf("  messages[0..4]   :"); first_n(h_msg.data(), 5); std::printf("\n");
    std::vector<float> h_inv(n_atoms * arch.n_channels);
    cudaMemcpy(h_inv.data(), S.node_scalar_invariants, h_inv.size() * sizeof(float), cudaMemcpyDeviceToHost);
    std::printf("  invariants[0..4] :"); first_n(h_inv.data(), 5); std::printf("\n");
    std::vector<float> h_ne(n_atoms);
    cudaMemcpy(h_ne.data(), S.node_energy, h_ne.size() * sizeof(float), cudaMemcpyDeviceToHost);
    std::printf("  node_energy[0..4]:"); first_n(h_ne.data(), 5); std::printf("\n");

    // End-to-end energy diff.
    float h_energy = 0.0f;
    cudaMemcpy(&h_energy, S.energy, sizeof(float), cudaMemcpyDeviceToHost);
    double const ref_energy = gt.tensors.at("energy").size() == 1
                              ? gt.tensors.at("energy")[0]
                              : 0.0;
    std::printf("  %-32s  ref=%14.6f  got=%14.6f  diff=%.3e\n",
                "energy (eV)", ref_energy, static_cast<double>(h_energy),
                std::abs(ref_energy - static_cast<double>(h_energy)));

    cudaFree(d_pos); cudaFree(d_src); cudaFree(d_dst);
    cudaFree(d_atom_el); cudaFree(d_graph_of_atom);
}

} // namespace

int main(int argc, char **argv) {
    std::string export_dir = (argc > 1) ? argv[1] : "/tmp/mace_export";
    std::string weights_path = export_dir + "/MACE-OFF23-small.weights.bin";

    std::printf("=== Phase 0: weight loader ===\n");
    smallfield::parameters::mace_weights_t W;
    W.load(weights_path);
    auto const &arch = W.arch();
    std::printf("  arch: n_elements=%u n_radial=%u n_channels=%u L_max=%u num_interactions=%u r_max=%.3f\n",
                arch.n_elements, arch.n_radial, arch.n_channels, arch.l_max, arch.num_interactions, arch.r_max);
    std::printf("  device_bytes=%zu  scale=%.6f shift=%.6f cutoff_p=%.1f bessel_prefactor=%.6f\n",
                W.device_bytes(), arch.scale, arch.shift, arch.cutoff_p, arch.bessel_prefactor);
    std::printf("  OK\n");

    std::printf("\n=== Phase 1: end-to-end forward on the 5 reference goldens ===\n");
    char const *names[] = {"caffeine", "paracetamol", "ibuprofen", "propranolol", "aspirin"};
    for (char const *name : names) {
        run_one(W, export_dir + "/golden_small_" + name + ".bin", name);
    }
    return 0;
}
