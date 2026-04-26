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
#include "smallfield/cuda/mace_fp16_io_kernel.cuh"
#include "smallfield/cuda/mace_graph_capture.cuh"

#include <cuda_fp16.h>

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
    double ref_max = 0.0, ref_ss = 0.0;
    double got_max = 0.0, got_ss = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double const d = std::abs(ref[i] - static_cast<double>(got[i]));
        max_abs = std::max(max_abs, d);
        ss += d * d;
        mean_abs += d;
        ref_max = std::max(ref_max, std::abs(ref[i]));
        ref_ss += ref[i] * ref[i];
        got_max = std::max(got_max, std::abs(static_cast<double>(got[i])));
        got_ss += static_cast<double>(got[i]) * static_cast<double>(got[i]);
    }
    double const rms = std::sqrt(ss / static_cast<double>(n));
    double const ref_rms = std::sqrt(ref_ss / static_cast<double>(n));
    double const got_rms = std::sqrt(got_ss / static_cast<double>(n));
    mean_abs /= static_cast<double>(n);
    std::printf("  %-32s  max_abs=%.3e  rms=%.3e  mean_abs=%.3e  ref_rms=%.3e  got_rms=%.3e  (n=%zu)\n",
                label, max_abs, rms, mean_abs, ref_rms, got_rms, n);
    (void)ref_max; (void)got_max;
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

    // Force-adjoint validation against the grad_positions golden.
    //   forces[a, k]      = -grad_positions[a, k]
    //   ref grad_positions = MACE Python torch.autograd.grad(E, positions)
    //   We compare per-atom force RMS in eV/Å; ≤ 1 meV/Å is the plan target.
    smallfield::cuda::mace_inference_run_force_adjoint(arch, W.device(), batch, S, /*stream*/0);
    cudaDeviceSynchronize();
    std::vector<float> h_forces(n_atoms * 3u);
    cudaMemcpy(h_forces.data(), S.forces, h_forces.size() * sizeof(float), cudaMemcpyDeviceToHost);
    if (gt.tensors.count("grad_positions")) {
        // forces = -grad_positions in the Python golden.
        auto const &grad_pos = gt.tensors.at("grad_positions");
        std::vector<double> ref_forces(grad_pos.size());
        for (std::size_t i = 0; i < grad_pos.size(); ++i) ref_forces[i] = -grad_pos[i];
        diff_report("forces (vs -grad_positions)", ref_forces.data(), h_forces.data(), n_atoms * 3u);
        // Per-atom RMS: sqrt(mean over atoms of |f - f_ref|^2 summed over xyz).
        double accumulator_sq = 0.0;
        for (std::uint32_t a = 0; a < n_atoms; ++a) {
            double const dxv = h_forces[3 * a + 0] - ref_forces[3 * a + 0];
            double const dyv = h_forces[3 * a + 1] - ref_forces[3 * a + 1];
            double const dzv = h_forces[3 * a + 2] - ref_forces[3 * a + 2];
            accumulator_sq += dxv * dxv + dyv * dyv + dzv * dzv;
        }
        double const rms_per_atom_eV_per_A = std::sqrt(accumulator_sq / static_cast<double>(n_atoms));
        std::printf("  %-32s  per-atom force RMS = %.3e eV/Å (target ≤ 1e-3)\n",
                    "force adjoint", rms_per_atom_eV_per_A);
        // First few atoms: print ref vs got side-by-side to expose sign/scale errors.
        std::printf("    ref[0..3]:");
        for (std::uint32_t a = 0; a < std::min<std::uint32_t>(3u, n_atoms); ++a)
            std::printf(" (%.4f,%.4f,%.4f)", ref_forces[3*a+0], ref_forces[3*a+1], ref_forces[3*a+2]);
        std::printf("\n    got[0..3]:");
        for (std::uint32_t a = 0; a < std::min<std::uint32_t>(3u, n_atoms); ++a)
            std::printf(" (%.4f,%.4f,%.4f)", h_forces[3*a+0], h_forces[3*a+1], h_forces[3*a+2]);
        std::printf("\n");

        // Dump intermediate gradients to compare against the golden — helps
        // localise which path is broken when the final forces don't match.
        if (gt.tensors.count("grad_radial_basis")) {
            std::vector<float> h_grad_rb(n_edges * arch.n_radial);
            cudaMemcpy(h_grad_rb.data(), S.grad_radial_basis,
                       h_grad_rb.size() * sizeof(float), cudaMemcpyDeviceToHost);
            diff_report("grad_radial_basis", gt.tensors.at("grad_radial_basis").data(),
                        h_grad_rb.data(), n_edges * arch.n_radial);
        }
        if (gt.tensors.count("grad_sph_harm")) {
            std::vector<float> h_grad_sph(n_edges * 16u);
            cudaMemcpy(h_grad_sph.data(), S.grad_sph_harm,
                       h_grad_sph.size() * sizeof(float), cudaMemcpyDeviceToHost);
            diff_report("grad_sph_harm", gt.tensors.at("grad_sph_harm").data(),
                        h_grad_sph.data(), n_edges * 16u);
        }
        for (std::uint32_t k = 0; k < arch.num_interactions && k < 2u; ++k) {
            std::string key = "grad_node_features_layer_" + std::to_string(k);
            if (gt.tensors.count(key)) {
                // Our internal grad_node_features mid-loop is overwritten between
                // layers, so we can only easily compare the final layer's value
                // captured at end of the reverse pass — not implemented here yet.
                (void)key;
            }
        }
    } else {
        std::printf("  (no grad_positions golden — skipping force adjoint diff)\n");
    }

    cudaFree(d_pos); cudaFree(d_src); cudaFree(d_dst);
    cudaFree(d_atom_el); cudaFree(d_graph_of_atom);
}

struct batched_sample_tensor_t {
    std::string name;
    std::string dtype;
    std::vector<std::int64_t> shape;
    std::uint64_t offset;
    std::uint64_t nbytes;
};

/**
 *  @brief Tiny manifest scanner for the fp16-batch artefact.
 *
 *  Cannot reuse `mace_detail_::parse_manifest_` because the batched-sample
 *  manifest uses JSON arrays inside the header (e.g. graph_atom_offset),
 *  which the weight-blob parser does not need to handle and currently
 *  rejects with "expected string".  This local scanner is name-driven —
 *  it walks the raw text looking for the keys we care about and stops
 *  once it has located each `tensors` entry.  Sufficient because the
 *  producer (`mace_export.py`) emits a known schema with one tensor per
 *  object inside a single top-level array.
 */
inline std::vector<batched_sample_tensor_t> scan_batched_manifest_tensors_(std::string const &raw) {
    std::vector<batched_sample_tensor_t> out;
    std::size_t cursor = raw.find("\"tensors\"");
    if (cursor == std::string::npos) return out;
    cursor = raw.find('[', cursor);
    if (cursor == std::string::npos) return out;
    auto find_string_field = [&](std::size_t obj_start, std::size_t obj_end,
                                 char const *key, std::string &dst) {
        std::string needle = std::string("\"") + key + "\"";
        std::size_t kp = raw.find(needle, obj_start);
        if (kp == std::string::npos || kp >= obj_end) return false;
        std::size_t qopen = raw.find('"', kp + needle.size());
        if (qopen == std::string::npos || qopen >= obj_end) return false;
        std::size_t qclose = raw.find('"', qopen + 1);
        if (qclose == std::string::npos || qclose >= obj_end) return false;
        dst = raw.substr(qopen + 1, qclose - qopen - 1);
        return true;
    };
    auto find_uint_field = [&](std::size_t obj_start, std::size_t obj_end,
                               char const *key, std::uint64_t &dst) {
        std::string needle = std::string("\"") + key + "\"";
        std::size_t kp = raw.find(needle, obj_start);
        if (kp == std::string::npos || kp >= obj_end) return false;
        std::size_t cp = raw.find(':', kp + needle.size());
        if (cp == std::string::npos || cp >= obj_end) return false;
        std::size_t pos = cp + 1;
        while (pos < obj_end && (raw[pos] == ' ' || raw[pos] == '\n' || raw[pos] == '\t')) ++pos;
        std::size_t start = pos;
        while (pos < obj_end && (std::isdigit(static_cast<unsigned char>(raw[pos])) || raw[pos] == '-')) ++pos;
        if (pos == start) return false;
        dst = static_cast<std::uint64_t>(std::stoll(raw.substr(start, pos - start)));
        return true;
    };
    auto find_shape = [&](std::size_t obj_start, std::size_t obj_end,
                          std::vector<std::int64_t> &dst) {
        std::size_t kp = raw.find("\"shape\"", obj_start);
        if (kp == std::string::npos || kp >= obj_end) return false;
        std::size_t lb = raw.find('[', kp);
        if (lb == std::string::npos || lb >= obj_end) return false;
        std::size_t rb = raw.find(']', lb);
        if (rb == std::string::npos || rb > obj_end) return false;
        std::size_t pos = lb + 1;
        while (pos < rb) {
            while (pos < rb && (raw[pos] == ' ' || raw[pos] == '\n' || raw[pos] == '\t' || raw[pos] == ',')) ++pos;
            if (pos >= rb) break;
            std::size_t start = pos;
            while (pos < rb && (std::isdigit(static_cast<unsigned char>(raw[pos])) || raw[pos] == '-')) ++pos;
            if (pos > start) dst.push_back(static_cast<std::int64_t>(std::stoll(raw.substr(start, pos - start))));
            else break;
        }
        return true;
    };

    // Walk objects { ... } inside the tensors array.
    std::size_t pos = cursor + 1;
    while (pos < raw.size()) {
        std::size_t ob = raw.find('{', pos);
        std::size_t end_array = raw.find(']', pos);
        if (ob == std::string::npos || (end_array != std::string::npos && ob > end_array)) break;
        // Match closing brace at depth 0 from `ob`.
        std::size_t scan = ob + 1;
        int depth = 1;
        while (scan < raw.size() && depth > 0) {
            if (raw[scan] == '{') ++depth;
            else if (raw[scan] == '}') --depth;
            ++scan;
        }
        std::size_t oe = scan; // one past closing '}'
        batched_sample_tensor_t entry{};
        find_string_field(ob, oe, "name",   entry.name);
        find_string_field(ob, oe, "dtype",  entry.dtype);
        find_shape       (ob, oe,           entry.shape);
        find_uint_field  (ob, oe, "offset", entry.offset);
        find_uint_field  (ob, oe, "nbytes", entry.nbytes);
        if (!entry.name.empty()) out.push_back(std::move(entry));
        pos = oe;
    }
    return out;
}

inline std::uint64_t scan_batched_manifest_uint_(std::string const &raw, char const *key,
                                                 std::uint64_t fallback) {
    std::string needle = std::string("\"") + key + "\"";
    std::size_t kp = raw.find(needle);
    if (kp == std::string::npos) return fallback;
    std::size_t cp = raw.find(':', kp + needle.size());
    if (cp == std::string::npos) return fallback;
    std::size_t pos = cp + 1;
    while (pos < raw.size() && (raw[pos] == ' ' || raw[pos] == '\n' || raw[pos] == '\t')) ++pos;
    std::size_t start = pos;
    while (pos < raw.size() && (std::isdigit(static_cast<unsigned char>(raw[pos])) || raw[pos] == '-')) ++pos;
    if (pos == start) return fallback;
    return static_cast<std::uint64_t>(std::stoll(raw.substr(start, pos - start)));
}

/**
 *  @brief Track B IO-cast round-trip: load the fp16 batched-sample artefact
 *  produced by `python/smallfield/mace_export.py`, run the
 *  `launch_mace_positions_fp16_to_fp32` kernel, and validate the result
 *  against the fp32 reference tensor stored alongside in the same artefact.
 *
 *  This test does NOT yet run the full MACE forward on the batched input —
 *  the driver swap to consume the new schema lands in a follow-up commit.
 *  It exists to lock down the fp16 ⇄ fp32 cast layer ahead of that change.
 */
void run_batched(smallfield::parameters::mace_weights_t const & /*W*/,
                 std::string const &export_dir) {
    std::string const sample_path = export_dir + "/sample_fp16_batch.bin";
    std::ifstream f(sample_path, std::ios::binary);
    if (!f) {
        std::printf("\n=== Track B batched IO-cast test ===\n");
        std::printf("  skipping batched test (sample_fp16_batch.bin not present, run mace_export.py first)\n");
        return;
    }

    std::printf("\n=== Track B batched IO-cast test ===\n");

    // Same SF_MACE container layout used by the weights and golden files.
    char magic[8];
    std::uint32_t version;
    std::uint64_t mo, ml;
    f.read(magic, 8);
    f.read(reinterpret_cast<char *>(&version), 4);
    f.read(reinterpret_cast<char *>(&mo), 8);
    f.read(reinterpret_cast<char *>(&ml), 8);
    if (std::memcmp(magic, "SF_MACE\0", 8) != 0) {
        std::printf("  bad magic in %s — aborting batched test\n", sample_path.c_str());
        return;
    }

    std::string raw(ml, '\0');
    f.seekg(static_cast<std::streamoff>(mo), std::ios::beg);
    f.read(raw.data(), static_cast<std::streamsize>(ml));

    std::uint32_t const n_atoms_total = static_cast<std::uint32_t>(
        scan_batched_manifest_uint_(raw, "n_atoms_total", 0));
    std::uint32_t const n_graphs = static_cast<std::uint32_t>(
        scan_batched_manifest_uint_(raw, "n_graphs", 0));
    std::printf("  manifest: n_atoms_total=%u  n_graphs=%u\n", n_atoms_total, n_graphs);

    auto const tensors_meta = scan_batched_manifest_tensors_(raw);

    // Pull the named tensors from the artefact.
    std::vector<std::uint16_t> h_positions_fp16_raw;
    std::vector<float>         h_positions_fp32_reference;
    std::vector<std::int32_t>  h_graph_atom_offset;
    bool found_fp16 = false;
    bool found_ref  = false;
    for (auto const &t : tensors_meta) {
        std::uint64_t n = 1;
        for (auto d : t.shape) n *= static_cast<std::uint64_t>(d);
        f.seekg(static_cast<std::streamoff>(28 + t.offset), std::ios::beg);

        if (t.name == "positions_fp16" && t.dtype == "float16") {
            h_positions_fp16_raw.resize(n);
            f.read(reinterpret_cast<char *>(h_positions_fp16_raw.data()),
                   static_cast<std::streamsize>(n * sizeof(std::uint16_t)));
            found_fp16 = true;
        } else if (t.name == "positions_fp32_reference" &&
                   (t.dtype == "float32" || t.dtype == "float64")) {
            if (t.dtype == "float32") {
                h_positions_fp32_reference.resize(n);
                f.read(reinterpret_cast<char *>(h_positions_fp32_reference.data()),
                       static_cast<std::streamsize>(n * sizeof(float)));
            } else {
                std::vector<double> tmp(n);
                f.read(reinterpret_cast<char *>(tmp.data()),
                       static_cast<std::streamsize>(n * sizeof(double)));
                h_positions_fp32_reference.resize(n);
                for (std::uint64_t i = 0; i < n; ++i)
                    h_positions_fp32_reference[i] = static_cast<float>(tmp[i]);
            }
            found_ref = true;
        } else if (t.name == "graph_atom_offset" &&
                   (t.dtype == "int32" || t.dtype == "int64")) {
            h_graph_atom_offset.resize(n);
            if (t.dtype == "int32") {
                f.read(reinterpret_cast<char *>(h_graph_atom_offset.data()),
                       static_cast<std::streamsize>(n * sizeof(std::int32_t)));
            } else {
                std::vector<std::int64_t> tmp(n);
                f.read(reinterpret_cast<char *>(tmp.data()),
                       static_cast<std::streamsize>(n * sizeof(std::int64_t)));
                for (std::uint64_t i = 0; i < n; ++i)
                    h_graph_atom_offset[i] = static_cast<std::int32_t>(tmp[i]);
            }
        }
    }

    if (!found_fp16 || !found_ref) {
        std::printf("  required tensors missing (fp16=%d, ref=%d) — aborting batched test\n",
                    found_fp16 ? 1 : 0, found_ref ? 1 : 0);
        return;
    }

    std::size_t const n_elements = h_positions_fp16_raw.size();
    if (h_positions_fp32_reference.size() != n_elements) {
        std::printf("  fp16/fp32 size mismatch (%zu vs %zu) — aborting batched test\n",
                    n_elements, h_positions_fp32_reference.size());
        return;
    }

    // Round-trip on device: H2D fp16 → cast → D2H fp32.
    __half *d_positions_fp16 = nullptr;
    float  *d_positions_fp32 = nullptr;
    cudaMalloc(reinterpret_cast<void **>(&d_positions_fp16), n_elements * sizeof(__half));
    cudaMalloc(reinterpret_cast<void **>(&d_positions_fp32), n_elements * sizeof(float));
    cudaMemcpy(d_positions_fp16, h_positions_fp16_raw.data(),
               n_elements * sizeof(__half), cudaMemcpyHostToDevice);

    smallfield::cuda::launch_mace_positions_fp16_to_fp32(
        static_cast<std::uint32_t>(n_elements), d_positions_fp16, d_positions_fp32, /*stream*/0);
    cudaDeviceSynchronize();

    std::vector<float> h_positions_fp32(n_elements);
    cudaMemcpy(h_positions_fp32.data(), d_positions_fp32,
               n_elements * sizeof(float), cudaMemcpyDeviceToHost);

    // Compare against the fp32 reference.
    double max_abs = 0.0;
    double sum_sq  = 0.0;
    for (std::size_t i = 0; i < n_elements; ++i) {
        double const d = std::abs(static_cast<double>(h_positions_fp32[i]) -
                                  static_cast<double>(h_positions_fp32_reference[i]));
        if (d > max_abs) max_abs = d;
        sum_sq += d * d;
    }
    double const rms = std::sqrt(sum_sq / static_cast<double>(n_elements));
    char const *verdict = (max_abs <= 2e-3) ? "PASS" : "FAIL";
    std::printf("  fp16 round-trip max_abs=%.3e  rms=%.3e  (n=%zu)  [%s, threshold 2e-3]\n",
                max_abs, rms, n_elements, verdict);

    cudaFree(d_positions_fp16);
    cudaFree(d_positions_fp32);
}

/**
 *  @brief Track E1 validation: capture the MACE forward pass into a
 *  `cudaGraph_t` and confirm replays produce the same energy as the
 *  eager driver.  Reports wall-time eager vs graph-replay across 50
 *  timed iterations to make the per-launch-overhead win measurable.
 *
 *  Picks one molecule (caffeine).  Captured graph references the device
 *  pointers in `batch` and `scratch` at capture time, so we keep the same
 *  buffers around for the eager-vs-replay comparison.
 */
void run_with_graph_capture(smallfield::parameters::mace_weights_t const &W,
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
    std::printf("\n--- graph-capture: %s: %u atoms, %u edges ---\n", name, n_atoms, n_edges);

    std::vector<std::int32_t> h_atom_element(n_atoms);
    for (std::uint32_t a = 0; a < n_atoms; ++a) {
        for (std::uint32_t z = 0; z < arch.n_elements; ++z) {
            if (node_attrs[a * arch.n_elements + z] > 0.5) { h_atom_element[a] = static_cast<std::int32_t>(z); break; }
        }
    }
    std::vector<std::int32_t> h_graph_of_atom(n_atoms, 0);

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

    // Use a non-default stream so capture is unambiguous (capture on legacy
    // stream 0 is technically allowed in `cudaStreamCaptureModeGlobal` but
    // a created stream is the recommended pattern).
    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    // 2. Eager reference.
    smallfield::cuda::mace_inference_reset_per_atom_energy_accumulator(S, n_atoms, stream);
    smallfield::cuda::mace_inference_forward(arch, W.device(), batch, S, stream);
    cudaStreamSynchronize(stream);
    float eager_energy = 0.0f;
    cudaMemcpy(&eager_energy, S.energy, sizeof(float), cudaMemcpyDeviceToHost);
    double const ref_energy = gt.tensors.at("energy").size() == 1
                              ? gt.tensors.at("energy")[0]
                              : 0.0;
    std::printf("  eager energy  ref=%14.6f  got=%14.6f  diff=%.3e\n",
                ref_energy, static_cast<double>(eager_energy),
                std::abs(ref_energy - static_cast<double>(eager_energy)));

    // 3. Capture the graph.  Any pending CUDA error is surfaced so that the
    //    failure mode is clear (the most common cause is a stray
    //    `cudaMalloc` inside the captured region — e.g. the TK atomic-fallback
    //    conv_tp_scatter path, which is incompatible with stream capture
    //    until Track B's segmented sort lands).
    cudaGetLastError();
    smallfield::cuda::mace_captured_graph_t captured;
    smallfield::cuda::mace_capture_forward(arch, W.device(), batch, S, stream, captured);
    cudaError_t const capture_err = cudaGetLastError();
    if (captured.graph_exec == nullptr) {
        std::printf("  graph instantiation FAILED (last cuda err: %s) — bailing out\n",
                    cudaGetErrorString(capture_err));
        cudaStreamDestroy(stream);
        cudaFree(d_pos); cudaFree(d_src); cudaFree(d_dst);
        cudaFree(d_atom_el); cudaFree(d_graph_of_atom);
        return;
    }
    std::printf("  graph instantiated  (n_atoms=%u n_edges=%u n_graphs=%u)\n",
                captured.n_atoms, captured.n_edges, captured.n_graphs);

    // 4. Replay 10 times and confirm bit-stability vs eager.
    double max_replay_diff = 0.0;
    for (int i = 0; i < 10; ++i) {
        captured.replay(stream);
        cudaStreamSynchronize(stream);
        float replay_energy = 0.0f;
        cudaMemcpy(&replay_energy, S.energy, sizeof(float), cudaMemcpyDeviceToHost);
        double const d = std::abs(static_cast<double>(replay_energy) -
                                  static_cast<double>(eager_energy));
        if (d > max_replay_diff) max_replay_diff = d;
    }
    char const *replay_verdict = (max_replay_diff <= 1e-3) ? "PASS" : "FAIL";
    std::printf("  replay vs eager max_abs=%.3e  [%s, threshold 1e-3]\n",
                max_replay_diff, replay_verdict);

    // 5. Wall-time comparison: 5 warmup + 50 timed iterations each.
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    constexpr int N_WARMUP = 5;
    constexpr int N_ITERS = 50;

    // Eager.
    for (int i = 0; i < N_WARMUP; ++i) {
        smallfield::cuda::mace_inference_reset_per_atom_energy_accumulator(S, n_atoms, stream);
        smallfield::cuda::mace_inference_forward(arch, W.device(), batch, S, stream);
    }
    cudaStreamSynchronize(stream);
    cudaEventRecord(t0, stream);
    for (int i = 0; i < N_ITERS; ++i) {
        smallfield::cuda::mace_inference_reset_per_atom_energy_accumulator(S, n_atoms, stream);
        smallfield::cuda::mace_inference_forward(arch, W.device(), batch, S, stream);
    }
    cudaEventRecord(t1, stream);
    cudaEventSynchronize(t1);
    float ms_eager = 0.0f;
    cudaEventElapsedTime(&ms_eager, t0, t1);

    // Graph replay.
    for (int i = 0; i < N_WARMUP; ++i) captured.replay(stream);
    cudaStreamSynchronize(stream);
    cudaEventRecord(t0, stream);
    for (int i = 0; i < N_ITERS; ++i) captured.replay(stream);
    cudaEventRecord(t1, stream);
    cudaEventSynchronize(t1);
    float ms_replay = 0.0f;
    cudaEventElapsedTime(&ms_replay, t0, t1);

    std::printf("  eager  %d iters total %.3f ms  (%.4f ms/forward)\n",
                N_ITERS, ms_eager, ms_eager / N_ITERS);
    std::printf("  graph  %d iters total %.3f ms  (%.4f ms/replay)\n",
                N_ITERS, ms_replay, ms_replay / N_ITERS);
    std::printf("  speedup %.2fx (per-launch overhead saved)\n",
                ms_eager / std::max(ms_replay, 1e-6f));

    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaStreamDestroy(stream);
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

    run_batched(W, export_dir);

    std::printf("\n=== Track E1: CUDA Graph capture validation ===\n");
    run_with_graph_capture(W, export_dir + "/golden_small_caffeine.bin", "caffeine");

    return 0;
}
