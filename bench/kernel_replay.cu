/**
 *  @file bench/bench_kernel_replay.cu
 *  @brief Replays one captured atom-limited bin through the two ETKDG warp kernels, linking only CUDA.
 *
 *  Every NVIDIA profiler currently dies with a SIGSEGV at the first CUDA call inside a SmallField binary — the
 *  injection library and `dlopen` recurse until the stack is gone, 321k frames in — while a program that links the
 *  CUDA runtime alone profiles cleanly on the same box. The difference is the conda-forge stack: RDKit, boost,
 *  Arrow, `{fmt}`, and conda's own `libstdc++` and `libgcc_s`. This binary links none of it, which is the whole
 *  reason it exists, why it prints through `std::printf` rather than `fmt::println` as every other `bench/` binary
 *  does, and why `bench/CMakeLists.txt` builds it as its own target rather than through `define_smallfield_target`
 *  — `smallfield_settings` for the flags the shipped kernels compile under, `CUDA::cudart` for the runtime, nothing
 *  else. The house grammar is unchanged: a verb at column zero, ASCII keys, `key=value` throughout.
 *  `SMALLFIELD_CAPTURE_DIR=<dir> SMALLFIELD_CAPTURE_BIN=48 bench_profile --smi <corpus> --conformers 3` writes the
 *  snapshot this reads; `src/smallfield/api/wave_capture.hpp` is the writer and defines the manifest keys.
 */

#include <cstdint> // `std::uint32_t`, `std::uint64_t`
#include <cstdio>  // `std::fopen`, `std::printf`
#include <cstdlib> // `std::strtoull`
#include <cstring> // `std::strcmp`, `std::strncmp`

#include <algorithm> // `std::sort`
#include <string>    // `std::string`
#include <vector>    // `std::vector`

#include <cuda_runtime.h>

#include "smallfield/cuda/etkdg_pipeline.cuh"

namespace sfc = smallfield::cuda;
namespace sft = smallfield::types;

/** Name printed on every line, so a concatenated log stays attributable. */
static constexpr char const *executable_name_k = "bench_kernel_replay";

/** Warp size, spelled once, matching the block size both kernels are launched at. */
static constexpr std::uint32_t warp_size_k = 32;

/** The byte a status slot carries before either kernel has written it. Zero is `conformer_status_t::success_k`, so a
 *  slot memset to zero and never reached would count as a conformer the replay produced -- the one measurement
 *  separating a representative capture from one the polish kernel skipped. */
static constexpr std::uint8_t unwritten_status_k = 0xFF;

#pragma region Capture Manifest

/** Every scalar `wave_capture.hpp` wrote, in the order the manifest lists it. Byte offsets are into `batch.bin`,
 *  `embedding_offset_floats` into `embedding.bin`; both blobs upload whole with the offsets applied on the device
 *  pointer, so a kernel argument here is the pipeline's own address arithmetic. */
struct capture_manifest_t {
    std::uint64_t bin_size = 0;
    std::uint64_t conformer_count = 0;
    std::uint64_t max_stereo_attempts = 0;
    std::uint64_t bin_max_atoms = 0;
    std::uint64_t gpu_max_atoms = 0;
    std::uint64_t grid_etkdg = 0;
    std::uint64_t grid_mmff = 0;
    std::uint64_t shared_bytes_etkdg = 0;
    std::uint64_t shared_bytes_mmff = 0;
    std::uint64_t random_seed = 0;
    std::uint64_t batch_bytes = 0;
    std::uint64_t embedding_floats = 0;
    std::uint64_t embedding_offset_floats = 0;
    std::uint64_t offset_bounds_matrices = 0;
    std::uint64_t offset_chiral_centers = 0;
    std::uint64_t offset_etk_torsions = 0;
    std::uint64_t offset_distance_constraints = 0;
    std::uint64_t offset_etk_interactions = 0;
    std::uint64_t offset_etk_atom_offsets = 0;
    std::uint64_t offset_molecule_metadata = 0;
    std::uint64_t offset_input_indices = 0;
    std::uint64_t offset_mmff_atom_vdw = 0;
    std::uint64_t offset_mmff_atom_charge = 0;
    std::uint64_t offset_bond_distance = 0;
    std::uint64_t offset_mmff_bonds = 0;
    std::uint64_t offset_mmff_angles = 0;
    std::uint64_t offset_mmff_torsions = 0;
    std::uint64_t offset_mmff_oops = 0;
    std::uint64_t offset_mmff_stretchbends = 0;
    std::uint64_t offset_mmff_interactions = 0;
    std::uint64_t offset_mmff_atom_offsets = 0;
};

/** What a manifest read ended as; a missing key and an unreadable file are different failures. */
enum class manifest_read_t { parsed_k, unreadable_k, incomplete_k };

/** Bind every manifest key to its field, so the reader and the writer share one list. */
static std::vector<std::pair<char const *, std::uint64_t *>> manifest_bindings_(capture_manifest_t &manifest) {
    return {{"bin_size", &manifest.bin_size},
            {"conformer_count", &manifest.conformer_count},
            {"max_stereo_attempts", &manifest.max_stereo_attempts},
            {"bin_max_atoms", &manifest.bin_max_atoms},
            {"gpu_max_atoms", &manifest.gpu_max_atoms},
            {"grid_etkdg", &manifest.grid_etkdg},
            {"grid_mmff", &manifest.grid_mmff},
            {"shared_bytes_etkdg", &manifest.shared_bytes_etkdg},
            {"shared_bytes_mmff", &manifest.shared_bytes_mmff},
            {"random_seed", &manifest.random_seed},
            {"batch_bytes", &manifest.batch_bytes},
            {"embedding_floats", &manifest.embedding_floats},
            {"embedding_offset_floats", &manifest.embedding_offset_floats},
            {"offset_bounds_matrices", &manifest.offset_bounds_matrices},
            {"offset_chiral_centers", &manifest.offset_chiral_centers},
            {"offset_etk_torsions", &manifest.offset_etk_torsions},
            {"offset_distance_constraints", &manifest.offset_distance_constraints},
            {"offset_etk_interactions", &manifest.offset_etk_interactions},
            {"offset_etk_atom_offsets", &manifest.offset_etk_atom_offsets},
            {"offset_molecule_metadata", &manifest.offset_molecule_metadata},
            {"offset_input_indices", &manifest.offset_input_indices},
            {"offset_mmff_atom_vdw", &manifest.offset_mmff_atom_vdw},
            {"offset_mmff_atom_charge", &manifest.offset_mmff_atom_charge},
            {"offset_bond_distance", &manifest.offset_bond_distance},
            {"offset_mmff_bonds", &manifest.offset_mmff_bonds},
            {"offset_mmff_angles", &manifest.offset_mmff_angles},
            {"offset_mmff_torsions", &manifest.offset_mmff_torsions},
            {"offset_mmff_oops", &manifest.offset_mmff_oops},
            {"offset_mmff_stretchbends", &manifest.offset_mmff_stretchbends},
            {"offset_mmff_interactions", &manifest.offset_mmff_interactions},
            {"offset_mmff_atom_offsets", &manifest.offset_mmff_atom_offsets}};
}

/** Fill @p manifest from `<directory>/manifest.txt`, refusing a file that leaves any key unseen. */
static manifest_read_t read_manifest_(std::string const &directory, capture_manifest_t &manifest) {
    std::string const path = directory + "/manifest.txt";
    std::FILE *const file = std::fopen(path.c_str(), "r");
    if (file == nullptr) return manifest_read_t::unreadable_k;

    auto const bindings = manifest_bindings_(manifest);
    std::vector<bool> seen(bindings.size(), false);
    char line[256];
    while (std::fgets(line, sizeof(line), file) != nullptr) {
        char *const separator = std::strchr(line, '=');
        if (separator == nullptr) continue;
        *separator = '\0';
        for (std::size_t binding = 0; binding < bindings.size(); ++binding) {
            if (std::strcmp(line, bindings[binding].first) != 0) continue;
            *bindings[binding].second = std::strtoull(separator + 1, nullptr, 10);
            seen[binding] = true;
        }
    }
    std::fclose(file);

    for (std::size_t binding = 0; binding < bindings.size(); ++binding)
        if (!seen[binding]) {
            std::printf("FAIL manifest key=%s missing\n", bindings[binding].first);
            return manifest_read_t::incomplete_k;
        }
    return manifest_read_t::parsed_k;
}

/** Read the whole of @p path into @p bytes, refusing a length the manifest did not predict. */
static bool read_blob_(std::string const &path, std::uint64_t expected_bytes, std::vector<char> &bytes) {
    std::FILE *const file = std::fopen(path.c_str(), "rb");
    if (file == nullptr) {
        std::printf("FAIL blob path=%s reason=unreadable\n", path.c_str());
        return false;
    }
    bytes.resize(expected_bytes);
    std::size_t const read = std::fread(bytes.data(), 1, expected_bytes, file);
    std::fclose(file);
    if (read != expected_bytes) {
        std::printf("FAIL blob path=%s expected=%llu got=%zu\n", path.c_str(),
                    static_cast<unsigned long long>(expected_bytes), read);
        return false;
    }
    return true;
}

#pragma endregion Capture Manifest

#pragma region Device Replay

/** Every device allocation the replay owns, freed together whichever way the run leaves. */
struct replay_device_t {
    char *batch = nullptr;
    float *embedding = nullptr;
    float *coordinates = nullptr;
    float *energies = nullptr;
    std::uint8_t *status = nullptr;
    float *diagnostics = nullptr;

    ~replay_device_t() {
        cudaFree(batch);
        cudaFree(embedding);
        cudaFree(coordinates);
        cudaFree(energies);
        cudaFree(status);
        cudaFree(diagnostics);
    }
};

/** Launch the ETKDG kernel exactly as `enqueue_binned_kernels_` does, on the default stream. */
static void launch_etkdg_(capture_manifest_t const &manifest, replay_device_t const &device) {
    auto const pool_at = [&device](std::uint64_t offset) { return device.batch + offset; };
    std::uint32_t const bin_size = static_cast<std::uint32_t>(manifest.bin_size);
    std::uint32_t const conformer_count = static_cast<std::uint32_t>(manifest.conformer_count);
    std::uint32_t const gpu_max_atoms = static_cast<std::uint32_t>(manifest.gpu_max_atoms);

    sfc::etkdg_pipeline_warp_kernel<<<static_cast<std::uint32_t>(manifest.grid_etkdg), warp_size_k,
                                      static_cast<std::uint32_t>(manifest.shared_bytes_etkdg)>>>(
        reinterpret_cast<float const *>(pool_at(manifest.offset_bounds_matrices)),
        device.embedding + manifest.embedding_offset_floats,
        reinterpret_cast<sft::chiral_center_descriptor_t const *>(pool_at(manifest.offset_chiral_centers)),
        reinterpret_cast<sft::etk_torsion_descriptor_t const *>(pool_at(manifest.offset_etk_torsions)),
        reinterpret_cast<sft::distance_constraint_t const *>(pool_at(manifest.offset_distance_constraints)),
        reinterpret_cast<sft::etk_atom_interaction_t const *>(pool_at(manifest.offset_etk_interactions)),
        reinterpret_cast<sft::etk_atom_offsets_t const *>(pool_at(manifest.offset_etk_atom_offsets)),
        reinterpret_cast<sft::etkdg_molecule_metadata_t const *>(pool_at(manifest.offset_molecule_metadata)),
        reinterpret_cast<std::uint32_t const *>(pool_at(manifest.offset_input_indices)), device.coordinates,
        device.energies, device.status, bin_size, conformer_count,
        static_cast<std::uint32_t>(manifest.max_stereo_attempts), static_cast<std::uint32_t>(manifest.bin_max_atoms),
        gpu_max_atoms, gpu_max_atoms, manifest.random_seed);
}

/** The shape this build's polish kernel launches under, derived rather than replayed. Taking grid and shared size
 *  from the manifest would measure every variant under the capturing build's occupancy without saying so; both come
 *  from the compiled kernel, and `main` prints them beside the captured pair. */
struct polish_launch_shape_t {
    std::uint32_t grid = 0;
    std::uint32_t shared_bytes = 0;
};

static polish_launch_shape_t polish_launch_shape_(capture_manifest_t const &manifest) {
    std::uint32_t const bin_max_atoms = static_cast<std::uint32_t>(manifest.bin_max_atoms);
    std::uint32_t const shared_bytes = sfc::compute_mmff_polish_shared_memory_bytes(bin_max_atoms);

    int blocks_per_multiprocessor = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_multiprocessor, sfc::mmff_polish_warp_kernel,
                                                  static_cast<int>(warp_size_k), shared_bytes);
    int multiprocessor_count = 0;
    cudaDeviceGetAttribute(&multiprocessor_count, cudaDevAttrMultiProcessorCount, 0);

    std::uint32_t const bin_pair_count = static_cast<std::uint32_t>(manifest.bin_size * manifest.conformer_count);
    std::uint32_t const occupancy_grid = static_cast<std::uint32_t>(blocks_per_multiprocessor) *
                                         static_cast<std::uint32_t>(multiprocessor_count);
    return {.grid = occupancy_grid < bin_pair_count ? occupancy_grid : bin_pair_count, .shared_bytes = shared_bytes};
}

/** Launch the polish kernel over the coordinates and statuses its sibling has just written. */
static void launch_mmff_(capture_manifest_t const &manifest, replay_device_t const &device,
                         polish_launch_shape_t shape) {
    auto const pool_at = [&device](std::uint64_t offset) { return device.batch + offset; };
    std::uint32_t const bin_size = static_cast<std::uint32_t>(manifest.bin_size);
    std::uint32_t const conformer_count = static_cast<std::uint32_t>(manifest.conformer_count);
    std::uint32_t const gpu_max_atoms = static_cast<std::uint32_t>(manifest.gpu_max_atoms);

    sfc::mmff_polish_warp_kernel<<<shape.grid, warp_size_k, shape.shared_bytes>>>(
        reinterpret_cast<sft::mmff94_atom_vdw_params_t const *>(pool_at(manifest.offset_mmff_atom_vdw)),
        reinterpret_cast<float const *>(pool_at(manifest.offset_mmff_atom_charge)),
        reinterpret_cast<std::uint8_t const *>(pool_at(manifest.offset_bond_distance)),
        reinterpret_cast<sft::mmff94_bond_t const *>(pool_at(manifest.offset_mmff_bonds)),
        reinterpret_cast<sft::mmff94_angle_t const *>(pool_at(manifest.offset_mmff_angles)),
        reinterpret_cast<sft::mmff94_torsion_t const *>(pool_at(manifest.offset_mmff_torsions)),
        reinterpret_cast<sft::mmff94_oop_t const *>(pool_at(manifest.offset_mmff_oops)),
        reinterpret_cast<sft::mmff94_stretchbend_t const *>(pool_at(manifest.offset_mmff_stretchbends)),
        reinterpret_cast<sft::mmff94_atom_interaction_t const *>(pool_at(manifest.offset_mmff_interactions)),
        reinterpret_cast<sft::mmff94_atom_offsets_t const *>(pool_at(manifest.offset_mmff_atom_offsets)),
        reinterpret_cast<sft::chiral_center_descriptor_t const *>(pool_at(manifest.offset_chiral_centers)),
        reinterpret_cast<sft::etkdg_molecule_metadata_t const *>(pool_at(manifest.offset_molecule_metadata)),
        device.coordinates, device.energies, device.status, bin_size, conformer_count, gpu_max_atoms,
        static_cast<std::uint32_t>(manifest.bin_max_atoms), device.diagnostics);
}

/** Upload both blobs and allocate the three output arrays this bin writes. Outputs start zeroed on every repeat,
 *  which is what makes the pair self-consistent: the polish kernel reads what its sibling has just written, exactly
 *  as in the pipeline, so nothing has to be captured mid-flight. */
static bool upload_(capture_manifest_t const &manifest, std::vector<char> const &batch,
                    std::vector<char> const &embedding, replay_device_t &device) {
    std::uint64_t const slot_count = manifest.bin_size * manifest.conformer_count;
    std::uint64_t const coordinate_count = slot_count * manifest.gpu_max_atoms * 3ull;
    bool allocated = cudaMalloc(&device.batch, manifest.batch_bytes) == cudaSuccess;
    allocated &= cudaMalloc(&device.embedding, embedding.size()) == cudaSuccess;
    allocated &= cudaMalloc(&device.coordinates, coordinate_count * sizeof(float)) == cudaSuccess;
    allocated &= cudaMalloc(&device.energies, slot_count * sizeof(float)) == cudaSuccess;
    allocated &= cudaMalloc(&device.status, slot_count) == cudaSuccess;
    allocated &= cudaMalloc(&device.diagnostics, slot_count * sfc::polish_diagnostic_floats_k * sizeof(float)) ==
                 cudaSuccess;
    if (!allocated) {
        std::printf("FAIL allocation bytes=%llu\n", static_cast<unsigned long long>(manifest.batch_bytes));
        return false;
    }
    cudaMemcpy(device.batch, batch.data(), batch.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(device.embedding, embedding.data(), embedding.size(), cudaMemcpyHostToDevice);
    cudaMemset(device.coordinates, 0, coordinate_count * sizeof(float));
    cudaMemset(device.energies, 0, slot_count * sizeof(float));
    cudaMemset(device.status, unwritten_status_k, slot_count);
    cudaMemset(device.diagnostics, 0, slot_count * sfc::polish_diagnostic_floats_k * sizeof(float));
    return cudaDeviceSynchronize() == cudaSuccess;
}

#pragma endregion Device Replay

#pragma region Polish Histogram

/** One decade per bucket, shared by the timestep and the RMS gradient — both are read as decades. */
static constexpr std::size_t decade_bucket_count_k = 7;
static constexpr float decade_edges_k[decade_bucket_count_k] = {1e-6f, 1e-5f, 1e-4f, 1e-3f, 1e-2f, 1e-1f, 1e30f};
static constexpr char const *decade_labels_k[decade_bucket_count_k] = {"lt_1e-6", "1e-6", "1e-5",   "1e-4",
                                                                       "1e-3",    "1e-2", "ge_1e-1"};

/** Index of the decade @p value falls in, saturating into the last bucket. */
static std::size_t decade_bucket_(float value) {
    for (std::size_t bucket = 0; bucket + 1 < decade_bucket_count_k; ++bucket)
        if (value < decade_edges_k[bucket]) return bucket;
    return decade_bucket_count_k - 1;
}

/** Print the median, the upper decile and the worst of @p samples, sorting them in place. */
static void print_quantiles_(char const *name, std::vector<float> &samples) {
    if (samples.empty()) return;
    std::sort(samples.begin(), samples.end());
    auto const quantile = [&samples](double fraction) {
        double const last = static_cast<double>(samples.size() - 1);
        return static_cast<double>(samples[static_cast<std::size_t>(fraction * last)]);
    };
    std::printf(" %s_p50=%.4g %s_p90=%.4g %s_max=%.4g", name, quantile(0.50), name, quantile(0.90), name,
                static_cast<double>(samples.back()));
}

/** Print how the polish kernel's conformer slots divided across its four exits, which the mean energy cannot
 *  separate: three ship a geometry MMFF94 never improved, and all four write energies. */
static void report_polish_diagnostics_(std::vector<float> const &diagnostics, std::vector<float> const &energies,
                                       std::uint64_t slot_count) {
    std::uint64_t path_counts[sfc::polish_path_count_k] = {};
    std::uint64_t timestep_counts[decade_bucket_count_k] = {};
    std::uint64_t rms_counts[decade_bucket_count_k] = {};
    std::uint64_t converged = 0, exhausted = 0;
    double iteration_sum = 0.0, drop_sum = 0.0, rms_sum = 0.0;
    float worst_iterations = 0.0f;
    std::uint64_t relaxed = 0;
    std::vector<float> final_energies, final_gradients;

    custd::span<float const> const rows {diagnostics.data(), diagnostics.size()};
    for (std::uint64_t slot = 0; slot < slot_count; ++slot) {
        sfc::polish_row_t const row = sfc::decode_polish_row(rows, slot);
        ++path_counts[static_cast<int>(row.path)];
        if (row.path != sfc::polish_path_t::write_back_declined_k &&
            row.path != sfc::polish_path_t::write_back_accepted_k)
            continue;

        ++relaxed;
        if (row.status == sfc::fire_status_t::converged_k) ++converged;
        else ++exhausted;
        iteration_sum += static_cast<double>(row.iterations);
        if (row.iterations > worst_iterations) worst_iterations = row.iterations;
        rms_sum += static_cast<double>(row.rms_gradient);
        drop_sum += static_cast<double>(row.staged_energy) - static_cast<double>(energies[slot]);
        ++timestep_counts[decade_bucket_(row.final_timestep)];
        ++rms_counts[decade_bucket_(row.rms_gradient)];
        final_energies.push_back(energies[slot]);
        final_gradients.push_back(row.rms_gradient);
    }

    std::printf("PATH unreached=%llu status_skipped=%llu no_terms=%llu declined=%llu accepted=%llu\n",
                static_cast<unsigned long long>(path_counts[0]), static_cast<unsigned long long>(path_counts[1]),
                static_cast<unsigned long long>(path_counts[2]), static_cast<unsigned long long>(path_counts[3]),
                static_cast<unsigned long long>(path_counts[4]));

    double const relaxed_denominator = relaxed == 0 ? 1.0 : static_cast<double>(relaxed);
    std::printf("STAGE relaxed=%llu converged=%llu exhausted=%llu\n", static_cast<unsigned long long>(relaxed),
                static_cast<unsigned long long>(converged), static_cast<unsigned long long>(exhausted));
    std::printf("SPEND iters_mean=%.1f iters_max=%.0f rms_mean=%.3g drop_mean=%.4g\n",
                iteration_sum / relaxed_denominator, static_cast<double>(worst_iterations),
                rms_sum / relaxed_denominator, drop_sum / relaxed_denominator);

    std::printf("TIMESTEP");
    for (std::size_t bucket = 0; bucket < decade_bucket_count_k; ++bucket)
        std::printf(" %s=%llu", decade_labels_k[bucket], static_cast<unsigned long long>(timestep_counts[bucket]));
    std::printf("\n");

    std::printf("GRADIENT");
    for (std::size_t bucket = 0; bucket < decade_bucket_count_k; ++bucket)
        std::printf(" %s=%llu", decade_labels_k[bucket], static_cast<unsigned long long>(rms_counts[bucket]));
    std::printf("\n");

    // A mean over conformer energies is dominated by whichever slot kept a van der Waals clash, whose r^-12 term runs
    // to millions, so two minimizers can only be compared on the quantiles.
    std::printf("QUANTILE");
    print_quantiles_("energy", final_energies);
    print_quantiles_("rms", final_gradients);
    std::printf("\n");
}

#pragma endregion Polish Histogram

#pragma region Command Line

/** Everything this driver takes from the command line. */
struct replay_arguments_t {
    std::string capture_directory;
    std::uint32_t repeats = 5;
};

static void print_usage_(char const *invocation) { std::printf("usage: %s --capture DIR [--repeat N]\n", invocation); }

/** Fill @p arguments from `argv`, refusing anything it does not recognize. */
static bool parse_arguments_(int argc, char **argv, replay_arguments_t &arguments) {
    for (int argument = 1; argument < argc; ++argument) {
        bool const has_value = argument + 1 < argc;
        if (std::strcmp(argv[argument], "--capture") == 0 && has_value) arguments.capture_directory = argv[++argument];
        else if (std::strcmp(argv[argument], "--repeat") == 0 && has_value)
            arguments.repeats = static_cast<std::uint32_t>(std::strtoull(argv[++argument], nullptr, 10));
        else {
            print_usage_(argv[0]);
            return false;
        }
    }
    if (arguments.capture_directory.empty()) {
        print_usage_(argv[0]);
        return false;
    }
    return arguments.repeats != 0;
}

#pragma endregion Command Line

int main(int argc, char **argv) {
    replay_arguments_t arguments;
    if (!parse_arguments_(argc, argv, arguments)) return 2;

    std::printf("%s: replaying one captured atom-limited bin through the two ETKDG warp kernels\n", executable_name_k);

    capture_manifest_t manifest;
    if (read_manifest_(arguments.capture_directory, manifest) != manifest_read_t::parsed_k) {
        std::printf("FAIL manifest directory=%s\n", arguments.capture_directory.c_str());
        return 1;
    }

    std::vector<char> batch, embedding;
    if (!read_blob_(arguments.capture_directory + "/batch.bin", manifest.batch_bytes, batch)) return 1;
    if (!read_blob_(arguments.capture_directory + "/embedding.bin", manifest.embedding_floats * sizeof(float),
                    embedding))
        return 1;

    replay_device_t device;
    if (!upload_(manifest, batch, embedding, device)) return 1;

    std::printf("SHAPE bin_max_atoms=%llu bin_size=%llu conformers=%llu pairs=%llu grid_etkdg=%llu " //
                "grid_mmff=%llu shared_etkdg=%llu shared_mmff=%llu output_stride=%llu\n",            //
                static_cast<unsigned long long>(manifest.bin_max_atoms),                             //
                static_cast<unsigned long long>(manifest.bin_size),                                  //
                static_cast<unsigned long long>(manifest.conformer_count),                           //
                static_cast<unsigned long long>(manifest.bin_size * manifest.conformer_count),       //
                static_cast<unsigned long long>(manifest.grid_etkdg),                                //
                static_cast<unsigned long long>(manifest.grid_mmff),                                 //
                static_cast<unsigned long long>(manifest.shared_bytes_etkdg),                        //
                static_cast<unsigned long long>(manifest.shared_bytes_mmff),                         //
                static_cast<unsigned long long>(manifest.gpu_max_atoms));

    // What this build's polish kernel launches under, beside what the capture recorded: a variant that changes
    // shared memory or occupancy moves these apart, and the captured shape would report the wrong number silently.
    polish_launch_shape_t const shape = polish_launch_shape_(manifest);
    std::printf("SHAPE polish_grid=%llu polish_shared=%llu captured_grid=%llu captured_shared=%llu\n", //
                static_cast<unsigned long long>(shape.grid),                                           //
                static_cast<unsigned long long>(shape.shared_bytes),                                   //
                static_cast<unsigned long long>(manifest.grid_mmff),                                   //
                static_cast<unsigned long long>(manifest.shared_bytes_mmff));

    cudaEvent_t before, between, after;
    cudaEventCreate(&before);
    cudaEventCreate(&between);
    cudaEventCreate(&after);

    std::uint64_t const slot_count = manifest.bin_size * manifest.conformer_count;

    // One untimed pass, so the driver's first-launch setup is not charged to the steady state.
    launch_etkdg_(manifest, device);
    launch_mmff_(manifest, device, shape);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::printf("FAIL launch reason=%s\n", cudaGetErrorString(cudaGetLastError()));
        return 1;
    }

    for (std::uint32_t repeat = 0; repeat < arguments.repeats; ++repeat) {
        cudaMemset(device.status, unwritten_status_k, slot_count);
        cudaMemset(device.diagnostics, 0, slot_count * sfc::polish_diagnostic_floats_k * sizeof(float));
        cudaEventRecord(before);
        launch_etkdg_(manifest, device);
        cudaEventRecord(between);
        launch_mmff_(manifest, device, shape);
        cudaEventRecord(after);
        cudaEventSynchronize(after);
        if (cudaError_t const failure = cudaGetLastError(); failure != cudaSuccess) {
            std::printf("FAIL launch reason=%s\n", cudaGetErrorString(failure));
            return 1;
        }
        float etkdg_ms = 0.0f, mmff_ms = 0.0f;
        cudaEventElapsedTime(&etkdg_ms, before, between);
        cudaEventElapsedTime(&mmff_ms, between, after);
        std::printf("TIME repeat=%u etkdg_ms=%.1f mmff_ms=%.1f\n", repeat, static_cast<double>(etkdg_ms),
                    static_cast<double>(mmff_ms));
    }

    cudaEventDestroy(before);
    cudaEventDestroy(between);
    cudaEventDestroy(after);

    // A snapshot of unrepresentative inputs replays at the right speed for the wrong reason — ETKDG burning every
    // stereo attempt, the polish skipping every slot — which the `success_k` count and the written energies separate.
    std::vector<std::uint8_t> status(slot_count, unwritten_status_k);
    std::vector<float> energies(slot_count);
    cudaMemcpy(status.data(), device.status, slot_count, cudaMemcpyDeviceToHost);
    cudaMemcpy(energies.data(), device.energies, slot_count * sizeof(float), cudaMemcpyDeviceToHost);
    std::uint64_t succeeded = 0;
    double energy_sum = 0.0;
    for (std::uint64_t slot = 0; slot < slot_count; ++slot)
        if (status[slot] == static_cast<std::uint8_t>(sft::conformer_status_t::success_k)) {
            ++succeeded;
            energy_sum += static_cast<double>(energies[slot]);
        }
    std::printf("OUTPUT slots=%llu succeeded=%llu mean_energy=%.4g\n", static_cast<unsigned long long>(slot_count),
                static_cast<unsigned long long>(succeeded),
                static_cast<double>(succeeded == 0 ? 0.0 : energy_sum / static_cast<double>(succeeded)));

    std::vector<float> diagnostics(slot_count * sfc::polish_diagnostic_floats_k);
    cudaMemcpy(diagnostics.data(), device.diagnostics, diagnostics.size() * sizeof(float), cudaMemcpyDeviceToHost);
    report_polish_diagnostics_(diagnostics, energies, slot_count);

    std::printf("PASS repeats=%u\n", arguments.repeats);
    return 0;
}
