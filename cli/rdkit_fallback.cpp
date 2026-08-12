/**
 *  @file cli/rdkit_fallback.cpp
 *  @brief Runs RDKit ETKDG embedding and MMFF minimization, in an OpenMP-parallel loop, over the
 *  conformer slots the GPU pipeline could not fill.
 */

#include <cstring> // `std::memset`

#include <atomic>  // `std::atomic`
#include <memory>  // `std::unique_ptr`
#include <sstream> // RDKit's `AtomTyper.h` uses `std::stringstream` without including it
#include <vector>  // `std::vector`

#include <omp.h>
#include <cuda_fp16.h>
#include <GraphMol/Chirality.h>
#include <GraphMol/DistGeomHelpers/Embedder.h>
#include <GraphMol/ForceFieldHelpers/MMFF/AtomTyper.h>
#include <GraphMol/ForceFieldHelpers/MMFF/MMFF.h>
#include <GraphMol/GraphMol.h>
#include <GraphMol/MolOps.h>
#include <GraphMol/RWMol.h>
#include <GraphMol/SmilesParse/SmilesParse.h>

#include "smallfield/api/smallfield.hpp"
#include "smallfield/molecule_parse.hpp"

#include "cli/rdkit_fallback.hpp"

namespace sfapi = smallfield::api;

namespace smallfield::cli {

/** Single-threaded warm-up that forces every RDKit lazy-init singleton the fallback OMP loop touches to
 *  populate its caches before the pool of ~188 fallback threads starts racing on first access. See matching
 *  `warm_up_rdkit_globals_once_` in `etkdg_prepare_batch.hpp`. */
static void warm_up_rdkit_globals_for_fallback_() {
    [[maybe_unused]] static bool const warmed = [] {
        try {
            auto warm = RDKit::SmilesToMol("C[C@H](O)[C@@H](Cl)c1ccccc1");
            if (warm) {
                RDKit::MolOps::addHs(*warm);
                RDKit::MolOps::assignStereochemistry(*warm, /*cleanIt=*/true, /*force=*/true,
                                                     /*flagPossibleStereoCenters=*/true);
                // Constructing the properties is the warm-up: it forces RDKit's lazy MMFF
                // parameter singletons to load before the parallel region opens on them.
                RDKit::MMFF::MMFFMolProperties props(*warm);
            }
        }
        catch (...) {
            // Best-effort.
        }
        return true;
    }();
}

#pragma region One Molecule

/** This molecule's conformer slots, as one view into the batch-wide arrays. */
template <typename element_type_>
static custd::span<element_type_> slots_of_(custd::span<element_type_> batch,
                                            sfapi::conformer_slot_layout_t const &layout,
                                            std::uint32_t molecule_index) {
    std::size_t const first = static_cast<std::size_t>(molecule_index) * layout.num_conformers;
    return batch.subspan(first, layout.num_conformers);
}

/** Conformer slots of one molecule the GPU did not fill, which is how many fresh ones to embed. */
static std::uint32_t unfilled_slots_of_(custd::span<sfapi::conformer_status_t const> status) {
    std::uint32_t unfilled = 0;
    for (sfapi::conformer_status_t const slot_status : status)
        if (slot_status != sfapi::conformer_status_t::success_k) ++unfilled;
    return unfilled;
}

/**
 *  @brief Parse one SMILES the way the GPU path does and confirm MMFF can type the result.
 *
 *  Embedding is the dominant CPU cost here and it is wasted on a molecule the force field cannot type:
 *  `constructForceField` would throw below and every conformer would fail anyway, so asking first costs a
 *  typing check instead of K embeddings.
 *
 *  @param[out] molecule Receives the parsed molecule on `success_k`, untouched on every refusal.
 *  @return `success_k`, or the status every slot of this molecule takes.
 */
static sfapi::conformer_status_t admit_molecule_(std::string_view smiles, std::uint32_t max_atoms,
                                                 std::unique_ptr<RDKit::RWMol> &molecule) {
    if (smiles.empty()) return sfapi::conformer_status_t::invalid_smiles_k;

    // Same parse contract as the GPU path, counterion removal included, so a molecule recovered here
    // carries the atom count and geometry the rest of the pipeline expects.
    smallfield::parse_options_t parse_options;
    parse_options.max_atoms = max_atoms;
    smallfield::parse_result_t parsed = smallfield::parse_smiles(smiles, parse_options);
    if (parsed.status != sfapi::conformer_status_t::success_k) return parsed.status;
    if (!RDKit::MMFF::MMFFMolProperties(*parsed.molecule).isValid())
        return sfapi::conformer_status_t::unsupported_element_k;

    molecule = std::move(parsed.molecule);
    return sfapi::conformer_status_t::success_k;
}

/**
 *  @brief MMFF-minimize one embedded conformer in place and read back its energy.
 *
 *  @param[out] energy_kcal_per_mol Minimized MMFF94 energy, untouched unless the field converged.
 */
static bool minimize_conformer_(RDKit::RWMol &molecule, int conformer_id, float &energy_kcal_per_mol) {
    try {
        auto force_field = std::unique_ptr<ForceFields::ForceField>(
            RDKit::MMFF::constructForceField(molecule, /*nonBondedThresh=*/100.0, conformer_id));
        if (!force_field) return false;
        force_field->initialize();
        force_field->minimize(200, 1e-4, 1e-6);
        energy_kcal_per_mol = static_cast<float>(force_field->calcEnergy());
        return true;
    }
    catch (...) {
        return false;
    }
}

/** Write one RDKit conformer into its structure-of-arrays slot, zeroing the padding atoms first. */
static void write_conformer_into_slot_(RDKit::RWMol const &molecule, int conformer_id,
                                       sfapi::conformer_slot_layout_t const &layout, __half *slot) {
    std::memset(reinterpret_cast<void *>(slot), 0, sizeof(__half) * layout.slot_stride());
    auto const &conformer = molecule.getConformer(conformer_id);
    for (unsigned int atom = 0; atom < molecule.getNumAtoms(); ++atom) {
        auto const &position = conformer.getAtomPos(atom);
        slot[layout.component_offset(0u, atom)] = __float2half(static_cast<float>(position.x));
        slot[layout.component_offset(1u, atom)] = __float2half(static_cast<float>(position.y));
        slot[layout.component_offset(2u, atom)] = __float2half(static_cast<float>(position.z));
    }
}

/**
 *  @brief Embed fresh conformers for one molecule and lay them into the slots the GPU left unfilled.
 *
 *  Only as many conformers are embedded as there are unfilled slots: `EmbedMultipleConfs` is the dominant
 *  CPU cost in this path, and failures cluster in one of K, so halving the count halves the per-molecule
 *  wall in the common case. Slots the GPU did fill keep their coordinates, energy and status untouched.
 *
 *  @return Slots this molecule recovered.
 */
static std::uint32_t recover_molecule_(RDKit::RWMol &molecule, std::uint64_t random_seed,
                                       sfapi::conformer_slot_layout_t const &layout,
                                       custd::span<sfapi::conformer_status_t> status, custd::span<float> energies,
                                       custd::span<__half> coordinates) {
    RDKit::DGeomHelpers::EmbedParameters embed_params = RDKit::DGeomHelpers::ETKDGv3;
    embed_params.randomSeed = static_cast<int>(random_seed);
    embed_params.maxIterations = 200;

    std::vector<int> conformer_ids;
    try {
        conformer_ids = RDKit::DGeomHelpers::EmbedMultipleConfs(
            molecule, static_cast<unsigned int>(unfilled_slots_of_(status)), embed_params);
    }
    catch (...) { /* an embedding RDKit refused leaves every slot as the GPU left it */
    }

    std::uint32_t recovered = 0;
    std::size_t fresh_index = 0;
    for (std::uint32_t conformer_slot = 0; conformer_slot < layout.num_conformers; ++conformer_slot) {
        if (status[conformer_slot] == sfapi::conformer_status_t::success_k) continue; // the GPU filled this one
        if (fresh_index >= conformer_ids.size()) break;                               // nothing left to lay in
        int const conformer_id = conformer_ids[fresh_index++];

        float energy_kcal_per_mol = 0.0f;
        if (!minimize_conformer_(molecule, conformer_id, energy_kcal_per_mol)) continue;

        write_conformer_into_slot_(molecule, conformer_id, layout,
                                   coordinates.data() + conformer_slot * layout.slot_stride());
        energies[conformer_slot] = energy_kcal_per_mol;
        status[conformer_slot] = sfapi::conformer_status_t::success_k;
        ++recovered;
    }
    return recovered;
}

#pragma endregion One Molecule

std::uint64_t rdkit_fallback(rdkit_fallback_inputs_t const &inputs, rdkit_fallback_outputs_t &outputs) {
    sfapi::conformer_slot_layout_t const &layout = inputs.layout;
    std::uint64_t const num_retries = inputs.retry_indices.size();
    if (num_retries == 0) return 0;

    std::atomic<std::uint64_t> num_recovered {0};
    int const num_threads = inputs.num_threads > 0 ? inputs.num_threads : omp_get_max_threads();
    warm_up_rdkit_globals_for_fallback_();

#pragma omp parallel for schedule(dynamic, 16) num_threads(num_threads) if (num_threads > 1)
    for (std::uint64_t retry_entry = 0; retry_entry < num_retries; ++retry_entry) {
        std::uint32_t const molecule_index = inputs.retry_indices[retry_entry];
        custd::span<sfapi::conformer_status_t> const status = slots_of_(outputs.status, layout, molecule_index);
        if (unfilled_slots_of_(status) == 0) continue; // the GPU filled every slot after all

        std::unique_ptr<RDKit::RWMol> molecule;
        if (sfapi::conformer_status_t const admission = admit_molecule_(inputs.smiles[molecule_index], layout.max_atoms,
                                                                        molecule);
            admission != sfapi::conformer_status_t::success_k) {
            // Only the unfilled slots take the refusal; a slot the GPU already filled keeps its geometry,
            // which a whole-molecule overwrite used to discard along with the status that described it.
            for (sfapi::conformer_status_t &slot_status : status)
                if (slot_status != sfapi::conformer_status_t::success_k) slot_status = admission;
            continue;
        }

        std::uint32_t const recovered = recover_molecule_(
            *molecule, inputs.random_seed, layout, status, slots_of_(outputs.energies, layout, molecule_index),
            outputs.coordinates.subspan(layout.slot_offset(molecule_index, 0),
                                        layout.num_conformers * layout.slot_stride()));
        if (recovered == 0) continue;

        // Only a molecule that actually produced a conformer gets an atom count. Writing it before the
        // embed left unrecovered molecules carrying a non-zero count over all-zero coordinates, which
        // defeats the `num_atoms == 0` guard downstream and buys them a full shape-context parse plus four
        // substructure searches for a row that has no geometry.
        outputs.num_atoms_per_molecule[molecule_index] = static_cast<std::uint8_t>(molecule->getNumAtoms());
        num_recovered.fetch_add(1, std::memory_order_relaxed);
    }

    return num_recovered.load();
}

} // namespace smallfield::cli
