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

std::uint64_t rdkit_fallback(rdkit_fallback_inputs_t const &inputs, rdkit_fallback_outputs_t &outputs) {
    std::uint32_t const num_conformers = inputs.num_conformers;
    std::uint32_t const max_atoms = inputs.max_atoms;
    sfapi::conformer_slot_layout_t const slot_layout {max_atoms, num_conformers};
    std::uint32_t const coordinate_stride = static_cast<std::uint32_t>(slot_layout.slot_stride());
    std::uint64_t const num_retries = inputs.retry_indices.size();

    if (num_retries == 0) return 0;

    std::atomic<std::uint64_t> num_recovered {0};

    int const num_threads = inputs.num_threads > 0 ? inputs.num_threads : omp_get_max_threads();

    warm_up_rdkit_globals_for_fallback_();

#pragma omp parallel for schedule(dynamic, 16) num_threads(num_threads) if (num_threads > 1)
    for (std::uint64_t retry_entry = 0; retry_entry < num_retries; ++retry_entry) {
        std::uint32_t const molecule_index = inputs.retry_indices[retry_entry];
        std::string_view const smiles = inputs.smiles[molecule_index];

        // Successful GPU conformers (solver==0, status==success) are preserved untouched — they keep
        // their GPU coordinates, energy, and status. Anything not flipped to success by this loop keeps
        // its existing failed status; the solver code below is only updated when a slot is successfully
        // recovered.
        std::uint32_t needed_slots = 0;
        for (std::uint32_t conformer_index = 0; conformer_index < num_conformers; ++conformer_index) {
            std::uint8_t const slot_status = outputs.status[molecule_index * num_conformers + conformer_index];
            if (slot_status != static_cast<std::uint8_t>(sfapi::conformer_status_t::success_k)) ++needed_slots;
        }
        if (needed_slots == 0) continue;

        if (smiles.empty()) {
            for (std::uint32_t conformer_index = 0; conformer_index < num_conformers; ++conformer_index)
                outputs.status[molecule_index * num_conformers + conformer_index] = static_cast<std::uint8_t>(
                    sfapi::conformer_status_t::invalid_smiles_k);
            continue;
        }

        // Same parse contract as the GPU path, counterion removal included, so a molecule recovered
        // here carries the atom count and geometry the rest of the pipeline expects.
        smallfield::parse_options_t parse_options;
        parse_options.max_atoms = max_atoms;
        smallfield::parse_result_t parsed = smallfield::parse_smiles(smiles, parse_options);
        if (parsed.status != sfapi::conformer_status_t::success_k) {
            for (std::uint32_t conformer_index = 0; conformer_index < num_conformers; ++conformer_index)
                outputs.status[molecule_index * num_conformers + conformer_index] = static_cast<std::uint8_t>(
                    parsed.status);
            continue;
        }
        std::unique_ptr<RDKit::RWMol> molecule = std::move(parsed.molecule);

        // Embedding is the dominant CPU cost here, and it is wasted on a molecule the force field
        // cannot type: `constructForceField` throws below and every conformer fails anyway. Ask
        // first, so an untypeable molecule costs a typing check instead of K embeddings.
        if (!RDKit::MMFF::MMFFMolProperties(*molecule).isValid()) {
            for (std::uint32_t conformer_index = 0; conformer_index < num_conformers; ++conformer_index)
                outputs.status[molecule_index * num_conformers + conformer_index] = static_cast<std::uint8_t>(
                    sfapi::conformer_status_t::unsupported_element_k);
            continue;
        }

        RDKit::DGeomHelpers::EmbedParameters embed_params = RDKit::DGeomHelpers::ETKDGv3;
        embed_params.randomSeed = static_cast<int>(inputs.random_seed);
        embed_params.maxIterations = 200;

        // Only embed as many fresh conformers as are still needed. EmbedMultipleConfs is the dominant CPU
        // cost in this path; halving the count halves the per-molecule fallback wall when failures cluster
        // in one of K, the common case.
        std::vector<int> conformer_ids;
        try {
            conformer_ids = RDKit::DGeomHelpers::EmbedMultipleConfs(*molecule, static_cast<unsigned int>(needed_slots),
                                                                    embed_params);
        }
        catch (...) { /* ignore */
        }

        if (conformer_ids.empty()) continue;

        bool recovered_any = false;
        std::uint32_t fresh_index = 0;
        for (std::uint32_t conformer_slot = 0; conformer_slot < num_conformers; ++conformer_slot) {
            std::uint8_t const slot_status = outputs.status[molecule_index * num_conformers + conformer_slot];
            if (slot_status == static_cast<std::uint8_t>(sfapi::conformer_status_t::success_k)) continue;
            if (fresh_index >= conformer_ids.size()) break;
            int const rdkit_conformer_id = conformer_ids[fresh_index++];

            // MMFF optimise this conformer in place.
            float energy_kcal_per_mol = 0.0f;
            try {
                auto force_field = std::unique_ptr<ForceFields::ForceField>(
                    RDKit::MMFF::constructForceField(*molecule, /*nonBondedThresh=*/100.0, rdkit_conformer_id));
                if (!force_field) continue;
                force_field->initialize();
                force_field->minimize(200, 1e-4, 1e-6);
                energy_kcal_per_mol = static_cast<float>(force_field->calcEnergy());
            }
            catch (...) {
                continue;
            }

            auto const &conformer = molecule->getConformer(rdkit_conformer_id);
            std::size_t const coord_base =
                (static_cast<std::size_t>(molecule_index) * num_conformers + conformer_slot) * coordinate_stride;
            // Zero the trailing atoms (pad region).
            std::memset(reinterpret_cast<void *>(outputs.coordinates.data() + coord_base), 0,
                        sizeof(__half) * coordinate_stride);

            // Slot layout is shared with the GPU producer — see `conformer_slot_layout_t`.
            for (unsigned int atom_index = 0; atom_index < molecule->getNumAtoms(); ++atom_index) {
                auto const &position = conformer.getAtomPos(atom_index);
                outputs.coordinates[coord_base + slot_layout.component_offset(0u, atom_index)] = __float2half(
                    static_cast<float>(position.x));
                outputs.coordinates[coord_base + slot_layout.component_offset(1u, atom_index)] = __float2half(
                    static_cast<float>(position.y));
                outputs.coordinates[coord_base + slot_layout.component_offset(2u, atom_index)] = __float2half(
                    static_cast<float>(position.z));
            }

            outputs.energies[molecule_index * num_conformers + conformer_slot] = energy_kcal_per_mol;
            outputs.status[molecule_index * num_conformers + conformer_slot] = static_cast<std::uint8_t>(
                sfapi::conformer_status_t::success_k);
            outputs.solver[molecule_index * num_conformers + conformer_slot] = 1;
            recovered_any = true;
        }
        // Only a molecule that actually produced a conformer gets an atom count. Writing it before
        // the embed left unrecovered molecules carrying a non-zero count over all-zero coordinates,
        // which defeats the `num_atoms == 0` guard downstream and buys them a full shape-context
        // parse plus four substructure searches for a row that has no geometry.
        if (recovered_any) {
            outputs.num_atoms_per_molecule[molecule_index] = static_cast<std::uint8_t>(molecule->getNumAtoms());
            num_recovered.fetch_add(1, std::memory_order_relaxed);
        }
    }

    return num_recovered.load();
}

} // namespace smallfield::cli
