#include "rdkit_fallback.hpp"

#include <atomic>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

#include <omp.h>

#include <GraphMol/DistGeomHelpers/Embedder.h>
#include <GraphMol/ForceFieldHelpers/MMFF/MMFF.h>
#include <GraphMol/GraphMol.h>
#include <GraphMol/MolOps.h>
#include <GraphMol/RWMol.h>
#include <GraphMol/SmilesParse/SmilesParse.h>

#include "api/smallfield.hpp"

namespace smallfield::batch {

std::uint64_t rdkit_fallback(rdkit_fallback_inputs_t const &inputs,
                             rdkit_fallback_outputs_t &outputs) {
    std::uint32_t const num_conformers = inputs.num_conformers;
    std::uint32_t const max_atoms = inputs.max_atoms;
    std::uint32_t const coordinate_stride = max_atoms * 3;
    std::uint64_t const num_retries = inputs.retry_indices.size();

    if (num_retries == 0) return 0;

    std::atomic<std::uint64_t> num_recovered {0};

    int const num_threads = inputs.num_threads > 0 ? inputs.num_threads : 1;

#pragma omp parallel for schedule(dynamic, 16) num_threads(num_threads) if (num_threads > 1)
    for (std::uint64_t retry_entry = 0; retry_entry < num_retries; ++retry_entry) {
        std::uint32_t const molecule_index = inputs.retry_indices[retry_entry];
        std::string const &smiles = inputs.smiles[molecule_index];

        // Smallest mutation that keeps `solver=2` on failures below: default everything
        // to the "failed" sentinel and flip only on success.
        for (std::uint32_t conformer_index = 0; conformer_index < num_conformers; ++conformer_index) {
            outputs.status[molecule_index * num_conformers + conformer_index]
                = static_cast<std::uint8_t>(conformer_status_t::mmff_unconverged);
            outputs.solver[molecule_index * num_conformers + conformer_index] = 2;
        }

        if (smiles.empty()) {
            for (std::uint32_t conformer_index = 0; conformer_index < num_conformers; ++conformer_index)
                outputs.status[molecule_index * num_conformers + conformer_index]
                    = static_cast<std::uint8_t>(conformer_status_t::invalid_smiles);
            continue;
        }

        std::unique_ptr<RDKit::RWMol> mol;
        try {
            RDKit::SmilesParserParams params;
            params.sanitize = true;
            params.removeHs = false;
            params.strictCXSMILES = false;
            mol.reset(RDKit::SmilesToMol(smiles, params));
        } catch (...) { /* ignore */ }

        if (!mol) {
            for (std::uint32_t conformer_index = 0; conformer_index < num_conformers; ++conformer_index)
                outputs.status[molecule_index * num_conformers + conformer_index]
                    = static_cast<std::uint8_t>(conformer_status_t::invalid_smiles);
            continue;
        }

        try { RDKit::MolOps::addHs(*mol); } catch (...) { mol.reset(); }
        if (!mol) continue;
        if (mol->getNumAtoms() > max_atoms) {
            for (std::uint32_t conformer_index = 0; conformer_index < num_conformers; ++conformer_index)
                outputs.status[molecule_index * num_conformers + conformer_index]
                    = static_cast<std::uint8_t>(conformer_status_t::too_many_atoms);
            continue;
        }
        outputs.num_atoms_per_molecule[molecule_index] = static_cast<std::uint8_t>(mol->getNumAtoms());

        RDKit::DGeomHelpers::EmbedParameters embed_params = RDKit::DGeomHelpers::ETKDGv3;
        embed_params.randomSeed = static_cast<int>(inputs.random_seed);
        embed_params.maxIterations = 200;

        std::vector<int> conformer_ids;
        try {
            conformer_ids = RDKit::DGeomHelpers::EmbedMultipleConfs(
                *mol, static_cast<unsigned int>(num_conformers), embed_params);
        } catch (...) { /* ignore */ }

        if (conformer_ids.empty()) continue;

        bool recovered_any = false;
        for (std::uint32_t conformer_slot = 0; conformer_slot < num_conformers; ++conformer_slot) {
            if (conformer_slot >= conformer_ids.size()) break;
            int const rdkit_conformer_id = conformer_ids[conformer_slot];

            // MMFF optimise this conformer in place.
            float energy_kcal_per_mol = 0.0f;
            try {
                auto force_field = std::unique_ptr<ForceFields::ForceField>(
                    RDKit::MMFF::constructForceField(*mol, /*nonBondedThresh=*/100.0,
                                                      rdkit_conformer_id));
                if (!force_field) continue;
                force_field->initialize();
                force_field->minimize(200, 1e-4, 1e-6);
                energy_kcal_per_mol = static_cast<float>(force_field->calcEnergy());
            } catch (...) {
                continue;
            }

            auto const &conformer = mol->getConformer(rdkit_conformer_id);
            std::size_t const coord_base = (static_cast<std::size_t>(molecule_index) * num_conformers
                                             + conformer_slot) * coordinate_stride;
            // Zero the trailing atoms (pad region).
            std::memset(reinterpret_cast<void *>(outputs.coordinates.data() + coord_base),
                        0, sizeof(__half) * coordinate_stride);

            for (unsigned int atom_index = 0; atom_index < mol->getNumAtoms(); ++atom_index) {
                auto const &position = conformer.getAtomPos(atom_index);
                outputs.coordinates[coord_base + atom_index * 3 + 0] = __float2half(static_cast<float>(position.x));
                outputs.coordinates[coord_base + atom_index * 3 + 1] = __float2half(static_cast<float>(position.y));
                outputs.coordinates[coord_base + atom_index * 3 + 2] = __float2half(static_cast<float>(position.z));
            }

            outputs.energies[molecule_index * num_conformers + conformer_slot] = energy_kcal_per_mol;
            outputs.status[molecule_index * num_conformers + conformer_slot]
                = static_cast<std::uint8_t>(conformer_status_t::success);
            outputs.solver[molecule_index * num_conformers + conformer_slot] = 1;
            recovered_any = true;
        }
        if (recovered_any) num_recovered.fetch_add(1, std::memory_order_relaxed);
    }

    return num_recovered.load();
}

} // namespace smallfield::batch
