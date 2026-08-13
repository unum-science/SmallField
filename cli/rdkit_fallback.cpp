/**
 *  @file cli/rdkit_fallback.cpp
 *  @brief Runs RDKit ETKDG embedding and MMFF minimization, in an OpenMP-parallel loop, over the
 *  conformer slots the GPU pipeline could not fill.
 */

#include <cstddef> // `std::size_t`
#include <cstring> // `std::memset`

#include <chrono>   // `std::chrono::steady_clock`
#include <memory>   // `std::unique_ptr`
#include <optional> // `std::optional`
#include <sstream>  // RDKit's `AtomTyper.h` uses `std::stringstream` without including it
#include <vector>   // `std::vector`

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
                // Constructing the properties is the warm-up: it loads RDKit's lazy MMFF parameter
                // singletons before the parallel region opens on them.
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

/**
 *  @brief Folds one stage's host time into a per-molecule tally when it leaves scope.
 *
 *  The tally is a worker thread's own local, so nothing here is atomic; `rdkit_fallback` folds the finished
 *  tally into the shared report once per molecule.
 */
class scoped_stage_span_t {
    std::uint64_t &nanoseconds_;
    std::chrono::steady_clock::time_point const start_;

  public:
    scoped_stage_span_t(fallback_stage_nanoseconds_t &tally, fallback_stage_t stage) noexcept
        : nanoseconds_(nanoseconds_of(tally, stage)), start_(std::chrono::steady_clock::now()) {}
    ~scoped_stage_span_t() noexcept {
        auto const elapsed = std::chrono::steady_clock::now() - start_;
        nanoseconds_ += static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count());
    }

    scoped_stage_span_t(scoped_stage_span_t const &) = delete;
    scoped_stage_span_t &operator=(scoped_stage_span_t const &) = delete;
};

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
 *  @brief A molecule the fallback admitted, together with the MMFF typing it was admitted by.
 *
 *  The two travel together because they are one fact: MMFF typing is a function of the graph alone, so the
 *  typing that decided admission is the typing every conformer of this molecule minimizes under. Splitting
 *  them is what makes a second typing pass look free.
 */
struct admitted_molecule_t {
    std::unique_ptr<RDKit::RWMol> molecule;
    /** Engaged on exactly the paths `molecule` is non-null, so `constructForceField` never re-derives it. */
    std::optional<RDKit::MMFF::MMFFMolProperties> mmff_properties;
};

/**
 *  @brief Parse one SMILES the way the GPU path does and confirm MMFF can type the result.
 *
 *  Typing is checked up front because `constructForceField` would throw below and every conformer would
 *  fail anyway, so a molecule MMFF cannot type costs one typing check instead of K embeddings.
 *
 *  @param[out] admitted Receives the molecule and its typing on `success_k`, untouched on every refusal.
 *  @param[inout] tally Receives this molecule's parse and typing time.
 *  @return `success_k`, or the status every slot of this molecule takes.
 */
static sfapi::conformer_status_t admit_molecule_(std::string_view smiles, std::uint32_t max_atoms,
                                                 admitted_molecule_t &admitted, fallback_stage_nanoseconds_t &tally) {
    if (smiles.empty()) return sfapi::conformer_status_t::invalid_smiles_k;

    // Same parse contract as the GPU path, counterion removal included, so a molecule recovered here
    // carries the atom count and geometry the rest of the pipeline expects.
    smallfield::parse_result_t parsed;
    {
        scoped_stage_span_t const timed {tally, fallback_stage_t::parse_k};
        smallfield::parse_options_t parse_options;
        parse_options.max_atoms = max_atoms;
        parsed = smallfield::parse_smiles(smiles, parse_options);
    }
    if (parsed.status != sfapi::conformer_status_t::success_k) return parsed.status;

    {
        // Constructing the typing rewrites the molecule moved out below: it kekulizes and re-marks
        // aromaticity by MMFF's own pi-electron count, so the molecule the caller receives is the one MMFF
        // typed rather than the one the parser produced.
        //
        // The `try` is what keeps that rewrite from ending the process: the constructor kekulizes, and a
        // ring RDKit's aromaticity model marks but cannot kekulize — a silacyclopropene the SMILES writes
        // aromatic, say — leaves by `KekulizeException` rather than by `isValid`. Escaping here would
        // escape the `omp parallel for` below, and an exception leaving a parallel region is a
        // `std::terminate` that takes the whole run's unwritten shards with it.
        scoped_stage_span_t const timed {tally, fallback_stage_t::type_k};
        try {
            admitted.mmff_properties.emplace(*parsed.molecule);
        }
        catch (...) {
            admitted.mmff_properties.reset();
            return sfapi::conformer_status_t::unsupported_element_k;
        }
        if (!admitted.mmff_properties->isValid()) {
            admitted.mmff_properties.reset();
            return sfapi::conformer_status_t::unsupported_element_k;
        }
    }

    admitted.molecule = std::move(parsed.molecule);
    return sfapi::conformer_status_t::success_k;
}

/**
 *  @brief MMFF-minimize one embedded conformer in place and read back its energy.
 *
 *  @p mmff_properties is the typing admission already computed. The overload that omits it types the
 *  molecule again per conformer, and typing is a function of the graph rather than of the coordinates, so
 *  a molecule with K unfilled slots would pay K+1 aromaticity perceptions, ring builds and charge passes
 *  for one answer.
 *
 *  @param[out] energy_kcal_per_mol Minimized MMFF94 energy, untouched unless the field converged.
 *  @param[inout] tally Receives this conformer's field-construction and minimization time.
 */
static bool minimize_conformer_(RDKit::RWMol &molecule, RDKit::MMFF::MMFFMolProperties &mmff_properties,
                                int conformer_id, float &energy_kcal_per_mol, fallback_stage_nanoseconds_t &tally) {
    try {
        std::unique_ptr<ForceFields::ForceField> force_field;
        {
            scoped_stage_span_t const timed {tally, fallback_stage_t::build_k};
            force_field.reset(RDKit::MMFF::constructForceField(molecule, &mmff_properties,
                                                               /*nonBondedThresh=*/100.0, conformer_id));
        }
        if (!force_field) return false;
        scoped_stage_span_t const timed {tally, fallback_stage_t::minimize_k};
        force_field->initialize();
        force_field->minimize(200, 1e-4, 1e-6);
        energy_kcal_per_mol = static_cast<float>(force_field->calcEnergy());
        return true;
    }
    catch (...) {
        return false;
    }
}

/**
 *  @brief Write one RDKit conformer into its structure-of-arrays slot, zeroing the padding atoms first.
 *
 *  The three components are written out rather than gathered through `scatter_interleaved_into_slot`,
 *  because RDKit's `RDGeom::Point3D` derives from a polymorphic base and indexes by axis through a virtual
 *  `operator[]`, so reaching that entry point costs a staged interleaved copy per conformer inside the
 *  fallback's parallel region.
 */
static void write_conformer_into_slot_(RDKit::RWMol const &molecule, int conformer_id,
                                       sfapi::conformer_slot_layout_t const &layout, __half *slot) {
    std::memset(reinterpret_cast<void *>(slot), 0, sizeof(__half) * layout.slot_stride());
    std::uint32_t const num_atoms = molecule.getNumAtoms();
    auto const &conformer = molecule.getConformer(conformer_id);
    types::strided_matrix<__half> const destination = layout.slot_view(slot, num_atoms);
    for (std::uint32_t atom = 0; atom < num_atoms; ++atom) {
        auto const &position = conformer.getAtomPos(atom);
        destination(0u, atom) = __float2half(static_cast<float>(position.x));
        destination(1u, atom) = __float2half(static_cast<float>(position.y));
        destination(2u, atom) = __float2half(static_cast<float>(position.z));
    }
}

/**
 *  @brief Embed fresh conformers for one molecule and lay them into the slots the GPU left unfilled.
 *
 *  Only as many conformers are embedded as there are unfilled slots, since `EmbedMultipleConfs` is the
 *  dominant CPU cost here. Slots the GPU did fill keep their coordinates, energy and status untouched.
 *
 *  @param[inout] tally Receives this molecule's embedding, minimization and slot-writing time.
 *  @return Slots this molecule recovered.
 */
static std::uint32_t recover_molecule_(admitted_molecule_t &admitted, std::uint64_t random_seed,
                                       sfapi::conformer_slot_layout_t const &layout,
                                       custd::span<sfapi::conformer_status_t> status, custd::span<float> energies,
                                       custd::span<__half> coordinates, fallback_stage_nanoseconds_t &tally) {
    RDKit::RWMol &molecule = *admitted.molecule;
    RDKit::DGeomHelpers::EmbedParameters embed_params = RDKit::DGeomHelpers::ETKDGv3;
    embed_params.randomSeed = static_cast<int>(random_seed);
    embed_params.maxIterations = 200;

    std::vector<int> conformer_ids;
    {
        scoped_stage_span_t const timed {tally, fallback_stage_t::embed_k};
        try {
            conformer_ids = RDKit::DGeomHelpers::EmbedMultipleConfs(
                molecule, static_cast<unsigned int>(unfilled_slots_of_(status)), embed_params);
        }
        catch (...) { /* an embedding RDKit refused leaves every slot as the GPU left it */
        }
    }

    std::uint32_t recovered = 0;
    std::size_t fresh_index = 0;
    for (std::uint32_t conformer_slot = 0; conformer_slot < layout.num_conformers; ++conformer_slot) {
        if (status[conformer_slot] == sfapi::conformer_status_t::success_k) continue; // the GPU filled this one
        if (fresh_index >= conformer_ids.size()) break;                               // nothing left to lay in
        int const conformer_id = conformer_ids[fresh_index++];

        float energy_kcal_per_mol = 0.0f;
        if (!minimize_conformer_(molecule, *admitted.mmff_properties, conformer_id, energy_kcal_per_mol, tally))
            continue;

        {
            scoped_stage_span_t const timed {tally, fallback_stage_t::store_k};
            write_conformer_into_slot_(molecule, conformer_id, layout,
                                       coordinates.data() + conformer_slot * layout.slot_stride());
        }
        energies[conformer_slot] = energy_kcal_per_mol;
        status[conformer_slot] = sfapi::conformer_status_t::success_k;
        ++recovered;
    }
    return recovered;
}

/**
 *  @brief One dispatched molecule's whole attempt: admit it, embed into its unfilled slots, minimize each.
 *
 *  @param[inout] tally Receives every stage this molecule spent host time in.
 *  @return Slots recovered, zero both when admission refused the molecule and when nothing embedded.
 */
static std::uint32_t attempt_molecule_(rdkit_fallback_inputs_t const &inputs, rdkit_fallback_outputs_t &outputs,
                                       std::uint32_t molecule_index, custd::span<sfapi::conformer_status_t> status,
                                       fallback_stage_nanoseconds_t &tally) {
    sfapi::conformer_slot_layout_t const &layout = inputs.layout;

    admitted_molecule_t admitted;
    if (sfapi::conformer_status_t const admission = admit_molecule_(inputs.smiles[molecule_index], layout.max_atoms,
                                                                    admitted, tally);
        admission != sfapi::conformer_status_t::success_k) {
        // Only the unfilled slots take the refusal, so a slot the GPU already filled keeps its geometry.
        for (sfapi::conformer_status_t &slot_status : status)
            if (slot_status != sfapi::conformer_status_t::success_k) slot_status = admission;
        return 0;
    }

    std::uint32_t const recovered = recover_molecule_(
        admitted, inputs.random_seed, layout, status, slots_of_(outputs.energies, layout, molecule_index),
        outputs.coordinates.subspan(layout.slot_offset(molecule_index, 0),
                                    layout.num_conformers * layout.slot_stride()),
        tally);
    if (recovered == 0) return 0;

    // Only a molecule that produced a conformer gets an atom count: a non-zero count over all-zero
    // coordinates defeats the `num_atoms == 0` guard the descriptor pass downstream skips on.
    outputs.num_atoms_per_molecule[molecule_index] = static_cast<std::uint8_t>(admitted.molecule->getNumAtoms());
    return recovered;
}

/** Fold one finished attempt's per-stage tally into the batch-wide report. */
static void fold_molecule_cost_into_(fallback_stage_nanoseconds_t const &tally, std::uint32_t molecule_index,
                                     fallback_report_t &report) {
    std::uint64_t attempt_nanoseconds = 0;
    for (std::size_t stage = 0; stage < fallback_stage_count_k; ++stage) {
        attempt_nanoseconds += tally[stage];
#pragma omp atomic
        report.nanoseconds[stage] += tally[stage];
    }
#pragma omp atomic
    ++report.molecules_by_duration[fallback_duration_bin(attempt_nanoseconds)];
#pragma omp critical(fallback_slowest)
    {
        if (attempt_nanoseconds > report.slowest_nanoseconds) {
            report.slowest_nanoseconds = attempt_nanoseconds;
            report.slowest_molecule_index = molecule_index;
        }
    }
}

#pragma endregion One Molecule

void rdkit_fallback(rdkit_fallback_inputs_t const &inputs, rdkit_fallback_outputs_t &outputs,
                    fallback_report_t &report) {
    report.attempted = {};
    report.recovered = {};
    report.nanoseconds = {};
    report.molecules_by_duration = {};
    report.slowest_nanoseconds = 0;
    report.slowest_molecule_index = 0;
    sfapi::conformer_slot_layout_t const &layout = inputs.layout;
    std::uint64_t const num_retries = inputs.retry_indices.size();
    if (num_retries == 0) return;

    int const num_threads = inputs.num_threads > 0 ? inputs.num_threads : omp_get_max_threads();
    warm_up_rdkit_globals_for_fallback_();

#pragma omp parallel for schedule(dynamic, 16) num_threads(num_threads) if (num_threads > 1)
    for (std::uint64_t retry_entry = 0; retry_entry < num_retries; ++retry_entry) {
        std::uint32_t const molecule_index = inputs.retry_indices[retry_entry];
        custd::span<sfapi::conformer_status_t> const status = slots_of_(outputs.status, layout, molecule_index);
        // Read before anything below overwrites it: this is the status the recovery is attributed to.
        sfapi::conformer_status_t const cause = fallback_dispatch_cause(status);
        if (cause == sfapi::conformer_status_t::success_k) continue; // no slot this role can fill

        fallback_stage_nanoseconds_t tally {};
        std::uint32_t const recovered = attempt_molecule_(inputs, outputs, molecule_index, status, tally);
        fold_molecule_cost_into_(tally, molecule_index, report);

        std::size_t const cause_index = static_cast<std::size_t>(cause);
#pragma omp atomic
        ++report.attempted[cause_index];
        if (recovered == 0) continue;
#pragma omp atomic
        ++report.recovered[cause_index];
    }
}

} // namespace smallfield::cli
