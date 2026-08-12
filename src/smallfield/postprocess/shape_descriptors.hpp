/**
 *  @file src/smallfield/postprocess/shape_descriptors.hpp
 *  @brief Native USR / USRCAT molecular shape descriptors.
 *
 *  Implements the 12-number Ultrafast Shape Recognition descriptor, USR, and its 60-number CREDO-pharmacophore
 *  extension, USRCAT, so the conformer pipeline can emit shape fingerprints without a round trip through RDKit
 *  per conformer.
 *
 *  The numeric definitions match `RDKit::Descriptors::USR` and `USRCAT` bit for bit, as implemented in RDKit's
 *  `USRDescriptor.cpp`. There are four reference points — the centroid, the atom closest to the centroid, the
 *  atom farthest from the centroid, and the atom farthest from that one — and for each reference point three
 *  moments of the distance distribution: mean, standard deviation, and the signed cube root of the standardized
 *  third central moment, the skewness. The descriptor is reference-point major.
 *
 *  RDKit's `GetUSRCAT` re-types the four CREDO pharmacophore subsets on every call. A molecule with `k`
 *  conformers therefore pays for `k` substructure searches. Here `build_shape_context` types the subsets and
 *  labels connected components exactly once; `usrcat_descriptor` then consumes that context per conformer.
 *
 *  This header also owns `molecule_shape_context_t` and the coordinate primitives, because
 *  `build_shape_context` is the only producer of the connected-component labels that
 *  `postprocess/fragment_separation.hpp` consumes.
 *
 *  @code{.cpp}
 *  auto context = smallfield::postprocess::build_shape_context("CC(=O)Oc1ccccc1C(=O)O");
 *  std::array<double, 60> fingerprint;
 *  smallfield::postprocess::usrcat_descriptor(coordinates, context, fingerprint);
 *  @endcode
 */

#pragma once

#include <cmath>   // `std::sqrt`, `std::cbrt`
#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint32_t`

#include <array>       // `std::array`
#include <memory>      // `std::unique_ptr`
#include <numeric>     // `std::iota`
#include <string_view> // `std::string_view`
#include <vector>      // `std::vector`

#include <GraphMol/GraphMol.h>
#include <GraphMol/MolOps.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <GraphMol/Substruct/SubstructMatch.h>

#include "smallfield/molecule_parse.hpp"
#include "smallfield/types/views.hpp"

namespace smallfield::postprocess {

#pragma region Extents

/** Reference points a USR block measures against: centroid, nearest to it, farthest from it, farthest from that. */
inline constexpr std::size_t usr_reference_point_count_k = 4;

/** Moments taken of each reference point's distance distribution: mean, standard deviation, skewness. */
inline constexpr std::size_t usr_moment_count_k = 3;

/** Numbers one USR block carries, reference-point major. */
inline constexpr std::size_t usr_components_k = usr_reference_point_count_k * usr_moment_count_k;

/** CREDO pharmacophore subsets USRCAT adds, in RDKit's order: hydrophobic, aromatic, acceptor, donor. */
inline constexpr std::size_t credo_subset_count_k = 4;

/** Numbers USRCAT carries: the all-atom block, then one block per CREDO subset. */
inline constexpr std::size_t usrcat_components_k = usr_components_k * (credo_subset_count_k + 1);

/** Connected components a context can label, set by the width of `component_label`. */
inline constexpr std::uint32_t max_components_k = 255;

#pragma endregion Extents

#pragma region Shape Context

/**
 *  @brief Per-molecule shape typing computed once and reused across all conformers.
 *
 *  The four CREDO subsets follow RDKit's USRCAT order: hydrophobic, aromatic, acceptor, donor. Each holds the
 *  atom indices that the corresponding SMARTS pattern matched, indexing into the hydrogen-augmented molecule and
 *  matching stored coordinate order. Duplicate matches are preserved exactly as RDKit collects them so subset
 *  distance distributions are identical.
 *
 *  `ok` is the only success signal: a refused molecule carries an otherwise default-constructed context. It is
 *  not derivable from `num_atoms`, because an empty SMILES parses into a real molecule of no atoms.
 */
struct molecule_shape_context_t {
    std::uint32_t num_atoms = 0;
    /** Hydrophobic, aromatic, acceptor, donor. */
    std::array<std::vector<std::uint32_t>, credo_subset_count_k> credo_subsets;
    /** Per-atom connected-component id, so `num_components` never exceeds `max_components_k`. */
    std::vector<std::uint8_t> component_label;
    std::uint32_t num_components = 0;
    bool ok = false;
};

#pragma endregion Shape Context

#pragma region Coordinate Primitives

/** A double-precision 3D point read out of the caller's float coordinate buffer. */
struct point3_t {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

/** Reads atom @p atom_index from a tight interleaved (num_atoms, 3) float buffer. */
inline point3_t read_point_(custd::span<float const> coords_interleaved, std::uint32_t atom_index) {
    std::size_t const base = static_cast<std::size_t>(atom_index) * 3u;
    return point3_t {static_cast<double>(coords_interleaved[base + 0u]),
                     static_cast<double>(coords_interleaved[base + 1u]),
                     static_cast<double>(coords_interleaved[base + 2u])};
}

/** Euclidean distance between two points. */
inline double point_distance_(point3_t const &first, point3_t const &second) {
    double const dx = first.x - second.x;
    double const dy = first.y - second.y;
    double const dz = first.z - second.z;
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

#pragma endregion Coordinate Primitives

#pragma region Internal

/** Centroid of @p num_atoms atoms; the caller guarantees at least one. */
inline point3_t centroid_of_(custd::span<float const> coords_interleaved, std::uint32_t num_atoms) {
    point3_t centroid {};
    for (std::uint32_t atom_index = 0; atom_index < num_atoms; ++atom_index) {
        point3_t const point = read_point_(coords_interleaved, atom_index);
        centroid.x += point.x;
        centroid.y += point.y;
        centroid.z += point.z;
    }
    centroid.x /= num_atoms;
    centroid.y /= num_atoms;
    centroid.z /= num_atoms;
    return centroid;
}

/** Atom nearest @p reference, first index winning a tie as RDKit's `smallestValId` does; at least one atom. */
inline std::uint32_t nearest_atom_to_(custd::span<float const> coords_interleaved, std::uint32_t num_atoms,
                                      point3_t const &reference) {
    std::uint32_t nearest = 0;
    double nearest_distance = point_distance_(read_point_(coords_interleaved, 0), reference);
    for (std::uint32_t atom_index = 1; atom_index < num_atoms; ++atom_index) {
        double const distance = point_distance_(read_point_(coords_interleaved, atom_index), reference);
        if (distance >= nearest_distance) continue;
        nearest_distance = distance;
        nearest = atom_index;
    }
    return nearest;
}

/** Atom farthest from @p reference, first index winning a tie as RDKit's `largestValId` does; at least one atom. */
inline std::uint32_t farthest_atom_from_(custd::span<float const> coords_interleaved, std::uint32_t num_atoms,
                                         point3_t const &reference) {
    std::uint32_t farthest = 0;
    double farthest_distance = point_distance_(read_point_(coords_interleaved, 0), reference);
    for (std::uint32_t atom_index = 1; atom_index < num_atoms; ++atom_index) {
        double const distance = point_distance_(read_point_(coords_interleaved, atom_index), reference);
        if (distance <= farthest_distance) continue;
        farthest_distance = distance;
        farthest = atom_index;
    }
    return farthest;
}

/**
 *  @brief The four USR reference points, in RDKit order: ctd, cst, fct, ftf.
 *
 *  A molecule of no atoms has no reference points and yields four origins, which the moment passes then turn into
 *  zeros; reading an atom out of an empty coordinate span is the alternative.
 */
inline std::array<point3_t, usr_reference_point_count_k> reference_points_(custd::span<float const> coords_interleaved,
                                                                           std::uint32_t num_atoms) {
    std::array<point3_t, usr_reference_point_count_k> points {};
    if (num_atoms == 0) return points;
    points[0] = centroid_of_(coords_interleaved, num_atoms);
    points[1] = read_point_(coords_interleaved, nearest_atom_to_(coords_interleaved, num_atoms, points[0]));
    points[2] = read_point_(coords_interleaved, farthest_atom_from_(coords_interleaved, num_atoms, points[0]));
    points[3] = read_point_(coords_interleaved, farthest_atom_from_(coords_interleaved, num_atoms, points[2]));
    return points;
}

/**
 *  @brief Three USR moments of the distances from @p atom_indices to @p reference, matching RDKit's `calcMoments`.
 *
 *  Order: [0] mean, [1] standard deviation — the square root of the mean squared deviation from the mean, [2]
 *  signed cube root of the standardized third central moment. An empty subset yields three zeros; a degenerate
 *  one with zero standard deviation yields a zero third moment, RDKit's guard against dividing by zero.
 *
 *  The distances are walked twice rather than materialized, so a per-conformer descriptor allocates nothing.
 */
inline std::array<double, usr_moment_count_k> moments_about_(custd::span<float const> coords_interleaved,
                                                             custd::span<std::uint32_t const> atom_indices,
                                                             point3_t const &reference) {
    std::array<double, usr_moment_count_k> moments {0.0, 0.0, 0.0};
    if (atom_indices.empty()) return moments;
    double const num_points = static_cast<double>(atom_indices.size());

    double sum = 0.0;
    for (std::uint32_t const atom_index : atom_indices)
        sum += point_distance_(read_point_(coords_interleaved, atom_index), reference);
    double const mean = sum / num_points;
    moments[0] = mean;

    double sum_squared = 0.0;
    double sum_cubed = 0.0;
    for (std::uint32_t const atom_index : atom_indices) {
        double const diff = point_distance_(read_point_(coords_interleaved, atom_index), reference) - mean;
        sum_squared += diff * diff;
        sum_cubed += diff * diff * diff;
    }
    double const standard_deviation = std::sqrt(sum_squared / num_points);
    moments[1] = standard_deviation;
    if (standard_deviation == 0.0) return moments;

    double const third_central = sum_cubed / num_points;
    moments[2] = std::cbrt(third_central / (standard_deviation * standard_deviation * standard_deviation));
    return moments;
}

/**
 *  @brief Writes one atom subset's block of moments, reference-point major.
 *
 *  @param[in] coords_interleaved Tight (num_atoms, 3) row-major xyz for one conformer.
 *  @param[in] points The four all-atom reference points every block measures against.
 *  @param[in] atom_indices Atoms whose distances feed the distributions — every atom for the plain USR block, a
 *             CREDO subset for a USRCAT extension block. An empty subset writes zeros, as RDKit's empty
 *             `reducedCoords` path does.
 *  @param[out] block Exactly `usr_components_k` numbers of the caller's descriptor row.
 */
inline void moments_for_subset_(custd::span<float const> coords_interleaved,
                                std::array<point3_t, usr_reference_point_count_k> const &points,
                                custd::span<std::uint32_t const> atom_indices, custd::span<double> block) {
    sf_assert_(block.size() == usr_components_k);
    for (std::size_t reference = 0; reference < usr_reference_point_count_k; ++reference) {
        std::array<double, usr_moment_count_k> const moments = moments_about_(coords_interleaved, atom_indices,
                                                                              points[reference]);
        for (std::size_t moment = 0; moment < usr_moment_count_k; ++moment)
            block[reference * usr_moment_count_k + moment] = moments[moment];
    }
}

/**
 *  @brief The four CREDO pharmacophore SMARTS matchers, compiled once.
 *
 *  Order matches RDKit's USRCAT default typing: hydrophobic, aromatic, acceptor, donor. Parsed on first use into
 *  a function-local static so the SMARTS are never re-parsed per molecule.
 */
inline std::array<RDKit::ROMol, credo_subset_count_k> const &credo_matchers_() {
    static char const *const patterns[credo_subset_count_k] = {
        "[#6+0!$(*~[#7,#8,F]),SH0+0v2,s+0,S^3,Cl+0,Br+0,I+0]",             //  hydrophobic
        "[a]",                                                             //  aromatic
        "[$([O,S;H1;v2]-[!$(*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),"     //  acceptor
        "$([N&v3;H1,H2]-[!$(*=[O,N,P,S])]),$([N;v3;H0]),$([n,o,s;+0]),F]", //
        "[N!H0v3,N!H0+v4,OH+0,SH+0,nH+0]",                                 //  donor
    };
    // `RDKit::ROMol` has a copy constructor but a deleted assignment operator, so the array is built by
    // initialization rather than filled in a loop.
    static auto const parse_one = [](char const *smarts) {
        std::unique_ptr<RDKit::ROMol const> const parsed {RDKit::SmartsToMol(smarts)};
        return RDKit::ROMol(*parsed);
    };
    static std::array<RDKit::ROMol, credo_subset_count_k> const compiled {
        parse_one(patterns[0]), parse_one(patterns[1]), parse_one(patterns[2]), parse_one(patterns[3])};
    return compiled;
}

#pragma endregion Internal

#pragma region Public API

/**
 *  @brief Parses @p smiles and types it for USRCAT, once per molecule.
 *
 *  Uses the same parse contract as the rest of the library — sanitize on, keep explicit hydrogens, tolerant
 *  CXSMILES — then `MolOps::addHs` so the atom order matches the coordinates the conformer generator stores. The
 *  four CREDO subsets are matched with RDKit's own default USRCAT SMARTS, and connected components are labeled
 *  with `MolOps::getMolFrags`.
 *
 *  @return A populated context with `ok == true` on success; `ok == false`, with empty fields, if parsing or
 *          typing threw, or if the molecule falls into more fragments than `component_label` can name.
 */
inline molecule_shape_context_t build_shape_context(
    std::string_view smiles, smallfield::salt_policy_t salt_policy = smallfield::keep_largest_fragment_k) {
    molecule_shape_context_t context;
    try {
        // Same contract as the preprocess path, counterion removal included: this atom count is compared against
        // the one the pipeline recorded, so the two parses have to agree. The cap is left off because every
        // molecule reaching here already passed the pipeline's own gate.
        smallfield::parse_options_t parse_options;
        parse_options.max_atoms = 0;
        parse_options.salt_policy = salt_policy;
        smallfield::parse_result_t parsed = smallfield::parse_smiles(smiles, parse_options);
        if (!parsed.molecule) return context;
        std::unique_ptr<RDKit::RWMol> molecule = std::move(parsed.molecule);

        context.num_atoms = parsed.num_atoms;

        // Type the four CREDO subsets exactly as RDKit's `getAtomIdsForFeatures` does: collect every matched
        // atom index, preserving RDKit's collection order.
        std::array<RDKit::ROMol, credo_subset_count_k> const &matchers = credo_matchers_();
        for (std::size_t subset_index = 0; subset_index < credo_subset_count_k; ++subset_index) {
            std::vector<RDKit::MatchVectType> matches;
            RDKit::SubstructMatch(*molecule, matchers[subset_index], matches);
            for (auto const &match : matches)
                for (auto const &pair : match)
                    context.credo_subsets[subset_index].push_back(static_cast<std::uint32_t>(pair.second));
        }

        // Label connected components — disconnected fragments and salt counterions. Past
        // `max_components_k` a label does not fit its byte and would alias another component, so the
        // molecule is refused rather than mislabelled.
        std::vector<int> mapping;
        std::uint32_t const num_components = RDKit::MolOps::getMolFrags(*molecule, mapping);
        if (num_components > max_components_k) return molecule_shape_context_t {};
        context.num_components = num_components;
        context.component_label.resize(mapping.size());
        for (std::size_t atom_index = 0; atom_index < mapping.size(); ++atom_index)
            context.component_label[atom_index] = static_cast<std::uint8_t>(mapping[atom_index]);

        context.ok = true;
    }
    catch (...) {
        context = molecule_shape_context_t {};
    }
    return context;
}

/**
 *  @brief The 12-number USR shape descriptor for one conformer.
 *
 *  @param[in] coords_interleaved Tight (num_atoms, 3) row-major xyz for one conformer.
 *  @param[in] num_atoms Atom count; the span carries three floats per atom.
 *  @param[out] out `usr_components_k` doubles, reference-point major (ctd, cst, fct, ftf).
 */
inline void usr_descriptor(custd::span<float const> coords_interleaved, std::uint32_t num_atoms,
                           std::array<double, usr_components_k> &out) {
    sf_assert_(coords_interleaved.size() >= static_cast<std::size_t>(num_atoms) * 3u);
    std::array<point3_t, usr_reference_point_count_k> const points = reference_points_(coords_interleaved, num_atoms);
    std::vector<std::uint32_t> all_atoms(num_atoms);
    std::iota(all_atoms.begin(), all_atoms.end(), 0u);
    moments_for_subset_(coords_interleaved, points, all_atoms, custd::span<double> {out.data(), out.size()});
}

/**
 *  @brief The 60-number USRCAT shape descriptor for one conformer.
 *
 *  Block 0 is the plain USR over all atoms. Blocks 1 through 4 repeat the same moments against the same four
 *  all-atom reference points, restricting the distance distributions to each CREDO subset in order hydrophobic,
 *  aromatic, acceptor, donor. Empty subsets contribute a block of zeros, matching RDKit.
 *
 *  @param[in] coords_interleaved Tight (num_atoms, 3) row-major xyz for one conformer.
 *  @param[in] context Context from `build_shape_context` for this molecule.
 *  @param[out] out `usrcat_components_k` doubles, one `usr_components_k` block per subset.
 */
inline void usrcat_descriptor(custd::span<float const> coords_interleaved, molecule_shape_context_t const &context,
                              std::array<double, usrcat_components_k> &out) {
    std::uint32_t const num_atoms = context.num_atoms;
    sf_assert_(coords_interleaved.size() >= static_cast<std::size_t>(num_atoms) * 3u);
    std::array<point3_t, usr_reference_point_count_k> const points = reference_points_(coords_interleaved, num_atoms);
    custd::span<double> const row {out.data(), out.size()};

    std::vector<std::uint32_t> all_atoms(num_atoms);
    std::iota(all_atoms.begin(), all_atoms.end(), 0u);
    moments_for_subset_(coords_interleaved, points, all_atoms, row.subspan(0, usr_components_k));

    for (std::size_t subset_index = 0; subset_index < credo_subset_count_k; ++subset_index)
        moments_for_subset_(coords_interleaved, points, context.credo_subsets[subset_index],
                            row.subspan((subset_index + 1) * usr_components_k, usr_components_k));
}

#pragma endregion Public API

} // namespace smallfield::postprocess
