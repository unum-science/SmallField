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
 *  conformers therefore pays for `k` substructure searches. Here `build_shape_context_into` types the subsets
 *  and labels connected components exactly once; `usrcat_descriptor` then consumes that context per conformer.
 *
 *  This header also owns `molecule_shape_context_t` and the coordinate primitives, because
 *  `build_shape_context_into` is the only producer of the connected-component labels that
 *  `postprocess/fragment_separation.hpp` consumes.
 *
 *  @code{.cpp}
 *  namespace sfpost = smallfield::postprocess;
 *  sfpost::molecule_shape_context_t context;  // hoisted above the molecule loop, one per worker
 *  if (sfpost::build_shape_context_into(smiles, num_atoms, smallfield::keep_largest_fragment_k, context) ==
 *      sfpost::shape_typing_t::typed_k)
 *      sfpost::usrcat_descriptor(coordinates, context,
 *                                {rows.data() + slot * sfpost::usrcat_components_k, sfpost::usrcat_components_k});
 *  @endcode
 */

#pragma once

#include <cmath>   // `std::sqrt`, `std::cbrt`
#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint32_t`

#include <array>       // `std::array`
#include <memory>      // `std::unique_ptr`
#include <ranges>      // `std::views::iota`
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

/** Connected components a context describes; past it `separate_fragments` would string more bounding spheres
 *  along +x than one shape means anything for, so the molecule is refused instead. */
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
 *  `build_shape_context_into` is the only producer and its `shape_typing_t` verdict the only success signal, so
 *  the context carries no flag of its own. A refusal leaves every field empty, capacity aside.
 */
struct molecule_shape_context_t {
    /** Hydrophobic, aromatic, acceptor, donor. */
    std::array<std::vector<std::uint32_t>, credo_subset_count_k> credo_subsets;
    /** Per-atom connected-component id, in the width `MolOps::getMolFrags` writes, so that call labels the atoms
     *  in place and nothing narrows its output afterwards. */
    std::vector<int> component_label;
    std::uint32_t num_components = 0;

    /** Atoms the context describes. Read off the per-atom labeling rather than stored beside it, so an
     *  atom count that disagrees with the labels it indexes is not representable. */
    [[nodiscard]] std::uint32_t num_atoms() const noexcept {
        return static_cast<std::uint32_t>(component_label.size());
    }
};

/**
 *  @brief Whether a molecule's shape context can describe the conformers already stored for it.
 *
 *  A refusal is not a repairable error; it names which way the typing failed to reproduce the atom order the
 *  stored coordinates were written in, so a caller reports and skips rather than describing the wrong atoms.
 */
enum class shape_typing_t : std::uint8_t {
    /** Typed, and the reparse reproduced the atom count the caller's coordinates were written for. */
    typed_k,
    /** The SMILES did not parse under the library's contract, or typing threw on it. */
    unparsed_k,
    /** More fragments than `max_components_k`, which is more than one shape describes. */
    too_many_fragments_k,
    /** The reparse produced a different atom count, so the stored coordinate order cannot be trusted. */
    atom_count_divergence_k,
};

/** Sentence fragment naming the verdict, meant to follow the molecule's SMILES in a diagnostic. */
inline char const *describe_shape_typing(shape_typing_t typing) {
    switch (typing) {
    case shape_typing_t::typed_k: return "typed";
    case shape_typing_t::unparsed_k: return "does not parse, or threw while being typed";
    case shape_typing_t::too_many_fragments_k: return "falls into more fragments than one shape describes";
    case shape_typing_t::atom_count_divergence_k: return "reparses to a different atom count";
    }
    return "was refused";
}

#pragma endregion Shape Context

#pragma region Coordinate Primitives

/** A double-precision 3D point read out of the caller's float coordinate buffer. */
struct point3_t {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

/** One conformer's atom-major coordinates for a caller that only reads them. */
using conformer_reader_t = types::coordinates_aos<types::coordinate_role_t::position_k, float const>;

/*
 *  A conformer reaches the public entry points as a flat buffer beside an atom count, two facts nothing
 *  compares. These reconcile them at the boundary so everything below addresses `(atom, axis)`, and the second
 *  form is an exact match for a mutable span, so a writer never spells index arithmetic of its own either.
 */

/** Views a tight interleaved (num_atoms, 3) float buffer as one conformer. */
inline conformer_reader_t conformer_over_(custd::span<float const> coords_interleaved, std::uint32_t num_atoms) {
    sf_assert_(coords_interleaved.size() >= static_cast<std::size_t>(num_atoms) * types::conformer_dimensions_k);
    return conformer_reader_t {coords_interleaved.data(), num_atoms};
}

/** The same view for a caller that writes it. */
inline types::conformer_aos_t conformer_over_(custd::span<float> coords_interleaved, std::uint32_t num_atoms) {
    sf_assert_(coords_interleaved.size() >= static_cast<std::size_t>(num_atoms) * types::conformer_dimensions_k);
    return types::conformer_aos_t {coords_interleaved.data(), num_atoms};
}

/** Reads atom @p atom_index out of a conformer. */
inline point3_t read_point_(conformer_reader_t coords, std::uint32_t atom_index) {
    return point3_t {static_cast<double>(coords(atom_index, 0)), static_cast<double>(coords(atom_index, 1)),
                     static_cast<double>(coords(atom_index, 2))};
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

/** Centroid of a conformer's atoms; the caller guarantees at least one. */
inline point3_t centroid_of_(conformer_reader_t coords) {
    point3_t centroid {};
    std::uint32_t const num_atoms = coords.extent(0);
    for (std::uint32_t atom_index = 0; atom_index < num_atoms; ++atom_index) {
        point3_t const point = read_point_(coords, atom_index);
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
inline std::uint32_t nearest_atom_to_(conformer_reader_t coords, point3_t const &reference) {
    std::uint32_t nearest = 0;
    double nearest_distance = point_distance_(read_point_(coords, 0), reference);
    for (std::uint32_t atom_index = 1; atom_index < coords.extent(0); ++atom_index) {
        double const distance = point_distance_(read_point_(coords, atom_index), reference);
        if (distance >= nearest_distance) continue;
        nearest_distance = distance;
        nearest = atom_index;
    }
    return nearest;
}

/** Atom farthest from @p reference, first index winning a tie as RDKit's `largestValId` does; at least one atom. */
inline std::uint32_t farthest_atom_from_(conformer_reader_t coords, point3_t const &reference) {
    std::uint32_t farthest = 0;
    double farthest_distance = point_distance_(read_point_(coords, 0), reference);
    for (std::uint32_t atom_index = 1; atom_index < coords.extent(0); ++atom_index) {
        double const distance = point_distance_(read_point_(coords, atom_index), reference);
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
 *  zeros; reading an atom out of an empty coordinate view is the alternative.
 */
inline std::array<point3_t, usr_reference_point_count_k> reference_points_(conformer_reader_t coords) {
    std::array<point3_t, usr_reference_point_count_k> points {};
    if (coords.extent(0) == 0) return points;
    points[0] = centroid_of_(coords);
    points[1] = read_point_(coords, nearest_atom_to_(coords, points[0]));
    points[2] = read_point_(coords, farthest_atom_from_(coords, points[0]));
    points[3] = read_point_(coords, farthest_atom_from_(coords, points[2]));
    return points;
}

/**
 *  @brief Three USR moments of the distances from @p atom_indices to @p reference, matching RDKit's `calcMoments`.
 *
 *  Order: [0] mean, [1] standard deviation — the square root of the mean squared deviation from the mean, [2]
 *  signed cube root of the standardized third central moment. An empty subset yields three zeros; a degenerate
 *  one with zero standard deviation yields a zero third moment, RDKit's guard against dividing by zero.
 *
 *  The distances are walked twice rather than materialized and @p atom_indices is borrowed rather than built, so
 *  a per-conformer descriptor allocates nothing.
 */
template <typename atom_indices_type_>
inline std::array<double, usr_moment_count_k> moments_about_(conformer_reader_t coords,
                                                             atom_indices_type_ const &atom_indices,
                                                             point3_t const &reference) {
    std::array<double, usr_moment_count_k> moments {0.0, 0.0, 0.0};
    if (atom_indices.empty()) return moments;
    double const num_points = static_cast<double>(atom_indices.size());

    double sum = 0.0;
    for (std::uint32_t const atom_index : atom_indices)
        sum += point_distance_(read_point_(coords, atom_index), reference);
    double const mean = sum / num_points;
    moments[0] = mean;

    double sum_squared = 0.0;
    double sum_cubed = 0.0;
    for (std::uint32_t const atom_index : atom_indices) {
        double const diff = point_distance_(read_point_(coords, atom_index), reference) - mean;
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
 *  @param[in] coords One conformer's coordinates.
 *  @param[in] points The four all-atom reference points every block measures against.
 *  @param[in] atom_indices Atoms whose distances feed the distributions — every atom for the plain USR block, a
 *             CREDO subset for a USRCAT extension block. An empty subset writes zeros, as RDKit's empty
 *             `reducedCoords` path does.
 *  @param[out] block Exactly `usr_components_k` numbers of the caller's descriptor row.
 */
template <typename atom_indices_type_>
inline void moments_for_subset_(conformer_reader_t coords,
                                std::array<point3_t, usr_reference_point_count_k> const &points,
                                atom_indices_type_ const &atom_indices, custd::span<double> block) {
    sf_assert_(block.size() == usr_components_k);
    for (std::size_t reference = 0; reference < usr_reference_point_count_k; ++reference) {
        std::array<double, usr_moment_count_k> const moments = moments_about_(coords, atom_indices, points[reference]);
        for (std::size_t moment = 0; moment < usr_moment_count_k; ++moment)
            block[reference * usr_moment_count_k + moment] = moments[moment];
    }
}

/**
 *  @brief Writes the all-atom USR block and hands back the reference points every further block measures against.
 *
 *  @param[in] coords One conformer's coordinates.
 *  @param[out] block Exactly `usr_components_k` numbers, reference-point major.
 *  @return The four all-atom reference points, in RDKit order.
 */
inline std::array<point3_t, usr_reference_point_count_k> usr_block_into_(conformer_reader_t coords,
                                                                         custd::span<double> block) {
    std::array<point3_t, usr_reference_point_count_k> const points = reference_points_(coords);
    moments_for_subset_(coords, points, std::views::iota(std::uint32_t {0}, coords.extent(0)), block);
    return points;
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

/**
 *  @brief Puts @p out back into its empty state, keeping the capacity its five vectors already grew to.
 *
 *  `out = {}` would release them, which is the allocation a hoisted context exists to avoid: the refusal
 *  paths are exactly where the next molecule's buffers would otherwise be freed.
 */
inline void clear_shape_context_(molecule_shape_context_t &out) {
    for (std::vector<std::uint32_t> &subset : out.credo_subsets) subset.clear();
    out.component_label.clear();
    out.num_components = 0;
}

#pragma endregion Internal

#pragma region Public API

/**
 *  @brief Parses @p smiles and types it for USRCAT into @p out, once per molecule.
 *
 *  Uses the same parse contract as the rest of the library — sanitized, explicit hydrogens kept, tolerant
 *  CXSMILES, and stereochemistry left unperceived because nothing here reads it — then `MolOps::addHs` so the
 *  atom order matches the coordinates the conformer generator stores. The
 *  four CREDO subsets are matched with RDKit's own default USRCAT SMARTS, and connected components are labeled
 *  with `MolOps::getMolFrags`.
 *
 *  @param[in] smiles Molecule to type.
 *  @param[in] expected_atoms Atoms the caller's stored coordinates were written for. This parse has to agree
 *      with the one that produced them, so a divergence is refused here rather than described against the
 *      wrong atom order. It is not derivable from the context, because an empty SMILES parses into a real
 *      molecule of no atoms.
 *  @param[in] salt_policy Counterion handling, matching the parse the stored coordinates came from.
 *  @param[out] out Typed context on `typed_k`, emptied on every refusal. Its vectors keep the capacity they
 *      grew to, so a context hoisted above a molecule loop stops allocating once the loop has warmed.
 *  @return Which of the three refusals applies, or `typed_k`.
 */
[[nodiscard]] inline shape_typing_t build_shape_context_into(std::string_view smiles, std::uint32_t expected_atoms,
                                                             smallfield::salt_policy_t salt_policy,
                                                             molecule_shape_context_t &out) {
    clear_shape_context_(out);
    try {
        // Same contract as the preprocess path, counterion removal included. The cap is left off because every
        // molecule reaching here already passed the pipeline's own gate.
        smallfield::parse_options_t parse_options;
        parse_options.max_atoms = 0;
        parse_options.salt_policy = salt_policy;
        // Nothing below reads a chiral tag or a bond stereo: the four SMARTS name elements, charges,
        // hydrogen counts, valences, aromaticity and hybridization, and the labeling and the atom count
        // are pure connectivity. Perceiving stereochemistry here is a CIP ranking of every molecule that
        // no consumer can observe.
        parse_options.stereo_policy = smallfield::stereo_policy_t::skip_k;
        smallfield::parse_result_t parsed = smallfield::parse_smiles(smiles, parse_options);
        if (!parsed.molecule) return shape_typing_t::unparsed_k;
        // Checked before the four substructure searches, which a molecule nobody can describe need not pay for.
        if (parsed.num_atoms != expected_atoms) return shape_typing_t::atom_count_divergence_k;
        std::unique_ptr<RDKit::RWMol> molecule = std::move(parsed.molecule);

        // Type the four CREDO subsets exactly as RDKit's `getAtomIdsForFeatures` does: collect every matched
        // atom index, preserving RDKit's collection order.
        std::array<RDKit::ROMol, credo_subset_count_k> const &matchers = credo_matchers_();
        // `SubstructMatch` move-assigns a vector it built itself, so the buffer behind these four searches is
        // RDKit's and the hoist keeps the object rather than its capacity.
        std::vector<RDKit::MatchVectType> matches;
        for (std::size_t subset_index = 0; subset_index < credo_subset_count_k; ++subset_index) {
            RDKit::SubstructMatch(*molecule, matchers[subset_index], matches);
            for (auto const &match : matches)
                for (auto const &pair : match)
                    out.credo_subsets[subset_index].push_back(static_cast<std::uint32_t>(pair.second));
        }

        // Label connected components — disconnected fragments and salt counterions. `getMolFrags` resizes the
        // destination to the atom count and writes every entry, so a hoisted context is labeled in place and
        // keeps the capacity it grew to. Past `max_components_k` the molecule is refused rather than described.
        std::uint32_t const num_components = RDKit::MolOps::getMolFrags(*molecule, out.component_label);
        if (num_components > max_components_k) {
            clear_shape_context_(out);
            return shape_typing_t::too_many_fragments_k;
        }
        out.num_components = num_components;
    }
    catch (...) {
        clear_shape_context_(out);
        return shape_typing_t::unparsed_k;
    }
    return shape_typing_t::typed_k;
}

/**
 *  @brief The 12-number USR shape descriptor for one conformer.
 *
 *  @param[in] coords_interleaved Tight (num_atoms, 3) row-major xyz for one conformer.
 *  @param[in] num_atoms Atom count; the span carries three floats per atom.
 *  @param[out] out Exactly `usr_components_k` doubles of the caller's own destination, reference-point major
 *      (ctd, cst, fct, ftf). Every one is written, so nothing stages through a temporary.
 */
inline void usr_descriptor(custd::span<float const> coords_interleaved, std::uint32_t num_atoms,
                           custd::span<double> out) {
    sf_assert_(out.size() == usr_components_k);
    [[maybe_unused]] std::array<point3_t, usr_reference_point_count_k> const points = usr_block_into_(
        conformer_over_(coords_interleaved, num_atoms), out);
}

/**
 *  @brief The 60-number USRCAT shape descriptor for one conformer.
 *
 *  Block 0 is the plain USR over all atoms. Blocks 1 through 4 repeat the same moments against the same four
 *  all-atom reference points, restricting the distance distributions to each CREDO subset in order hydrophobic,
 *  aromatic, acceptor, donor. Empty subsets contribute a block of zeros, matching RDKit.
 *
 *  @param[in] coords_interleaved Tight (num_atoms, 3) row-major xyz for one conformer.
 *  @param[in] context Context from `build_shape_context_into` for this molecule.
 *  @param[out] out Exactly `usrcat_components_k` doubles of the caller's own destination, one
 *      `usr_components_k` block per subset. Every one is written, so nothing stages through a temporary.
 */
inline void usrcat_descriptor(custd::span<float const> coords_interleaved, molecule_shape_context_t const &context,
                              custd::span<double> out) {
    sf_assert_(out.size() == usrcat_components_k);
    conformer_reader_t const coords = conformer_over_(coords_interleaved, context.num_atoms());
    std::array<point3_t, usr_reference_point_count_k> const points = usr_block_into_(coords,
                                                                                     out.subspan(0, usr_components_k));

    for (std::size_t subset_index = 0; subset_index < credo_subset_count_k; ++subset_index)
        moments_for_subset_(coords, points, context.credo_subsets[subset_index],
                            out.subspan((subset_index + 1) * usr_components_k, usr_components_k));
}

#pragma endregion Public API

} // namespace smallfield::postprocess
