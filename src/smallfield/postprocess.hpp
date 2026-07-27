/**
 *  @file postprocess.hpp
 *  @brief Native USR / USRCAT molecular shape descriptors and fragment separation.
 *
 *  Implements the Ultrafast Shape Recognition descriptor (USR, 12 numbers) and its
 *  CREDO-pharmacophore extension (USRCAT, 60 numbers) so the conformer pipeline can
 *  emit shape fingerprints without a round trip through RDKit per conformer.
 *
 *  The numeric definitions match `RDKit::Descriptors::USR` / `USRCAT` bit for bit
 *  (see RDKit's `USRDescriptor.cpp`): four reference points (centroid; atom closest
 *  to the centroid; atom farthest from the centroid; atom farthest from that one),
 *  and for each reference point three moments of the distance distribution -- mean,
 *  standard deviation, and the signed cube root of the standardized third central
 *  moment (skewness).  The descriptor is reference-point major.
 *
 *  RDKit's `GetUSRCAT` re-types the four CREDO pharmacophore subsets on every call.
 *  A molecule with @a k conformers therefore pays for @a k substructure searches.
 *  Here @ref build_shape_context types the subsets and labels connected components
 *  exactly once; @ref usrcat_descriptor then consumes that context per conformer.
 *
 *  @code{.cpp}
 *  auto ctx = smallfield::postprocess::build_shape_context("CC(=O)Oc1ccccc1C(=O)O");
 *  std::array<double, 60> fingerprint;
 *  smallfield::postprocess::usrcat_descriptor(coords, ctx, fingerprint);
 *  @endcode
 */

#pragma once

#include <cmath>    // `std::sqrt`, `std::cbrt`
#include <cstddef>  // `std::size_t`
#include <cstdint>  // `std::uint32_t`

#include <array>       // `std::array`
#include <memory>      // `std::unique_ptr`
#include <span>        // `std::span`
#include <string>      // `std::string`
#include <string_view> // `std::string_view`
#include <vector>      // `std::vector`

#include <GraphMol/GraphMol.h>
#include <GraphMol/MolOps.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <GraphMol/Substruct/SubstructMatch.h>

namespace smallfield::postprocess {

#pragma region Shape Context

/**
 *  @brief Per-molecule shape typing computed once and reused across all conformers.
 *
 *  The four CREDO subsets follow RDKit's USRCAT order: hydrophobic, aromatic,
 *  acceptor, donor.  Each holds the atom indices (into the hydrogen-augmented
 *  molecule, matching stored coordinate order) that the corresponding SMARTS
 *  pattern matched.  Duplicate matches are preserved exactly as RDKit collects
 *  them so subset distance distributions are identical.
 */
struct molecule_shape_context_t {
    std::uint32_t num_atoms = 0;
    std::array<std::vector<std::uint32_t>, 4> credo_subsets; ///< hydrophobic, aromatic, acceptor, donor
    std::vector<std::uint8_t> component_label;               ///< per-atom connected-component id
    std::uint32_t num_components = 0;
    bool ok = false;
};

#pragma endregion Shape Context

#pragma region Internal

namespace detail {

/** @brief A double-precision 3D point read out of the caller's float coordinate buffer. */
struct point3_t {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

/** @brief Reads atom @p atom_index from a tight interleaved (num_atoms, 3) float buffer. */
inline point3_t read_point(std::span<float const> coords_interleaved, std::uint32_t atom_index) {
    std::size_t const base = static_cast<std::size_t>(atom_index) * 3u;
    return point3_t {static_cast<double>(coords_interleaved[base + 0u]),
                     static_cast<double>(coords_interleaved[base + 1u]),
                     static_cast<double>(coords_interleaved[base + 2u])};
}

/** @brief Euclidean distance between two points. */
inline double distance(point3_t const &first, point3_t const &second) {
    double const dx = first.x - second.x;
    double const dy = first.y - second.y;
    double const dz = first.z - second.z;
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

/**
 *  @brief The four USR reference points, in RDKit order.
 *
 *  Index 0 = centroid (ctd); 1 = atom closest to the centroid (cst);
 *  2 = atom farthest from the centroid (fct); 3 = atom farthest from fct (ftf).
 *  Tie-breaking matches RDKit's `smallestValId` / `largestValId`: strict
 *  comparison, first winning index kept.
 */
inline std::array<point3_t, 4> reference_points(std::span<float const> coords_interleaved,
                                                std::uint32_t num_atoms) {
    std::array<point3_t, 4> points {};

    // ctd: centroid of all atoms.
    point3_t centroid {};
    for (std::uint32_t atom_index = 0; atom_index < num_atoms; ++atom_index) {
        point3_t const p = read_point(coords_interleaved, atom_index);
        centroid.x += p.x;
        centroid.y += p.y;
        centroid.z += p.z;
    }
    if (num_atoms > 0) {
        centroid.x /= num_atoms;
        centroid.y /= num_atoms;
        centroid.z /= num_atoms;
    }
    points[0] = centroid;

    // Distances to the centroid, then closest (cst) and farthest (fct).
    std::uint32_t closest_index = 0;
    std::uint32_t farthest_index = 0;
    double closest_distance = 0.0;
    double farthest_distance = 0.0;
    for (std::uint32_t atom_index = 0; atom_index < num_atoms; ++atom_index) {
        double const d = distance(read_point(coords_interleaved, atom_index), centroid);
        if (atom_index == 0) {
            closest_distance = d;
            farthest_distance = d;
        }
        else {
            if (d < closest_distance) {
                closest_distance = d;
                closest_index = atom_index;
            }
            if (d > farthest_distance) {
                farthest_distance = d;
                farthest_index = atom_index;
            }
        }
    }
    points[1] = read_point(coords_interleaved, closest_index);
    points[2] = read_point(coords_interleaved, farthest_index);

    // ftf: atom farthest from fct.
    std::uint32_t farthest_from_fct_index = 0;
    double farthest_from_fct_distance = 0.0;
    for (std::uint32_t atom_index = 0; atom_index < num_atoms; ++atom_index) {
        double const d = distance(read_point(coords_interleaved, atom_index), points[2]);
        if (atom_index == 0) { farthest_from_fct_distance = d; }
        else if (d > farthest_from_fct_distance) {
            farthest_from_fct_distance = d;
            farthest_from_fct_index = atom_index;
        }
    }
    points[3] = read_point(coords_interleaved, farthest_from_fct_index);

    return points;
}

/**
 *  @brief Three USR moments of a distance list, matching RDKit's `calcMoments`.
 *
 *  Order: [0] mean, [1] standard deviation `sqrt(mean((d-mean)^2))`, [2] signed
 *  cube root of the standardized third central moment `cbrt(mean((d-mean)^3) /
 *  std^3)`.  An empty list yields three zeros; a degenerate list with zero
 *  standard deviation yields a zero third moment (RDKit's guard against dividing
 *  by zero).
 */
inline std::array<double, 3> calc_moments(std::vector<double> const &distances) {
    std::array<double, 3> moments {0.0, 0.0, 0.0};
    std::size_t const num_points = distances.size();
    if (num_points == 0) return moments;

    double mean = 0.0;
    for (double const d : distances) mean += d;
    mean /= static_cast<double>(num_points);
    moments[0] = mean;

    double sum_squared = 0.0;
    double sum_cubed = 0.0;
    for (double const d : distances) {
        double const diff = d - mean;
        sum_squared += diff * diff;
        sum_cubed += diff * diff * diff;
    }
    double const standard_deviation = std::sqrt(sum_squared / static_cast<double>(num_points));
    moments[1] = standard_deviation;

    double const third_central = sum_cubed / static_cast<double>(num_points);
    if (standard_deviation == 0.0) moments[2] = 0.0;
    else
        moments[2] = std::cbrt(third_central /
                               (standard_deviation * standard_deviation * standard_deviation));
    return moments;
}

/**
 *  @brief Writes the 12 moments for one atom subset relative to four reference points.
 *
 *  @p atom_indices lists the atoms whose distances feed the distributions (all atoms
 *  for the plain USR block, a CREDO subset for USRCAT extension blocks).  Output is
 *  written reference-point major starting at @p out_offset.  An empty subset writes
 *  twelve zeros, exactly as RDKit's empty `reducedCoords` path does.
 */
inline void moments_for_subset(std::span<float const> coords_interleaved,
                               std::array<point3_t, 4> const &points,
                               std::span<std::uint32_t const> atom_indices, double *out,
                               std::size_t out_offset) {
    std::vector<double> distances;
    distances.reserve(atom_indices.size());
    for (std::size_t reference = 0; reference < 4; ++reference) {
        distances.clear();
        for (std::uint32_t const atom_index : atom_indices)
            distances.push_back(distance(read_point(coords_interleaved, atom_index), points[reference]));
        std::array<double, 3> const moments = calc_moments(distances);
        out[out_offset + reference * 3u + 0u] = moments[0];
        out[out_offset + reference * 3u + 1u] = moments[1];
        out[out_offset + reference * 3u + 2u] = moments[2];
    }
}

/**
 *  @brief The four CREDO pharmacophore SMARTS matchers, compiled once.
 *
 *  Order matches RDKit's USRCAT default typing: hydrophobic, aromatic, acceptor,
 *  donor.  Parsed on first use into a function-local static so the SMARTS are never
 *  re-parsed per molecule.
 */
inline std::array<RDKit::ROMol const *, 4> const &credo_matchers() {
    static std::array<std::unique_ptr<RDKit::ROMol const>, 4> const owned = []() {
        static char const *const patterns[4] = {
            "[#6+0!$(*~[#7,#8,F]),SH0+0v2,s+0,S^3,Cl+0,Br+0,I+0]",                                     // hydrophobic
            "[a]",                                                                                     // aromatic
            "[$([O,S;H1;v2]-[!$(*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),$([N&v3;H1,H2]-[!$(*=[O,N,P,"
            "S])]),$([N;v3;H0]),$([n,o,s;+0]),F]",                                                     // acceptor
            "[N!H0v3,N!H0+v4,OH+0,SH+0,nH+0]"                                                          // donor
        };
        std::array<std::unique_ptr<RDKit::ROMol const>, 4> compiled;
        for (std::size_t pattern_index = 0; pattern_index < 4; ++pattern_index)
            compiled[pattern_index].reset(RDKit::SmartsToMol(patterns[pattern_index]));
        return compiled;
    }();
    static std::array<RDKit::ROMol const *, 4> const views = {owned[0].get(), owned[1].get(), owned[2].get(),
                                                              owned[3].get()};
    return views;
}

} // namespace detail

#pragma endregion Internal

#pragma region Public API

/**
 *  @brief Parses @p smiles and types it for USRCAT, once per molecule.
 *
 *  Uses the same parse contract as the rest of the library -- sanitize on,
 *  keep explicit hydrogens, tolerant CXSMILES -- then `MolOps::addHs` so the
 *  atom order matches the coordinates the conformer generator stores.  The four
 *  CREDO subsets are matched with RDKit's own default USRCAT SMARTS, and
 *  connected components are labeled with `MolOps::getMolFrags`.
 *
 *  @return A populated context with @c ok == true on success; @c ok == false
 *          (and empty fields) if parsing or typing threw.
 */
inline molecule_shape_context_t build_shape_context(std::string_view smiles) {
    molecule_shape_context_t context;
    try {
        RDKit::SmilesParserParams params;
        params.sanitize = true;
        params.removeHs = false;
        params.strictCXSMILES = false;
        std::unique_ptr<RDKit::RWMol> mol(RDKit::SmilesToMol(std::string(smiles), params));
        if (!mol) return context;
        RDKit::MolOps::addHs(*mol);

        context.num_atoms = mol->getNumAtoms();

        // Type the four CREDO subsets exactly as RDKit's getAtomIdsForFeatures does:
        // collect every matched atom index, preserving RDKit's collection order.
        std::array<RDKit::ROMol const *, 4> const &matchers = detail::credo_matchers();
        for (std::size_t subset_index = 0; subset_index < 4; ++subset_index) {
            std::vector<RDKit::MatchVectType> matches;
            RDKit::SubstructMatch(*mol, *matchers[subset_index], matches);
            for (auto const &match : matches)
                for (auto const &pair : match)
                    context.credo_subsets[subset_index].push_back(static_cast<std::uint32_t>(pair.second));
        }

        // Label connected components (disconnected fragments / salt counterions).
        std::vector<int> mapping;
        context.num_components = RDKit::MolOps::getMolFrags(*mol, mapping);
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
 *  @param coords_interleaved tight (num_atoms, 3) row-major xyz for one conformer.
 *  @param num_atoms          atom count (must equal `coords_interleaved.size() / 3`).
 *  @param out                12 doubles, reference-point major (ctd, cst, fct, ftf).
 */
inline void usr_descriptor(std::span<float const> coords_interleaved, std::uint32_t num_atoms,
                           std::array<double, 12> &out) {
    std::array<detail::point3_t, 4> const points = detail::reference_points(coords_interleaved, num_atoms);
    std::vector<std::uint32_t> all_atoms(num_atoms);
    for (std::uint32_t atom_index = 0; atom_index < num_atoms; ++atom_index) all_atoms[atom_index] = atom_index;
    detail::moments_for_subset(coords_interleaved, points, all_atoms, out.data(), 0);
}

/**
 *  @brief The 60-number USRCAT shape descriptor for one conformer.
 *
 *  Block 0 (indices 0..11) is the plain USR over all atoms.  Blocks 1..4 repeat the
 *  12 moments using the same four all-atom reference points, but restricting the
 *  distance distributions to each CREDO subset in order hydrophobic, aromatic,
 *  acceptor, donor.  Empty subsets contribute twelve zeros, matching RDKit.
 *
 *  @param coords_interleaved tight (num_atoms, 3) row-major xyz for one conformer.
 *  @param ctx                context from @ref build_shape_context for this molecule.
 *  @param out                60 doubles.
 */
inline void usrcat_descriptor(std::span<float const> coords_interleaved, molecule_shape_context_t const &ctx,
                              std::array<double, 60> &out) {
    std::array<detail::point3_t, 4> const points = detail::reference_points(coords_interleaved, ctx.num_atoms);

    std::vector<std::uint32_t> all_atoms(ctx.num_atoms);
    for (std::uint32_t atom_index = 0; atom_index < ctx.num_atoms; ++atom_index) all_atoms[atom_index] = atom_index;
    detail::moments_for_subset(coords_interleaved, points, all_atoms, out.data(), 0);

    std::size_t out_offset = 12;
    for (std::size_t subset_index = 0; subset_index < 4; ++subset_index) {
        detail::moments_for_subset(coords_interleaved, points, ctx.credo_subsets[subset_index], out.data(),
                                   out_offset);
        out_offset += 12;
    }
}

/**
 *  @brief Rigidly translates each connected component so bounding spheres do not overlap.
 *
 *  Lays every component's centroid on the +x axis in label order, spacing consecutive
 *  centroids by the sum of their bounding-sphere radii plus a fixed margin.  Each
 *  component's atoms move by the same delta (pure translation), so intra-component
 *  distances are preserved.  Single-component molecules are left untouched.
 */
inline void separate_fragments(std::span<float> coords_interleaved, molecule_shape_context_t const &ctx) {
    if (ctx.num_components <= 1) return;

    double constexpr margin = 2.0;
    std::uint32_t const num_components = ctx.num_components;

    // Per-component centroid and count.
    std::vector<detail::point3_t> centroid(num_components);
    std::vector<std::uint32_t> count(num_components, 0);
    for (std::uint32_t atom_index = 0; atom_index < ctx.num_atoms; ++atom_index) {
        std::uint8_t const label = ctx.component_label[atom_index];
        std::size_t const base = static_cast<std::size_t>(atom_index) * 3u;
        centroid[label].x += coords_interleaved[base + 0u];
        centroid[label].y += coords_interleaved[base + 1u];
        centroid[label].z += coords_interleaved[base + 2u];
        ++count[label];
    }
    for (std::uint32_t component = 0; component < num_components; ++component) {
        if (count[component] > 0) {
            centroid[component].x /= count[component];
            centroid[component].y /= count[component];
            centroid[component].z /= count[component];
        }
    }

    // Bounding-sphere radius of each component about its centroid.
    std::vector<double> radius(num_components, 0.0);
    for (std::uint32_t atom_index = 0; atom_index < ctx.num_atoms; ++atom_index) {
        std::uint8_t const label = ctx.component_label[atom_index];
        double const d = detail::distance(detail::read_point(coords_interleaved, atom_index), centroid[label]);
        if (d > radius[label]) radius[label] = d;
    }

    // Target centroids strung along +x so consecutive spheres clear each other.
    double cursor = 0.0;
    for (std::uint32_t component = 0; component < num_components; ++component) {
        double const target_x = cursor + radius[component];
        double const delta_x = target_x - centroid[component].x;
        double const delta_y = -centroid[component].y;
        double const delta_z = -centroid[component].z;
        for (std::uint32_t atom_index = 0; atom_index < ctx.num_atoms; ++atom_index) {
            if (ctx.component_label[atom_index] != component) continue;
            std::size_t const base = static_cast<std::size_t>(atom_index) * 3u;
            coords_interleaved[base + 0u] += static_cast<float>(delta_x);
            coords_interleaved[base + 1u] += static_cast<float>(delta_y);
            coords_interleaved[base + 2u] += static_cast<float>(delta_z);
        }
        cursor = target_x + radius[component] + margin;
    }
}

#pragma endregion Public API

} // namespace smallfield::postprocess
