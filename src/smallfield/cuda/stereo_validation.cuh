/**
 *  @file src/smallfield/cuda/stereo_validation.cuh
 *  @brief Re-read a conformer's stereochemistry out of the coordinates that will actually ship.
 *
 *  `mmff_polish_warp_kernel` rewrites every coordinate after `etkdg_pipeline_warp_kernel` stamped its verdict
 *  and carries no stereochemistry term of its own, so a centre or bond it inverts leaves no trace in the status
 *  byte or any counter; this is the only stage that looks at the geometry the writer will pack. It cannot live
 *  in `etkdg_pipeline.cuh` because it reads `molecule_descriptor_t`, whose header pulls in RDKit, while the
 *  replay bench consumes the pipeline header and deliberately links none.
 */

#pragma once

#if defined(__CUDACC__)

#include <cstdint> // `std::uint32_t`, `std::uint8_t`

#include "smallfield/cuda/dg_kernel.cuh"
#include "smallfield/types/conformer_status.hpp"
#include "smallfield/types/etkdg_types.hpp"
#include "smallfield/types/molecule_descriptor.hpp"

namespace smallfield::cuda {

using smallfield::types::chiral_center_descriptor_t;
using smallfield::types::conformer_status_t;
using smallfield::types::etkdg_molecule_metadata_t;
using smallfield::types::molecule_descriptor_t;

/**
 *  Smallest `|cos phi|` across a declared double bond before its sense is read at all. Twisted near ninety
 *  degrees a bond carries no cis-or-trans answer, and calling it wrong would refuse a geometry for being
 *  undecided rather than incorrect. At 0.5 the band is ±60° from planar, where 90 % of declared bonds land.
 */
constexpr float bond_stereo_planarity_floor_k = 0.5f;

/**
 *  How close two atoms may sit before the geometry is refused, in angstrom. Below the 0.74 Å bond of H2,
 *  the shortest in chemistry, so no separation a molecule can actually form falls under it and a bonded
 *  pair never has to be excluded from the walk. The `fallback-role` applies no such floor, so a collapsed
 *  geometry from the CPU embedder still ships as `success_k`.
 */
constexpr float min_physical_atom_separation_k = 0.70f;

/**
 *  @brief Count the declared centers a re-perception of the shipped geometry will read the other way.
 *
 *  A tetrahedral center carries two signed volumes over the same three neighbors, and the pipeline and its
 *  consumer read different ones: `count_invalid_chiral_centers_` takes the triple product from the @b fourth
 *  neighbor — RDKit's `DistGeom::ChiralSet` expression, the quantity the embedding restrains — while
 *  `assignChiralTypesFrom3D` takes it from the @b center atom and never reads the fourth. They agree wherever
 *  the center lies on the same side of its neighbors' plane as the fourth, every ordinary tetrahedron.
 *
 *  Both are taken: the pivot-based volume must be handed correctly and clear the flatness floor, which the
 *  embedding gate asks, and the center-based one must agree, which the consumer asks. The MMFF polish
 *  restrains both, so this stage audits rather than discovers the gap — `BENCH.md`, _Restraining Only One Of
 *  The Two Volumes_. A three-coordinate center is its own fourth neighbor, so there the second test is free.
 *
 *  Centers are spread across the warp, so the count is warp-reduced and every lane returns it.
 *
 *  @param[in] positions SoA coordinate array.
 *  @param[in] stride Distance between coordinate planes, which need not be the atom count.
 *  @param[in] chiral_centers Declared centers of this molecule.
 *  @param[in] num_chiral_centers How many there are.
 *  @param[in] lane Warp lane index (0-31).
 *  @return How many centers will not read back as declared; zero means the geometry ships as asked.
 */
__device__ __forceinline__ std::uint32_t count_unreadable_chiral_centers_(
    float const *positions, std::uint32_t stride, chiral_center_descriptor_t const *chiral_centers,
    std::uint32_t num_chiral_centers, std::uint32_t lane) {
    std::uint32_t lane_invalid = 0;
    for (std::uint32_t center_index = lane; center_index < num_chiral_centers; center_index += warp_size_k) {
        auto const &center = chiral_centers[center_index];
        float const restrained = compute_signed_tetrahedral_volume_(positions, stride, center.neighbor_atoms[3],
                                                                    center.neighbor_atoms[0], center.neighbor_atoms[1],
                                                                    center.neighbor_atoms[2]);
        float const perceived = compute_signed_tetrahedral_volume_(positions, stride, center.center_atom,
                                                                   center.neighbor_atoms[0], center.neighbor_atoms[1],
                                                                   center.neighbor_atoms[2]);
        bool const target_negative = center.target_sign < 0;
        bool const wrong_handed = (signbit(restrained) != 0) != target_negative;
        bool const wrong_perceived = (signbit(perceived) != 0) != target_negative;
        bool const too_flat = fabsf(restrained) < chiral_volume_slack_k * center.minimum_volume;
        lane_invalid += (wrong_handed || wrong_perceived || too_flat) ? 1u : 0u;
    }
    for (std::uint32_t offset = warp_size_k / 2; offset > 0; offset >>= 1)
        lane_invalid += __shfl_xor_sync(full_mask_k, lane_invalid, offset);
    return lane_invalid;
}

/**
 *  @brief Count the declared double bonds whose geometry contradicts their flag.
 *
 *  No chiral descriptor is ever emitted for a double bond, so nothing else in the pipeline can refuse one.
 *  The flag relates the two @b recorded reference substituents, one per end, so the test is the sign of
 *  `cos phi` over (reference, end, end, reference) — positive is cis, which the Z flag asserts. Bonds are
 *  spread across the warp and the count is warp-reduced.
 *
 *  @param[in] positions SoA coordinate array.
 *  @param[in] stride Distance between coordinate planes, which need not be the atom count.
 *  @param[in] descriptor Molecule whose bond flags and reference atoms are read.
 *  @param[in] lane Warp lane index (0-31).
 *  @return How many declared bonds came out with the opposite sense.
 */
__device__ __forceinline__ std::uint32_t count_invalid_bond_stereo_(float const *positions, std::uint32_t stride,
                                                                    molecule_descriptor_t const &descriptor,
                                                                    std::uint32_t lane) {
    std::uint32_t const num_bonds = descriptor.num_bonds;
    std::uint32_t lane_invalid = 0;
    for (std::uint32_t bond = lane; bond < num_bonds; bond += warp_size_k) {
        std::uint8_t const flags = descriptor.bond_flags[bond];
        bool const declared_e = (flags & types::bond_flag_stereo_e_k) != 0;
        bool const declared_z = (flags & types::bond_flag_stereo_z_k) != 0;
        if (!declared_e && !declared_z) continue;

        std::uint8_t const reference_i = descriptor.bond_stereo_atom_i[bond];
        std::uint8_t const reference_j = descriptor.bond_stereo_atom_j[bond];
        // A flagged bond whose reference pair never resolved is unperceived, not violated.
        if (reference_i == types::no_stereo_atom_k || reference_j == types::no_stereo_atom_k) continue;

        float3 const begin_position = load_position_(positions, stride, descriptor.bond_atom_i[bond]);
        float3 const end_position = load_position_(positions, stride, descriptor.bond_atom_j[bond]);
        float3 const position_i = load_position_(positions, stride, reference_i);
        float3 const position_j = load_position_(positions, stride, reference_j);

        // Which reference hangs off which end is read from the geometry rather than assumed from the
        // order the two were stored, matching how the bounds builder resolves the same pair.
        float3 const offset_near = sub_float3_(position_i, begin_position);
        float3 const offset_far = sub_float3_(position_i, end_position);
        bool const i_is_near = dot_float3_(offset_near, offset_near) < dot_float3_(offset_far, offset_far);
        float3 const first = i_is_near ? position_i : position_j;
        float3 const fourth = i_is_near ? position_j : position_i;

        float3 const axis = sub_float3_(end_position, begin_position);
        float3 const normal_first = cross_float3_(sub_float3_(begin_position, first), axis);
        float3 const normal_second = cross_float3_(axis, sub_float3_(fourth, end_position));
        float const numerator = dot_float3_(normal_first, normal_second);
        float const denominator =
            sqrtf(dot_float3_(normal_first, normal_first) * dot_float3_(normal_second, normal_second)) + 1e-12f;
        float const cosine = numerator / denominator;

        if (fabsf(cosine) < bond_stereo_planarity_floor_k) continue;
        lane_invalid += ((cosine > 0.0f) != declared_z) ? 1u : 0u;
    }
    for (std::uint32_t offset = warp_size_k / 2; offset > 0; offset >>= 1)
        lane_invalid += __shfl_xor_sync(full_mask_k, lane_invalid, offset);
    return lane_invalid;
}

/**
 *  @brief Count atom pairs sitting closer than `min_physical_atom_separation_k`.
 *
 *  `mmff_polish_warp_kernel` can leave two atoms on top of each other where the embedding started them badly
 *  and the minimizer walked downhill into the singularity rather than out of it. Nothing else refuses that
 *  geometry: the buffered 14-7 term makes the energy enormous but finite, which `energy_is_physical` accepts,
 *  so the slot ships as a success carrying coordinates no molecule occupies.
 *
 *  Every pair is walked, bonded ones included, because the floor sits below the shortest bond in chemistry
 *  and a real bond can never trip it. The ring walk is the one `mmff94_energy_kernel.cuh` uses for nonbonded
 *  pairs: at a fixed offset the 32 lanes address 32 distinct atoms, where a triangular split leaves lane 0
 *  walking `num_atoms` rows against lane 31's handful.
 *
 *  @param[in] positions SoA coordinate array.
 *  @param[in] stride Distance between coordinate planes, which need not be the atom count.
 *  @param[in] num_atoms Atoms this molecule actually carries, so the padding past it is never read.
 *  @param[in] lane Warp lane index (0-31).
 *  @return How many pairs are closer than the floor; zero means the geometry is one atoms could occupy.
 */
__device__ __forceinline__ std::uint32_t count_clashing_atom_pairs_(float const *positions, std::uint32_t stride,
                                                                    std::uint32_t num_atoms, std::uint32_t lane) {
    float const floor_squared = min_physical_atom_separation_k * min_physical_atom_separation_k;
    std::uint32_t lane_clashes = 0;
    std::uint32_t const half_shell = num_atoms / 2;
    bool const middle_offset_is_mirrored = (num_atoms % 2u) == 0u;
    for (std::uint32_t offset = 1; offset <= half_shell; ++offset) {
        bool const mirrored = middle_offset_is_mirrored && offset == half_shell;
        for (std::uint32_t atom = lane; atom < num_atoms; atom += warp_size_k) {
            std::uint32_t const other = (atom + offset) % num_atoms;
            // At the mirrored offset each pair is reachable from both ends, so only the lower half speaks.
            if (mirrored && atom >= other) continue;
            float3 const delta = sub_float3_(load_position_(positions, stride, other),
                                             load_position_(positions, stride, atom));
            float const separation_squared = delta.x * delta.x + delta.y * delta.y + delta.z * delta.z;
            lane_clashes += (separation_squared < floor_squared) ? 1u : 0u;
        }
    }
    for (std::uint32_t offset = warp_size_k / 2; offset > 0; offset >>= 1)
        lane_clashes += __shfl_xor_sync(full_mask_k, lane_clashes, offset);
    return lane_clashes;
}

/**
 *  @brief Re-read each conformer's stereochemistry out of the coordinates that will actually ship.
 *
 *  One warp per slot against the post-polish geometry, in single precision and before the half-precision
 *  narrowing. The predicates are not the pipeline kernel's: this stage asks what a re-perception of the
 *  shipped coordinates will say, which for a tetrahedral center is a different signed volume.
 *
 *  Only slots already holding `success_k` are read and demoted, so a more specific refusal keeps its cause.
 *  The verdict names which stereochemistry moved — `polish_inverted_center_k` or `polish_inverted_bond_k` —
 *  rather than the `stereo_failure_k` the pipeline writes when the embedding never satisfied a constraint at
 *  all. Both route to the CPU retry, which recovers them; the split separates a stage that cannot reach a
 *  geometry from one that walked out of the geometry it had.
 *
 *  Launched between the last polish and `stamp_refused_molecules_kernel`, so the stamp stays the last
 *  word and a molecule the metric prepass refused outright is not relabelled by this.
 */
__global__ __launch_bounds__(32, 8) void validate_polished_stereo_kernel(
    float const *__restrict__ conformer_coordinates, chiral_center_descriptor_t const *__restrict__ chiral_centers,
    molecule_descriptor_t const *__restrict__ descriptors,
    etkdg_molecule_metadata_t const *__restrict__ molecule_metadata, std::uint8_t *__restrict__ conformer_status,
    std::uint32_t total_molecules, std::uint32_t num_conformers_per_molecule, std::uint32_t output_atoms_stride) {
    std::uint32_t const lane = threadIdx.x;
    std::uint32_t const total_slots = total_molecules * num_conformers_per_molecule;
    for (std::uint32_t slot = blockIdx.x; slot < total_slots; slot += gridDim.x) {
        if (conformer_status[slot] != static_cast<std::uint8_t>(conformer_status_t::success_k)) continue;
        std::uint32_t const molecule = slot / num_conformers_per_molecule;
        auto const &meta = molecule_metadata[molecule];

        // The counter uses its second argument as a plane stride, so the padded slot stride goes here.
        float const *const positions = conformer_coordinates + static_cast<std::size_t>(slot) * output_atoms_stride *
                                                                   types::conformer_dimensions_k;
        // Decided ahead of the stereo walks and short-circuiting them: a centre read out of two atoms
        // sharing a position carries no handedness to be right or wrong about, and reporting that as an
        // inversion would name the symptom while the collapse goes unrecorded.
        if (count_clashing_atom_pairs_(positions, output_atoms_stride, meta.num_atoms, lane) != 0) {
            if (lane == 0)
                conformer_status[slot] = static_cast<std::uint8_t>(conformer_status_t::nonphysical_geometry_k);
            continue;
        }

        std::uint32_t invalid_centers = 0;
        if (meta.num_chiral_centers != 0)
            invalid_centers = count_unreadable_chiral_centers_(
                positions, output_atoms_stride, chiral_centers + meta.chiral_offset, meta.num_chiral_centers, lane);
        // No metadata counts declared double bonds, so the walk is the test: one flag byte, out on non-E/Z.
        std::uint32_t const invalid_bonds = count_invalid_bond_stereo_(positions, output_atoms_stride,
                                                                       descriptors[molecule], lane);
        // Counted apart rather than summed: an inverted center and an inverted bond have different causes
        // and fixes, and a corpus can be wrong in both at rates a sum cannot separate. The split costs one
        // branch and rides the parquet `status` column and every histogram built on it.
        if (lane == 0 && (invalid_centers != 0 || invalid_bonds != 0))
            conformer_status[slot] = static_cast<std::uint8_t>(invalid_centers != 0
                                                                   ? conformer_status_t::polish_inverted_center_k
                                                                   : conformer_status_t::polish_inverted_bond_k);
    }
}

} // namespace smallfield::cuda

#endif // defined(__CUDACC__)
