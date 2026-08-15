/**
 *  @file src/smallfield/cuda/stereo_validation.cuh
 *  @brief Re-read a conformer's stereochemistry out of the coordinates that will actually ship.
 *
 *  `mmff_polish_warp_kernel` rewrites every coordinate after `etkdg_pipeline_warp_kernel` has already
 *  stamped its verdict, and carries no stereochemistry term of its own, so a centre or bond it inverts
 *  leaves no trace anywhere — not in the status byte, not in any counter. This is the only stage that
 *  looks at the geometry the writer will pack.
 *
 *  Separate from `etkdg_pipeline.cuh` because it reads `molecule_descriptor_t`, whose header pulls in
 *  RDKit: the replay bench consumes the pipeline header and deliberately links no RDKit, so the two
 *  cannot share one file.
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
 *  Smallest `|cos phi|` across a declared double bond before its sense is read at all.
 *  A bond twisted near ninety degrees carries no cis-or-trans answer to compare against, and calling
 *  one wrong would refuse a geometry for being undecided rather than for being incorrect. At 0.5 the
 *  band is sixty degrees either side of planar, which is where 90 % of declared bonds land.
 */
constexpr float bond_stereo_planarity_floor_k = 0.5f;

/**
 *  @brief Count the declared double bonds whose geometry contradicts their flag.
 *
 *  `count_invalid_chiral_centers_` is blind to these: it walks tetrahedral centers, and no chiral
 *  descriptor is ever emitted for a double bond, so `stereo_failure_k` was unreachable for one.
 *
 *  The flag states the relationship of the two @b recorded reference substituents, one per end, so the
 *  test is the sign of `cos phi` over (reference, end, end, reference) — positive is cis, which is what
 *  the Z flag asserts. Bonds are spread across the warp and the count is warp-reduced.
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
 *  @brief Re-read each conformer's stereochemistry out of the coordinates that will actually ship.
 *
 *  One warp per slot, re-running the same predicates the pipeline kernel gates on against the
 *  post-polish geometry, in single precision and before the half-precision narrowing.
 *
 *  Only slots already holding `success_k` are read, and only those are demoted, so a more specific
 *  refusal keeps its own cause. The verdict names which kind of stereochemistry moved —
 *  `polish_inverted_center_k` or `polish_inverted_bond_k` — rather than the `stereo_failure_k` the
 *  pipeline kernel writes when the embedding could not satisfy a constraint to begin with. Both route
 *  to the CPU retry, which recovers them; the split is what separates a stage that cannot reach a
 *  geometry from a stage that walks out of one it had.
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

        // `count_invalid_chiral_centers_` reads its second argument only as the dimension stride, so
        // the slot's padded stride is what it wants rather than the molecule's own atom count.
        float const *const positions = conformer_coordinates + static_cast<std::size_t>(slot) * output_atoms_stride *
                                                                   types::conformer_dimensions_k;
        std::uint32_t invalid_centers = 0;
        if (meta.num_chiral_centers != 0)
            invalid_centers = count_invalid_chiral_centers_(
                positions, output_atoms_stride, chiral_centers + meta.chiral_offset, meta.num_chiral_centers, lane);
        // No metadata field counts declared double bonds, so the bond walk is the membership test —
        // it reads one flag byte per bond and leaves immediately on the ones carrying no E/Z.
        std::uint32_t const invalid_bonds = count_invalid_bond_stereo_(positions, output_atoms_stride,
                                                                       descriptors[molecule], lane);
        // The two counts were summed here and the distinction died in the sum. They answer different
        // questions — an inverted tetrahedral center and an inverted double bond have different causes
        // and different fixes — and the shipped corpus was wrong in both at once, at rates nobody could
        // separate. Naming them costs one branch on a value both counters already carry, and the split
        // then rides the parquet `status` column and every histogram built on it.
        if (lane == 0 && (invalid_centers != 0 || invalid_bonds != 0))
            conformer_status[slot] = static_cast<std::uint8_t>(invalid_centers != 0
                                                                   ? conformer_status_t::polish_inverted_center_k
                                                                   : conformer_status_t::polish_inverted_bond_k);
    }
}

} // namespace smallfield::cuda

#endif // defined(__CUDACC__)
