/**
 *  @file test/parameters/mmff94_lookup_probes.hpp
 *  @brief Query/result structs for probing the vdW, bond, angle and torsion table lookups in isolation.
 */

#pragma once

#include <cstdint> // `std::uint8_t`

namespace smallfield::tests {

struct vdw_query_t {
    std::uint8_t atom_type;
};

struct vdw_result_t {
    std::uint8_t hit;
    float alpha;
    float effective_electron_count;
    float polarizability_scale;
    float slater_kirkwood_factor;
    float r_star;
    std::uint8_t donor_acceptor;
};

struct bond_query_t {
    std::uint8_t bond_type;
    std::uint8_t atom_type_i;
    std::uint8_t atom_type_j;
};

struct bond_result_t {
    std::uint8_t hit;
    float force_constant;
    float equilibrium_distance;
};

struct angle_query_t {
    std::uint8_t angle_type;
    std::uint8_t atom_type_i;
    std::uint8_t atom_type_j_center;
    std::uint8_t atom_type_k;
};

struct angle_result_t {
    std::uint8_t hit;
    float force_constant;
    float equilibrium_angle_degrees;
};

struct torsion_query_t {
    std::uint8_t primary_torsion_type;
    std::uint8_t secondary_torsion_type;
    std::uint8_t atom_type_i;
    std::uint8_t atom_type_j;
    std::uint8_t atom_type_k;
    std::uint8_t atom_type_l;
};

struct torsion_result_t {
    std::uint8_t hit;
    float torsion_v1;
    float torsion_v2;
    float torsion_v3;
    std::uint8_t canonical_torsion_type;
};

} // namespace smallfield::tests
