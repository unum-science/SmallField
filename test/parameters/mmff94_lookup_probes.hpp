/**
 *  @file mmff94_lookup_probes.hpp
 *  @brief Query/result structs for probing each MMFF94 parameter table lookup in isolation.
 */

#pragma once

#include <cstdint> // `std::uint8_t`

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
    float kb;
    float r0;
};

struct angle_query_t {
    std::uint8_t angle_type;
    std::uint8_t atom_type_i;
    std::uint8_t atom_type_j_center;
    std::uint8_t atom_type_k;
};

struct angle_result_t {
    std::uint8_t hit;
    float ka;
    float theta0_deg;
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
    float v1;
    float v2;
    float v3;
    std::uint8_t canonical_torsion_type;
};

struct oop_query_t {
    std::uint8_t atom_type_i;
    std::uint8_t atom_type_j_center;
    std::uint8_t atom_type_k;
    std::uint8_t atom_type_l;
};

struct oop_result_t {
    std::uint8_t hit;
    float koop;
};

struct stbn_query_t {
    std::uint8_t stretch_bend_type;
    std::uint8_t bond_type_ij;
    std::uint8_t bond_type_kj;
    std::uint8_t atom_type_i;
    std::uint8_t atom_type_j_center;
    std::uint8_t atom_type_k;
};

struct stbn_result_t {
    std::uint8_t hit;
    std::uint8_t canonical_order_swapped;
    float kba_ijk;
    float kba_kji;
};
