/**
 *  @file src/smallfield/parameters/mmff94_tables_rdkit.hpp
 *  @brief RDKit-dependent MMFF94 table extraction.
 *
 *  Separated from `mmff94_tables.hpp` so that CUDA translation units can include the POD structs and
 *  lookup functions without pulling in RDKit, which triggers nvcc internal errors via `boost::tti`.
 *  Only host-side `.cpp` files that call `extract_mmff94_tables_from_rdkit` need this header.
 *
 *  This file is the declared library-spelling boundary: `crd`, `mltb`, `pilp`, `linh`, `sbmb`, `kbaIJK`
 *  and their siblings on the right of every assignment are RDKit's own field names, while the spelled-out
 *  names on the left are ours. One extractor per collection, and each one `assign`s its destination so a
 *  freshly value-initialized entry already carries the zeroes its padding needs.
 */

#pragma once

#include <algorithm> // `std::max`, `std::sort`

#include <ForceField/MMFF/Params.h>
#include <GraphMol/ForceFieldHelpers/MMFF/AtomTyper.h>

#include "smallfield/parameters/mmff94_tables.hpp"
#include <sstream> // RDKit's `AtomTyper.h` needs it but does not include it

namespace RDKit::MMFF::DefaultParameters {
ForceFields::MMFF::MMFFBondCollection const *getMMFFBond();
ForceFields::MMFF::MMFFChgCollection const *getMMFFChg();
ForceFields::MMFF::MMFFBndkCollection const *getMMFFBndk();
ForceFields::MMFF::MMFFCovRadPauEleCollection const *getMMFFCovRadPauEle();
} // namespace RDKit::MMFF::DefaultParameters

namespace smallfield::parameters {

#pragma region Sparse Tables

/**
 *  @brief Extract the VdW entries and the five collection-wide scalars that arrive with them.
 *
 *  @param[out] out Container whose `vdw` table and `vdw_*` scalars are filled.
 *  @return Highest MMFF atom type the table mentions, which sizes the two dense tables below.
 */
inline std::uint8_t extract_vdw_table_(mmff94_tables_host_t &out) {
    auto const *collection = RDKit::MMFF::DefaultParameters::getMMFFVdW();
    out.vdw_power = static_cast<float>(collection->power);
    out.vdw_b = static_cast<float>(collection->B);
    out.vdw_beta = static_cast<float>(collection->Beta);
    out.vdw_darad = static_cast<float>(collection->DARAD);
    out.vdw_daeps = static_cast<float>(collection->DAEPS);

    std::size_t const count = collection->d_atomType.size();
    out.vdw.assign(count, mmff94_vdw_entry_t {});
    std::uint8_t max_atom_type = 0;
    for (std::size_t entry_index = 0; entry_index < count; ++entry_index) {
        auto &entry = out.vdw[entry_index];
        auto const &params = collection->d_params[entry_index];
        entry.atom_type = collection->d_atomType[entry_index];
        entry.alpha_i = static_cast<float>(params.alpha_i);
        entry.n_effective = static_cast<float>(params.N_i);
        entry.a_scale = static_cast<float>(params.A_i);
        entry.g_factor = static_cast<float>(params.G_i);
        entry.r_star = static_cast<float>(params.R_star);
        entry.donor_acceptor = (params.DA == 'D') ? 2u : (params.DA == 'A') ? 1u : 0u;
        if (entry.atom_type > max_atom_type) max_atom_type = entry.atom_type;
    }
    return max_atom_type;
}

/** Extract the bond-stretch table, sorted by (i, j, bond_type). */
inline void extract_bond_table_(std::vector<mmff94_bond_entry_t> &out) {
    auto const *collection = RDKit::MMFF::DefaultParameters::getMMFFBond();
    std::size_t const count = collection->d_iAtomType.size();
    out.assign(count, mmff94_bond_entry_t {});
    for (std::size_t entry_index = 0; entry_index < count; ++entry_index) {
        auto &entry = out[entry_index];
        auto const &params = collection->d_params[entry_index];
        entry.i_type = collection->d_iAtomType[entry_index];
        entry.j_type = collection->d_jAtomType[entry_index];
        entry.bond_type = collection->d_bondType[entry_index];
        entry.force_constant = static_cast<float>(params.kb);
        entry.equilibrium_distance = static_cast<float>(params.r0);
    }
}

/** Extract the angle-bend table, sorted by (j, i, k, angle_type). */
inline void extract_angle_table_(std::vector<mmff94_angle_entry_t> &out) {
    auto const *collection = RDKit::MMFF::DefaultParameters::getMMFFAngle();
    std::size_t const count = collection->d_iAtomType.size();
    out.assign(count, mmff94_angle_entry_t {});
    for (std::size_t entry_index = 0; entry_index < count; ++entry_index) {
        auto &entry = out[entry_index];
        auto const &params = collection->d_params[entry_index];
        entry.j_type = collection->d_jAtomType[entry_index];
        entry.i_type = collection->d_iAtomType[entry_index];
        entry.k_type = collection->d_kAtomType[entry_index];
        entry.angle_type = collection->d_angleType[entry_index];
        entry.force_constant = static_cast<float>(params.ka);
        entry.theta0_deg = static_cast<float>(params.theta0);
    }
}

/** Extract the torsion table, sorted by (j, k, i, l, tor_type). */
inline void extract_torsion_table_(std::vector<mmff94_torsion_entry_t> &out) {
    auto const *collection = RDKit::MMFF::DefaultParameters::getMMFFTor(false);
    std::size_t const count = collection->d_iAtomType.size();
    out.assign(count, mmff94_torsion_entry_t {});
    for (std::size_t entry_index = 0; entry_index < count; ++entry_index) {
        auto &entry = out[entry_index];
        auto const &params = collection->d_params[entry_index];
        entry.j_type = collection->d_jAtomType[entry_index];
        entry.k_type = collection->d_kAtomType[entry_index];
        entry.i_type = collection->d_iAtomType[entry_index];
        entry.l_type = collection->d_lAtomType[entry_index];
        entry.tor_type = collection->d_torType[entry_index];
        entry.torsion_v1 = static_cast<float>(params.V1);
        entry.torsion_v2 = static_cast<float>(params.V2);
        entry.torsion_v3 = static_cast<float>(params.V3);
    }
}

/** Extract the out-of-plane table, sorted by (j, i, k, l). */
inline void extract_oop_table_(std::vector<mmff94_oop_entry_t> &out) {
    auto const *collection = RDKit::MMFF::DefaultParameters::getMMFFOop(false);
    std::size_t const count = collection->d_iAtomType.size();
    out.assign(count, mmff94_oop_entry_t {});
    for (std::size_t entry_index = 0; entry_index < count; ++entry_index) {
        auto &entry = out[entry_index];
        entry.j_type = collection->d_jAtomType[entry_index];
        entry.i_type = collection->d_iAtomType[entry_index];
        entry.k_type = collection->d_kAtomType[entry_index];
        entry.l_type = collection->d_lAtomType[entry_index];
        entry.force_constant = static_cast<float>(collection->d_params[entry_index].koop);
    }
}

/** Extract the stretch-bend table, sorted by (j, i, k, stretch_bend_type). */
inline void extract_stbn_table_(std::vector<mmff94_stbn_entry_t> &out) {
    auto const *collection = RDKit::MMFF::DefaultParameters::getMMFFStbn();
    std::size_t const count = collection->d_iAtomType.size();
    out.assign(count, mmff94_stbn_entry_t {});
    for (std::size_t entry_index = 0; entry_index < count; ++entry_index) {
        auto &entry = out[entry_index];
        auto const &params = collection->d_params[entry_index];
        entry.j_type = collection->d_jAtomType[entry_index];
        entry.i_type = collection->d_iAtomType[entry_index];
        entry.k_type = collection->d_kAtomType[entry_index];
        entry.stretch_bend_type = collection->d_stretchBendType[entry_index];
        entry.kba_ijk = static_cast<float>(params.kbaIJK);
        entry.kba_kji = static_cast<float>(params.kbaKJI);
    }
}

/**
 *  @brief Flatten the default stretch-bend fallback out of RDKit's three nested maps and sort it.
 *
 *  The lookup cascade binary-searches (row_i, row_j, row_k), so the map iteration order — already
 *  ascending for `std::map` but not guaranteed by the interface — is re-established explicitly.
 *
 *  @param[out] out Flat table, sorted by (row_i, row_j, row_k).
 */
inline void extract_dfsb_table_(std::vector<mmff94_dfsb_entry_t> &out) {
    auto const *collection = RDKit::MMFF::DefaultParameters::getMMFFDfsb();
    out.clear();
    for (auto const &[row_i, by_row_j] : collection->d_params)
        for (auto const &[row_j, by_row_k] : by_row_j)
            for (auto const &[row_k, params] : by_row_k) {
                mmff94_dfsb_entry_t entry {};
                entry.row_i = static_cast<std::uint8_t>(row_i);
                entry.row_j = static_cast<std::uint8_t>(row_j);
                entry.row_k = static_cast<std::uint8_t>(row_k);
                entry.kba_ijk = static_cast<float>(params.kbaIJK);
                entry.kba_kji = static_cast<float>(params.kbaKJI);
                out.push_back(entry);
            }
    std::sort(out.begin(), out.end(), [](mmff94_dfsb_entry_t const &left, mmff94_dfsb_entry_t const &right) {
        if (left.row_i != right.row_i) return left.row_i < right.row_i;
        if (left.row_j != right.row_j) return left.row_j < right.row_j;
        return left.row_k < right.row_k;
    });
}

/** Extract the partial bond-charge increments, sorted by atom type. */
inline void extract_pbci_table_(std::vector<mmff94_pbci_entry_t> &out) {
    auto const *collection = RDKit::MMFF::DefaultParameters::getMMFFPBCI();
    std::size_t const count = collection->d_params.size();
    out.assign(count, mmff94_pbci_entry_t {});
    for (std::size_t type_index = 0; type_index < count; ++type_index) {
        auto &entry = out[type_index];
        entry.atom_type = static_cast<std::uint8_t>(type_index + 1);
        entry.pbci = static_cast<float>(collection->d_params[type_index].pbci);
        entry.fcadj = static_cast<float>(collection->d_params[type_index].fcadj);
    }
}

/** Extract the bond-charge increments, sorted by (i, j, bond_type). */
inline void extract_chg_table_(std::vector<mmff94_chg_entry_t> &out) {
    auto const *collection = RDKit::MMFF::DefaultParameters::getMMFFChg();
    std::size_t const count = collection->d_iAtomType.size();
    out.assign(count, mmff94_chg_entry_t {});
    for (std::size_t entry_index = 0; entry_index < count; ++entry_index) {
        auto &entry = out[entry_index];
        entry.i_type = collection->d_iAtomType[entry_index];
        entry.j_type = collection->d_jAtomType[entry_index];
        entry.bond_type = collection->d_bondType[entry_index];
        entry.bond_charge_increment = static_cast<float>(collection->d_params[entry_index].bci);
    }
}

/** Extract the default bond parameters, sorted by atomic-number pair. */
inline void extract_bndk_table_(std::vector<mmff94_bndk_entry_t> &out) {
    auto const *collection = RDKit::MMFF::DefaultParameters::getMMFFBndk();
    std::size_t const count = collection->d_iAtomicNum.size();
    out.assign(count, mmff94_bndk_entry_t {});
    for (std::size_t entry_index = 0; entry_index < count; ++entry_index) {
        auto &entry = out[entry_index];
        auto const &params = collection->d_params[entry_index];
        entry.atomic_num_i = collection->d_iAtomicNum[entry_index];
        entry.atomic_num_j = collection->d_jAtomicNum[entry_index];
        entry.force_constant = static_cast<float>(params.kb);
        entry.equilibrium_distance = static_cast<float>(params.r0);
    }
}

/** Extract the Herschbach-Laurie parameters, sorted by periodic-table-row pair. */
inline void extract_hl_table_(std::vector<mmff94_hl_entry_t> &out) {
    auto const *collection = RDKit::MMFF::DefaultParameters::getMMFFHerschbachLaurie();
    std::size_t const count = collection->d_iRow.size();
    out.assign(count, mmff94_hl_entry_t {});
    for (std::size_t entry_index = 0; entry_index < count; ++entry_index) {
        auto &entry = out[entry_index];
        auto const &params = collection->d_params[entry_index];
        entry.row_i = collection->d_iRow[entry_index];
        entry.row_j = collection->d_jRow[entry_index];
        entry.a_ij = static_cast<float>(params.a_ij);
        entry.d_ij = static_cast<float>(params.d_ij);
        entry.dp_ij = static_cast<float>(params.dp_ij);
    }
}

/** Extract covalent radii and Pauling electronegativities, sorted by atomic number. */
inline void extract_covrad_table_(std::vector<mmff94_covrad_entry_t> &out) {
    auto const *collection = RDKit::MMFF::DefaultParameters::getMMFFCovRadPauEle();
    std::size_t const count = collection->d_atomicNum.size();
    out.assign(count, mmff94_covrad_entry_t {});
    for (std::size_t entry_index = 0; entry_index < count; ++entry_index) {
        auto &entry = out[entry_index];
        auto const &params = collection->d_params[entry_index];
        entry.atomic_num = collection->d_atomicNum[entry_index];
        entry.covalent_radius = static_cast<float>(params.r0);
        entry.pauling_electronegativity = static_cast<float>(params.chi);
    }
}

#pragma endregion Sparse Tables

#pragma region Dense Tables

/**
 *  @brief Extract the MMFFDef equivalence levels into a table indexed directly by atom type.
 *
 *  @param[out] out Dense table, one entry per atom type in [0, max_atom_type], zeroed where unused.
 *  @param[in] max_atom_type Highest type the VdW table mentions; the table must reach at least that far
 *             because the angle, torsion and out-of-plane cascades index it by an arbitrary type.
 */
inline void extract_def_table_(std::vector<mmff94_def_entry_t> &out, std::uint8_t max_atom_type) {
    auto const *collection = RDKit::MMFF::DefaultParameters::getMMFFDef();
    std::size_t const count = collection->d_params.size();
    out.assign(std::max<std::size_t>(max_atom_type, count) + 1, mmff94_def_entry_t {});
    for (std::size_t type_index = 0; type_index < count; ++type_index) {
        auto const &params = collection->d_params[type_index];
        auto &entry = out[type_index + 1];
        for (unsigned level = 0; level < mmff94_equivalence_levels_k; ++level)
            entry.eq_level[level] = params.eqLevel[level];
    }
}

/**
 *  @brief Extract the MMFFProp per-atom-type properties into a table indexed directly by atom type.
 *
 *  @param[out] out Dense table, one entry per atom type in [0, max_atom_type], zeroed where unused.
 *  @param[in] max_atom_type Highest type the VdW table mentions; a property row past it is dropped
 *             rather than growing the table, since no lookup can reach it.
 */
inline void extract_prop_table_(std::vector<mmff94_prop_entry_t> &out, std::uint8_t max_atom_type) {
    auto const *collection = RDKit::MMFF::DefaultParameters::getMMFFProp();
    out.assign(static_cast<std::size_t>(max_atom_type) + 1, mmff94_prop_entry_t {});
    std::size_t const count = collection->d_iAtomType.size();
    for (std::size_t entry_index = 0; entry_index < count; ++entry_index) {
        std::uint8_t const atom_type = collection->d_iAtomType[entry_index];
        if (atom_type >= out.size()) continue;
        auto const &params = collection->d_params[entry_index];
        auto &entry = out[atom_type];
        entry.atomic_num = params.atno;
        entry.coordination_number = params.crd;
        entry.valence = params.val;
        entry.pi_lone_pair = params.pilp;
        entry.multiple_bond_designator = params.mltb;
        entry.aromatic = params.arom;
        entry.linear = params.linh;
        entry.single_bond_multiple_bond = params.sbmb;
    }
}

#pragma endregion Dense Tables

/**
 *  @brief Extract every MMFF94 parameter table from RDKit's defaults collection into flat sorted
 *  `std::vector`s. Only called once per process.
 *
 *  The VdW extraction runs first and hands its highest atom type to the two dense tables, so the one
 *  ordering constraint between the fourteen is a value the compiler carries rather than a comment.
 *
 *  @param[out] out Container filled from scratch; every table is assigned rather than appended.
 */
inline void extract_mmff94_tables_from_rdkit(mmff94_tables_host_t &out) {
    out.max_atom_type = extract_vdw_table_(out);
    extract_bond_table_(out.bond);
    extract_angle_table_(out.angle);
    extract_torsion_table_(out.torsion);
    extract_oop_table_(out.oop);
    extract_stbn_table_(out.stbn);
    extract_dfsb_table_(out.dfsb);
    extract_pbci_table_(out.pbci);
    extract_chg_table_(out.chg);
    extract_bndk_table_(out.bndk);
    extract_hl_table_(out.hl);
    extract_covrad_table_(out.covrad);
    extract_def_table_(out.def, out.max_atom_type);
    extract_prop_table_(out.prop, out.max_atom_type);
}

} // namespace smallfield::parameters
