/**
 *  @file src/smallfield/parameters/mmff94_tables_rdkit.hpp
 *  @brief RDKit-dependent MMFF94 table extraction.
 *
 *  Separated from `mmff94_tables.hpp` so that CUDA translation units can include the POD structs and
 *  lookup functions without pulling in RDKit, which triggers nvcc internal errors via `boost::tti`.
 *  Only host-side `.cpp` files that call `extract_mmff94_tables_from_rdkit` need this header.
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

/**
 *  @brief Extract every MMFF94 parameter table from RDKit's defaults collection into flat sorted
 *  `std::vector`s. Only called once per process.
 */
inline void extract_mmff94_tables_from_rdkit(mmff94_tables_host_t &out) {
    // VdW.
    {
        auto const *vdw_collection = RDKit::MMFF::DefaultParameters::getMMFFVdW();
        out.vdw_power = static_cast<float>(vdw_collection->power);
        out.vdw_b = static_cast<float>(vdw_collection->B);
        out.vdw_beta = static_cast<float>(vdw_collection->Beta);
        out.vdw_darad = static_cast<float>(vdw_collection->DARAD);
        out.vdw_daeps = static_cast<float>(vdw_collection->DAEPS);
        std::size_t const count = vdw_collection->d_atomType.size();
        out.vdw.resize(count);
        for (std::size_t entry_index = 0; entry_index < count; ++entry_index) {
            auto &entry = out.vdw[entry_index];
            entry.atom_type = vdw_collection->d_atomType[entry_index];
            auto const &params = vdw_collection->d_params[entry_index];
            entry.alpha_i = static_cast<float>(params.alpha_i);
            entry.n_effective = static_cast<float>(params.N_i);
            entry.a_scale = static_cast<float>(params.A_i);
            entry.g_factor = static_cast<float>(params.G_i);
            entry.r_star = static_cast<float>(params.R_star);
            entry.donor_acceptor = (params.DA == 'D') ? 2u : (params.DA == 'A') ? 1u : 0u;
            entry.padding[0] = entry.padding[1] = 0;
            entry.padding_float = 0.0f;
            if (entry.atom_type > out.max_atom_type) out.max_atom_type = entry.atom_type;
        }
    }

    // Bond.
    {
        auto const *bond_collection = RDKit::MMFF::DefaultParameters::getMMFFBond();
        std::size_t const count = bond_collection->d_iAtomType.size();
        out.bond.resize(count);
        for (std::size_t entry_index = 0; entry_index < count; ++entry_index) {
            auto &entry = out.bond[entry_index];
            entry.i_type = bond_collection->d_iAtomType[entry_index];
            entry.j_type = bond_collection->d_jAtomType[entry_index];
            entry.bond_type = bond_collection->d_bondType[entry_index];
            entry.padding = 0;
            entry.force_constant = static_cast<float>(bond_collection->d_params[entry_index].kb);
            entry.equilibrium_distance = static_cast<float>(bond_collection->d_params[entry_index].r0);
            entry.padding_float = 0.0f;
        }
    }

    // Angle.
    {
        auto const *angle_collection = RDKit::MMFF::DefaultParameters::getMMFFAngle();
        std::size_t const count = angle_collection->d_iAtomType.size();
        out.angle.resize(count);
        for (std::size_t entry_index = 0; entry_index < count; ++entry_index) {
            auto &entry = out.angle[entry_index];
            entry.j_type = angle_collection->d_jAtomType[entry_index];
            entry.i_type = angle_collection->d_iAtomType[entry_index];
            entry.k_type = angle_collection->d_kAtomType[entry_index];
            entry.angle_type = angle_collection->d_angleType[entry_index];
            auto const &params = angle_collection->d_params[entry_index];
            entry.force_constant = static_cast<float>(params.ka);
            entry.theta0_deg = static_cast<float>(params.theta0);
            entry.padding_float = 0.0f;
        }
    }

    // Torsion.
    {
        auto const *torsion_collection = RDKit::MMFF::DefaultParameters::getMMFFTor(false);
        std::size_t const count = torsion_collection->d_iAtomType.size();
        out.torsion.resize(count);
        for (std::size_t entry_index = 0; entry_index < count; ++entry_index) {
            auto &entry = out.torsion[entry_index];
            entry.j_type = torsion_collection->d_jAtomType[entry_index];
            entry.k_type = torsion_collection->d_kAtomType[entry_index];
            entry.i_type = torsion_collection->d_iAtomType[entry_index];
            entry.l_type = torsion_collection->d_lAtomType[entry_index];
            entry.tor_type = torsion_collection->d_torType[entry_index];
            entry.padding[0] = entry.padding[1] = entry.padding[2] = 0;
            auto const &params = torsion_collection->d_params[entry_index];
            entry.torsion_v1 = static_cast<float>(params.V1);
            entry.torsion_v2 = static_cast<float>(params.V2);
            entry.torsion_v3 = static_cast<float>(params.V3);
        }
    }

    // Out-of-plane.
    {
        auto const *oop_collection = RDKit::MMFF::DefaultParameters::getMMFFOop(false);
        std::size_t const count = oop_collection->d_iAtomType.size();
        out.oop.resize(count);
        for (std::size_t entry_index = 0; entry_index < count; ++entry_index) {
            auto &entry = out.oop[entry_index];
            entry.j_type = oop_collection->d_jAtomType[entry_index];
            entry.i_type = oop_collection->d_iAtomType[entry_index];
            entry.k_type = oop_collection->d_kAtomType[entry_index];
            entry.l_type = oop_collection->d_lAtomType[entry_index];
            entry.force_constant = static_cast<float>(oop_collection->d_params[entry_index].koop);
        }
    }

    // Stretch-bend.
    {
        auto const *stbn_collection = RDKit::MMFF::DefaultParameters::getMMFFStbn();
        std::size_t const count = stbn_collection->d_iAtomType.size();
        out.stbn.resize(count);
        for (std::size_t entry_index = 0; entry_index < count; ++entry_index) {
            auto &entry = out.stbn[entry_index];
            entry.j_type = stbn_collection->d_jAtomType[entry_index];
            entry.i_type = stbn_collection->d_iAtomType[entry_index];
            entry.k_type = stbn_collection->d_kAtomType[entry_index];
            entry.stretch_bend_type = stbn_collection->d_stretchBendType[entry_index];
            entry.kba_ijk = static_cast<float>(stbn_collection->d_params[entry_index].kbaIJK);
            entry.kba_kji = static_cast<float>(stbn_collection->d_params[entry_index].kbaKJI);
            entry.padding_float = 0.0f;
        }
    }

    // DFSB.
    {
        auto const *dfsb_collection = RDKit::MMFF::DefaultParameters::getMMFFDfsb();
        for (auto const &[row_i, by_row_j] : dfsb_collection->d_params) {
            for (auto const &[row_j, by_row_k] : by_row_j) {
                for (auto const &[row_k, params] : by_row_k) {
                    mmff94_dfsb_entry_t entry;
                    entry.row_i = static_cast<std::uint8_t>(row_i);
                    entry.row_j = static_cast<std::uint8_t>(row_j);
                    entry.row_k = static_cast<std::uint8_t>(row_k);
                    entry.padding = 0;
                    entry.kba_ijk = static_cast<float>(params.kbaIJK);
                    entry.kba_kji = static_cast<float>(params.kbaKJI);
                    out.dfsb.push_back(entry);
                }
            }
        }
        std::sort(out.dfsb.begin(), out.dfsb.end(), [](auto const &left, auto const &right) {
            if (left.row_i != right.row_i) return left.row_i < right.row_i;
            if (left.row_j != right.row_j) return left.row_j < right.row_j;
            return left.row_k < right.row_k;
        });
    }

    // MMFFDef.
    {
        auto const *def_collection = RDKit::MMFF::DefaultParameters::getMMFFDef();
        std::size_t const count = def_collection->d_params.size();
        std::size_t const dense_count = std::max<std::size_t>(out.max_atom_type, static_cast<std::uint8_t>(count)) + 1;
        out.def.assign(dense_count, mmff94_def_entry_t {});
        for (std::size_t type_index = 0; type_index < count; ++type_index) {
            auto const &def_params = def_collection->d_params[type_index];
            std::size_t const atom_type = type_index + 1;
            out.def[atom_type].eq_level[0] = def_params.eqLevel[0];
            out.def[atom_type].eq_level[1] = def_params.eqLevel[1];
            out.def[atom_type].eq_level[2] = def_params.eqLevel[2];
            out.def[atom_type].eq_level[3] = def_params.eqLevel[3];
        }
    }

    // PBCI.
    {
        auto const *pbci_collection = RDKit::MMFF::DefaultParameters::getMMFFPBCI();
        std::size_t const count = pbci_collection->d_params.size();
        out.pbci.reserve(count);
        for (std::size_t type_index = 0; type_index < count; ++type_index) {
            mmff94_pbci_entry_t entry {};
            entry.atom_type = static_cast<std::uint8_t>(type_index + 1);
            entry.pbci = static_cast<float>(pbci_collection->d_params[type_index].pbci);
            entry.fcadj = static_cast<float>(pbci_collection->d_params[type_index].fcadj);
            out.pbci.push_back(entry);
        }
    }

    // Chg.
    {
        auto const *chg_collection = RDKit::MMFF::DefaultParameters::getMMFFChg();
        std::size_t const count = chg_collection->d_iAtomType.size();
        out.chg.resize(count);
        for (std::size_t entry_index = 0; entry_index < count; ++entry_index) {
            auto &entry = out.chg[entry_index];
            entry.i_type = chg_collection->d_iAtomType[entry_index];
            entry.j_type = chg_collection->d_jAtomType[entry_index];
            entry.bond_type = chg_collection->d_bondType[entry_index];
            entry.padding = 0;
            entry.bond_charge_increment = static_cast<float>(chg_collection->d_params[entry_index].bci);
        }
    }

    // Bndk.
    {
        auto const *bndk_collection = RDKit::MMFF::DefaultParameters::getMMFFBndk();
        std::size_t const count = bndk_collection->d_iAtomicNum.size();
        out.bndk.resize(count);
        for (std::size_t entry_index = 0; entry_index < count; ++entry_index) {
            auto &entry = out.bndk[entry_index];
            entry.atomic_num_i = bndk_collection->d_iAtomicNum[entry_index];
            entry.atomic_num_j = bndk_collection->d_jAtomicNum[entry_index];
            entry.padding[0] = entry.padding[1] = 0;
            entry.force_constant = static_cast<float>(bndk_collection->d_params[entry_index].kb);
            entry.equilibrium_distance = static_cast<float>(bndk_collection->d_params[entry_index].r0);
        }
    }

    // Herschbach-Laurie.
    {
        auto const *herschbach_laurie_collection = RDKit::MMFF::DefaultParameters::getMMFFHerschbachLaurie();
        std::size_t const count = herschbach_laurie_collection->d_iRow.size();
        out.hl.resize(count);
        for (std::size_t entry_index = 0; entry_index < count; ++entry_index) {
            auto &entry = out.hl[entry_index];
            entry.row_i = herschbach_laurie_collection->d_iRow[entry_index];
            entry.row_j = herschbach_laurie_collection->d_jRow[entry_index];
            entry.padding[0] = entry.padding[1] = 0;
            entry.a_ij = static_cast<float>(herschbach_laurie_collection->d_params[entry_index].a_ij);
            entry.d_ij = static_cast<float>(herschbach_laurie_collection->d_params[entry_index].d_ij);
            entry.dp_ij = static_cast<float>(herschbach_laurie_collection->d_params[entry_index].dp_ij);
        }
    }

    // CovRadPauEle.
    {
        auto const *covrad_collection = RDKit::MMFF::DefaultParameters::getMMFFCovRadPauEle();
        std::size_t const count = covrad_collection->d_atomicNum.size();
        out.covrad.resize(count);
        for (std::size_t entry_index = 0; entry_index < count; ++entry_index) {
            auto &entry = out.covrad[entry_index];
            entry.atomic_num = covrad_collection->d_atomicNum[entry_index];
            entry.padding[0] = entry.padding[1] = entry.padding[2] = 0;
            entry.covalent_radius = static_cast<float>(covrad_collection->d_params[entry_index].r0);
            entry.pauling_electronegativity = static_cast<float>(covrad_collection->d_params[entry_index].chi);
        }
    }

    // MMFFProp.
    {
        auto const *prop_collection = RDKit::MMFF::DefaultParameters::getMMFFProp();
        std::size_t const dense_count = static_cast<std::size_t>(out.max_atom_type) + 1;
        out.prop.assign(dense_count, mmff94_prop_entry_t {});
        std::size_t const count = prop_collection->d_iAtomType.size();
        for (std::size_t entry_index = 0; entry_index < count; ++entry_index) {
            std::uint8_t const atom_type = prop_collection->d_iAtomType[entry_index];
            if (atom_type >= dense_count) continue;
            auto const &params = prop_collection->d_params[entry_index];
            auto &entry = out.prop[atom_type];
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
}

} // namespace smallfield::parameters
