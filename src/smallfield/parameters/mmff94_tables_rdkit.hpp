/**
 *  @file mmff94_tables_rdkit.hpp
 *  @brief RDKit-dependent MMFF94 table extraction.
 *
 *  Separated from mmff94_tables.hpp so that CUDA translation units can
 *  include the POD structs and lookup functions without pulling in RDKit
 *  (which triggers nvcc internal errors via boost::tti).
 *  Only host-side .cpp files that call extract_mmff94_tables_from_rdkit()
 *  need this header.
 */

#pragma once

#include "smallfield/parameters/mmff94_tables.hpp"

#include <ForceField/MMFF/Params.h>
#include <GraphMol/ForceFieldHelpers/MMFF/AtomTyper.h>

namespace RDKit {
namespace MMFF {
namespace DefaultParameters {
ForceFields::MMFF::MMFFBondCollection const *getMMFFBond();
ForceFields::MMFF::MMFFChgCollection const *getMMFFChg();
ForceFields::MMFF::MMFFBndkCollection const *getMMFFBndk();
ForceFields::MMFF::MMFFCovRadPauEleCollection const *getMMFFCovRadPauEle();
} // namespace DefaultParameters
} // namespace MMFF
} // namespace RDKit

namespace smallfield::parameters {

/**
 *  Extract every MMFF94 parameter table from RDKit's defaults collection
 *  into flat sorted std::vectors.  Only called once per process.
 */
inline void extract_mmff94_tables_from_rdkit(mmff94_tables_host_t &out) {
    namespace MMFF = ForceFields::MMFF;

    // VdW.
    {
        auto const *vc = RDKit::MMFF::DefaultParameters::getMMFFVdW();
        out.vdw_power = static_cast<float>(vc->power);
        out.vdw_b = static_cast<float>(vc->B);
        out.vdw_beta = static_cast<float>(vc->Beta);
        out.vdw_darad = static_cast<float>(vc->DARAD);
        out.vdw_daeps = static_cast<float>(vc->DAEPS);
        std::size_t const n = vc->d_atomType.size();
        out.vdw.resize(n);
        for (std::size_t k = 0; k < n; ++k) {
            auto &e = out.vdw[k];
            e.atom_type = vc->d_atomType[k];
            auto const &p = vc->d_params[k];
            e.alpha_i = static_cast<float>(p.alpha_i);
            e.n_i = static_cast<float>(p.N_i);
            e.a_i = static_cast<float>(p.A_i);
            e.g_i = static_cast<float>(p.G_i);
            e.r_star = static_cast<float>(p.R_star);
            e.donor_acceptor = (p.DA == 'D') ? 2u : (p.DA == 'A') ? 1u : 0u;
            e.padding[0] = e.padding[1] = 0;
            e.padding_float = 0.0f;
            if (e.atom_type > out.max_atom_type) out.max_atom_type = e.atom_type;
        }
    }

    // Bond.
    {
        auto const *bc = RDKit::MMFF::DefaultParameters::getMMFFBond();
        std::size_t const n = bc->d_iAtomType.size();
        out.bond.resize(n);
        for (std::size_t k = 0; k < n; ++k) {
            auto &e = out.bond[k];
            e.i_type = bc->d_iAtomType[k];
            e.j_type = bc->d_jAtomType[k];
            e.bond_type = bc->d_bondType[k];
            e.padding = 0;
            e.kb = static_cast<float>(bc->d_params[k].kb);
            e.r0 = static_cast<float>(bc->d_params[k].r0);
            e.padding_float = 0.0f;
        }
    }

    // Angle.
    {
        auto const *ac = RDKit::MMFF::DefaultParameters::getMMFFAngle();
        std::size_t const n = ac->d_iAtomType.size();
        out.angle.resize(n);
        for (std::size_t k = 0; k < n; ++k) {
            auto &e = out.angle[k];
            e.j_type = ac->d_jAtomType[k];
            e.i_type = ac->d_iAtomType[k];
            e.k_type = ac->d_kAtomType[k];
            e.angle_type = ac->d_angleType[k];
            auto const &p = ac->d_params[k];
            e.ka = static_cast<float>(p.ka);
            e.theta0_deg = static_cast<float>(p.theta0);
            e.padding_float = 0.0f;
        }
    }

    // Torsion.
    {
        auto const *type_counts = RDKit::MMFF::DefaultParameters::getMMFFTor(false);
        std::size_t const n = type_counts->d_iAtomType.size();
        out.torsion.resize(n);
        for (std::size_t k = 0; k < n; ++k) {
            auto &e = out.torsion[k];
            e.j_type = type_counts->d_jAtomType[k];
            e.k_type = type_counts->d_kAtomType[k];
            e.i_type = type_counts->d_iAtomType[k];
            e.l_type = type_counts->d_lAtomType[k];
            e.tor_type = type_counts->d_torType[k];
            e.padding[0] = e.padding[1] = e.padding[2] = 0;
            auto const &p = type_counts->d_params[k];
            e.v1 = static_cast<float>(p.V1);
            e.v2 = static_cast<float>(p.V2);
            e.v3 = static_cast<float>(p.V3);
        }
    }

    // Out-of-plane.
    {
        auto const *oc = RDKit::MMFF::DefaultParameters::getMMFFOop(false);
        std::size_t const n = oc->d_iAtomType.size();
        out.oop.resize(n);
        for (std::size_t k = 0; k < n; ++k) {
            auto &e = out.oop[k];
            e.j_type = oc->d_jAtomType[k];
            e.i_type = oc->d_iAtomType[k];
            e.k_type = oc->d_kAtomType[k];
            e.l_type = oc->d_lAtomType[k];
            e.koop = static_cast<float>(oc->d_params[k].koop);
        }
    }

    // Stretch-bend.
    {
        auto const *sc = RDKit::MMFF::DefaultParameters::getMMFFStbn();
        std::size_t const n = sc->d_iAtomType.size();
        out.stbn.resize(n);
        for (std::size_t k = 0; k < n; ++k) {
            auto &e = out.stbn[k];
            e.j_type = sc->d_jAtomType[k];
            e.i_type = sc->d_iAtomType[k];
            e.k_type = sc->d_kAtomType[k];
            e.stretch_bend_type = sc->d_stretchBendType[k];
            e.kba_ijk = static_cast<float>(sc->d_params[k].kbaIJK);
            e.kba_kji = static_cast<float>(sc->d_params[k].kbaKJI);
            e.padding_float = 0.0f;
        }
    }

    // DFSB.
    {
        auto const *dc = RDKit::MMFF::DefaultParameters::getMMFFDfsb();
        for (auto const &[row_i, inner1] : dc->d_params) {
            for (auto const &[row_j, inner2] : inner1) {
                for (auto const &[row_k, p] : inner2) {
                    mmff94_dfsb_entry_t e;
                    e.row_i = static_cast<std::uint8_t>(row_i);
                    e.row_j = static_cast<std::uint8_t>(row_j);
                    e.row_k = static_cast<std::uint8_t>(row_k);
                    e.padding = 0;
                    e.kba_ijk = static_cast<float>(p.kbaIJK);
                    e.kba_kji = static_cast<float>(p.kbaKJI);
                    out.dfsb.push_back(e);
                }
            }
        }
        std::sort(out.dfsb.begin(), out.dfsb.end(), [](auto const &a, auto const &b) {
            if (a.row_i != b.row_i) return a.row_i < b.row_i;
            if (a.row_j != b.row_j) return a.row_j < b.row_j;
            return a.row_k < b.row_k;
        });
    }

    // MMFFDef.
    {
        auto const *dc = RDKit::MMFF::DefaultParameters::getMMFFDef();
        std::size_t const n = dc->d_params.size();
        std::size_t const dense_n = std::max<std::size_t>(out.max_atom_type, static_cast<std::uint8_t>(n)) + 1;
        out.def.assign(dense_n, mmff94_def_entry_t {});
        for (std::size_t k = 0; k < n; ++k) {
            auto const &d = dc->d_params[k];
            std::size_t const at = k + 1;
            out.def[at].eq_level[0] = d.eqLevel[0];
            out.def[at].eq_level[1] = d.eqLevel[1];
            out.def[at].eq_level[2] = d.eqLevel[2];
            out.def[at].eq_level[3] = d.eqLevel[3];
        }
    }

    // PBCI.
    {
        auto const *pc = RDKit::MMFF::DefaultParameters::getMMFFPBCI();
        std::size_t const n = pc->d_params.size();
        out.pbci.reserve(n);
        for (std::size_t k = 0; k < n; ++k) {
            mmff94_pbci_entry_t e {};
            e.atom_type = static_cast<std::uint8_t>(k + 1);
            e.pbci = static_cast<float>(pc->d_params[k].pbci);
            e.fcadj = static_cast<float>(pc->d_params[k].fcadj);
            out.pbci.push_back(e);
        }
    }

    // Chg.
    {
        auto const *cc = RDKit::MMFF::DefaultParameters::getMMFFChg();
        std::size_t const n = cc->d_iAtomType.size();
        out.chg.resize(n);
        for (std::size_t k = 0; k < n; ++k) {
            auto &e = out.chg[k];
            e.i_type = cc->d_iAtomType[k];
            e.j_type = cc->d_jAtomType[k];
            e.bond_type = cc->d_bondType[k];
            e.padding = 0;
            e.bci = static_cast<float>(cc->d_params[k].bci);
        }
    }

    // Bndk.
    {
        auto const *bc = RDKit::MMFF::DefaultParameters::getMMFFBndk();
        std::size_t const n = bc->d_iAtomicNum.size();
        out.bndk.resize(n);
        for (std::size_t k = 0; k < n; ++k) {
            auto &e = out.bndk[k];
            e.atomic_num_i = bc->d_iAtomicNum[k];
            e.atomic_num_j = bc->d_jAtomicNum[k];
            e.padding[0] = e.padding[1] = 0;
            e.kb = static_cast<float>(bc->d_params[k].kb);
            e.r0 = static_cast<float>(bc->d_params[k].r0);
        }
    }

    // Herschbach-Laurie.
    {
        auto const *hc = RDKit::MMFF::DefaultParameters::getMMFFHerschbachLaurie();
        std::size_t const n = hc->d_iRow.size();
        out.hl.resize(n);
        for (std::size_t k = 0; k < n; ++k) {
            auto &e = out.hl[k];
            e.row_i = hc->d_iRow[k];
            e.row_j = hc->d_jRow[k];
            e.padding[0] = e.padding[1] = 0;
            e.a_ij = static_cast<float>(hc->d_params[k].a_ij);
            e.d_ij = static_cast<float>(hc->d_params[k].d_ij);
            e.dp_ij = static_cast<float>(hc->d_params[k].dp_ij);
        }
    }

    // CovRadPauEle.
    {
        auto const *cc = RDKit::MMFF::DefaultParameters::getMMFFCovRadPauEle();
        std::size_t const n = cc->d_atomicNum.size();
        out.covrad.resize(n);
        for (std::size_t k = 0; k < n; ++k) {
            auto &e = out.covrad[k];
            e.atomic_num = cc->d_atomicNum[k];
            e.padding[0] = e.padding[1] = e.padding[2] = 0;
            e.r0 = static_cast<float>(cc->d_params[k].r0);
            e.chi = static_cast<float>(cc->d_params[k].chi);
        }
    }

    // MMFFProp.
    {
        auto const *pc = RDKit::MMFF::DefaultParameters::getMMFFProp();
        std::size_t const dense_n = static_cast<std::size_t>(out.max_atom_type) + 1;
        out.prop.assign(dense_n, mmff94_prop_entry_t {});
        std::size_t const n = pc->d_iAtomType.size();
        for (std::size_t k = 0; k < n; ++k) {
            std::uint8_t const at = pc->d_iAtomType[k];
            if (at >= dense_n) continue;
            auto const &p = pc->d_params[k];
            auto &e = out.prop[at];
            e.atomic_num = p.atno;
            e.crd = p.crd;
            e.val = p.val;
            e.pilp = p.pilp;
            e.mltb = p.mltb;
            e.arom = p.arom;
            e.linh = p.linh;
            e.sbmb = p.sbmb;
        }
    }
}

} // namespace smallfield::parameters
