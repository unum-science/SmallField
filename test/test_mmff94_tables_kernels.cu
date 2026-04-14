/**
 *  @file test_mmff94_tables_kernels.cu
 *  @brief MG-A device-side validation kernels for test_mmff94_tables.
 *
 *  Launches one device lookup per probe key (vdw / bond / angle / torsion
 *  / oop / stbn) and writes the resolved parameters into a device output
 *  buffer.  The host then bit-equals the device result against the
 *  host-side lookup over the same inputs.
 */

#include <cstdint>
#include <cuda_runtime.h>

// Exclude the RDKit-dependent `extract_mmff94_tables_from_rdkit` from
// the .cu compilation — it pulls in `<GraphMol/.../AtomTyper.h>` which
// the nvcc host pass cannot compile.  The extraction runs on the host
// side (compiled by g++) in the .cpp; here we only need the POD entry
// structs + the lookup functions.
#define SMALLFIELD_MMFF_TABLES_NO_RDKIT 1
#include "mmff94_tables.hpp"
#include "mmff94_tables_device.cuh"

namespace smallfield {

struct vdw_query_t { std::uint8_t atom_type; };
struct vdw_result_t {
    std::uint8_t hit; float alpha; float n_i; float a_i; float g_i; float r_star;
    std::uint8_t da;
};

struct bond_query_t { std::uint8_t bt; std::uint8_t i; std::uint8_t j; };
struct bond_result_t { std::uint8_t hit; float kb; float r0; };

struct angle_query_t { std::uint8_t at; std::uint8_t i; std::uint8_t j; std::uint8_t k; };
struct angle_result_t { std::uint8_t hit; float ka; float theta0_deg; };

struct torsion_query_t {
    std::uint8_t tt1; std::uint8_t tt2;
    std::uint8_t i; std::uint8_t j; std::uint8_t k; std::uint8_t l;
};
struct torsion_result_t { std::uint8_t hit; float v1; float v2; float v3; std::uint8_t canTT; };

struct oop_query_t { std::uint8_t i; std::uint8_t j; std::uint8_t k; std::uint8_t l; };
struct oop_result_t { std::uint8_t hit; float koop; };

struct stbn_query_t {
    std::uint8_t sbt; std::uint8_t bt1; std::uint8_t bt2;
    std::uint8_t i; std::uint8_t j; std::uint8_t k;
};
struct stbn_result_t { std::uint8_t hit; std::uint8_t swap; float kba_ijk; float kba_kji; };

__global__ void vdw_lookup_kernel(mmff94_tables_view_t view,
                                  vdw_query_t const * __restrict__ q,
                                  vdw_result_t * __restrict__ out,
                                  std::uint32_t n) {
    std::uint32_t const idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    auto const *hit = lookup_vdw(view, q[idx].atom_type);
    vdw_result_t r {};
    if (hit) {
        r.hit = 1; r.alpha = hit->alpha_i; r.n_i = hit->n_i;
        r.a_i = hit->a_i; r.g_i = hit->g_i; r.r_star = hit->r_star;
        r.da = hit->donor_acceptor;
    }
    out[idx] = r;
}

__global__ void bond_lookup_kernel(mmff94_tables_view_t view,
                                   bond_query_t const * __restrict__ q,
                                   bond_result_t * __restrict__ out,
                                   std::uint32_t n) {
    std::uint32_t const idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    auto const *hit = lookup_bond(view, q[idx].bt, q[idx].i, q[idx].j);
    bond_result_t r {};
    if (hit) { r.hit = 1; r.kb = hit->kb; r.r0 = hit->r0; }
    out[idx] = r;
}

__global__ void angle_lookup_kernel(mmff94_tables_view_t view,
                                    angle_query_t const * __restrict__ q,
                                    angle_result_t * __restrict__ out,
                                    std::uint32_t n) {
    std::uint32_t const idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    auto const *hit = lookup_angle(view, q[idx].at, q[idx].i, q[idx].j, q[idx].k);
    angle_result_t r {};
    if (hit) { r.hit = 1; r.ka = hit->ka; r.theta0_deg = hit->theta0_deg; }
    out[idx] = r;
}

__global__ void torsion_lookup_kernel(mmff94_tables_view_t view,
                                      torsion_query_t const * __restrict__ q,
                                      torsion_result_t * __restrict__ out,
                                      std::uint32_t n) {
    std::uint32_t const idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    auto const res = lookup_torsion(view, q[idx].tt1, q[idx].tt2,
                                     q[idx].i, q[idx].j, q[idx].k, q[idx].l);
    torsion_result_t r {};
    if (res.second) {
        r.hit = 1;
        r.v1 = res.second->v1; r.v2 = res.second->v2; r.v3 = res.second->v3;
        r.canTT = (std::uint8_t)res.first;
    }
    out[idx] = r;
}

__global__ void oop_lookup_kernel(mmff94_tables_view_t view,
                                  oop_query_t const * __restrict__ q,
                                  oop_result_t * __restrict__ out,
                                  std::uint32_t n) {
    std::uint32_t const idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    auto const *hit = lookup_oop(view, q[idx].i, q[idx].j, q[idx].k, q[idx].l);
    oop_result_t r {};
    if (hit) { r.hit = 1; r.koop = hit->koop; }
    out[idx] = r;
}

__global__ void stbn_lookup_kernel(mmff94_tables_view_t view,
                                   stbn_query_t const * __restrict__ q,
                                   stbn_result_t * __restrict__ out,
                                   std::uint32_t n) {
    std::uint32_t const idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    auto const res = lookup_stbn(view, q[idx].sbt, q[idx].bt1, q[idx].bt2,
                                  q[idx].i, q[idx].j, q[idx].k);
    stbn_result_t r {};
    if (res.second) {
        r.hit = 1; r.swap = res.first ? 1 : 0;
        r.kba_ijk = res.second->kba_ijk; r.kba_kji = res.second->kba_kji;
    }
    out[idx] = r;
}

// ---- Host-callable launchers -------------------------------------------

void launch_device_vdw_lookup(mmff94_tables_view_t view, vdw_query_t const *q_dev,
                              vdw_result_t *out_dev, std::uint32_t n) {
    vdw_lookup_kernel<<<(n + 127) / 128, 128>>>(view, q_dev, out_dev, n);
}
void launch_device_bond_lookup(mmff94_tables_view_t view, bond_query_t const *q_dev,
                               bond_result_t *out_dev, std::uint32_t n) {
    bond_lookup_kernel<<<(n + 127) / 128, 128>>>(view, q_dev, out_dev, n);
}
void launch_device_angle_lookup(mmff94_tables_view_t view, angle_query_t const *q_dev,
                                angle_result_t *out_dev, std::uint32_t n) {
    angle_lookup_kernel<<<(n + 127) / 128, 128>>>(view, q_dev, out_dev, n);
}
void launch_device_torsion_lookup(mmff94_tables_view_t view, torsion_query_t const *q_dev,
                                  torsion_result_t *out_dev, std::uint32_t n) {
    torsion_lookup_kernel<<<(n + 127) / 128, 128>>>(view, q_dev, out_dev, n);
}
void launch_device_oop_lookup(mmff94_tables_view_t view, oop_query_t const *q_dev,
                              oop_result_t *out_dev, std::uint32_t n) {
    oop_lookup_kernel<<<(n + 127) / 128, 128>>>(view, q_dev, out_dev, n);
}
void launch_device_stbn_lookup(mmff94_tables_view_t view, stbn_query_t const *q_dev,
                               stbn_result_t *out_dev, std::uint32_t n) {
    stbn_lookup_kernel<<<(n + 127) / 128, 128>>>(view, q_dev, out_dev, n);
}

void upload_tables_for_test(mmff94_tables_host_t const &host, mmff94_tables_device_t &out) {
    upload_mmff94_tables_to_device(host, out);
}

/** C-compatible lifecycle wrappers so the test .cpp (compiled by g++)
 *  never touches CUDA types directly.  `device_tables_create` allocates
 *  the arena and returns an opaque handle + view; `device_tables_destroy`
 *  frees it.  The view remains valid between these two calls. */
void *device_tables_create(mmff94_tables_host_t const &host,
                           mmff94_tables_view_t &out_view) {
    auto *dev = new mmff94_tables_device_t;
    upload_mmff94_tables_to_device(host, *dev);
    if (dev->arena == nullptr) {
        delete dev;
        return nullptr;
    }
    out_view = dev->view;
    return dev;
}

void device_tables_destroy(void *handle) {
    delete static_cast<mmff94_tables_device_t *>(handle);
}

} // namespace smallfield
