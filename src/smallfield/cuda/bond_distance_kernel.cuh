/**
 *  @file src/smallfield/cuda/bond_distance_kernel.cuh
 *  @brief Warp-parallel all-pairs bond-hop matrix over the batch pool's bond list.
 *
 *  The MMFF nonbonded sweep enumerates pairs on the fly and needs the hop count between
 *  every atom pair: 1-2 and 1-3 are excluded, 1-4 is scaled, everything further apart is
 *  a full interaction, and an unreachable pair — the two halves of a salt — is a full
 *  interaction too. That matrix is N x N bytes per molecule, sixteen kilobytes at the
 *  128-atom cap, against the two bytes per bond it is derived from, so it is cheaper
 *  to rebuild it on the device than to ship it across PCIe.
 *
 *  The bond list comes from the molecule descriptor rather than from the MMFF bond terms, so this
 *  pass depends on nothing but the upload and can run while the MMFF extractor is still working.
 *
 *  One warp owns one molecule and one lane owns one source atom, striding by the warp
 *  width when the molecule has more than 32 atoms. A lane keeps its visited and frontier
 *  sets as four 32-bit words covering the 128-atom cap and expands a level by sweeping the
 *  molecule's bond list once: every lane of the warp reads the same bond, so the sweep is a
 *  broadcast out of L1 rather than 32 independent streams. Bond counts are small — a
 *  drug-like molecule with explicit hydrogens carries well under 160 — and the level count
 *  is the graph's eccentricity, so the sweep costs far less than the N x N stores it feeds.
 */

#pragma once

#if defined(__CUDACC__)

#include <cstdint> // `std::uint32_t`, `std::uint8_t`

#include <cuda_runtime.h>

#include "smallfield/cuda/warp_primitives.cuh"
#include "smallfield/types/etkdg_types.hpp"
#include "smallfield/types/molecule_descriptor.hpp"

namespace smallfield::cuda {

using smallfield::types::etkdg_molecule_metadata_t;
using smallfield::types::max_atoms_per_molecule_k;
using smallfield::types::molecule_descriptor_t;

/** 32-bit words needed to hold one bit per atom at the `max_atoms_per_molecule_k` cap. */
inline constexpr std::uint32_t bond_distance_mask_words_k = max_atoms_per_molecule_k / 32u;

/** Hop count written for a pair with no bonded path between its atoms. */
inline constexpr std::uint8_t bond_distance_unreachable_k = types::unreachable_hops_k;

/**
 *  @brief Fill the batch pool's per-molecule N x N hop-count blocks.
 *
 *  Reads the molecule's own bond list out of its descriptor and writes `num_atoms` squared
 *  bytes at `bond_distance_offset`, packed with no padding stride: row `i` of molecule
 *  `m` starts at `bond_distance_offset + i * num_atoms`.
 *
 *  @param[in] descriptors Batch-wide molecule descriptors; only the bond atom pairs are read.
 *  @param[in] metadata One entry per molecule, carrying both region offsets and the atom count.
 *  @param[out] bond_distance Batch-wide hop-count region, one N x N block per molecule.
 *  @param[in] molecule_count Number of molecules in @p metadata.
 *
 *  Internal linkage: every translation unit that includes this header gets its own copy, so a
 *  test that compiles the header and also links the library does not hand nvlink two definitions
 *  of one device symbol.
 */
static __global__ void bond_distance_bfs_kernel(molecule_descriptor_t const *__restrict__ descriptors,
                                                etkdg_molecule_metadata_t const *__restrict__ metadata,
                                                std::uint8_t *__restrict__ bond_distance,
                                                std::uint32_t molecule_count) {
    std::uint32_t const lane = threadIdx.x % warp_size_k;
    std::uint32_t const warps_per_block = blockDim.x / warp_size_k;
    std::uint32_t const warp_stride = warps_per_block * gridDim.x;
    std::uint32_t const first_warp = blockIdx.x * warps_per_block + threadIdx.x / warp_size_k;

    for (std::uint32_t molecule = first_warp; molecule < molecule_count; molecule += warp_stride) {
        etkdg_molecule_metadata_t const meta = metadata[molecule];
        std::uint32_t const num_atoms = meta.num_atoms;
        if (num_atoms == 0) continue;
        std::uint8_t *const block = bond_distance + meta.bond_distance_offset;
        molecule_descriptor_t const &descriptor = descriptors[molecule];
        std::uint32_t const bond_count = descriptor.num_bonds;

        // Every pair starts unreachable; the sweeps below overwrite the ones a path reaches.
        for (std::uint32_t entry = lane; entry < num_atoms * num_atoms; entry += warp_size_k)
            block[entry] = bond_distance_unreachable_k;
        __syncwarp();

        std::uint32_t const words = (num_atoms + 31u) / 32u;
        for (std::uint32_t source = lane; source < num_atoms; source += warp_size_k) {
            std::uint8_t *const row = block + source * num_atoms;
            row[source] = 0;

            std::uint32_t visited[bond_distance_mask_words_k] = {0u, 0u, 0u, 0u};
            std::uint32_t frontier[bond_distance_mask_words_k] = {0u, 0u, 0u, 0u};
            visited[source / 32u] = 1u << (source % 32u);
            frontier[source / 32u] = visited[source / 32u];

            for (std::uint32_t depth = 1; depth < num_atoms; ++depth) {
                std::uint32_t next[bond_distance_mask_words_k] = {0u, 0u, 0u, 0u};
                for (std::uint32_t bond = 0; bond < bond_count; ++bond) {
                    std::uint32_t const atom_i = descriptor.bond_atom_i[bond];
                    std::uint32_t const atom_j = descriptor.bond_atom_j[bond];
                    bool const reached_i = (frontier[atom_i / 32u] >> (atom_i % 32u)) & 1u;
                    bool const reached_j = (frontier[atom_j / 32u] >> (atom_j % 32u)) & 1u;
                    if (reached_i) next[atom_j / 32u] |= 1u << (atom_j % 32u);
                    if (reached_j) next[atom_i / 32u] |= 1u << (atom_i % 32u);
                }

                std::uint32_t reached_any = 0;
                for (std::uint32_t word = 0; word < words; ++word) {
                    next[word] &= ~visited[word];
                    visited[word] |= next[word];
                    frontier[word] = next[word];
                    reached_any |= next[word];
                }
                if (reached_any == 0) break;

                std::uint8_t const value = depth < bond_distance_unreachable_k ? static_cast<std::uint8_t>(depth)
                                                                               : bond_distance_unreachable_k;
                for (std::uint32_t word = 0; word < words; ++word) {
                    std::uint32_t bits = next[word];
                    while (bits != 0) {
                        std::uint32_t const bit_index = __ffs(static_cast<int>(bits)) - 1;
                        bits &= bits - 1u;
                        row[word * 32u + bit_index] = value;
                    }
                }
            }
        }
        __syncwarp();
    }
}

/**
 *  @brief Enqueue `bond_distance_bfs_kernel` over a whole batch.
 *
 *  @param[in] descriptors Device pointer to the batch's molecule-descriptor region.
 *  @param[in] metadata Device pointer to the batch's per-molecule metadata region.
 *  @param[out] bond_distance Device pointer to the batch's hop-count region.
 *  @param[in] molecule_count Number of molecules in the batch.
 *  @param[in] stream Stream to enqueue on.
 */
inline void launch_bond_distance_bfs(molecule_descriptor_t const *descriptors,
                                     etkdg_molecule_metadata_t const *metadata, std::uint8_t *bond_distance,
                                     std::uint32_t molecule_count, cudaStream_t stream = nullptr) {
    if (molecule_count == 0) return;
    constexpr std::uint32_t block_threads_k = 128;
    constexpr std::uint32_t warps_per_block_k = block_threads_k / warp_size_k;
    // One block per four molecules saturates a wave-sized batch without launching empty warps
    // for the tail; the grid-stride loop in the kernel covers whatever the cap leaves over.
    std::uint32_t const warp_blocks = (molecule_count + warps_per_block_k - 1u) / warps_per_block_k;
    std::uint32_t const blocks = warp_blocks < 4096u ? warp_blocks : 4096u;
    bond_distance_bfs_kernel<<<blocks, block_threads_k, 0, stream>>>(descriptors, metadata, bond_distance,
                                                                     molecule_count);
}

} // namespace smallfield::cuda

#endif // __CUDACC__
