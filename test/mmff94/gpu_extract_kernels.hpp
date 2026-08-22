/**
 *  @file test/mmff94/test_mmff_gpu_extract_kernels.hpp
 *  @brief Device harness for the MMFF94 GPU term-extraction pipeline, for use by its unit tests.
 */

#pragma once

#include <cstdint> // `std::uint32_t`, `std::uint8_t`

#include <string_view> // `std::string_view`
#include <vector>      // `std::vector`

#include "smallfield/mmff94/mmff94_extract_device.cuh"
#include "smallfield/parameters/mmff94_tables.hpp"
#include "smallfield/types/etkdg_types.hpp"
#include "smallfield/types/molecule_descriptor.hpp"
#include "smallfield/types/views.hpp"

namespace smallfield::tests {

/** Which step of the extraction round trip refused, so a caller can say why it compared nothing. Zero is the
 *  typer's own verdict for a molecule it accepted, and no path here writes the status array before the kernels do,
 *  so a caller reading it after a refusal is told every molecule was typed and scores them all against the CPU
 *  extractor's real answer. */
enum class extract_probe_t {
    completed_k,
    table_upload_failed_k,
    allocation_failed_k,
    transfer_failed_k,
    launch_failed_k,
};

/** The sentence an `extract_probe_t` reads as, for the `NOTE` line above a skipped comparison. */
std::string_view extract_probe_reason(extract_probe_t outcome) noexcept;

/** Host mirror of the packed, variable-length MMFF term arrays the compaction pass emits. Each vector is sized from
 *  the batch totals the harness sums out of the per-molecule counts, so the aggregate is a wave's real device
 *  footprint rather than the fixed-stride staging one. */
struct mmff94_packed_host_result_t {
    std::vector<smallfield::types::mmff94_bond_t> bonds;
    std::vector<smallfield::types::mmff94_angle_t> angles;
    std::vector<smallfield::types::mmff94_torsion_t> torsions;
    std::vector<smallfield::types::mmff94_oop_t> oops;
    std::vector<smallfield::types::mmff94_stretchbend_t> stretchbends;
    std::vector<smallfield::types::mmff94_atom_interaction_t> interactions;
    std::vector<smallfield::types::mmff94_atom_offsets_t> atom_offsets;
    std::vector<smallfield::types::etkdg_molecule_metadata_t> metadata;
};

/** Type, extract, build the CSR, and compact one batch on the device, then copy it all back. Every staging array is
 *  fixed-stride, so each output span states the stride by its own extent and no count travels beside it:
 *  `out_bonds` spans `max_mmff_bonds_per_mol_k` slots per descriptor, `out_atom_offsets` one slot per atom, which
 *  is what carries the batch's atom total.
 *  @param[in] tables_host MMFF94 parameter tables, uploaded here.
 *  @param[in] atom_offsets Each molecule's first atom in the batch-global atom array.
 *  @param[out] out_counts Receives the per-molecule term counts.
 *  @param[out] out_status Receives one typing status per molecule.
 *  @param[out] out_atom_offsets Receives one CSR offset record per atom of the batch.
 *  @param[out] packed_out Receives the compacted arrays, resized to the totals the counts sum to.
 *  @return Whether the round trip completed, and which step refused it. */
[[nodiscard]] extract_probe_t launch_mmff94_gpu_extract_pipeline(
    smallfield::parameters::mmff94_tables_host_t const &tables_host,
    custd::span<smallfield::types::molecule_descriptor_t const> descriptors,
    custd::span<std::uint32_t const> atom_offsets, custd::span<smallfield::mmff94::mmff94_per_mol_counts_t> out_counts,
    custd::span<std::uint8_t> out_status, custd::span<smallfield::types::mmff94_bond_t> out_bonds,
    custd::span<smallfield::types::mmff94_angle_t> out_angles,
    custd::span<smallfield::types::mmff94_torsion_t> out_torsions,
    custd::span<smallfield::types::mmff94_oop_t> out_oops,
    custd::span<smallfield::types::mmff94_stretchbend_t> out_stbns,
    custd::span<smallfield::types::mmff94_atom_interaction_t> out_interactions,
    custd::span<smallfield::types::mmff94_atom_offsets_t> out_atom_offsets, mmff94_packed_host_result_t &packed_out);

} // namespace smallfield::tests
