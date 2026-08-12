/**
 *  @file test/cuda/test_bounds_infeasible_kernels.hpp
 *  @brief Device harness driving the metric prepass and the pipeline kernel over one bounds matrix.
 */

#pragma once

#include "smallfield/types/conformer_status.hpp"
#include "smallfield/types/views.hpp"

namespace smallfield::tests {

/**
 *  @brief Embed and refine one hand-built bounds matrix, and report the status the three launches left.
 *
 *  Runs `build_metric_initial_coordinates_on_gpu`, `etkdg_pipeline_warp_kernel` and
 *  `stamp_refused_molecules_kernel` exactly as a wave does, over a single molecule of one conformer carrying
 *  no chiral centers and no torsion terms — so the only verdict any of them can reach is the one the bounds
 *  matrix earns. The status slot starts at `mmff_unconverged_k`, which is what the caller reads back when
 *  nothing wrote it.
 *
 *  @param[inout] bounds Square bounds matrix over the molecule's atoms, smoothed in place by the prepass.
 *  @return The conformer slot's status, in the same numbering the device wrote.
 */
smallfield::types::conformer_status_t gpu_pipeline_status_of_bounds(smallfield::types::packed_matrix<float> bounds);

} // namespace smallfield::tests
