/**
 *  @file mace_tk_tile.cuh
 *  @brief Thin wrapper over ThunderKittens for the MACE-OFF inference module.
 *
 *  ThunderKittens (TK) is a tile-based register/shared-memory abstraction for
 *  Hopper and Blackwell tensor cores.  We use it as the single MMA library
 *  for MACE-OFF inference: the radial MLP, every per-irrep linear, the
 *  fused conv_tp + scatter, the element-conditioned residual, the symmetric
 *  contraction, and the readouts all go through TK kernels.  See
 *  `MACE_INFERENCE.md` for the rationale.
 *
 *  This header isolates TK behind project-flavoured aliases.  The rest of
 *  the MACE module includes this file (not `<kittens.cuh>` directly), so:
 *
 *    - if TK upstream renames a type or moves a header we patch one place,
 *    - if a future iteration replaces TK with cuBLASLt or CUTLASS the call
 *      sites stay stable (tile and op aliases are decoupled from the impl),
 *    - the `KITTENS_HOPPER` / `KITTENS_BLACKWELL` arch flag is selected
 *      automatically based on the CUDA arch at compile time.
 *
 *  Pinned ThunderKittens commit (validated against the local compile path):
 *      0b55588d2769dde0c7a9a606ffc015b0ca7a9551
 *  Re-pin only after rebuild + golden-trace pass; record the new commit
 *  hash here and in `MACE_INFERENCE.md`.
 */

#pragma once

#include "smallfield/module_fwd.hpp"

// Only meaningful in CUDA translation units; host-only includes that touch
// this header (e.g. through `mace_inference.cuh`) still pick up the namespace
// declarations cleanly.
#if defined(__CUDACC__)

// Select the TK arch macro from the CUDA __CUDA_ARCH__ at compile time.
// Hopper sm_90 / sm_90a both use KITTENS_HOPPER; Blackwell sm_100 / sm_120 use
// KITTENS_BLACKWELL.  We deliberately do not let the user override these from
// the command line — the arch is a property of the compile target, not a
// switch.
#if !defined(KITTENS_HOPPER) && !defined(KITTENS_BLACKWELL)
#  if defined(__CUDA_ARCH__)
#    if __CUDA_ARCH__ >= 1000
#      define KITTENS_BLACKWELL 1
#    elif __CUDA_ARCH__ >= 900
#      define KITTENS_HOPPER 1
#    endif
#  else
// Host-side translation unit: define KITTENS_HOPPER as the default so the TK
// host helpers compile cleanly when this header is pulled in by a .cpp file.
#    define KITTENS_HOPPER 1
#  endif
#endif

#include <kittens.cuh>

namespace smallfield::cuda {

/**
 *  @brief Project-flavoured aliases over the ThunderKittens tile types.
 *
 *  TK's `st_bf<rows, cols>` takes the row and column count in actual
 *  elements (with `rows` and `cols` required to be a multiple of 16, the
 *  native MMA tile granularity on Hopper / Blackwell — TK enforces this
 *  via static_assert).  Our aliases pass the element counts straight
 *  through and only exist to give the call sites project-style names and
 *  one place to swap the precision template if a future iteration needs it.
 */
namespace mace_tk {

// Underlying type aliases (re-exported from kittens namespace).
using kittens::bf16;
using kittens::half;
// fp32 type from TK is plain `float`.

// Shared-memory tiles.  Sizes in elements; must be multiples of 16.
template <int rows_in_elements, int cols_in_elements>
using shared_tile_bf16 = kittens::st_bf<rows_in_elements, cols_in_elements>;

template <int rows_in_elements, int cols_in_elements>
using shared_tile_fp16 = kittens::st_hf<rows_in_elements, cols_in_elements>;

template <int rows_in_elements, int cols_in_elements>
using shared_tile_fp32 = kittens::st_fl<rows_in_elements, cols_in_elements>;

// Register tiles.  TK's `rt_bf<H, W, layout>` expresses (H, W) in tile
// units of 16 (i.e. `rt_bf<2, 4>` is 32 rows × 64 cols), opposite to the
// shared-tile convention.  We keep that as-is to match TK directly.
template <int rows_in_tiles, int cols_in_tiles,
          typename layout = kittens::ducks::rt_layout::row>
using register_tile_bf16 = kittens::rt_bf<rows_in_tiles, cols_in_tiles, layout>;

template <int rows_in_tiles, int cols_in_tiles,
          typename layout = kittens::ducks::rt_layout::row>
using register_tile_fp32 = kittens::rt_fl<rows_in_tiles, cols_in_tiles, layout>;

// Compile-time constants for our static MACE-OFF shapes.  These are used
// as template parameters at the call sites to spell out tile sizes
// in a single place.
constexpr int mace_channels_small_k    = 96;
constexpr int mace_channels_medium_k   = 128;
constexpr int mace_channels_large_k    = 224;   // MACE-OFF23 large is 224 channels (verified at export time)
constexpr int mace_irrep_dim_total_k   = 16;   // 1 + 3 + 5 + 7
constexpr int mace_radial_hidden_dim_k = 64;
constexpr int mace_radial_input_dim_k  = 8;    // n_radial Bessel basis functions
constexpr int mace_n_paths_k           = 4;    // tensor-product output paths
constexpr int mace_radial_output_dim_k = mace_n_paths_k * mace_channels_small_k; // = 384

// Activation epilogue helpers — one entry per non-linear epilogue we emit
// from a TK persistent kernel.  Defined as functors so they can be folded
// into `kt::warpgroup::store` epilogues without separate kernel launches.
struct silu_normalize2mom_epilogue_t {
    // The constant matches the e3nn `normalize2mom(silu)` value computed via
    // the same procedure (1 / sqrt(<silu(z)^2>_{z~N(0,1)})) — see
    // `mace_linear_kernel.cuh::mace_silu_normalize_to_unit_second_moment_constant_k`.
    static constexpr float normalize_constant = 1.679176792398942f;

    __device__ inline float operator()(float pre_activation_value) const {
        float const sigmoid_value = 1.0f / (1.0f + __expf(-pre_activation_value));
        return pre_activation_value * sigmoid_value * normalize_constant;
    }
};

} // namespace mace_tk

} // namespace smallfield::cuda

#endif // __CUDACC__
