/**
 *  @file src/smallfield/utilities.hpp
 *  @brief Type-generic helpers shared by more than one SmallField subsystem.
 *
 *  Admission rule, applied to every candidate before it lands here. A helper belongs in this header only
 *  when all three hold:
 *   - it is generic over its operand type rather than tied to one struct or one table layout,
 *   - it needs no `#include` at all — this file has none, and that is the property being maintained rather
 *     than an accident of what has landed so far,
 *   - it has two or more `src/` subsystems as consumers; `cli/` and `bench/` are tooling, not subsystems.
 *  A helper failing any of the three belongs to the subsystem that owns it, where its name sits next to
 *  the data it serves. `lower_bound_` in `parameters/mmff94_tables.hpp` is the worked example: generic and
 *  standalone, but read by that one header, so it stays there. Without the rule a file named "utilities"
 *  becomes the junk drawer every subsystem reaches into and nobody prunes.
 *
 *  @b Why the second clause is stricter than "standard headers only". `cuda/etkdg_pipeline.cuh`,
 *  `cuda/warp_primitives.cuh` and `mmff94/mmff94_extract_device.cuh` all pull this header in, so every
 *  include here reaches three translation units `nvcc` compiles. A host-only standard header such as
 *  `<charconv>` costs all three of them to serve a helper no kernel can call — which is why the
 *  command-line number parser lives in `number_parse.hpp` at the root instead of here.
 *
 *  Neither helper is marked `__host__ __device__`. `parameters/mmff94_tables.hpp` pulls this header in and
 *  is compiled as plain C++ by `test/parameters/test_mmff94_tables.cpp` and `test/mmff94/test_mmff_typer.cpp`,
 *  where a CUDA attribute is not a keyword g++ knows. Device code reaches both helpers as `constexpr`
 *  through `--expt-relaxed-constexpr`, the same route the `lookup_*` cascades take.
 */

#pragma once

/**
 *  @brief Debug-only assertion that compiles to nothing in Release.
 *
 *  Guards pointer-math accesses so an out-of-bounds deref traps immediately under compute-sanitizer or a
 *  debugger instead of silently corrupting memory and crashing later somewhere unrelated. Note that
 *  `cuda::std::mdspan` performs no such check of its own, even with `CCCL_ENABLE_ASSERTIONS` set, so raw
 *  index arithmetic and every view whose extents must agree still want this. The device path calls
 *  `__trap()` so compute-sanitizer reports the offending thread and block coordinates; the host path calls
 *  `__builtin_trap()` so gdb or a core dump points at the bad caller.
 *
 *  @b Only the `debug` preset carries these. `dev`, `release` and `profile` all define `NDEBUG`, where this
 *  compiles to nothing — so a check written here is a check `cmake --preset debug` runs and the inner loop
 *  does not. Anything that must hold in a shipped build belongs in the code rather than in this macro.
 *
 *  It lives here rather than beside the warp primitives because `types/views.hpp` needs it and host code
 *  includes that header, while everything in `cuda/` sits behind a `__CUDACC__` guard.
 */
#if defined(NDEBUG)
#define sf_assert_(cond) ((void)0)
#elif defined(__CUDA_ARCH__)
#define sf_assert_(cond)           \
    do {                           \
        if (!(cond)) { __trap(); } \
    } while (0)
#else
#define sf_assert_(cond)                   \
    do {                                   \
        if (!(cond)) { __builtin_trap(); } \
    } while (0)
#endif

namespace smallfield {

/**
 *  @brief Round `value` up to the next multiple of `alignment`, which must be a power of two.
 *
 *  Two template parameters rather than one, because every call site mixes widths: the shared-memory layout
 *  in `cuda/etkdg_pipeline.cuh` passes `(std::uint32_t, std::size_t)` via `alignof`, and the pool layouts in
 *  `api/smallfield.cu` pass `(std::size_t, int)`. A single deduced parameter would fail to deduce at all of
 *  them. The result takes the width of `value`, so a `std::uint32_t` offset stays a `std::uint32_t`.
 */
template <typename value_, typename alignment_>
constexpr auto align_up_(value_ value, alignment_ alignment) noexcept {
    value_ const step = static_cast<value_>(alignment);
    return static_cast<value_>((value + step - 1) & ~(step - 1));
}

/** Exchange two values. Used by the MMFF94 key canonicalisation steps, on host and device alike. */
template <typename value_>
constexpr void swap_(value_ &first, value_ &second) noexcept {
    value_ const held = first;
    first = second;
    second = held;
}

} // namespace smallfield
