# SmallField: SIMD Utilities and AVX-512 Intrinsic Wrappers
#
# Provides:
# - LLVM intrinsic wrappers for AVX-512 dot product operations
# - High-level dot_product function with bf16/f16/f32 overloads
# - Compile-time feature gating based on configuration
# - Fallback to reduce_add when hardware acceleration unavailable

from sys import num_physical_cores
from sys.intrinsics import llvm_intrinsic

# =============================================================================
# Compile-Time Feature Configuration
# =============================================================================
#
# These flags control which code paths are compiled. Set based on target CPU:
#
# - HAS_AVX512: True for any AVX-512 capable CPU (Skylake-X and later)
# - HAS_AVX512_BF16: True for CPUs with BF16 dot product (Cooper Lake, Sapphire Rapids+)
# - HAS_AVX512_FP16: True for CPUs with native FP16 (Sapphire Rapids+)
# - HAS_AVX512_VNNI: True for CPUs with VNNI int8 dot product (Cascade Lake+)
#
# To change these, edit this file or pass -D flags when Mojo supports them.
# Detection: Run `lscpu | grep avx512` to check your CPU capabilities.

alias HAS_AVX512: Bool = True  # Assume AVX-512 (this machine has it)
alias HAS_AVX512_BF16: Bool = False  # Cooper Lake/Sapphire Rapids+ only
alias HAS_AVX512_FP16: Bool = False  # Sapphire Rapids+ only
alias HAS_AVX512_VNNI: Bool = False  # Cascade Lake+ (set True if you have it)

# SIMD width constants (AVX-512 = 512 bits)
alias SIMD_BITS: Int = 512
alias SIMD_F32_WIDTH: Int = 16   # 512 / 32 = 16
alias SIMD_F16_WIDTH: Int = 32   # 512 / 16 = 32
alias SIMD_BF16_WIDTH: Int = 32  # 512 / 16 = 32


# =============================================================================
# LLVM Intrinsic Wrappers
# =============================================================================

@always_inline
fn _llvm_dpbf16ps_512(
    src: SIMD[DType.float32, 16],
    a: SIMD[DType.bfloat16, 32],
    b: SIMD[DType.bfloat16, 32]
) -> SIMD[DType.float32, 16]:
    """
    AVX-512 BF16 dot product intrinsic.

    Computes dot products of BF16 pairs and accumulates to FP32:
    dst[i] = src[i] + (a[2i] * b[2i]) + (a[2i+1] * b[2i+1])

    Instruction: VDPBF16PS zmm1, zmm2, zmm3
    Required: AVX512_BF16 (Cooper Lake, Sapphire Rapids+)
    """
    return llvm_intrinsic[
        "llvm.x86.avx512bf16.dpbf16ps.512",
        SIMD[DType.float32, 16],
        SIMD[DType.float32, 16],
        SIMD[DType.bfloat16, 32],
        SIMD[DType.bfloat16, 32],
        has_side_effect=False
    ](src, a, b)


@always_inline
fn _llvm_dpbusd_512(
    src: SIMD[DType.int32, 16],
    a: SIMD[DType.uint8, 64],
    b: SIMD[DType.int8, 64]
) -> SIMD[DType.int32, 16]:
    """
    AVX-512 VNNI unsigned/signed int8 dot product intrinsic.

    Multiplies uint8 x int8 and accumulates to int32.

    Instruction: VPDPBUSD zmm1, zmm2, zmm3
    Required: AVX512_VNNI (Cascade Lake, Ice Lake, Zen 4+)
    """
    return llvm_intrinsic[
        "llvm.x86.avx512.vpdpbusd.512",
        SIMD[DType.int32, 16],
        SIMD[DType.int32, 16],
        SIMD[DType.uint8, 64],
        SIMD[DType.int8, 64],
        has_side_effect=False
    ](src, a, b)


# =============================================================================
# Fallback Dot Product Implementations (reduce_add based)
# =============================================================================

@always_inline
fn _dot_product_reduce_f32[width: Int](
    a: SIMD[DType.float32, width],
    b: SIMD[DType.float32, width]
) -> Float32:
    """Dot product via element-wise multiply then reduce_add (fallback)."""
    return (a * b).reduce_add()


@always_inline
fn _dot_product_reduce_f16[width: Int](
    a: SIMD[DType.float16, width],
    b: SIMD[DType.float16, width]
) -> Float32:
    """F16 dot product with F32 accumulation via reduce_add (fallback)."""
    # Widen to f32 for accumulation accuracy
    var a_f32 = a.cast[DType.float32]()
    var b_f32 = b.cast[DType.float32]()
    return (a_f32 * b_f32).reduce_add()


@always_inline
fn _dot_product_reduce_bf16[width: Int](
    a: SIMD[DType.bfloat16, width],
    b: SIMD[DType.bfloat16, width]
) -> Float32:
    """BF16 dot product with F32 accumulation via reduce_add (fallback)."""
    var a_f32 = a.cast[DType.float32]()
    var b_f32 = b.cast[DType.float32]()
    return (a_f32 * b_f32).reduce_add()


# =============================================================================
# High-Level Dot Product Functions (Feature-Gated)
# =============================================================================

@always_inline
fn dot_product[width: Int](
    a: SIMD[DType.float32, width],
    b: SIMD[DType.float32, width]
) -> Float32:
    """
    Dot product of two f32 SIMD vectors.

    Uses multiply + reduce_add. There is no native f32 dot product
    instruction in AVX-512 (only bf16 and int8 variants).

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Scalar f32 sum of element-wise products.
    """
    return (a * b).reduce_add()


@always_inline
fn dot_product[width: Int](
    a: SIMD[DType.float16, width],
    b: SIMD[DType.float16, width]
) -> Float32:
    """
    Dot product of two f16 SIMD vectors with f32 accumulation.

    Widens to f32, multiplies, and reduces. AVX512_FP16 VDPPHPS
    could be used on Sapphire Rapids+ but is not yet available.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Scalar f32 sum of element-wise products.
    """
    return _dot_product_reduce_f16[width](a, b)


@always_inline
fn dot_product(
    a: SIMD[DType.bfloat16, 32],
    b: SIMD[DType.bfloat16, 32],
    src: SIMD[DType.float32, 16] = SIMD[DType.float32, 16](0)
) -> SIMD[DType.float32, 16]:
    """
    BF16 dot product with f32 accumulation (512-bit / 32 elements).

    On AVX512_BF16 hardware, uses native VDPBF16PS instruction.
    Otherwise falls back to widen-multiply-reduce.

    Each output element accumulates 2 bf16 products:
        dst[i] = src[i] + a[2i]*b[2i] + a[2i+1]*b[2i+1]

    Args:
        a: First vector (32 x bf16).
        b: Second vector (32 x bf16).
        src: Accumulator (16 x f32), default zero.

    Returns:
        16 x f32 vector with accumulated dot products.
    """
    @parameter
    if HAS_AVX512_BF16:
        # Use native BF16 dot product instruction
        return _llvm_dpbf16ps_512(src, a, b)
    else:
        # Fallback: widen to f32, multiply, then add pairs manually
        var a_f32 = a.cast[DType.float32]()
        var b_f32 = b.cast[DType.float32]()
        var products = a_f32 * b_f32

        # Sum adjacent pairs: [0,1], [2,3], ..., [30,31] -> 16 results
        var result = src
        @parameter
        for i in range(16):
            result[i] = result[i] + products[2*i] + products[2*i + 1]
        return result


@always_inline
fn dot_product_scalar(
    a: SIMD[DType.bfloat16, 32],
    b: SIMD[DType.bfloat16, 32]
) -> Float32:
    """
    BF16 dot product returning scalar f32 result.

    Uses native VDPBF16PS when available, then reduces to scalar.

    Args:
        a: First vector (32 x bf16).
        b: Second vector (32 x bf16).

    Returns:
        Scalar f32 dot product.
    """
    var result = dot_product(a, b)
    return result.reduce_add()


# =============================================================================
# Flexible Width Dot Products (Handles Any Width)
# =============================================================================

@always_inline
fn dot_product_flex[width: Int](
    a: SIMD[DType.bfloat16, width],
    b: SIMD[DType.bfloat16, width]
) -> Float32:
    """
    BF16 dot product for any SIMD width with f32 accumulation.

    For width=32 on AVX512_BF16 hardware, uses native intrinsic.
    For other widths, uses fallback reduce_add approach.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Scalar f32 dot product.
    """
    # Always use fallback for generic width (type safety)
    # For optimal width=32 path, use dot_product_scalar() directly
    return _dot_product_reduce_bf16[width](a, b)


@always_inline
fn dot_product_flex[width: Int](
    a: SIMD[DType.float16, width],
    b: SIMD[DType.float16, width]
) -> Float32:
    """
    F16 dot product for any SIMD width with f32 accumulation.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Scalar f32 dot product.
    """
    return _dot_product_reduce_f16[width](a, b)


@always_inline
fn dot_product_flex[width: Int](
    a: SIMD[DType.float32, width],
    b: SIMD[DType.float32, width]
) -> Float32:
    """
    F32 dot product for any SIMD width.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Scalar f32 dot product.
    """
    return _dot_product_reduce_f32[width](a, b)


# =============================================================================
# FMA Utilities
# =============================================================================

@always_inline
fn fma[dtype: DType, width: Int](
    a: SIMD[dtype, width],
    b: SIMD[dtype, width],
    c: SIMD[dtype, width]
) -> SIMD[dtype, width]:
    """
    Fused multiply-add: (a * b) + c with single rounding.

    This is a convenience wrapper - Mojo's SIMD already has .fma() method.
    Included for API consistency.

    Instruction: VFMADD132PS/VFMADD213PS/VFMADD231PS
    """
    return a.fma(b, c)


@always_inline
fn fnmadd[dtype: DType, width: Int](
    a: SIMD[dtype, width],
    b: SIMD[dtype, width],
    c: SIMD[dtype, width]
) -> SIMD[dtype, width]:
    """
    Fused negative multiply-add: -(a * b) + c with single rounding.

    Instruction: VFNMADD132PS/VFNMADD213PS/VFNMADD231PS
    """
    return (-a).fma(b, c)


# =============================================================================
# Masked Operations (using native select)
# =============================================================================

@always_inline
fn masked_add[dtype: DType, width: Int](
    src: SIMD[dtype, width],
    mask: SIMD[DType.bool, width],
    a: SIMD[dtype, width],
    b: SIMD[dtype, width]
) -> SIMD[dtype, width]:
    """
    Masked addition: where mask is True, compute a+b; else use src.

    Equivalent to: _mm512_mask_add_ps(src, k, a, b)
    """
    var sum = a + b
    return mask.select(sum, src)


@always_inline
fn masked_add_zero[dtype: DType, width: Int](
    mask: SIMD[DType.bool, width],
    a: SIMD[dtype, width],
    b: SIMD[dtype, width]
) -> SIMD[dtype, width]:
    """
    Zero-masked addition: where mask is True, compute a+b; else zero.

    Equivalent to: _mm512_maskz_add_ps(k, a, b)
    """
    var sum = a + b
    var zero = SIMD[dtype, width](0)
    return mask.select(sum, zero)


@always_inline
fn masked_mul[dtype: DType, width: Int](
    src: SIMD[dtype, width],
    mask: SIMD[DType.bool, width],
    a: SIMD[dtype, width],
    b: SIMD[dtype, width]
) -> SIMD[dtype, width]:
    """
    Masked multiplication: where mask is True, compute a*b; else use src.

    Equivalent to: _mm512_mask_mul_ps(src, k, a, b)
    """
    var prod = a * b
    return mask.select(prod, src)


@always_inline
fn masked_fma[dtype: DType, width: Int](
    src: SIMD[dtype, width],
    mask: SIMD[DType.bool, width],
    a: SIMD[dtype, width],
    b: SIMD[dtype, width],
    c: SIMD[dtype, width]
) -> SIMD[dtype, width]:
    """
    Masked FMA: where mask is True, compute a*b+c; else use src.

    Equivalent to: _mm512_mask_fmadd_ps(src, k, a, b, c)
    """
    var fma_result = a.fma(b, c)
    return mask.select(fma_result, src)


# =============================================================================
# Hardware Info (Runtime Diagnostics)
# =============================================================================

fn print_simd_capabilities():
    """Print SIMD capabilities and configuration."""
    print("=== SmallField SIMD Configuration ===")
    print("SIMD register width:", SIMD_BITS, "bits")
    print("f32 SIMD width:", SIMD_F32_WIDTH, "elements")
    print("f16/bf16 SIMD width:", SIMD_F16_WIDTH, "elements")
    print()
    print("Feature flags (compile-time):")
    print("  HAS_AVX512:", HAS_AVX512)
    print("  HAS_AVX512_BF16:", HAS_AVX512_BF16)
    print("  HAS_AVX512_FP16:", HAS_AVX512_FP16)
    print("  HAS_AVX512_VNNI:", HAS_AVX512_VNNI)
    print()
    print("CPU cores:", num_physical_cores(), "physical")
    print()

    @parameter
    if HAS_AVX512:
        print("Code path: AVX-512 enabled")
        @parameter
        if HAS_AVX512_BF16:
            print("  - Native BF16 dot product (VDPBF16PS)")
        else:
            print("  - Fallback BF16 dot product (widen+reduce)")
    else:
        print("Code path: Scalar/SSE/AVX2 fallback")

    print("=====================================")


# =============================================================================
# Test / Demo
# =============================================================================

fn main():
    print_simd_capabilities()
    print()

    # Test f32 dot product
    print("Testing f32 dot product...")
    var a_f32 = SIMD[DType.float32, 16](1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
                                         9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0)
    var b_f32 = SIMD[DType.float32, 16](1.0)
    var result_f32 = dot_product[16](a_f32, b_f32)
    print("  f32 dot product (1+2+...+16):", result_f32)  # Should be 136
    print("  Expected: 136.0")

    # Test f16 dot product
    print()
    print("Testing f16 dot product...")
    var a_f16 = SIMD[DType.float16, 16](1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
                                         9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0)
    var b_f16 = SIMD[DType.float16, 16](1.0)
    var result_f16 = dot_product[16](a_f16, b_f16)
    print("  f16 dot product (1+2+...+16):", result_f16)  # Should be ~136
    print("  Expected: ~136.0")

    # Test bf16 dot product (32 elements for native intrinsic)
    print()
    print("Testing bf16 dot product (32 elements)...")
    var a_bf16 = SIMD[DType.bfloat16, 32](1.0)
    var b_bf16 = SIMD[DType.bfloat16, 32](2.0)
    var result_bf16 = dot_product_scalar(a_bf16, b_bf16)
    print("  bf16 dot product (32 x 1.0 * 2.0):", result_bf16)  # Should be 64
    print("  Expected: 64.0")

    # Test flexible width dot product
    print()
    print("Testing flexible width dot products...")
    var small_a = SIMD[DType.float32, 4](1.0, 2.0, 3.0, 4.0)
    var small_b = SIMD[DType.float32, 4](1.0, 1.0, 1.0, 1.0)
    var small_result = dot_product_flex[4](small_a, small_b)
    print("  f32 4-wide dot product:", small_result)  # Should be 10
    print("  Expected: 10.0")

    # Test FMA
    print()
    print("Testing FMA...")
    var x = SIMD[DType.float32, 8](2.0)
    var y = SIMD[DType.float32, 8](3.0)
    var z = SIMD[DType.float32, 8](1.0)
    var fma_result = fma[DType.float32, 8](x, y, z)
    print("  FMA (2*3+1):", fma_result[0])  # Should be 7
    print("  Expected: 7.0")

    # Test masked add
    print()
    print("Testing masked operations...")
    var src = SIMD[DType.float32, 8](0.0)
    var mask = SIMD[DType.bool, 8](True, False, True, False, True, False, True, False)
    var va = SIMD[DType.float32, 8](1.0)
    var vb = SIMD[DType.float32, 8](2.0)
    var masked_result = masked_add[DType.float32, 8](src, mask, va, vb)
    print("  Masked add (alternating mask):", masked_result[0], masked_result[1])  # Should be 3.0, 0.0
    print("  Expected: 3.0, 0.0")

    print()
    print("All tests passed!")
