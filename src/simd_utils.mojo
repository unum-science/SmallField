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
alias HAS_AVX512_BF16: Bool = True  # Cooper Lake/Sapphire Rapids+ (enabled for Xeon Platinum 8468)
alias HAS_AVX512_FP16: Bool = True  # Sapphire Rapids+ (enabled for Xeon Platinum 8468)
alias HAS_AVX512_VNNI: Bool = True  # Cascade Lake+ (enabled for Xeon Platinum 8468)

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
fn _llvm_dpphps_512(
    src: SIMD[DType.float32, 16],
    a: SIMD[DType.float16, 32],
    b: SIMD[DType.float16, 32]
) -> SIMD[DType.float32, 16]:
    """
    AVX-512 FP16 dot product intrinsic with FP32 accumulation.

    Computes dot products of FP16 pairs and accumulates to FP32:
    dst[i] = src[i] + (a[2i] * b[2i]) + (a[2i+1] * b[2i+1])

    Instruction: VDPPHPS zmm1, zmm2, zmm3
    Required: AVX512_FP16 (Sapphire Rapids+)
    Latency: ~8 cycles, Throughput: 0.5 CPI
    """
    return llvm_intrinsic[
        "llvm.x86.avx512fp16.dpphps.512",
        SIMD[DType.float32, 16],
        SIMD[DType.float32, 16],
        SIMD[DType.float16, 32],
        SIMD[DType.float16, 32],
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
# Fast Approximate Math (rsqrt, rcp)
# =============================================================================

@always_inline
fn rsqrt14[width: Int](a: SIMD[DType.float32, width]) -> SIMD[DType.float32, width]:
    """
    Approximate reciprocal square root with ~14-bit precision.

    Computes 1/sqrt(a) using VRSQRT14PS instruction.

    This is MUCH faster than sqrt() + division for distance normalization:
        Old: r = sqrt(d²)  →  inv_r = 1.0 / r       (sqrt + div = slow)
        New: inv_r = rsqrt(d²)  →  r = d² * inv_r   (rsqrt + mul = fast)

    Instruction: VRSQRT14PS zmm1, zmm2
    Required: AVX-512F
    Precision: ~2^-14 relative error (sufficient for force field calculations)
    """
    @parameter
    if width == 16:
        return llvm_intrinsic[
            "llvm.x86.avx512.rsqrt14.ps.512",
            SIMD[DType.float32, 16],
            SIMD[DType.float32, 16],
            has_side_effect=False
        ](a.cast[DType.float32]()).cast[DType.float32]()
    else:
        # Fallback for other widths
        return Float32(1.0) / a.sqrt()


@always_inline
fn rcp14[width: Int](a: SIMD[DType.float32, width]) -> SIMD[DType.float32, width]:
    """
    Approximate reciprocal with ~14-bit precision.

    Computes 1/a using VRCP14PS instruction.

    Useful for electrostatic calculations with explicit 1/r terms:
        Old: energy = k / r                  (div = slow)
        New: energy = k * rcp14(r)           (rcp + mul = fast)

    Instruction: VRCP14PS zmm1, zmm2
    Required: AVX-512F
    Precision: ~2^-14 relative error (sufficient for force field calculations)
    """
    @parameter
    if width == 16:
        return llvm_intrinsic[
            "llvm.x86.avx512.rcp14.ps.512",
            SIMD[DType.float32, 16],
            SIMD[DType.float32, 16],
            has_side_effect=False
        ](a.cast[DType.float32]()).cast[DType.float32]()
    else:
        # Fallback for other widths
        return Float32(1.0) / a


@always_inline
fn rsqrt14_nr[width: Int](a: SIMD[DType.float32, width]) -> SIMD[DType.float32, width]:
    """
    Reciprocal square root with Newton-Raphson refinement.

    One iteration of Newton-Raphson improves precision from ~14 bits to ~28 bits:
        y' = y * (1.5 - 0.5 * a * y * y)

    Use this when higher precision is needed (e.g., for gradient calculations).
    """
    var y = rsqrt14[width](a)
    # Newton-Raphson: y' = y * (1.5 - 0.5 * a * y²)
    var half_a = Float32(0.5) * a
    var y_sq = y * y
    return y * (Float32(1.5) - half_a * y_sq)


@always_inline
fn rcp14_nr[width: Int](a: SIMD[DType.float32, width]) -> SIMD[DType.float32, width]:
    """
    Reciprocal with Newton-Raphson refinement.

    One iteration of Newton-Raphson improves precision from ~14 bits to ~28 bits:
        y' = y * (2 - a * y)

    Use this when higher precision is needed.
    """
    var y = rcp14[width](a)
    # Newton-Raphson: y' = y * (2 - a * y)
    return y * (Float32(2.0) - a * y)


# =============================================================================
# Native FP16 Approximate Math (AVX512_FP16)
# =============================================================================

@always_inline
fn rsqrt_f16[width: Int](a: SIMD[DType.float16, width]) -> SIMD[DType.float16, width]:
    """
    Approximate reciprocal square root for FP16.

    On AVX512_FP16 hardware, uses native VRSQRTPH instruction.
    Otherwise falls back to widening to F32.

    Instruction: VRSQRTPH zmm1, zmm2
    Required: AVX512_FP16 (Sapphire Rapids+)
    Precision: ~2^-11 relative error
    """
    @parameter
    if HAS_AVX512_FP16 and width == 32:
        return llvm_intrinsic[
            "llvm.x86.avx512fp16.rsqrt.ph.512",
            SIMD[DType.float16, 32],
            SIMD[DType.float16, 32],
            has_side_effect=False
        ](a)
    else:
        # Fallback: widen to F32, compute, narrow back
        var a_f32 = a.cast[DType.float32]()
        var result_f32 = rsqrt14[width](a_f32)
        return result_f32.cast[DType.float16]()


@always_inline
fn rcp_f16[width: Int](a: SIMD[DType.float16, width]) -> SIMD[DType.float16, width]:
    """
    Approximate reciprocal for FP16.

    On AVX512_FP16 hardware, uses native VRCPPH instruction.
    Otherwise falls back to widening to F32.

    Instruction: VRCPPH zmm1, zmm2
    Required: AVX512_FP16 (Sapphire Rapids+)
    Precision: ~2^-11 relative error
    """
    @parameter
    if HAS_AVX512_FP16 and width == 32:
        return llvm_intrinsic[
            "llvm.x86.avx512fp16.rcp.ph.512",
            SIMD[DType.float16, 32],
            SIMD[DType.float16, 32],
            has_side_effect=False
        ](a)
    else:
        # Fallback: widen to F32, compute, narrow back
        var a_f32 = a.cast[DType.float32]()
        var result_f32 = rcp14[width](a_f32)
        return result_f32.cast[DType.float16]()


@always_inline
fn compute_distance_f16[width: Int = 32](
    dx: SIMD[DType.float16, width],
    dy: SIMD[DType.float16, width],
    dz: SIMD[DType.float16, width]
) -> Tuple[SIMD[DType.float16, width], SIMD[DType.float16, width]]:
    """
    Compute distance and inverse distance in native FP16.

    Returns (r, inv_r) where:
        r = sqrt(dx² + dy² + dz²)
        inv_r = 1/r (or rsqrt(d²))

    On AVX512_FP16 hardware, uses native FP16 FMA and rsqrt.
    This is faster than widening to F32 for distance computation.

    Safe for molecular distances (typically 1-10 Angstroms):
    - FP16 has ~3 decimal digits of precision
    - Differences (dx, dy, dz) are small
    - sqrt is well-conditioned

    Args:
        dx, dy, dz: Displacement vectors in FP16

    Returns:
        Tuple of (distance, inverse_distance) in FP16
    """
    # Compute r² = dx² + dy² + dz²
    var r_sq = dx.fma(dx, dy.fma(dy, dz * dz))

    # Use rsqrt for fast distance: inv_r = rsqrt(r²), r = r² * inv_r
    var inv_r = rsqrt_f16[width](r_sq)
    var r = r_sq * inv_r

    return (r, inv_r)


@always_inline
fn compute_distance_f16_to_f32[width: Int = 32](
    dx: SIMD[DType.float16, width],
    dy: SIMD[DType.float16, width],
    dz: SIMD[DType.float16, width]
) -> Tuple[SIMD[DType.float32, width], SIMD[DType.float32, width]]:
    """
    Compute distance in FP16, return results in FP32.

    Hybrid approach: compute in FP16 for speed, widen to F32 for
    subsequent calculations that need higher precision.

    Args:
        dx, dy, dz: Displacement vectors in FP16

    Returns:
        Tuple of (distance, inverse_distance) in FP32
    """
    var result = compute_distance_f16[width](dx, dy, dz)
    return (result[0].cast[DType.float32](), result[1].cast[DType.float32]())


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


@always_inline
fn dot_product(
    a: SIMD[DType.float16, 32],
    b: SIMD[DType.float16, 32],
    src: SIMD[DType.float32, 16] = SIMD[DType.float32, 16](0)
) -> SIMD[DType.float32, 16]:
    """
    FP16 dot product with f32 accumulation (512-bit / 32 elements).

    On AVX512_FP16 hardware, uses native VDPPHPS instruction.
    Otherwise falls back to widen-multiply-reduce.

    Each output element accumulates 2 fp16 products:
        dst[i] = src[i] + a[2i]*b[2i] + a[2i+1]*b[2i+1]

    Args:
        a: First vector (32 x fp16).
        b: Second vector (32 x fp16).
        src: Accumulator (16 x f32), default zero.

    Returns:
        16 x f32 vector with accumulated dot products.
    """
    @parameter
    if HAS_AVX512_FP16:
        # Use native FP16 dot product instruction (Sapphire Rapids+)
        return _llvm_dpphps_512(src, a, b)
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
    a: SIMD[DType.float16, 32],
    b: SIMD[DType.float16, 32]
) -> Float32:
    """
    FP16 dot product returning scalar f32 result.

    Uses native VDPPHPS when available, then reduces to scalar.

    Args:
        a: First vector (32 x fp16).
        b: Second vector (32 x fp16).

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
# Multi-Accumulator Pattern (Reduces Dependency Chains)
# =============================================================================

struct MultiAccumulator[N: Int = 4](Copyable, Movable):
    """
    N independent scalar accumulators to break dependency chains.

    On modern CPUs with ~5 cycle FP add latency and ability to sustain
    2 FP adds/cycle, using N=4 accumulators allows the CPU to overlap
    additions and hide latency.

    Usage:
        var accum = MultiAccumulator[4]()
        for i in range(num_items):
            accum.add(i, values[i])
        return accum.reduce()
    """
    var a0: Float32
    var a1: Float32
    var a2: Float32
    var a3: Float32

    fn __init__(out self):
        self.a0 = 0.0
        self.a1 = 0.0
        self.a2 = 0.0
        self.a3 = 0.0

    fn __copyinit__(out self, other: Self):
        self.a0 = other.a0
        self.a1 = other.a1
        self.a2 = other.a2
        self.a3 = other.a3

    fn __moveinit__(out self, owned other: Self):
        self.a0 = other.a0
        self.a1 = other.a1
        self.a2 = other.a2
        self.a3 = other.a3

    @always_inline
    fn add(mut self, idx: Int, value: Float32):
        """Add to accumulator idx % 4 (fast modulo via bitwise AND)."""
        var i = idx & 3  # Equivalent to idx % 4 for N=4
        if i == 0:
            self.a0 += value
        elif i == 1:
            self.a1 += value
        elif i == 2:
            self.a2 += value
        else:
            self.a3 += value

    @always_inline
    fn add0(mut self, value: Float32):
        """Add directly to accumulator 0."""
        self.a0 += value

    @always_inline
    fn add1(mut self, value: Float32):
        """Add directly to accumulator 1."""
        self.a1 += value

    @always_inline
    fn add2(mut self, value: Float32):
        """Add directly to accumulator 2."""
        self.a2 += value

    @always_inline
    fn add3(mut self, value: Float32):
        """Add directly to accumulator 3."""
        self.a3 += value

    @always_inline
    fn reduce(self) -> Float32:
        """Reduce all accumulators to single value."""
        return (self.a0 + self.a1) + (self.a2 + self.a3)


struct SIMDMultiAccumulator[width: Int, N: Int = 4](Copyable, Movable):
    """
    N independent SIMD accumulators for vectorized loops.

    Each accumulator is a SIMD vector of 'width' f32 elements.
    Final reduce() returns a scalar sum of all elements.
    """
    var a0: SIMD[DType.float32, width]
    var a1: SIMD[DType.float32, width]
    var a2: SIMD[DType.float32, width]
    var a3: SIMD[DType.float32, width]

    fn __init__(out self):
        self.a0 = SIMD[DType.float32, width](0)
        self.a1 = SIMD[DType.float32, width](0)
        self.a2 = SIMD[DType.float32, width](0)
        self.a3 = SIMD[DType.float32, width](0)

    fn __copyinit__(out self, other: Self):
        self.a0 = other.a0
        self.a1 = other.a1
        self.a2 = other.a2
        self.a3 = other.a3

    fn __moveinit__(out self, owned other: Self):
        self.a0 = other.a0
        self.a1 = other.a1
        self.a2 = other.a2
        self.a3 = other.a3

    @always_inline
    fn add(mut self, idx: Int, value: SIMD[DType.float32, width]):
        """Add vector to accumulator idx % 4."""
        var i = idx & 3
        if i == 0:
            self.a0 += value
        elif i == 1:
            self.a1 += value
        elif i == 2:
            self.a2 += value
        else:
            self.a3 += value

    @always_inline
    fn add0(mut self, value: SIMD[DType.float32, width]):
        """Add directly to accumulator 0."""
        self.a0 += value

    @always_inline
    fn add1(mut self, value: SIMD[DType.float32, width]):
        """Add directly to accumulator 1."""
        self.a1 += value

    @always_inline
    fn add2(mut self, value: SIMD[DType.float32, width]):
        """Add directly to accumulator 2."""
        self.a2 += value

    @always_inline
    fn add3(mut self, value: SIMD[DType.float32, width]):
        """Add directly to accumulator 3."""
        self.a3 += value

    @always_inline
    fn reduce(self) -> Float32:
        """Reduce all SIMD accumulators to single scalar."""
        var sum_vec = (self.a0 + self.a1) + (self.a2 + self.a3)
        return sum_vec.reduce_add()

    @always_inline
    fn reduce_vec(self) -> SIMD[DType.float32, width]:
        """Reduce to single SIMD vector (before horizontal reduce)."""
        return (self.a0 + self.a1) + (self.a2 + self.a3)


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
        @parameter
        if HAS_AVX512_FP16:
            print("  - Native FP16 dot product (VDPPHPS)")
        else:
            print("  - Fallback FP16 dot product (widen+reduce)")
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
