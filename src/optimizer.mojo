# SmallField: Optimization
# Merged module: Kahan summation, line search, and L-BFGS optimizer
# Updated for Mojo 0.25.7

from math import sqrt
from memory import alloc


# ============================================================================
# CONSTANTS
# ============================================================================

# L-BFGS parameters
alias DEFAULT_HISTORY_SIZE: Int = 8
alias DEFAULT_MAX_ITERATIONS: Int = 200
alias DEFAULT_GRADIENT_TOLERANCE: Float32 = 0.01  # kcal/mol/A
alias DEFAULT_ENERGY_TOLERANCE: Float32 = 1e-6

# Line search parameters
alias DEFAULT_ARMIJO_C1: Float32 = 1e-4
alias DEFAULT_BACKTRACK_FACTOR: Float32 = 0.5
alias DEFAULT_INITIAL_STEP: Float32 = 1.0
alias DEFAULT_MIN_STEP: Float32 = 1e-10
alias DEFAULT_MAX_LINE_SEARCH_ITERS: Int = 20


# ============================================================================
# KAHAN SUMMATION
# ============================================================================

struct KahanAccumulator(Copyable, Movable):
    """
    Kahan summation accumulator for accurate FP32 addition.

    Maintains a compensation term to reduce floating-point errors
    when summing many values.
    """
    var sum: Float32
    var compensation: Float32

    fn __init__(out self):
        self.sum = 0.0
        self.compensation = 0.0

    fn __init__(out self, initial: Float32):
        self.sum = initial
        self.compensation = 0.0

    fn __copyinit__(out self, other: Self):
        self.sum = other.sum
        self.compensation = other.compensation

    fn __moveinit__(out self, deinit other: Self):
        self.sum = other.sum
        self.compensation = other.compensation

    fn add(mut self, value: Float32):
        """Add a value with compensation."""
        var y = value - self.compensation
        var t = self.sum + y
        self.compensation = (t - self.sum) - y
        self.sum = t

    fn result(self) -> Float32:
        """Get the accumulated sum."""
        return self.sum

    fn reset(mut self):
        """Reset accumulator to zero."""
        self.sum = 0.0
        self.compensation = 0.0


fn kahan_sum(values: type_of(alloc[Float32](1)), n: Int) -> Float32:
    """
    Compute sum of array using Kahan summation.
    """
    var sum: Float32 = 0.0
    var compensation: Float32 = 0.0

    for i in range(n):
        var y = values[i] - compensation
        var t = sum + y
        compensation = (t - sum) - y
        sum = t

    return sum


fn kahan_dot_product(
    a: type_of(alloc[Float32](1)),
    b: type_of(alloc[Float32](1)),
    n: Int
) -> Float32:
    """
    Compute dot product using Kahan summation.
    """
    var sum: Float32 = 0.0
    var compensation: Float32 = 0.0

    for i in range(n):
        var product = a[i] * b[i]
        var y = product - compensation
        var t = sum + y
        compensation = (t - sum) - y
        sum = t

    return sum


fn kahan_norm_squared(
    values: type_of(alloc[Float32](1)),
    n: Int
) -> Float32:
    """
    Compute squared L2 norm using Kahan summation.
    """
    var sum: Float32 = 0.0
    var compensation: Float32 = 0.0

    for i in range(n):
        var sq = values[i] * values[i]
        var y = sq - compensation
        var t = sum + y
        compensation = (t - sum) - y
        sum = t

    return sum


fn neumaier_sum(values: type_of(alloc[Float32](1)), n: Int) -> Float32:
    """
    Neumaier's improved Kahan summation.

    More accurate when large and small values are mixed.
    """
    var sum: Float32 = 0.0
    var compensation: Float32 = 0.0

    for i in range(n):
        var t = sum + values[i]

        # Check which is larger in absolute value
        var abs_sum = sum if sum >= Float32(0.0) else -sum
        var abs_val = values[i] if values[i] >= Float32(0.0) else -values[i]

        if abs_sum >= abs_val:
            compensation = compensation + (sum - t) + values[i]
        else:
            compensation = compensation + (values[i] - t) + sum

        sum = t

    return sum + compensation


# ============================================================================
# LINE SEARCH
# ============================================================================

struct LineSearchResult(Copyable, Movable):
    """Result of line search."""
    var step: Float32
    var new_energy: Float32
    var success: Bool
    var iterations: Int

    fn __init__(out self):
        self.step = 0.0
        self.new_energy = 0.0
        self.success = False
        self.iterations = 0

    fn __copyinit__(out self, other: Self):
        self.step = other.step
        self.new_energy = other.new_energy
        self.success = other.success
        self.iterations = other.iterations

    fn __moveinit__(out self, deinit other: Self):
        self.step = other.step
        self.new_energy = other.new_energy
        self.success = other.success
        self.iterations = other.iterations


fn directional_derivative(
    gradient: type_of(alloc[Float32](1)),
    direction: type_of(alloc[Float32](1)),
    n: Int
) -> Float32:
    """
    Compute directional derivative: g^T * d
    """
    var dot: Float32 = 0.0
    for i in range(n):
        dot += gradient[i] * direction[i]
    return dot


fn backtracking_line_search(
    x_current: type_of(alloc[Float32](1)),
    mut x_trial: type_of(alloc[Float32](1)),
    direction: type_of(alloc[Float32](1)),
    gradient: type_of(alloc[Float32](1)),
    current_energy: Float32,
    n: Int,
    # Energy function callback would be here in full implementation
    # For now, we return a structure that caller uses
    c1: Float32,
    backtrack_factor: Float32,
    initial_step: Float32,
    min_step: Float32,
    max_iterations: Int
) -> LineSearchResult:
    """
    Backtracking line search with Armijo condition.

    Finds step size alpha such that:
    f(x + alpha*d) <= f(x) + c1 * alpha * g^T * d

    Returns trial step and position. Caller must evaluate energy.
    """
    var result = LineSearchResult()

    # Compute initial directional derivative
    var dg = directional_derivative(gradient, direction, n)

    # Check descent direction
    if dg >= Float32(0.0):
        # Not a descent direction
        result.success = False
        return result

    var step = initial_step

    # Armijo threshold
    var threshold = c1 * dg

    # Set initial trial position
    for i in range(n):
        x_trial[i] = x_current[i] + step * direction[i]

    result.step = step
    result.success = True  # Caller will evaluate
    result.iterations = 0

    return result


fn update_trial_position(
    x_current: type_of(alloc[Float32](1)),
    mut x_trial: type_of(alloc[Float32](1)),
    direction: type_of(alloc[Float32](1)),
    step: Float32,
    n: Int
):
    """Update trial position: x_trial = x_current + step * direction"""
    for i in range(n):
        x_trial[i] = x_current[i] + step * direction[i]


fn check_armijo_condition(
    current_energy: Float32,
    trial_energy: Float32,
    step: Float32,
    directional_deriv: Float32,
    c1: Float32
) -> Bool:
    """
    Check if Armijo (sufficient decrease) condition is satisfied.

    f(x + alpha*d) <= f(x) + c1 * alpha * (g^T * d)
    """
    return trial_energy <= current_energy + c1 * step * directional_deriv


fn wolfe_curvature_condition(
    current_deriv: Float32,
    trial_deriv: Float32,
    c2: Float32
) -> Bool:
    """
    Check Wolfe curvature condition.

    g(x + alpha*d)^T * d >= c2 * g(x)^T * d

    Ensures we don't stop too early.
    """
    return trial_deriv >= c2 * current_deriv


fn strong_wolfe_curvature(
    current_deriv: Float32,
    trial_deriv: Float32,
    c2: Float32
) -> Bool:
    """
    Check strong Wolfe curvature condition.

    |g(x + alpha*d)^T * d| <= c2 * |g(x)^T * d|
    """
    var abs_trial = trial_deriv if trial_deriv >= Float32(0.0) else -trial_deriv
    var abs_current = current_deriv if current_deriv >= Float32(0.0) else -current_deriv
    return abs_trial <= c2 * abs_current


fn cubic_interpolation(
    alpha_lo: Float32, f_lo: Float32, g_lo: Float32,
    alpha_hi: Float32, f_hi: Float32
) -> Float32:
    """
    Cubic interpolation to find minimum.

    Used for more sophisticated line search.
    """
    var d1 = g_lo + (f_hi - f_lo) / (alpha_hi - alpha_lo)
    var d2_sq = d1 * d1 - g_lo * (f_hi - f_lo) / (alpha_hi - alpha_lo)

    if d2_sq < Float32(0.0):
        # Fall back to bisection
        return Float32(0.5) * (alpha_lo + alpha_hi)

    var d2 = sqrt(d2_sq)

    # Cubic minimum
    var alpha_new = alpha_lo + (alpha_hi - alpha_lo) * (d2 - g_lo - d1) / (Float32(2.0) * d2)

    # Ensure we stay in bounds
    var lower = alpha_lo + Float32(0.1) * (alpha_hi - alpha_lo)
    var upper = alpha_lo + Float32(0.9) * (alpha_hi - alpha_lo)

    if alpha_new < lower:
        alpha_new = lower
    elif alpha_new > upper:
        alpha_new = upper

    return alpha_new


# ============================================================================
# L-BFGS OPTIMIZER
# ============================================================================

struct LBFGSState(Copyable, Movable):
    """
    State for L-BFGS optimizer.

    Stores history of position and gradient differences for
    two-loop recursion without explicit Hessian.
    """
    var history_size: Int
    var num_vars: Int       # 3 * num_atoms
    var iteration: Int
    var history_count: Int  # How many history entries are filled

    # Circular buffer for s = x_{k+1} - x_k
    var s: type_of(alloc[Float32](1))  # [history_size * num_vars]

    # Circular buffer for y = g_{k+1} - g_k
    var y: type_of(alloc[Float32](1))  # [history_size * num_vars]

    # rho = 1 / (y^T * s)
    var rho: type_of(alloc[Float32](1))  # [history_size]

    # alpha for two-loop recursion (temporary)
    var alpha: type_of(alloc[Float32](1))  # [history_size]

    # Previous position and gradient
    var prev_x: type_of(alloc[Float32](1))
    var prev_g: type_of(alloc[Float32](1))

    # Search direction
    var direction: type_of(alloc[Float32](1))

    fn __init__(out self, num_atoms: Int, history_size: Int):
        """Initialize L-BFGS state."""
        self.history_size = history_size
        self.num_vars = 3 * num_atoms
        self.iteration = 0
        self.history_count = 0

        var buffer_size = history_size * self.num_vars

        self.s = alloc[Float32](buffer_size)
        self.y = alloc[Float32](buffer_size)
        self.rho = alloc[Float32](history_size)
        self.alpha = alloc[Float32](history_size)
        self.prev_x = alloc[Float32](self.num_vars)
        self.prev_g = alloc[Float32](self.num_vars)
        self.direction = alloc[Float32](self.num_vars)

        # Initialize to zero
        for i in range(buffer_size):
            self.s[i] = 0.0
            self.y[i] = 0.0

        for i in range(history_size):
            self.rho[i] = 0.0
            self.alpha[i] = 0.0

        for i in range(self.num_vars):
            self.prev_x[i] = 0.0
            self.prev_g[i] = 0.0
            self.direction[i] = 0.0

    fn __copyinit__(out self, other: Self):
        self.history_size = other.history_size
        self.num_vars = other.num_vars
        self.iteration = other.iteration
        self.history_count = other.history_count

        var buffer_size = self.history_size * self.num_vars

        self.s = alloc[Float32](buffer_size)
        self.y = alloc[Float32](buffer_size)
        self.rho = alloc[Float32](self.history_size)
        self.alpha = alloc[Float32](self.history_size)
        self.prev_x = alloc[Float32](self.num_vars)
        self.prev_g = alloc[Float32](self.num_vars)
        self.direction = alloc[Float32](self.num_vars)

        for i in range(buffer_size):
            self.s[i] = other.s[i]
            self.y[i] = other.y[i]

        for i in range(self.history_size):
            self.rho[i] = other.rho[i]
            self.alpha[i] = other.alpha[i]

        for i in range(self.num_vars):
            self.prev_x[i] = other.prev_x[i]
            self.prev_g[i] = other.prev_g[i]
            self.direction[i] = other.direction[i]

    fn __moveinit__(out self, deinit other: Self):
        self.history_size = other.history_size
        self.num_vars = other.num_vars
        self.iteration = other.iteration
        self.history_count = other.history_count
        self.s = other.s
        self.y = other.y
        self.rho = other.rho
        self.alpha = other.alpha
        self.prev_x = other.prev_x
        self.prev_g = other.prev_g
        self.direction = other.direction

    fn free(self):
        """Release allocated memory."""
        self.s.free()
        self.y.free()
        self.rho.free()
        self.alpha.free()
        self.prev_x.free()
        self.prev_g.free()
        self.direction.free()


fn lbfgs_two_loop(
    mut state: LBFGSState,
    gradient: type_of(alloc[Float32](1))
):
    """
    L-BFGS two-loop recursion to compute search direction.

    Computes H * g without forming the Hessian explicitly.
    Result is stored in state.direction.
    """
    var n = state.num_vars
    var m = state.history_count

    # Copy gradient to direction (q = g in algorithm)
    for i in range(n):
        state.direction[i] = gradient[i]

    if m == 0:
        # First iteration: use steepest descent
        for i in range(n):
            state.direction[i] = -gradient[i]
        return

    # First loop: from most recent to oldest
    for i in range(m - 1, -1, -1):
        var idx = i % state.history_size
        var offset = idx * n

        # alpha_i = rho_i * s_i^T * q
        var dot: Float32 = 0.0
        for j in range(n):
            dot += state.s[offset + j] * state.direction[j]
        state.alpha[idx] = state.rho[idx] * dot

        # q = q - alpha_i * y_i
        for j in range(n):
            state.direction[j] = state.direction[j] - state.alpha[idx] * state.y[offset + j]

    # Compute initial Hessian approximation: gamma * I
    # gamma = s^T * y / (y^T * y) from most recent update
    var latest_idx = (m - 1) % state.history_size
    var latest_offset = latest_idx * n

    var sy: Float32 = 0.0
    var yy: Float32 = 0.0
    for j in range(n):
        sy += state.s[latest_offset + j] * state.y[latest_offset + j]
        yy += state.y[latest_offset + j] * state.y[latest_offset + j]

    var gamma = sy / yy if yy > Float32(1e-10) else Float32(1.0)

    # r = gamma * q
    for j in range(n):
        state.direction[j] = gamma * state.direction[j]

    # Second loop: from oldest to most recent
    for i in range(m):
        var idx = i % state.history_size
        var offset = idx * n

        # beta = rho_i * y_i^T * r
        var dot: Float32 = 0.0
        for j in range(n):
            dot += state.y[offset + j] * state.direction[j]
        var beta = state.rho[idx] * dot

        # r = r + (alpha_i - beta) * s_i
        var diff = state.alpha[idx] - beta
        for j in range(n):
            state.direction[j] = state.direction[j] + diff * state.s[offset + j]

    # Negate to get descent direction
    for j in range(n):
        state.direction[j] = -state.direction[j]


fn lbfgs_update_history(
    mut state: LBFGSState,
    x_new: type_of(alloc[Float32](1)),
    g_new: type_of(alloc[Float32](1))
):
    """
    Update L-BFGS history with new position and gradient.
    """
    var n = state.num_vars
    var idx = state.iteration % state.history_size
    var offset = idx * n

    # s = x_new - x_old
    # y = g_new - g_old
    var sy: Float32 = 0.0
    var yy: Float32 = 0.0

    for i in range(n):
        state.s[offset + i] = x_new[i] - state.prev_x[i]
        state.y[offset + i] = g_new[i] - state.prev_g[i]
        sy += state.s[offset + i] * state.y[offset + i]
        yy += state.y[offset + i] * state.y[offset + i]

    # rho = 1 / (y^T * s)
    state.rho[idx] = Float32(1.0) / sy if sy > Float32(1e-10) else Float32(0.0)

    # Store current as previous
    for i in range(n):
        state.prev_x[i] = x_new[i]
        state.prev_g[i] = g_new[i]

    state.iteration += 1
    if state.history_count < state.history_size:
        state.history_count += 1


# ============================================================================
# COORDINATE UTILITIES
# ============================================================================

fn flatten_coordinates(
    x: type_of(alloc[Float32](1)),
    y: type_of(alloc[Float32](1)),
    z: type_of(alloc[Float32](1)),
    mut flat: type_of(alloc[Float32](1)),
    num_atoms: Int
):
    """Flatten xyz coordinates to 1D array."""
    for i in range(num_atoms):
        flat[3 * i + 0] = x[i]
        flat[3 * i + 1] = y[i]
        flat[3 * i + 2] = z[i]


fn unflatten_coordinates(
    flat: type_of(alloc[Float32](1)),
    mut x: type_of(alloc[Float32](1)),
    mut y: type_of(alloc[Float32](1)),
    mut z: type_of(alloc[Float32](1)),
    num_atoms: Int
):
    """Unflatten 1D array to xyz coordinates."""
    for i in range(num_atoms):
        x[i] = flat[3 * i + 0]
        y[i] = flat[3 * i + 1]
        z[i] = flat[3 * i + 2]


fn flatten_gradients(
    grad_x: type_of(alloc[Float32](1)),
    grad_y: type_of(alloc[Float32](1)),
    grad_z: type_of(alloc[Float32](1)),
    mut flat: type_of(alloc[Float32](1)),
    num_atoms: Int
):
    """Flatten xyz gradients to 1D array."""
    for i in range(num_atoms):
        flat[3 * i + 0] = grad_x[i]
        flat[3 * i + 1] = grad_y[i]
        flat[3 * i + 2] = grad_z[i]


fn gradient_norm(gradient: type_of(alloc[Float32](1)), n: Int) -> Float32:
    """Compute L2 norm of gradient."""
    var sum_sq: Float32 = 0.0
    for i in range(n):
        sum_sq += gradient[i] * gradient[i]
    return sqrt(sum_sq)


fn gradient_rms_from_flat(gradient: type_of(alloc[Float32](1)), n: Int) -> Float32:
    """Compute RMS gradient."""
    var sum_sq: Float32 = 0.0
    for i in range(n):
        sum_sq += gradient[i] * gradient[i]
    return sqrt(sum_sq / Float32(n))
