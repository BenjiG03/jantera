
import jax
import jax.numpy as jnp
from typing import NamedTuple, Optional, Callable, Any, Tuple
import equinox as eqx
from jax.experimental import sparse
from sparsejac import jacfwd as sparse_jacfwd
import lineax as lx



class BDFState(NamedTuple):
    t: float
    y: jnp.ndarray
    h: float
    h_old: float
    # Nordsieck history (simplest form: just previous values for now)
    # y_history: [y_{n}, y_{n-1}, ...]
    y_history: jnp.ndarray 
    order: int
    step: int
    # Jacobian management
    jacobian_val: jnp.ndarray # BCOO or Dense
    jac_is_sparse: bool
    last_jac_step: int
    # Statistics
    n_steps: int
    n_fevals: int
    n_jevals: int
    n_lu_decomps: int

def init_bdf_state(t0, y0, h0, jac_structure, order=1):
    # Initialize history with y0 repeated
    # shape: (order+1, dim)
    y_history = jnp.repeat(y0[None, :], order+1, axis=0)
    
    return BDFState(
        t=t0, y=y0, h=h0, h_old=h0, y_history=y_history, order=order, step=0,
        jacobian_val=jac_structure, 
        jac_is_sparse=isinstance(jac_structure, sparse.BCOO), 
        last_jac_step=-999,
        n_steps=0, n_fevals=0, n_jevals=0, n_lu_decomps=0
    )

def get_sparsity_pattern(func, y0):
    """Computes sparsity pattern of func(y) at y0."""
    # Use standard jacfwd to find pattern
    J_sample = jax.jacfwd(func)(y0)
    # Convert to BCOO for sparsejac
    return sparse.BCOO.fromdense(jnp.abs(J_sample) > 0)

# BDF Coefficients (Fixed Step, Order 1-5)
# Normalized form: y_n - sum(alpha_i * y_{n-i}) = beta * h * f(y_n)
# Note: My previous definition was slightly different. 
# Standard BDF2: y_n - (4/3 y_{n-1} - 1/3 y_{n-2}) = 2/3 h f(y_n)
# So for my formulation: G(y) = y - psi - beta * h * f(y)
# psi = sum(alpha_i * y_{n-i}) for i=1..q
# beta = 2/3

# Alphas for y_{n-1}, y_{n-2}, ...
# Order 1: y_n - y_{n-1} = h f(y_n) -> psi = y_{n-1}, beta = 1.0
# Order 2: y_n - 4/3 y_{n-1} + 1/3 y_{n-2} = 2/3 h f(y_n) -> psi = 4/3 y_{n-1} - 1/3 y_{n-2}, beta = 2/3

BDF_ALPHAS = [
    [0.0, 0.0], # Order 0
    [1.0, 0.0], # Order 1
    [4.0/3.0, -1.0/3.0] # Order 2
]
BDF_BETAS = [
    0.0,
    1.0, 
    2.0/3.0
]


# Convert to JAX arrays for JIT indexing
BDF_ALPHAS_ARR = jnp.array(BDF_ALPHAS)
BDF_BETAS_ARR = jnp.array(BDF_BETAS)

# @jax.jit removed: called inside while_loop@jax.jit
def bdf_step(state: BDFState, rhs_fn, t_target, args=None, jac_fn=None, rtol=1e-6, atol=1e-8):
    """Performs one step of BDF method with adaptive step size and rejection."""
    
    # 1. Predict (Linear Extrapolation with variable step size)
    y_n = state.y_history[0]
    y_nm1 = state.y_history[1]
    
    # y_pred = y_n + (y_n - y_{n-1}) * (h / h_old)
    # Limit ratio to avoid instability if h jumps too much? 
    # But h is controlled.
    ratio = state.h / state.h_old
    y_pred = y_n + (y_n - y_nm1) * ratio
    
    # 2. Setup constants
    h = state.h
    order = state.order
    # Ensure order is within bounds (max 2 for now)
    order = jnp.minimum(order, 2)
    
    # Calculate variable step coefficients
    # BDF1 is always fixed (Euler)
    # BDF2 depends on rho = h / h_old
    rho = state.h / state.h_old
    
    # Default to Fixed Coeffs (Order 1)
    beta = 1.0
    alpha0 = 1.0
    alpha1 = 0.0
    
    def get_bdf2_coeffs(r):
        # returns beta, alpha0 (for y_{n-1}), alpha1 (for y_{n-2})
        denom = 1.0 + 2.0 * r
        b = (1.0 + r) / denom
        a0 = (1.0 + r)**2 / denom
        a1 = -(r**2) / denom
        return b, a0, a1

    # Select coefficients based on order
    # Note: we use jax.lax.cond to select
    
    beta, alpha0, alpha1 = jax.lax.cond(
        order == 2,
        lambda: get_bdf2_coeffs(rho),
        lambda: (1.0, 1.0, 0.0) # Order 1
    )
    
    # psi = alpha0 * y_n + alpha1 * y_nm1
    # Note: my history indexing: y_history[0] is y_{n-1}, y_history[1] is y_{n-2}
    # Wait, in bdf_step, y_n variable is y_history[0] (which is y_{n-1}).
    # and y_nm1 is y_history[1] (which is y_{n-2}).
    # My naming in previous block: y_n = history[0], y_nm1 = history[1].
    # So psi = alpha0 * y_history[0] + alpha1 * y_history[1]
    
    psi = alpha0 * y_n + alpha1 * y_nm1 
    
    def residual(y):
        return y - h * beta * rhs_fn(state.t + h, y, args) - psi

    # ... (rest of function until new_order logic) ...



    # 3. Jacobian (same as before)
    should_update_jac = (state.step - state.last_jac_step) >= 20
    
    def compute_jac(y):
        t_eval = state.t + h
        if jac_fn is not None:
             return jac_fn(t_eval, y, args)
        else:
            return jax.jacfwd(lambda y_arg: rhs_fn(t_eval, y_arg, args))(y)

    current_jac = jax.lax.cond(
        should_update_jac,
        lambda: compute_jac(y_pred),
        lambda: state.jacobian_val
    )
    last_jac_step = jax.lax.cond(
        should_update_jac,
        lambda: state.step,
        lambda: state.last_jac_step
    )
    
    # 4. Iteration Matrix M = I - h * beta * J
    def get_iteration_matrix(J):
        if isinstance(J, sparse.BCOO):
            J_dense = J.todense()
            return jnp.eye(J_dense.shape[0]) - h * beta * J_dense
        else:
            return jnp.eye(J.shape[0]) - h * beta * J
            
    M = get_iteration_matrix(current_jac)
    
    # 5. Newton Iteration
    def newton_body(carry):
        y, i, converged = carry
        R = residual(y)
        dy = jnp.linalg.solve(M, -R)
        y_new = y + dy
        weight = 1.0 / (atol + rtol * jnp.abs(y_new))
        error = jnp.linalg.norm(dy * weight) / jnp.sqrt(y.size)
        return y_new, i + 1, error < 0.1 

    # Allow more iterations (up to 6) because sparse linear solve is fast
    # and predictor might still be far for large h
    y_new, iters, converged = jax.lax.while_loop(
        lambda c: (c[1] < 6) & (~c[2]),
        newton_body,
        (y_pred, 0, False)
    )
    
    # 6. Step Size Control & Rejection Logic
    step_accepted = converged
    
    # Calculate scale factor
    # Fast (<=2 iters): 2.0x
    # Moderate (3-4 iters): 1.2x (Grow slowly)
    # Slow (5-6 iters): 0.8x (Shrink gently)
    # Fail: 0.3x (Shrink hard)
    
    scale = jax.lax.cond(
        converged,
        lambda: jax.lax.cond(iters <= 2, lambda: 2.0, lambda: jax.lax.cond(iters <= 4, lambda: 1.2, lambda: 0.8)),
        lambda: 0.3
    )
    
    h_new = h * scale 
    h_new = jnp.minimum(h_new, t_target - state.t)
    
    # Auto-order increase?
    new_order = jax.lax.cond(
        (order < 2) & converged,
        lambda: order + 1,
        lambda: order
    )
    
    # Prepare states
    accepted_t = state.t + h
    accepted_y = y_new
    accepted_history = jnp.concatenate([y_new[None, :], state.y_history[:-1]], axis=0)
    
    # Next h_old is current h (if accepted)
    accepted_h_old = h
    
    # Rejected state: t, y same. h_old same (we haven't moved).
    # h for next attempt is h_new.
    rejected_h_old = state.h_old
    
    next_t = jax.lax.cond(step_accepted, lambda: accepted_t, lambda: state.t)
    next_y = jax.lax.cond(step_accepted, lambda: accepted_y, lambda: state.y)
    next_hist = jax.lax.cond(step_accepted, lambda: accepted_history, lambda: state.y_history)
    next_order = jax.lax.cond(step_accepted, lambda: new_order, lambda: state.order)
    next_h_old = jax.lax.cond(step_accepted, lambda: accepted_h_old, lambda: rejected_h_old)
    
    # Stats update
    new_n_steps = state.n_steps + 1 
    new_n_fevals = state.n_fevals + iters
    new_n_jevals = state.n_jevals + should_update_jac.astype(int)
    new_n_lu = state.n_lu_decomps + 1
    
    new_state = BDFState(
        t = next_t,
        y = next_y,
        h = h_new,
        h_old = next_h_old,
        y_history = next_hist,
        order = next_order,
        step = new_n_steps,
        jacobian_val = current_jac,
        jac_is_sparse = True,
        last_jac_step = last_jac_step,
        n_steps = new_n_steps,
        n_fevals = new_n_fevals,
        n_jevals = new_n_jevals,
        n_lu_decomps = new_n_lu
    )
    
    return new_state, step_accepted

def make_sparse_jac_fn(rhs_fn, sparsity_pattern, args_val=None):
    """Creates a sparse Jacobian function J(t, y, args) handling time dependency."""
    dim = sparsity_pattern.shape[0]
    
    # Augmented RHS: f_aug(z) where z = [t, y]
    def aug_rhs(z):
        t = z[0]
        y = z[1:]
        dy = rhs_fn(t, y, args_val)
        return jnp.concatenate([jnp.array([1.0]), dy])

    # Augmented sparsity: (dim+1, dim+1)
    indices = sparsity_pattern.indices
    indices_y = indices + 1
    # Add dependency on t (column 0) for all variables
    indices_t = jnp.stack([jnp.arange(1, dim+1), jnp.zeros(dim, dtype=indices.dtype)], axis=1)
    aug_indices = jnp.concatenate([indices_y, indices_t], axis=0)
    # Add diagonal for t (0,0) - technically 0=0+1*0
    aug_indices = jnp.concatenate([aug_indices, jnp.array([[0, 0]])], axis=0)
    
    aug_nse = aug_indices.shape[0]
    aug_shape = (dim+1, dim+1)
    
    aug_sparsity = sparse.BCOO(
        (jnp.ones(aug_nse, dtype=bool), aug_indices),
        shape=aug_shape
    )
    
    # Helper to create aug_jac_fn
    # Note: sparse_jacfwd might struggle with BCOO construction inside if traced?
    # But this function is called at setup time (not traced).
    aug_jac_fn = sparse_jacfwd(aug_rhs, aug_sparsity)
    
    def jac_fn(t, y, args):
        z = jnp.concatenate([jnp.array([t]), y])
        J_aug = aug_jac_fn(z)
        # Return J_y block [1:, 1:]
        return J_aug[1:, 1:]
        
    return jac_fn

def bdf_solve(rhs_fn, t0, y0, t_end, args=None, jac_fn=None, h_init=1e-8, rtol=1e-6, atol=1e-8):
    """Main solver loop."""
    
    # Initialize Jacobian structure
    if jac_fn is not None:
        # Evaluate at t0, y0 to get structure (BCOO)
        jac_structure = jac_fn(t0, y0, args)
    else:
        # Dense zero matrix
        jac_structure = jnp.zeros((y0.size, y0.size))

    # Initialize state
    state0 = init_bdf_state(t0, y0, h_init, jac_structure)
    
    def cond_fn(carry):
        state, _ = carry
        # Continue if time < t_end and steps < max_steps
        return (state.t < t_end) & (state.n_steps < 100000)

    def body_fn(carry):
        state, _ = carry
        # Use lax.stop_gradient for step size to avoid differentiability issues for now
        # or just keep it differentiable if possible.
        new_state, converged = bdf_step(state, rhs_fn, t_end, args, jac_fn, rtol, atol)
        
        # If not converged, we should technically reduce step and retry.
        # For this prototype, we just continue (step size controller inside bdf_step shrinks h)
        # In a real solver, we would have a retry loop inside the step.
        return new_state, converged

    state_final, converged_final = jax.lax.while_loop(
        cond_fn,
        body_fn,
        (state0, True)
    )
    
    return state_final, converged_final
