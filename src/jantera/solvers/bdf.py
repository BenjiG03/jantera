import jax
import jax.numpy as jnp
from jax import jacfwd, jit, lax
from jax.scipy.linalg import lu_factor, lu_solve
import equinox as eqx

class BDFState(eqx.Module):
    """Internal state for the BDF solver."""
    y: jax.Array        # Current state
    t: jax.Array        # Current time
    h: jax.Array        # Current step size
    order: jax.Array    # Current order (dynamic)
    z: jax.Array        # Nordsieck history (6, n) for orders 1-5
    
    # Jacobian and LU caching
    jac: jax.Array      # Cached Jacobian
    lu: jax.Array       # LU factorization of W = l1*I - h*l0*J
    piv: jax.Array      # Pivots for LU
    jac_t: jax.Array    # Time at which Jacobian was evaluated
    jac_h: jax.Array    # h at which Jacobian was evaluated
    
    # Stats
    n_steps: int
    n_fevals: int
    n_jevals: int
    n_niter: int
    
    # Error control
    error_ratio: jax.Array
    steps_at_current_order: int
    steps_since_jac: int

def get_pascal_matrix():
    """Pascal matrix for Nordsieck prediction (fixed 6x6)."""
    mat = jnp.array([
        [1, 1, 1, 1, 1, 1],
        [0, 1, 2, 3, 4, 5],
        [0, 0, 1, 3, 6, 10],
        [0, 0, 0, 1, 4, 10],
        [0, 0, 0, 0, 1, 5],
        [0, 0, 0, 0, 0, 1]
    ], dtype=jnp.float64)
    return mat

def get_bdf_coefficients(order):
    """BDF coefficients L for the Nordsieck form."""
    all_coeffs = jnp.array([
        [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],       # Order 1
        [1.0, 3/2, 1/2, 0.0, 0.0, 0.0],       # Order 2
        [1.0, 11/6, 1.0, 1/6, 0.0, 0.0],      # Order 3
        [1.0, 25/12, 35/24, 5/12, 1/24, 0.0], # Order 4
        [1.0, 137/60, 15/8, 17/24, 9/80, 1/120] # Order 5
    ])
    return all_coeffs[order - 1]

def rescale_nordsieck(z, h_ratio):
    """Rescale Nordsieck history when step size h changes."""
    powers = jnp.arange(6)
    factors = h_ratio ** powers
    return z * factors[:, jnp.newaxis]

def bdf_setup(h, order, jac):
    """Compute iteration matrix W and its LU factorization."""
    L = get_bdf_coefficients(order)
    l0, l1 = L[0], L[1]
    W = l1 * jnp.eye(jac.shape[0]) - h * l0 * jac
    lu, piv = lu_factor(W)
    return lu, piv

@eqx.filter_jit
def bdf_step(state, fun, args, rtol, atol):
    """Single BDF step with optional Jacobian and LU reuse."""
    order = state.order
    z = state.z
    h = state.h
    t = state.t
    
    # 1. Prediction
    z_pred = get_pascal_matrix() @ z
    y_pred = z_pred[0]
    t_next = t + h
    
    # 2. Correction (Newton)
    L = get_bdf_coefficients(order)
    l0, l1 = L[0], L[1]
    
    # Use cached LU/Piv if they match current h and order (Phase 2 logic)
    # For Phase 1, we just use what's in state.
    lu, piv = state.lu, state.piv
    
    def newton_body(val):
        delta, step_prev, count, converged = val
        y = y_pred + l0 * delta
        ydot = fun(t_next, y, args)
        res = z_pred[1] + l1 * delta - h * ydot
        
        # solve: (l1*I - h*l0*J) * step = res
        step = lu_solve((lu, piv), res)
        
        step_norm = jnp.sqrt(jnp.mean(jnp.square(step / (jnp.abs(y) * rtol + atol))))
        converged = step_norm < 0.3 # Relaxed from 0.05
        return (delta - step, step_norm, count + 1, converged)

    def newton_cond(val):
        delta, step_prev, count, converged = val
        return (~converged) & (count < 4)

    delta_final, _, n_iter, converged = lax.while_loop(
        newton_cond, newton_body, (jnp.zeros_like(y_pred), 1e10, 0, False)
    )
    
    # 3. Update result
    z_next = z_pred + jnp.outer(L, delta_final)
    
    # 4. Error estimation
    err_vec = delta_final / (jnp.abs(z_next[0]) * rtol + atol)
    error_ratio = jnp.sqrt(jnp.mean(jnp.square(err_vec))) / (order + 1.0)
    error_ratio = jnp.where(converged, error_ratio, 1e10)
    
    return BDFState(
        y=z_next[0], t=t_next, h=h, order=order, z=z_next,
        jac=state.jac, lu=state.lu, piv=state.piv,
        jac_t=state.jac_t, jac_h=state.jac_h,
        n_steps=state.n_steps + 1,
        n_fevals=state.n_fevals + n_iter + 1,
        n_jevals=state.n_jevals,
        n_niter=state.n_niter + n_iter,
        error_ratio=error_ratio,
        steps_at_current_order=state.steps_at_current_order + 1,
        steps_since_jac=state.steps_since_jac
    )

def bdf_solve(fun, t0, t1, y0, args=(), rtol=1e-6, atol=1e-9, max_steps=4000):
    """Adaptive BDF integration loop with Jacobian and LU caching."""
    n = y0.shape[0]
    h0 = 1e-8
    ydot0 = fun(t0, y0, args)
    z0 = jnp.zeros((6, n))
    z0 = z0.at[0].set(y0).at[1].set(h0 * ydot0)
    
    # Initial Jacobian and LU
    jac0 = jacfwd(fun, argnums=1)(t0, y0, args)
    lu0, piv0 = bdf_setup(h0, jnp.array(1), jac0)
    
    init_state = BDFState(
        y=y0, t=t0, h=h0, order=jnp.array(1), z=z0,
        jac=jac0, lu=lu0, piv=piv0,
        jac_t=t0, jac_h=h0,
        n_steps=0, n_fevals=1, n_jevals=1, n_niter=0,
        error_ratio=jnp.array(0.0), steps_at_current_order=0,
        steps_since_jac=0
    )
    
    def cond_fun(state):
        return (state.t < t1) & (state.n_steps < max_steps)
    
    def body_fun(state):
        # 1. Take step with cached Jacobian and LU
        ns = bdf_step(state, fun, args, rtol, atol)
        
        # 2. Handle failure or success
        newton_failed = ns.error_ratio >= 1e9
        
        def on_newton_fail(ns):
            # Refresh Jacobian and LU, retry same step
            new_jac = jacfwd(fun, argnums=1)(state.t, state.y, args)
            new_lu, new_piv = bdf_setup(state.h, state.order, new_jac)
            
            state_new_jac = BDFState(
                y=state.y, t=state.t, h=state.h, order=state.order, z=state.z,
                jac=new_jac, lu=new_lu, piv=new_piv,
                jac_t=state.t, jac_h=state.h,
                n_steps=state.n_steps, n_fevals=state.n_fevals, n_jevals=state.n_jevals + 1,
                n_niter=state.n_niter, error_ratio=state.error_ratio,
                steps_at_current_order=state.steps_at_current_order,
                steps_since_jac=0
            )
            return bdf_step(state_new_jac, fun, args, rtol, atol)
            
        ns = lax.cond(newton_failed, on_newton_fail, lambda x: x, ns)
        
        success = ns.error_ratio <= 1.0
        
        def on_success(ns):
            h_scale = jnp.clip(0.8 / (ns.error_ratio ** (1.0 / (ns.order + 1.0))), 0.1, 5.0)
            # Dampen growth if we just changed order or failed (steps_at_current_order is small)
            h_scale = jnp.where(ns.steps_at_current_order <= ns.order + 1, jnp.minimum(1.0, h_scale), h_scale)
            
            # Hold h constant if growth is small (helps LU reuse)
            h_scale = jnp.where((h_scale > 1.0) & (h_scale < 1.2), 1.0, h_scale)
            h_new = ns.h * h_scale
            
            # Order selection logic (every K+1 steps)
            should_change = ns.steps_at_current_order > (ns.order + 1)
            new_order = jnp.where(should_change & (ns.order < 5), ns.order + 1, ns.order)
            new_steps_at_order = jnp.where(should_change, 0, ns.steps_at_current_order)
            
            # Jacobian update logic: every 20 steps
            should_update_jac = (ns.steps_since_jac >= 20)
            def update_jac(_):
                return jacfwd(fun, argnums=1)(ns.t, ns.y, args), ns.n_jevals + 1, 0
            def reuse_jac(_):
                return ns.jac, ns.n_jevals, ns.steps_since_jac + 1
            
            new_jac, new_jevals, new_steps_since_jac = lax.cond(should_update_jac, update_jac, reuse_jac, None)
            
            # Setup LU for NEXT step (only if h/order/jac changed)
            # Use a slightly loose tolerance for h comparison to handle precision
            h_changed = jnp.abs(h_new - ns.h) > 1e-15 * ns.h
            should_setup = h_changed | (new_order != ns.order) | should_update_jac
            
            def setup_new_lu(_):
                return bdf_setup(h_new, new_order, new_jac)
            def reuse_old_lu(_):
                return ns.lu, ns.piv
            
            final_lu, final_piv = lax.cond(should_setup, setup_new_lu, reuse_old_lu, None)
            
            z_rescaled = rescale_nordsieck(ns.z, h_new / ns.h)
            return BDFState(
                y=ns.y, t=ns.t, h=h_new, order=new_order, z=z_rescaled,
                jac=new_jac, lu=final_lu, piv=final_piv,
                jac_t=ns.jac_t, jac_h=ns.jac_h,
                n_steps=ns.n_steps, n_fevals=ns.n_fevals, n_jevals=new_jevals,
                n_niter=ns.n_niter, error_ratio=ns.error_ratio,
                steps_at_current_order=new_steps_at_order,
                steps_since_jac=new_steps_since_jac
            )
            
        def on_failure(ns):
            h_new = state.h * 0.25
            new_order = jnp.maximum(jnp.array(1), state.order-1)
            # Failure always triggers a setup (usually)
            new_lu, new_piv = bdf_setup(h_new, new_order, state.jac)
            z_rescaled = rescale_nordsieck(state.z, h_new / state.h)
            return BDFState(
                y=state.y, t=state.t, h=h_new, order=new_order,
                z=z_rescaled, jac=state.jac, lu=new_lu, piv=new_piv,
                jac_t=state.jac_t, jac_h=state.jac_h,
                n_steps=state.n_steps, n_fevals=ns.n_fevals, n_jevals=ns.n_jevals,
                n_niter=ns.n_niter, error_ratio=ns.error_ratio, 
                steps_at_current_order=0,
                steps_since_jac=state.steps_since_jac
            )

        return lax.cond(success, on_success, on_failure, ns)

    return lax.while_loop(cond_fun, body_fun, init_state)
