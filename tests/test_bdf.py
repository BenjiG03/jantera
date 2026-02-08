
import jax
import jax.numpy as jnp
import time
from jantera.bdf import bdf_solve

def robertson_rhs(t, y, args):
    # Stiff problem (Robertson 1966)
    # k1 = 0.04, k2 = 3e7, k3 = 1e4
    dx = -0.04 * y[0] + 1e4 * y[1] * y[2]
    dz = 3e7 * y[1] * y[1]
    dy = -dx - dz
    return jnp.stack([dx, dy, dz])

def test_robertson():
    y0 = jnp.array([1.0, 0.0, 0.0])
    t0 = 0.0
    t_end = 1e6 # Very long time to test stability
    
    print("Solving Robertson problem with custom BDF...")
    t_start = time.time()
    
    from jantera.bdf import get_sparsity_pattern, make_sparse_jac_fn
    # Compute sparsity at y=1 to ensure all terms have non-zero gradients
    sparsity = get_sparsity_pattern(lambda y: robertson_rhs(t0, y, None), jnp.ones_like(y0))
    print(f"Sparsity NNZ: {sparsity.nse}")
    
    # Create the optimized Jacobian function (this runs coloring)
    # Note: robertson_rhs takes (t, y, args), make_sparse_jac_fn handles it
    jac_fn = make_sparse_jac_fn(robertson_rhs, sparsity, args_val=None)

    # Capture jac_fn in closure
    solve_jit = jax.jit(
        lambda rhs, t, y, tend, args: bdf_solve(rhs, t, y, tend, args, jac_fn=jac_fn),
        static_argnames=['rhs']
    )
    
    # Run
    final_state, converged = solve_jit(robertson_rhs, t0, y0, t_end, None)
    
    elapsed = time.time() - t_start
    print(f"Solved in {elapsed:.4f}s")
    print(f"Final t: {final_state.t:.2e}")
    print(f"Final y: {final_state.y}")
    print(f"Steps: {final_state.n_steps}")
    print(f"Jacobian Evals: {final_state.n_jevals}")
    print(f"LU Decomps: {final_state.n_lu_decomps}")
    print(f"Ratio LU/Steps: {final_state.n_lu_decomps/final_state.n_steps:.2f}")
    print(f"Ratio Jac/Steps: {final_state.n_jevals/final_state.n_steps:.2f}")

if __name__ == "__main__":
    test_robertson()
