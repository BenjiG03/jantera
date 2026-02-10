import jax
import jax.numpy as jnp
from canterax.solvers.bdf import bdf_solve
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

def test_bdf_simple():
    # dy/dt = -y
    def fun(t, y, args):
        return -y
    
    y0 = jnp.array([1.0])
    t0, t1 = 0.0, 1.0
    
    res = bdf_solve(fun, t0, t1, y0, max_steps=10)
    print(f"--- 10 Steps Stats ---")
    print(f"Final t: {res.t}")
    print(f"Final y: {res.y}")
    print(f"h: {res.h}")
    print(f"Error Ratio: {res.error_ratio}")
    print(f"n_iter count (total): {res.n_niter}")
    
    # Run to completion or failure
    res_full = bdf_solve(fun, t0, t1, y0, max_steps=1000)
    print(f"\n--- Full Run Stats ---")
    print(f"Final t: {res_full.t}")
    print(f"Final y: {res_full.y}")
    print(f"Exact y: {jnp.exp(-float(res_full.t))}")
    print(f"Steps: {res_full.n_steps}")

if __name__ == "__main__":
    test_bdf_simple()
