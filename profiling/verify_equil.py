import jax
import jax.numpy as jnp
import cantera as ct
import time
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from canterax.loader import load_mechanism
from canterax.solution import Solution as CanteraxSolution
from canterax.equilibrate import equilibrate

def test_equil_robustness():
    jax.config.update("jax_enable_x64", True)
    
    yaml_path = "gri30.yaml"
    sol = CanteraxSolution(yaml_path)
    
    T, P = 1500.0, 101325.0
    X = "CH4:1, O2:2, N2:7.52"
    
    sol.TPX = T, P, X
    
    print("Running Canterax Equilibrate (Basis Optimized)...")
    t0 = time.perf_counter()
    res = equilibrate(sol, rtol=1e-12)
    jax.block_until_ready(sol.Y)
    t_equil = time.perf_counter() - t0
    
    print(f"Equil Time: {t_equil*1000:.2f} ms")
    # print(f"Result type: {type(res)}")
    # In some optimistix versions it might be different. 
    # Usually it's 'result' or 'value'
    print(f"Result Info: {res.result}")
    print(f"Steps: {res.stats['num_steps']}")
    print(f"Final T: {sol.T:.2f} K")
    print(f"Final P: {sol.P:.2f} Pa")
    
    # Comparison with Cantera
    sol_ct = ct.Solution(yaml_path)
    sol_ct.TPX = T, P, X
    sol_ct.equilibrate("TP")
    
    err = jnp.linalg.norm(sol.Y - sol_ct.Y)
    print(f"L2 Error vs Cantera: {err:.2e}")

import optimistix as optx
if __name__ == "__main__":
    test_equil_robustness()
