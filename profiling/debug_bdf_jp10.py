import time
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import cantera as ct
import matplotlib.pyplot as plt
import os
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from jantera.loader import load_mechanism
from jantera.solution import Solution
from jantera.reactor import ReactorNet
from jantera.solvers.bdf import bdf_solve

def debug_jp10():
    print("Loading JP-10...")
    yaml_path = os.path.join(os.path.dirname(__file__), "..", "jp10.yaml")
    mech = load_mechanism(yaml_path)
    sol = Solution(yaml_path)
    
    T0, P0 = 1500.0, 101325.0
    # "C10H16:1, O2:14, N2:52.64"
    sol.TPX = T0, P0, "C10H16:1, O2:14, N2:52.64"
    y0 = sol.Y
    
    t_end = 1e-3
    
    print(f"Running BDF on JP-10 to {t_end}s...")
    
    # We want to capture statistics. 
    # Since bdf_solve is inside ReactorNet.advance, we can call it directly or modify it.
    # Let's call ReactorNet.advance with our solver, but we really want trace info.
    # To get trace info without modifying library code permanently, 
    # I will rely on standard JAX behavior or just observe the result first.
    
    net = ReactorNet(mech)
    
    start = time.time()
    # Advance returns a dict in the latest version: {"ts": ..., "ys": ..., "stats": ...}
    res = net.advance(T0, P0, y0, t_end, solver="bdf")
    res = jax.block_until_ready(res)
    end = time.time()
    
    steps = res["stats"]["num_steps"]
    print(f"Finished in {end - start:.4f}s")
    print(f"Steps taken: {steps}")
    print(f"Final Time: {res['ts'][-1] if 'ts' in res else 'N/A'}")
    print(f"Status: {res['stats']}")
    
    # If it hit 4000, it's likely stalled.
    # Let's try to run Cantera for comparison to see its steps
    print("\nRunning Cantera CVODE...")
    sol_ct = ct.Solution(yaml_path)
    sol_ct.TPX = T0, P0, "C10H16:1, O2:14, N2:52.64"
    r = ct.IdealGasConstPressureReactor(sol_ct)
    sim = ct.ReactorNet([r])
    
    start = time.time()
    sim.advance(t_end)
    end = time.time()
    print(f"Cantera finished in {end - start:.4f}s")
    
    # Cantera stats aren't easily available in simple interface, but we can try
    try:
        print(f"Cantera Stats: {sim.get_solver_stats()}")
    except:
        print("Cantera stats not available")

if __name__ == "__main__":
    debug_jp10()
