"""
Check step count and T_final discrepancy at 1500K.
"""
import os
import sys
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import cantera as ct
import diffrax
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from jantera.loader import load_mechanism
from jantera.reactor import ReactorNet

def debug_trajectory():
    yaml_path = "gri30.yaml"
    T0, P0 = 1500.0, 101325.0
    X0 = "CH4:1, O2:2, N2:7.52"
    t_end = 1e-3
    
    mech = load_mechanism(yaml_path)
    sol_ct = ct.Solution(yaml_path)
    sol_ct.TPX = T0, P0, X0
    Y0 = jnp.array(sol_ct.Y)
    
    net = ReactorNet(mech)
    
    # JIT the advance call to match intended usage and measure performance correctly
    @jax.jit
    def run_solve():
        return net.advance(T0, P0, Y0, t_end, rtol=1e-10, atol=1e-14)

    print("Warming up JIT...")
    res = run_solve()
    jax.block_until_ready(res)
    
    print(f"Running solve...")
    start = time.perf_counter()
    res = run_solve()
    jax.block_until_ready(res)
    elapsed = time.perf_counter() - start
    
    print(f"\n--- Jantera Results ---")
    print(f"  Execution Time: {elapsed*1000:.2f} ms")
    print(f"  T_final: {float(res.ys[-1, 0]):.4f} K")
    if hasattr(res, 'stats'):
        print(f"  Steps: {res.stats.get('num_steps', 'N/A')}")
        print(f"  Accepted: {res.stats.get('num_accepted_steps', 'N/A')}")
        
    # Cantera
    print(f"\n--- Cantera Results ---")
    sol_ct.TPX = T0, P0, X0
    reac = ct.IdealGasConstPressureReactor(sol_ct)
    sim = ct.ReactorNet([reac])
    sim.rtol, sim.atol = 1e-10, 1e-14
    
    start_ct = time.perf_counter()
    sim.advance(t_end)
    elapsed_ct = time.perf_counter() - start_ct
    print(f"  Execution Time: {elapsed_ct*1000:.2f} ms")
    print(f"  T_final: {sol_ct.T:.4f} K")

    print(f"\n--- Discrepancy ---")
    print(f"  dT_final: {abs(float(res.ys[-1, 0]) - sol_ct.T):.4f} K")

if __name__ == "__main__":
    debug_trajectory()
