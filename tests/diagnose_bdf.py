
import os
import sys
import time
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jantera.loader import load_mechanism
from jantera.reactor import ReactorNet

def diagnose():
    yaml = "gri30.yaml"
    T0, P0, X0 = 1500.0, 101325.0, "CH4:1, O2:2, N2:7.52"
    t_end = 1e-4 # 100 microseconds (same as benchmark)
    
    print(f"Diagnosing BDF Solver on {yaml} for {t_end} s...")
    
    mech = load_mechanism(yaml)
    # Convert X0 to Y0 manually for now (using Cantera loading behind scenes in loader usually, but let's assume loader works)
    # Actually loader returns MechData. We need Y0.
    # Use Cantera to get Y0 easily
    import cantera as ct
    sol = ct.Solution(yaml)
    sol.TPX = T0, P0, X0
    Y0 = jnp.array(sol.Y)
    
    net = ReactorNet(mech)
    
    print("Pre-compiling...")
    start = time.perf_counter()
    # Run for tiny step to warm up
    # Use custom solver
    res_warm = net.advance(T0, P0, Y0, 1e-9, solver="custom_bdf")
    jax.block_until_ready(res_warm.ys)
    print(f"Compilation took: {time.perf_counter() - start:.4f} s")
    print(f"Warmup stats: {res_warm.stats}")
    
    print("Running diagnosis...")
    start = time.perf_counter()
    res = net.advance(T0, P0, Y0, t_end, solver="custom_bdf")
    jax.block_until_ready(res.ys)
    elapsed = time.perf_counter() - start
    
    print(f"Elapsed: {elapsed:.4f} s")
    print("Stats:")
    for k, v in res.stats.items():
        print(f"  {k}: {v}")
        
    n_steps = int(res.stats['n_steps'])
    if n_steps > 0:
        print(f"Time/step: {elapsed/n_steps*1e6:.1f} us")
        print(f"Avg h: {t_end/n_steps:.2e}")
        
    print(f"Final T: {res.ys[0, 0]:.2f} K")

if __name__ == "__main__":
    diagnose()
