import jax
import jax.numpy as jnp
import cantera as ct
import time
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from canterax.loader import load_mechanism
from canterax.reactor import ReactorNet

def test_bdf_gri30():
    jax.config.update("jax_enable_x64", True)
    
    yaml_path = "gri30.yaml"
    mech = load_mechanism(yaml_path)
    net = ReactorNet(mech)
    
    T0, P0 = 1500.0, 101325.0
    X0 = "CH4:1, O2:2, N2:7.52"
    
    # Cantera for initial Y
    sol_ct = ct.Solution(yaml_path)
    sol_ct.TPX = T0, P0, X0
    y0 = jnp.array(sol_ct.Y)
    
    t_end = 1e-3
    
    print("Running Canterax BDF...")
    t0 = time.perf_counter()
    res_bdf = net.advance(T0, P0, y0, t_end, solver="bdf", rtol=1e-8, atol=1e-12)
    # Block until ready
    jax.block_until_ready(res_bdf["ys"])
    t_bdf = time.perf_counter() - t0
    
    print(f"BDF Time: {t_bdf*1000:.2f} ms")
    print(f"BDF Steps: {res_bdf['stats']['num_steps']}")
    print(f"Final T: {res_bdf['ys'][-1, 0]:.2f} K")
    
    print("\nRunning Canterax Kvaerno5 (Reference)...")
    t0 = time.perf_counter()
    res_ref = net.advance(T0, P0, y0, t_end, rtol=1e-8, atol=1e-12)
    jax.block_until_ready(res_ref.ys)
    t_ref = time.perf_counter() - t0
    
    print(f"Ref Time: {t_ref*1000:.2f} ms")
    print(f"Ref Steps: {res_ref.stats['num_steps']}")
    print(f"Final T: {res_ref.ys[-1, 0]:.2f} K")

if __name__ == "__main__":
    test_bdf_gri30()
