"""
Quick performance comparison with different SaveAt configurations.
"""
import os
import sys
import time
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import cantera as ct

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from canterax.loader import load_mechanism
from canterax.reactor import ReactorNet
from diffrax import Kvaerno5, SaveAt


def quick_perf_test():
    print("=" * 60)
    print("QUICK PERFORMANCE COMPARISON")
    print("=" * 60)
    
    yaml_path = "gri30.yaml"
    T0, P0 = 1500.0, 101325.0
    X0 = "CH4:1, O2:2, N2:7.52"
    t_end = 1e-3
    
    mech = load_mechanism(yaml_path)
    sol_ct = ct.Solution(yaml_path)
    sol_ct.TPX = T0, P0, X0
    Y0 = jnp.array(sol_ct.Y)
    
    net = ReactorNet(mech)
    
    # Warmup
    print("Warming up JIT...")
    net.advance(T0, P0, Y0, 1e-8)
    jax.block_until_ready(net.advance(T0, P0, Y0, 1e-8))
    
    # Test 1: SaveAt(t1=True) - only final state
    print("\nTest 1: SaveAt(t1=True) - final state only")
    saveat = SaveAt(t1=True)
    start = time.perf_counter()
    res = net.advance(T0, P0, Y0, t_end, solver=Kvaerno5(), saveat=saveat)
    jax.block_until_ready(res)
    t1 = time.perf_counter() - start
    print(f"  Time: {t1*1e3:.1f} ms, T_final: {float(res.ys[-1, 0]):.2f} K")
    
    # Test 2: Default saveat (t1=True is default in ReactorNet)
    print("\nTest 2: Default saveat (no explicit SaveAt)")
    start = time.perf_counter()
    res = net.advance(T0, P0, Y0, t_end, solver=Kvaerno5())
    jax.block_until_ready(res)
    t2 = time.perf_counter() - start
    print(f"  Time: {t2*1e3:.1f} ms, T_final: {float(res.ys[-1, 0]):.2f} K")
    
    # Test 3: SaveAt with 50 points
    print("\nTest 3: SaveAt(ts=linspace, 50 points)")
    saveat = SaveAt(ts=jnp.linspace(0, t_end, 50))
    start = time.perf_counter()
    res = net.advance(T0, P0, Y0, t_end, solver=Kvaerno5(), saveat=saveat)
    jax.block_until_ready(res)
    t3 = time.perf_counter() - start
    print(f"  Time: {t3*1e3:.1f} ms, T_final: {float(res.ys[-1, 0]):.2f} K")
    
    # Cantera comparison
    print("\n--- Cantera Reference ---")
    sol_ct.TPX = T0, P0, X0
    reac = ct.IdealGasConstPressureReactor(sol_ct)
    net_ct = ct.ReactorNet([reac])
    
    start = time.perf_counter()
    net_ct.advance(t_end)
    ct_time = time.perf_counter() - start
    print(f"  Time: {ct_time*1e3:.1f} ms, T_final: {sol_ct.T:.2f} K")


if __name__ == "__main__":
    quick_perf_test()
