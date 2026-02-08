"""
Profile Jantera reactor performance - GRI-30 only (fast).
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

from jantera.loader import load_mechanism
from jantera.reactor import ReactorNet
from diffrax import Kvaerno5, Tsit5, SaveAt


def profile_gri30():
    print("=" * 60)
    print("REACTOR PERFORMANCE PROFILING (GRI-30)")
    print("=" * 60)
    
    yaml = "gri30.yaml"
    T0, P0 = 1500.0, 101325.0
    X0 = "CH4:1, O2:2, N2:7.52"
    t_end = 1e-4  # 100 microseconds
    
    # Load mechanism
    mech = load_mechanism(yaml)
    sol_ct = ct.Solution(yaml)
    sol_ct.TPX = T0, P0, X0
    Y0 = jnp.array(sol_ct.Y)
    
    net = ReactorNet(mech)
    
    # JIT compilation (first call)
    print("\n1. JIT Compilation (first call)...")
    start_jit = time.perf_counter()
    res = net.advance(T0, P0, Y0, 1e-8, solver=Kvaerno5())
    jax.block_until_ready(res)
    jit_time = time.perf_counter() - start_jit
    print(f"   JIT time: {jit_time:.1f} s")
    
    # Warm execution (subsequent calls)
    print("\n2. Warm Execution (JIT-compiled)...")
    n_runs = 5
    times = []
    for i in range(n_runs):
        start = time.perf_counter()
        saveat = SaveAt(ts=jnp.linspace(0, t_end, 50))
        res = net.advance(T0, P0, Y0, t_end, rtol=1e-8, atol=1e-12, solver=Kvaerno5(), saveat=saveat)
        jax.block_until_ready(res)
        times.append(time.perf_counter() - start)
        print(f"   Run {i+1}: {times[-1]*1e3:.1f} ms")
    
    avg_time = np.mean(times[1:])  # Exclude first warm run
    print(f"   Average (runs 2-{n_runs}): {avg_time*1e3:.1f} ms")
    
    # Cantera comparison
    print("\n3. Cantera Comparison...")
    sol_ct.TPX = T0, P0, X0
    reac = ct.IdealGasConstPressureReactor(sol_ct, clone=False)
    net_ct = ct.ReactorNet([reac])
    net_ct.rtol, net_ct.atol = 1e-8, 1e-12
    
    ct_times = []
    for i in range(n_runs):
        sol_ct.TPX = T0, P0, X0
        reac = ct.IdealGasConstPressureReactor(sol_ct, clone=False)
        net_ct = ct.ReactorNet([reac])
        net_ct.rtol, net_ct.atol = 1e-8, 1e-12
        start = time.perf_counter()
        net_ct.advance(t_end)
        ct_times.append(time.perf_counter() - start)
    
    ct_avg = np.mean(ct_times)
    print(f"   Cantera average: {ct_avg*1e3:.2f} ms")
    
    # Final state comparison
    T_jt = float(res.ys[-1, 0])
    T_ct = sol_ct.T
    
    print("\n" + "=" * 60)
    print("SUMMARY: GRI-30 @ 1500K, t=100μs")
    print("=" * 60)
    print(f"  JIT compile time:    {jit_time:.1f} s")
    print(f"  Jantera (warm):      {avg_time*1e3:.1f} ms")
    print(f"  Cantera:             {ct_avg*1e3:.2f} ms")
    print(f"  Speedup (warm):      {ct_avg/avg_time:.2f}x" if avg_time < ct_avg else f"  Slowdown (warm):     {avg_time/ct_avg:.2f}x")
    print(f"  T_final (Jantera):   {T_jt:.2f} K")
    print(f"  T_final (Cantera):   {T_ct:.2f} K")
    print(f"  ΔT:                  {abs(T_jt - T_ct):.4f} K")

if __name__ == "__main__":
    profile_gri30()
