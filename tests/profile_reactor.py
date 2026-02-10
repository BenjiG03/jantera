"""
Profile Canterax reactor performance to identify bottlenecks.
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


def profile_reactor():
    print("=" * 60)
    print("REACTOR PERFORMANCE PROFILING")
    print("=" * 60)
    
    # Test configurations
    configs = [
        # (name, yaml, T, P, X, t_end)
        ("GRI-30 @ 1200K/1atm", "gri30.yaml", 1200.0, 101325.0, "CH4:1, O2:2, N2:7.52", 1e-3),
        ("GRI-30 @ 1500K/1atm", "gri30.yaml", 1500.0, 101325.0, "CH4:1, O2:2, N2:7.52", 1e-4),
        ("GRI-30 @ 2000K/10atm", "gri30.yaml", 2000.0, 10*101325.0, "CH4:1, O2:2, N2:7.52", 1e-3),
        ("JP-10 @ 1500K/1atm", "jp10.yaml", 1500.0, 101325.0, "C10H16:1, O2:14, N2:52.64", 1e-3),
    ]
    
    for name, yaml, T0, P0, X0, t_end in configs:
        print(f"\n--- {name} ---")
        
        # Load mechanism
        mech = load_mechanism(yaml)
        sol_ct = ct.Solution(yaml)
        sol_ct.TPX = T0, P0, X0
        Y0 = jnp.array(sol_ct.Y)
        
        net = ReactorNet(mech)
        
        # Warmup / JIT
        start_jit = time.perf_counter()
        res = net.advance(T0, P0, Y0, 1e-8)
        jax.block_until_ready(res)
        jit_time = time.perf_counter() - start_jit
        
        # Warm run
        start_warm = time.perf_counter()
        saveat = SaveAt(ts=jnp.linspace(0, t_end, 50))
        res = net.advance(T0, P0, Y0, t_end, rtol=1e-8, atol=1e-12, solver=Kvaerno5(), saveat=saveat)
        jax.block_until_ready(res)
        warm_time = time.perf_counter() - start_warm
        
        # Cantera comparison
        sol_ct.TPX = T0, P0, X0
        reac = ct.IdealGasConstPressureReactor(sol_ct)
        net_ct = ct.ReactorNet([reac])
        net_ct.rtol, net_ct.atol = 1e-8, 1e-12
        
        start_ct = time.perf_counter()
        net_ct.advance(t_end)
        ct_time = time.perf_counter() - start_ct
        
        # Final state comparison
        T_jt = float(res.ys[-1, 0])
        T_ct = sol_ct.T
        
        print(f"  JIT time:    {jit_time*1e3:.1f} ms")
        print(f"  Warm time:   {warm_time*1e3:.1f} ms")
        print(f"  Cantera:     {ct_time*1e3:.1f} ms")
        print(f"  Speedup:     {ct_time/warm_time:.2f}x" if warm_time < ct_time else f"  Slowdown:    {warm_time/ct_time:.2f}x")
        print(f"  T_final_JT:  {T_jt:.2f} K")
        print(f"  T_final_CT:  {T_ct:.2f} K")
        print(f"  dT_final:    {abs(T_jt - T_ct):.4f} K")


if __name__ == "__main__":
    profile_reactor()
