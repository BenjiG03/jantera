"""
Count ODE solver steps for different conditions to diagnose stiffness.
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
from diffrax import Kvaerno5, SaveAt


def count_steps():
    print("=" * 60)
    print("ODE SOLVER STEP COUNT ANALYSIS")
    print("=" * 60)
    
    # Test configurations - compare mild vs aggressive
    configs = [
        # (name, yaml, T, P, X, t_end)
        ("GRI-30 @ 1200K/1atm (mild)", "gri30.yaml", 1200.0, 101325.0, "CH4:1, O2:2, N2:7.52", 1e-3),
        ("GRI-30 @ 1500K/1atm (original)", "gri30.yaml", 1500.0, 101325.0, "CH4:1, O2:2, N2:7.52", 1e-4),
        ("GRI-30 @ 2000K/10atm (aggressive)", "gri30.yaml", 2000.0, 10*101325.0, "CH4:1, O2:2, N2:7.52", 1e-3),
    ]
    
    for name, yaml, T0, P0, X0, t_end in configs:
        print(f"\n--- {name} ---")
        
        # Load mechanism
        mech = load_mechanism(yaml)
        sol_ct = ct.Solution(yaml)
        sol_ct.TPX = T0, P0, X0
        Y0 = jnp.array(sol_ct.Y)
        
        net = ReactorNet(mech)
        
        # Run with stats
        start = time.perf_counter()
        saveat = SaveAt(t1=True)  # Only save final state
        res = net.advance(T0, P0, Y0, t_end, rtol=1e-8, atol=1e-12, solver=Kvaerno5(), saveat=saveat)
        jax.block_until_ready(res)
        elapsed = time.perf_counter() - start
        
        # Get stats
        n_steps = res.stats.get('num_steps', 'N/A') if hasattr(res, 'stats') else 'N/A'
        n_accepted = res.stats.get('num_accepted_steps', 'N/A') if hasattr(res, 'stats') else 'N/A'
        n_rejected = res.stats.get('num_rejected_steps', 'N/A') if hasattr(res, 'stats') else 'N/A'
        
        T_final = float(res.ys[-1, 0])
        
        print(f"  Time:       {elapsed*1e3:.1f} ms")
        print(f"  Steps:      {n_steps}")
        print(f"  Accepted:   {n_accepted}")
        print(f"  Rejected:   {n_rejected}")
        print(f"  T_initial:  {T0:.1f} K")
        print(f"  T_final:    {T_final:.1f} K")
        print(f"  dT:         {T_final - T0:.1f} K")


if __name__ == "__main__":
    count_steps()
