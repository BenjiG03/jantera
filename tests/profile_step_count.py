"""
ODE Step Count Comparison: Jantera vs Cantera.

Compares the number of ODE solver steps taken by diffrax and CVODE
to understand solver efficiency differences.

Usage:
    python profile_step_count.py
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
from diffrax import Kvaerno5, Tsit5, SaveAt, Dopri5


def count_jantera_steps(net, T0, P0, Y0, t_end, solver, rtol=1e-8, atol=1e-12):
    """Count solver steps for Jantera using solver stats."""
    # Use SaveAt(t1=True) for efficiency, step count is in stats
    saveat = SaveAt(t1=True)
    
    start = time.perf_counter()
    res = net.advance(T0, P0, Y0, t_end, rtol=rtol, atol=atol, solver=solver, saveat=saveat)
    
    # Ensure stats are ready
    if hasattr(res, 'ys'):
        jax.block_until_ready(res.ys)
    else:
        jax.block_until_ready(res)

    elapsed = time.perf_counter() - start
    
    if solver == "custom_bdf":
        n_steps = int(res.stats['n_steps'])
        n_rejected = 0 # Not tracked yet
        # CustomSolution ys is [1, dim], T is at 0,0
        T_final = float(res.ys[0, 0])
    else:
        # Get actual step count from solver statistics
        n_steps = int(res.stats['num_accepted_steps'])
        n_rejected = int(res.stats['num_rejected_steps'])
        # diffrax returns ys as (n_save, state_dim)
        T_final = float(res.ys[-1, 0])
    
    return n_steps, n_rejected, elapsed, T_final


def count_cantera_steps(yaml, T0, P0, X0, t_end, rtol=1e-8, atol=1e-12):
    """Count solver steps for Cantera."""
    sol = ct.Solution(yaml)
    sol.TPX = T0, P0, X0
    
    reactor = ct.IdealGasConstPressureReactor(sol)
    net = ct.ReactorNet([reactor])
    net.rtol = rtol
    net.atol = atol
    
    # Count steps by advancing step by step
    steps = 0
    start = time.perf_counter()
    while net.time < t_end:
        net.step()
        steps += 1
    elapsed = time.perf_counter() - start
    
    T_final = sol.T
    return steps, elapsed, T_final


def compare_solvers():
    """Compare step counts for different solvers."""
    print("=" * 70)
    print("ODE STEP COUNT COMPARISON: Jantera vs Cantera")
    print("=" * 70)
    
    configs = [
        ("GRI-30 @ 1500K", "gri30.yaml", 1500.0, 101325.0, "CH4:1, O2:2, N2:7.52", 1e-4),
        ("GRI-30 @ 1200K", "gri30.yaml", 1200.0, 101325.0, "CH4:1, O2:2, N2:7.52", 1e-3),
        #("JP-10 @ 1500K", "jp10.yaml", 1500.0, 101325.0, "C10H16:1, O2:14, N2:52.64", 1e-4),
    ]
    
    rtol, atol = 1e-8, 1e-12
    
    for name, yaml, T0, P0, X0, t_end in configs:
        print(f"\n{'='*70}")
        print(f"Test Case: {name}")
        print(f"t_end = {t_end*1e6:.0f} µs, rtol = {rtol}, atol = {atol}")
        print("-" * 70)
        
        # Load mechanism
        mech = load_mechanism(yaml)
        sol_ct = ct.Solution(yaml)
        sol_ct.TPX = T0, P0, X0
        Y0 = jnp.array(sol_ct.Y)
        
        net = ReactorNet(mech)
        
        # Warmup JIT for Custom BDF
        print("  Warming up JIT...")
        try:
             _ = net.advance(T0, P0, Y0, 1e-8, solver="custom_bdf")
             jax.block_until_ready(_.ys)
        except Exception as e:
             print(f"JIT Warmup Failed: {e}")
             import traceback
             traceback.print_exc()

        # Cantera (CVODE BDF)
        ct_steps, ct_time, ct_T_final = count_cantera_steps(yaml, T0, P0, X0, t_end, rtol, atol)
        print(f"\n  Cantera (CVODE BDF):")
        print(f"    Steps: {ct_steps:6d}")
        print(f"    Time:  {ct_time*1e3:8.2f} ms")
        print(f"    T_end: {ct_T_final:.2f} K")
        print(f"    µs/step: {ct_time/ct_steps*1e6:.1f}")
        
        # Jantera - Custom BDF (Sparse Jacobian + Reuse)
        try:
            jt_steps, jt_rejected, jt_time, jt_T_final = count_jantera_steps(net, T0, P0, Y0, t_end, "custom_bdf", rtol, atol)
            print(f"\n  Jantera (Custom BDF, Sparse):")
            print(f"    Steps: {jt_steps:6d}")
            print(f"    Time:  {jt_time*1e3:8.2f} ms")
            print(f"    T_end: {jt_T_final:.2f} K")
            print(f"    µs/step: {jt_time/max(jt_steps,1)*1e6:.1f}")
            print(f"    Steps ratio vs Cantera: {jt_steps/ct_steps:.2f}x")
            
            # Compare time
            ratio = jt_time / ct_time
            print(f"    Time ratio (Jantera/Cantera): {ratio:.2f}x")
        except Exception as e:
            print(f"\n  Jantera (Custom BDF): FAILED - {e}")
            import traceback
            traceback.print_exc()

        # Jantera - Kvaerno5 (ESDIRK5, implicit) - using default optimized solver
        # Skip for now if too slow, or run to compare
        try:
           pass
           # jt_steps_kv, jt_rejected_kv, jt_time_kv, jt_T_final_kv = count_jantera_steps(net, T0, P0, Y0, t_end, None, rtol, atol)
           # print(f"\n  Jantera (Kvaerno5 GMRES):")
           # print(f"    Steps: {jt_steps_kv:6d} (rejected: {jt_rejected_kv})")
           # print(f"    Time:  {jt_time_kv*1e3:8.2f} ms")
           # print(f"    T_end: {jt_T_final_kv:.2f} K")
           # print(f"    µs/step: {jt_time_kv/max(jt_steps_kv,1)*1e6:.1f}")
           # print(f"    Time ratio vs Cantera: {jt_time_kv/ct_time:.2f}x")
        except Exception as e:
            print(f"\n  Jantera (Kvaerno5): FAILED - {e}")


def analyze_step_sizes():
    """Analyze step size evolution for both solvers."""
    pass 

def main():
    compare_solvers()
    # analyze_step_sizes()
    
    print("\n" + "=" * 70)
    print("STEP COUNT PROFILING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
