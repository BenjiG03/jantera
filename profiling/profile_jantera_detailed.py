import os
import time
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import equinox as eqx
from diffrax import Kvaerno5, SaveAt, PIDController
import cProfile
import pstats
import io

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from jantera.loader import load_mechanism
from jantera.solution import Solution
from jantera.reactor import ReactorNet, reactor_rhs
from jantera.equilibrate import equilibrate

def profile_function(func, *args, **kwargs):
    """Run cProfile on a function."""
    pr = cProfile.Profile()
    pr.enable()
    result = func(*args, **kwargs)
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(20)
    print(s.getvalue())
    return result

def run_full_suite(yaml_path, name):
    print("\n" + "#"*40)
    print(f" PROFILING {name} ({yaml_path})")
    print("#"*40)
    
    try:
        mech = load_mechanism(yaml_path)
        sol = Solution(yaml_path)
    except Exception as e:
        print(f"Failed to load mechanism: {e}")
        return

    # Initial Condition
    if "gri30" in yaml_path.lower():
        T0, P0, X0_str = 1500.0, 101325.0, "CH4:1, O2:2, N2:7.52"
    else:
        # Assume JP-10
        T0, P0, X0_str = 1500.0, 101325.0, "C10H16:1, O2:14, N2:52.64"
        
    sol.TPX = T0, P0, X0_str
    y0 = sol.Y

    print("\n" + "="*30 + " EQUILIBRATE PROFILING " + "="*30)
    # 1. Equilibrate
    # Cold Start (JIT)
    print("Cold start (JIT) equilibrate...")
    sol.TPX = T0, P0, X0_str
    t0 = time.perf_counter()
    _ = equilibrate(sol, 'TP')
    jax.block_until_ready(_)
    jit_eq = time.perf_counter() - t0
    print(f"Equilibrate Cold Start (JIT): {jit_eq*1000:.2f} ms")

    # Measure warm average
    N_eq = 5
    eq_times = []
    for _ in range(N_eq):
        sol.TPX = T0, P0, X0_str
        start = time.perf_counter()
        res_eq = equilibrate(sol, 'TP')
        jax.block_until_ready(res_eq)
        eq_times.append(time.perf_counter() - start)
    
    avg_eq = np.mean(eq_times)
    steps = res_eq.stats['num_steps'] if hasattr(res_eq, 'stats') else -1
    print(f"Equilibrate Warm Avg Time ({N_eq} runs): {avg_eq*1000:.2f} ms")
    print(f"Equilibrate Steps: {steps}")
    
    print("\n" + "="*30 + " REACTOR RHS PROFILING " + "="*30)
    args = (P0, mech)
    state = jnp.concatenate([jnp.array([T0]), y0])
    
    # Cold Start (JIT)
    t0 = time.perf_counter()
    _ = jax.block_until_ready(reactor_rhs(0.0, state, args))
    jit_rhs = time.perf_counter() - t0
    print(f"Reactor RHS Cold Start (JIT): {jit_rhs*1000:.2f} ms")
    
    # Measure
    n_rhs = 1000 if "gri30" not in yaml_path.lower() else 10000
    start = time.perf_counter()
    for _ in range(n_rhs):
        _ = jax.block_until_ready(reactor_rhs(0.0, state, args))
    end = time.perf_counter()
    avg_rhs = (end - start) / n_rhs
    print(f"Reactor RHS Warm Avg Time: {avg_rhs*1e6:.2f} us")

    print("\n" + "="*30 + " REACTOR ADVANCE PROFILING " + "="*30)
    net = ReactorNet(mech)
    t_end = 1e-3
    
    # Cold Start (JIT)
    print("Cold start (JIT) ReactorNet.advance...")
    t0 = time.perf_counter()
    _ = net.advance(T0, P0, y0, t_end, rtol=1e-8, atol=1e-12, solver=Kvaerno5())
    jax.block_until_ready(_)
    jit_adv = time.perf_counter() - t0
    print(f"Advance Cold Start (JIT): {jit_adv*1000:.2f} ms")
    
    # Measure Warm Average
    N_adv = 3
    print(f"Benchmarking ReactorNet.advance (t_end={t_end}, N={N_adv})...")
    adv_times = []
    for _ in range(N_adv):
        start = time.perf_counter()
        res_adv = net.advance(T0, P0, y0, t_end, rtol=1e-8, atol=1e-12, solver=Kvaerno5())
        jax.block_until_ready(res_adv)
        adv_times.append(time.perf_counter() - start)
    
    avg_adv = np.mean(adv_times)
    adv_steps = res_adv.stats['num_steps']
    print(f"Advance Warm Avg Time: {avg_adv*1000:.2f} ms")
    print(f"Advance Steps: {adv_steps}")
    print(f"Time per Step (Total): {avg_adv*1000/adv_steps:.3f} ms/step")
    print(f"Difference (Total/Step - RHS): {avg_adv*1e6/adv_steps - avg_rhs*1e6:.2f} us (Overhead)")

def main():
    print(f"JAX Backend: {jax.default_backend()}")
    
    gri_yaml = os.path.join(os.path.dirname(__file__), "../tests/gri30.yaml")
    if not os.path.exists(gri_yaml): gri_yaml = "gri30.yaml"
    run_full_suite(gri_yaml, "GRI-30")

    jp10_yaml = os.path.join(os.path.dirname(__file__), "../jp10.yaml")
    if os.path.exists(jp10_yaml):
        run_full_suite(jp10_yaml, "JP-10")
    else:
        print(f"\nWarning: {jp10_yaml} not found.")

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
