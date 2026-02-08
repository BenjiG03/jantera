"""
Micro-benchmark for Jantera components to isolate performance bottlenecks.
Tests individual functions without running the full ODE solver.
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
from jantera.kinetics import compute_wdot, compute_kf, compute_Kc
from jantera.thermo import compute_mixture_props
from jantera.reactor import reactor_rhs


def benchmark_function(func, args, name, n_warmup=3, n_runs=100):
    """Benchmark a JAX function."""
    # Warmup / JIT
    for _ in range(n_warmup):
        result = func(*args)
        jax.block_until_ready(result)
    
    # Timed runs
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        result = func(*args)
        jax.block_until_ready(result)
        times.append(time.perf_counter() - start)
    
    mean_us = np.mean(times) * 1e6
    std_us = np.std(times) * 1e6
    print(f"  {name}: {mean_us:.1f} ± {std_us:.1f} µs")
    return mean_us


def main():
    print("=" * 60)
    print("MICRO-BENCHMARK: Individual Function Performance")
    print("=" * 60)
    
    # Test with GRI-30 (simpler mechanism)
    yaml_path = "gri30.yaml"
    T0, P0 = 1500.0, 101325.0
    X0 = "CH4:1, O2:2, N2:7.52"
    
    print(f"\nLoading {yaml_path}...")
    mech = load_mechanism(yaml_path)
    sol_ct = ct.Solution(yaml_path)
    sol_ct.TPX = T0, P0, X0
    Y0 = jnp.array(sol_ct.Y)
    
    # Prepare inputs
    rho = sol_ct.density
    conc = jnp.array(sol_ct.concentrations)
    state = jnp.concatenate([jnp.array([T0]), Y0])
    
    print(f"\nMechanism: {mech.n_species} species, {mech.n_reactions} reactions")
    print(f"State size: {len(state)}")
    
    print("\n--- Function Timings (GRI-30) ---")
    
    # 1. Thermo
    benchmark_function(
        lambda: compute_mixture_props(T0, P0, Y0, mech),
        (), "compute_mixture_props"
    )
    
    # 2. Forward rate constants
    benchmark_function(
        lambda: compute_kf(T0, conc, mech),
        (), "compute_kf"
    )
    
    # 3. Equilibrium constants
    benchmark_function(
        lambda: compute_Kc(T0, mech),
        (), "compute_Kc"
    )
    
    # 4. Full wdot (kinetics)
    benchmark_function(
        lambda: compute_wdot(T0, P0, Y0, mech),
        (), "compute_wdot"
    )
    
    # 5. Full RHS
    benchmark_function(
        lambda: reactor_rhs(0.0, state, (P0, mech)),
        (), "reactor_rhs"
    )
    
    # Now test JP-10
    print("\n" + "=" * 60)
    yaml_path = "jp10.yaml"
    T0, P0 = 1500.0, 101325.0
    X0 = "C10H16:1, O2:14, N2:52.64"
    
    print(f"\nLoading {yaml_path}...")
    mech = load_mechanism(yaml_path)
    sol_ct = ct.Solution(yaml_path)
    sol_ct.TPX = T0, P0, X0
    Y0 = jnp.array(sol_ct.Y)
    
    conc = jnp.array(sol_ct.concentrations)
    state = jnp.concatenate([jnp.array([T0]), Y0])
    
    print(f"Mechanism: {mech.n_species} species, {mech.n_reactions} reactions")
    print(f"State size: {len(state)}")
    
    print("\n--- Function Timings (JP-10) ---")
    
    benchmark_function(
        lambda: compute_mixture_props(T0, P0, Y0, mech),
        (), "compute_mixture_props"
    )
    
    benchmark_function(
        lambda: compute_kf(T0, conc, mech),
        (), "compute_kf"
    )
    
    benchmark_function(
        lambda: compute_Kc(T0, mech),
        (), "compute_Kc"
    )
    
    benchmark_function(
        lambda: compute_wdot(T0, P0, Y0, mech),
        (), "compute_wdot"
    )
    
    benchmark_function(
        lambda: reactor_rhs(0.0, state, (P0, mech)),
        (), "reactor_rhs"
    )
    
    # Estimate steps needed for 1ms simulation
    print("\n--- Estimating ODE Solver Overhead ---")
    # Typical stiff solver takes 500-2000 steps for 1ms
    rhs_time_gri = 50  # µs estimate from above
    rhs_time_jp10 = 50  # µs estimate
    
    for n_steps in [500, 1000, 2000, 5000]:
        print(f"  {n_steps:5d} steps × 50 µs/eval = {n_steps * 50 / 1000:.1f} ms")


if __name__ == "__main__":
    main()
