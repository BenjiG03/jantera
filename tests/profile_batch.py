"""
Batched Performance Profiling: jax.vmap Scaling.

Benchmarks how Jantera performance scales with batch size using jax.vmap,
to find the crossover point where it beats Cantera serial execution.

Usage:
    python profile_batch.py
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
from jantera.reactor import ReactorNet, reactor_rhs
from jantera.kinetics import compute_wdot
from diffrax import Kvaerno5, Tsit5, SaveAt
import diffrax


def create_batched_advance(mech, T0, P0, t_end, rtol=1e-8, atol=1e-12):
    """Create a vmapped reactor advance function."""
    
    def single_advance(Y0):
        state0 = jnp.concatenate([jnp.array([T0]), Y0])
        term = diffrax.ODETerm(reactor_rhs)
        solver = diffrax.Kvaerno5()
        stepsize_controller = diffrax.PIDController(rtol=rtol, atol=atol)
        
        sol = diffrax.diffeqsolve(
            term,
            solver,
            t0=0.0,
            t1=t_end,
            dt0=1e-12,
            y0=state0,
            args=(P0, mech),
            stepsize_controller=stepsize_controller,
            max_steps=100000,
            saveat=diffrax.SaveAt(t1=True),
            adjoint=diffrax.RecursiveCheckpointAdjoint()
        )
        return sol.ys[-1, 0]  # Return final temperature
    
    # vmap over Y0 (mass fractions)
    batched_advance = jax.vmap(single_advance)
    return jax.jit(batched_advance)


def benchmark_cantera_serial(yaml, T0, P0, X0, t_end, batch_size, rtol=1e-8, atol=1e-12):
    """Benchmark Cantera running serial simulations in a loop."""
    times = []
    T_finals = []
    
    for _ in range(min(batch_size, 10)):  # Cap at 10 for timing
        sol = ct.Solution(yaml)
        sol.TPX = T0, P0, X0
        reactor = ct.IdealGasConstPressureReactor(sol)
        net = ct.ReactorNet([reactor])
        net.rtol, net.atol = rtol, atol
        
        start = time.perf_counter()
        net.advance(t_end)
        times.append(time.perf_counter() - start)
        T_finals.append(sol.T)
    
    avg_time = np.mean(times)
    # Extrapolate to full batch
    total_time = avg_time * batch_size
    return total_time, np.mean(T_finals)


def profile_batch_scaling():
    """Profile how performance scales with batch size."""
    print("=" * 70)
    print("BATCHED PERFORMANCE PROFILING: jax.vmap Scaling")
    print("=" * 70)
    
    configs = [
        ("GRI-30", "gri30.yaml", 1500.0, 101325.0, "CH4:1, O2:2, N2:7.52", 1e-4),
        ("JP-10", "jp10.yaml", 1500.0, 101325.0, "C10H16:1, O2:14, N2:52.64", 1e-4),
    ]
    
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    rtol, atol = 1e-8, 1e-12
    
    for name, yaml, T0, P0, X0, t_end in configs:
        print(f"\n{'='*70}")
        print(f"Mechanism: {name}")
        print("-" * 70)
        
        # Load mechanism
        mech = load_mechanism(yaml)
        sol = ct.Solution(yaml)
        sol.TPX = T0, P0, X0
        Y0_base = jnp.array(sol.Y)
        
        # Create batched function
        print("Creating vmapped advance function...")
        batched_advance = create_batched_advance(mech, T0, P0, t_end, rtol, atol)
        
        # Warmup JIT
        print("Warming up JIT...")
        Y0_batch = jnp.tile(Y0_base, (2, 1))
        _ = batched_advance(Y0_batch)
        jax.block_until_ready(_)
        
        # Measure Cantera single-reactor time
        ct_single_time, ct_T_final = benchmark_cantera_serial(yaml, T0, P0, X0, t_end, 1, rtol, atol)
        print(f"\nCantera single reactor: {ct_single_time*1e3:.2f} ms")
        
        print(f"\n{'Batch':>8} {'Jantera (ms)':>14} {'Cantera (ms)':>14} {'Speedup':>10} {'ms/reactor':>12}")
        print("-" * 60)
        
        results = []
        
        for batch_size in batch_sizes:
            # Create batch with slight perturbations
            Y0_batch = jnp.tile(Y0_base, (batch_size, 1))
            # Add small perturbations to avoid caching artifacts
            key = jax.random.PRNGKey(42)
            perturbations = jax.random.uniform(key, (batch_size, len(Y0_base)), minval=0.999, maxval=1.001)
            Y0_batch = Y0_batch * perturbations
            Y0_batch = Y0_batch / jnp.sum(Y0_batch, axis=1, keepdims=True)  # Renormalize
            
            # Time Jantera batched
            n_runs = 3
            jt_times = []
            for _ in range(n_runs):
                start = time.perf_counter()
                T_finals = batched_advance(Y0_batch)
                jax.block_until_ready(T_finals)
                jt_times.append(time.perf_counter() - start)
            
            jt_time = np.mean(jt_times)
            jt_std = np.std(jt_times)
            
            # Cantera extrapolated time
            ct_time = ct_single_time * batch_size
            
            # Speedup
            speedup = ct_time / jt_time if jt_time > 0 else 0
            ms_per_reactor = jt_time * 1e3 / batch_size
            
            print(f"{batch_size:>8} {jt_time*1e3:>14.1f} {ct_time*1e3:>14.1f} {speedup:>10.2f}x {ms_per_reactor:>12.3f}")
            
            results.append({
                'batch_size': batch_size,
                'jantera_time': jt_time,
                'cantera_time': ct_time,
                'speedup': speedup,
                'ms_per_reactor': ms_per_reactor
            })
        
        # Find crossover point
        crossover = None
        for r in results:
            if r['speedup'] >= 1.0:
                crossover = r['batch_size']
                break
        
        if crossover:
            print(f"\n-> Crossover point: Jantera faster at batch size >= {crossover}")
        else:
            print(f"\n-> Cantera faster for all tested batch sizes")
        
        # Save results
        output_dir = os.path.join(os.path.dirname(__file__), "outputs")
        os.makedirs(output_dir, exist_ok=True)
        np.savez(
            os.path.join(output_dir, f"batch_scaling_{name.lower().replace('-', '')}.npz"),
            results=results
        )


def profile_memory_scaling():
    """Profile memory usage vs batch size."""
    print("\n" + "=" * 70)
    print("MEMORY SCALING ANALYSIS")
    print("=" * 70)
    
    yaml = "gri30.yaml"
    T0, P0, X0 = 1500.0, 101325.0, "CH4:1, O2:2, N2:7.52"
    t_end = 1e-4
    
    mech = load_mechanism(yaml)
    sol = ct.Solution(yaml)
    sol.TPX = T0, P0, X0
    Y0_base = jnp.array(sol.Y)
    
    n_species = len(Y0_base)
    n_state = n_species + 1
    
    print(f"\nMechanism: GRI-30 ({n_species} species)")
    print(f"\nEstimated memory per reactor:")
    print(f"  State vector:     {n_state * 8} bytes (float64)")
    print(f"  Jacobian (dense): {n_state * n_state * 8} bytes")
    
    print(f"\n{'Batch':>8} {'State (MB)':>12} {'Jacobian (MB)':>14} {'Total (MB)':>12}")
    print("-" * 50)
    
    for batch_size in [1, 10, 100, 1000, 10000]:
        state_mb = batch_size * n_state * 8 / 1e6
        jac_mb = batch_size * n_state * n_state * 8 / 1e6
        total_mb = state_mb + jac_mb
        print(f"{batch_size:>8} {state_mb:>12.2f} {jac_mb:>14.2f} {total_mb:>12.2f}")
    
    print("\nNote: Actual memory usage may be higher due to intermediate arrays")
    print("      and diffrax solver state.")


def main():
    profile_batch_scaling()
    profile_memory_scaling()
    
    print("\n" + "=" * 70)
    print("BATCH PROFILING COMPLETE")
    print("=" * 70)
    print("\nKey Findings:")
    print("  - Jantera's vmap enables parallel reactor simulation")
    print("  - Crossover point indicates where batching beats serial Cantera")
    print("  - Memory scales linearly with batch size (dense Jacobian)")


if __name__ == "__main__":
    main()
