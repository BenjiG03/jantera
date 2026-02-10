import time
import jax
import jax.numpy as jnp
import diffrax
import numpy as np
import os
import sys
from tabulate import tabulate

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from canterax.loader import load_mechanism
from canterax.solution import Solution
from canterax.reactor import ReactorNet

jax.config.update("jax_enable_x64", True)

def benchmark_solvers():
    print("Loading GRI-30...")
    yaml_path = "gri30.yaml"
    mech = load_mechanism(yaml_path)
    sol = Solution(yaml_path)
    
    T0, P0 = 1500.0, 101325.0
    sol.TPX = T0, P0, "CH4:1, O2:2, N2:7.52"
    y0 = sol.Y
    t_end = 1e-3
    
    net = ReactorNet(mech)
    
    # Selection of solvers to test
    # Implicit solvers are needed for stiff reactor trajectories
    solvers = [
        ("Kvaerno3", diffrax.Kvaerno3()),
        ("Kvaerno4", diffrax.Kvaerno4()),
        ("Kvaerno5", diffrax.Kvaerno5()),
        ("ImplicitEuler", diffrax.ImplicitEuler()),
    ]
    
    results = []
    
    for name, solver in solvers:
        print(f"\nBenchmarking {name}...")
        
        # Define a wrapper for JIT that takes the solver
        # Note: diffrax solvers are usually pytrees, so we can pass them in or capture them
        @jax.jit
        def run_sim(y0):
            return net.advance(T0, P0, y0, t_end, solver=solver)
        
        try:
            # Warmup
            print(f"  Warmup/JIT...")
            res = jax.block_until_ready(run_sim(y0))
            
            # Measure
            n_runs = 10
            times = []
            for _ in range(n_runs):
                start = time.perf_counter()
                res = jax.block_until_ready(run_sim(y0))
                times.append(time.perf_counter() - start)
            
            median_time = np.median(times)
            steps = res.stats['num_steps'] if hasattr(res, 'stats') else res.stats['num_steps'] # diffrax.Solution has .stats
            
            # Diffrax Solution stats handling
            if hasattr(res, 'stats'):
                steps = res.stats['num_steps']
            else:
                # If we returned the raw diffrax solution
                steps = res.stats['num_steps']

            results.append([name, median_time * 1000, int(steps), (median_time / steps) * 1e6])
            print(f"  Done: {median_time*1000:.2f} ms, {steps} steps")
            
        except Exception as e:
            print(f"  Failed: {e}")
            results.append([name, "FAILED", "-", "-"])

    # Add custom BDF for comparison
    print("\nBenchmarking Custom BDF...")
    @jax.jit
    def run_bdf(y0):
        return net.advance(T0, P0, y0, t_end, solver="bdf")
    
    try:
        res = jax.block_until_ready(run_bdf(y0))
        times = []
        for _ in range(10):
            start = time.perf_counter()
            res = jax.block_until_ready(run_bdf(y0))
            times.append(time.perf_counter() - start)
        median_time = np.median(times)
        steps = int(res["stats"]["num_steps"])
        results.append(["Custom BDF", median_time * 1000, steps, (median_time / steps) * 1e6])
        print(f"  Done: {median_time*1000:.2f} ms, {steps} steps")
    except Exception as e:
        print(f"  Failed: {e}")

    print("\n" + tabulate(results, headers=["Solver", "Time (ms)", "Steps", "Time/Step (µs)"], tablefmt="grid"))

if __name__ == "__main__":
    benchmark_solvers()
