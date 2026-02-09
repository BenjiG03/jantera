import time
import jax
import jax.numpy as jnp
import diffrax
import optimistix as optx
import numpy as np
import os
import sys
from tabulate import tabulate

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from jantera.loader import load_mechanism
from jantera.solution import Solution
from jantera.reactor import ReactorNet, reactor_rhs

jax.config.update("jax_enable_x64", True)

def benchmark_kvaerno_tuning():
    print("Loading GRI-30...")
    yaml_path = "gri30.yaml"
    mech = load_mechanism(yaml_path)
    sol = Solution(yaml_path)
    
    T0, P0 = 1500.0, 101325.0
    sol.TPX = T0, P0, "CH4:1, O2:2, N2:7.52"
    y0 = sol.Y
    t_end = 1e-3
    
    net = ReactorNet(mech)
    
    # Selection of solver configurations
    configs = [
        ("Kvaerno5 (Default)", diffrax.Kvaerno5()),
        ("Kvaerno5 (VeryChord, kappa=0.1)", diffrax.Kvaerno5(root_finder=diffrax.VeryChord(rtol=1e-7, atol=1e-10, kappa=0.1))),
        ("Kvaerno5 (VeryChord, kappa=0.001)", diffrax.Kvaerno5(root_finder=diffrax.VeryChord(rtol=1e-7, atol=1e-10, kappa=0.001))),
        ("Kvaerno5 (optx.Broyden, loose)", diffrax.Kvaerno5(root_finder=optx.Broyden(rtol=1e-3, atol=1e-6))),
    ]
    
    results = []
    
    for name, solver in configs:
        print(f"\nBenchmarking {name}...")
        
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
            steps = int(res.stats['num_steps'])
            
            results.append([name, median_time * 1000, steps, (median_time / steps) * 1e6])
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
    except Exception as e:
        print(f"  Failed: {e}")

    print("\n" + tabulate(results, headers=["Configuration", "Time (ms)", "Steps", "Time/Step (µs)"], tablefmt="grid"))

if __name__ == "__main__":
    benchmark_kvaerno_tuning()
