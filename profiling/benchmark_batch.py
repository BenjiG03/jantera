
import time
import jax
import jax.numpy as jnp
import diffrax
import lineax
import numpy as np
import os
import sys
from tabulate import tabulate

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from canterax.loader import load_mechanism
from canterax.solution import Solution
from canterax.reactor import ReactorNet, reactor_rhs

jax.config.update("jax_enable_x64", True)

def benchmark_batch():
    print("Loading GRI-30 for Batch Benchmark...")
    yaml_path = "gri30.yaml"
    mech = load_mechanism(yaml_path)
    sol = Solution(yaml_path)
    
    T0, P0 = 1500.0, 101325.0
    sol.TPX = T0, P0, "CH4:1, O2:2, N2:7.52"
    y0_single = sol.Y
    t_end = 1e-3
    
    net = ReactorNet(mech)
    
    # Vectorize the advance function over a batch dimension
    # state0 for advance is [T, Y...]
    # let's vectorize over different initial temperatures
    
    @jax.jit
    def run_batch(T_batch, P, Y_batch):
        # vmap over T and Y, broadcast P and t_end
        # advance(self, T0, P, Y0, t_end, ...)
        return jax.vmap(lambda t, y: net.advance(t, P, y, t_end))(T_batch, Y_batch)

    batch_sizes = [1, 10, 100, 1000, 10000]
    results = []

    print(f"{'Batch Size':<10} | {'Total Time (ms)':<15} | {'Time/Reactor (ms)':<18} | {'Throughput (react/s)':<20}")
    print("-" * 70)

    for b in batch_sizes:
        # Create batch data
        # Perturb T slightly randomly
        key = jax.random.PRNGKey(0)
        T_batch = T0 + jax.random.uniform(key, (b,), minval=-50.0, maxval=50.0)
        # Duplicate Y0
        Y_batch = jnp.tile(y0_single, (b, 1))

        try:
            # Warmup
            _ = jax.block_until_ready(run_batch(T_batch, P0, Y_batch))
            
            # Measure
            n_runs = 5
            times = []
            for _ in range(n_runs):
                start = time.perf_counter()
                res = jax.block_until_ready(run_batch(T_batch, P0, Y_batch))
                times.append(time.perf_counter() - start)
            
            median_time = np.median(times)
            time_per_reactor = median_time / b
            throughput = b / median_time
            
            print(f"{b:<10} | {median_time*1000:>15.2f} | {time_per_reactor*1000:>18.4f} | {throughput:>20.2f}")
            results.append([b, median_time*1000, time_per_reactor*1000, throughput])
            
        except Exception as e:
            print(f"{b:<10} | FAILED: {str(e)[:50]}")
            results.append([b, "FAILED", "-", "-"])

    print("\nBatch Benchmark Results:")
    print(tabulate(results, headers=["Batch Size", "Total Time (ms)", "Time/Reactor (ms)", "Throughput (react/s)"], tablefmt="github"))

if __name__ == "__main__":
    benchmark_batch()
