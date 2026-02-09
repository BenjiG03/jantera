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

from jantera.loader import load_mechanism
from jantera.solution import Solution
from jantera.reactor import ReactorNet

jax.config.update("jax_enable_x64", True)

def run_experiment():
    print("Loading GRI-30 Benchmark...")
    yaml_path = "gri30.yaml"
    mech = load_mechanism(yaml_path)
    sol = Solution(yaml_path)
    
    T0, P0 = 1500.0, 101325.0
    sol.TPX = T0, P0, "CH4:1, O2:2, N2:7.52"
    y0 = sol.Y
    t_end = 1e-3
    
    net = ReactorNet(mech)
    
    # Range of hyperparameters to sweep
    linear_solvers = [("Auto", None), ("LU", lineax.LU())]
    kappas = [0.001, 0.01, 0.1, 0.5]
    scan_kinds = [("none", None), ("lax", "lax")]
    
    results = []
    
    print(f"{'Solver Config':<40} | {'Time (ms)':<10} | {'Steps':<7} | {'µs/step':<10}")
    print("-" * 75)

    for ls_name, ls_val in linear_solvers:
        for kappa in kappas:
            for sk_name, sk_val in scan_kinds:
                
                # Create the specific solver config
                root_finder = diffrax.VeryChord(
                    rtol=1e-7, # Dummy, Diffrax overrides these from the controller
                    atol=1e-10,
                    kappa=kappa,
                    linear_solver=ls_val if ls_val else lineax.AutoLinearSolver(well_posed=None)
                )
                
                solver = diffrax.Kvaerno5(
                    scan_kind=sk_val,
                    root_finder=root_finder
                )
                
                config_name = f"LS:{ls_name}, K:{kappa}, SK:{sk_name}"
                
                @jax.jit
                def run_sim(y0):
                    return net.advance(T0, P0, y0, t_end, solver=solver)
                
                try:
                    # Warmup
                    _ = jax.block_until_ready(run_sim(y0))
                    
                    # Benchmark
                    n_runs = 5
                    times = []
                    for _ in range(n_runs):
                        start = time.perf_counter()
                        res = jax.block_until_ready(run_sim(y0))
                        times.append(time.perf_counter() - start)
                    
                    median_time = np.median(times)
                    steps = int(res.stats['num_steps'])
                    us_per_step = (median_time / steps) * 1e6
                    
                    print(f"{config_name:<40} | {median_time*1000:>10.2f} | {steps:>7} | {us_per_step:>10.2f}")
                    results.append([ls_name, kappa, sk_name, median_time * 1000, steps, us_per_step])
                    
                except Exception as e:
                    print(f"{config_name:<40} | FAILED | {str(e)[:30]}")

    # Sort results by time
    results.sort(key=lambda x: x[3] if isinstance(x[3], float) else float('inf'))
    
    print("\nBest 5 Configurations:")
    print(tabulate(results[:5], headers=["LS", "Kappa", "Scan", "Time (ms)", "Steps", "µs/step"], tablefmt="github"))

if __name__ == "__main__":
    run_experiment()
