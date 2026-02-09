import time
import jax
import jax.numpy as jnp
from jax import jit, lax
from jax.scipy.linalg import lu_factor, lu_solve
import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from jantera.solvers.bdf import bdf_solve
from jantera.kinetics import compute_wdot
from jantera.loader import load_mechanism

jax.config.update("jax_enable_x64", True)

def time_function(name, func, args, n_warmup=100, n_runs=1000):
    # JIT compile
    print(f"JIT compiling {name}...")
    _ = jax.block_until_ready(func(*args))
    
    # Warmup
    for _ in range(n_warmup):
        _ = jax.block_until_ready(func(*args))
        
    start = time.perf_counter()
    for _ in range(n_runs):
        _ = jax.block_until_ready(func(*args))
    end = time.perf_counter()
    
    avg_time = (end - start) / n_runs
    print(f"[{name}] Avg Time: {avg_time * 1e6:.2f} µs")
    return avg_time

def trivial_ode(t, y, args):
    # dy/dt = -y
    # Simulate a little bit of work to not be completely optimized away
    return -0.1 * y

def run_latency_test():
    yaml_path = os.path.join(os.path.dirname(__file__), "..", "jp10.yaml")
    mech = load_mechanism(yaml_path)
    N = mech.mol_weights.shape[0] + 1 # +1 for T
    print(f"Testing with state size N={N} (JP-10 size)")

    # 1. Pure Loop Overhead
    @jit
    def pure_loop(n_steps):
        def body(i, val):
            return val + 1.0
        return lax.fori_loop(0, n_steps, body, 0.0)
    
    time_function("Pure JAX Loop (1 step equivalent)", pure_loop, (1,), n_runs=10000)

    # 2. Linear Algebra (LU Factor + Solve)
    A = jax.random.normal(jax.random.PRNGKey(0), (N, N))
    b = jax.random.normal(jax.random.PRNGKey(1), (N,))
    
    @jit
    def dense_lu_solve(A, b):
        lu, piv = lu_factor(A)
        return lu_solve((lu, piv), b)

    time_function(f"Dense LU Factor+Solve ({N}x{N})", dense_lu_solve, (A, b), n_runs=5000)

    # 2b. Dense LU Solve Only
    lu_piv = lu_factor(A)
    @jit
    def dense_lu_solve_only(lu, piv, b):
        return lu_solve((lu, piv), b)
        
    time_function(f"Dense LU Solve Only ({N}x{N})", dense_lu_solve_only, (*lu_piv, b), n_runs=5000)

    # 3. Kinetics Kernel (compute_wdot)
    T = 1500.0
    P = 101325.0
    Y = jnp.ones(N-1) / (N-1)
    
    @jit
    def kinetics_kernel(T, P, Y):
        return compute_wdot(T, P, Y, mech)

    time_function("Kinetics Kernel (compute_wdot)", kinetics_kernel, (T, P, Y), n_runs=5000)

    # 4. Full BDF Solver Structure (Trivial ODE)
    # Solve dy/dt = -y for a short time
    y0 = jnp.ones(N)
    
    # We want to measure overhead per step, not just solving time.
    # So run for fixed number of steps.
    # But bdf_solve integrates to time t_end.
    # Let's run a short integration that takes a predictable number of steps.
    # Or measure total time for 100 steps.
    
    # Actually, let's just time a single `bdf_step` call to isolate step logic!
    # But `bdf_step` takes a state object.
    # Let's use `bdf_solve` on trivial ODE for 1 ms, it should take some steps.
    
    print("\nRunning Trivial ODE Benchmark (bdf_solve overhead)...")
    # Trivial ODE
    # Force max steps 100 to measure step cost
    # We need to ensure it takes steps.
    # Trivial ODE is not stiff, so it might take large steps.
    # Force small max step? No, bdf doesn't have max step size arg exposed easily yet.
    # Let's just integrate to t=1.0 which requires steps.
    
    # Pre-compile a wrapper that includes the trivial_ode in the graph
    @jit
    def run_bdf_trivial(y0, t_end):
        # We integrate from 0 to t_end
        return bdf_solve(trivial_ode, 0.0, t_end, y0, rtol=1e-6, atol=1e-9)

    print("\nRunning Trivial ODE Benchmark (bdf_solve JIT overhead)...")
    res = run_bdf_trivial(y0, 1e-5) # Warmup JIT
    
    # Measure
    t_end = 1e-2
    start = time.perf_counter()
    res = run_bdf_trivial(y0, t_end)
    res = jax.block_until_ready(res)
    end = time.perf_counter()
    
    steps = res.n_steps
    # The stats are arrays on device, need to pull them to CPU to print?
    # Actually checking n_steps might trigger a sync if we aren't careful, 
    # but block_until_ready on 'res' (struct) should wait for all fields.
    
    steps = int(steps) # Convert to python int
    total_time = end - start
    print(f"[{'BDF Solver (Trivial ODE)'}] Total Time: {total_time*1e3:.2f} ms")
    print(f"  Steps: {steps}")
    print(f"  Avg Time/Step: {(total_time / steps) * 1e6:.2f} µs")


if __name__ == "__main__":
    run_latency_test()
