"""
JAX Profiler Trace for Jantera Reactor.

Generates TensorBoard-compatible traces to analyze XLA kernel breakdown,
memory allocation patterns, and kernel fusion effectiveness.

Usage:
    python profile_jax_trace.py
    tensorboard --logdir=/tmp/jax_trace
"""
import os
import sys
import time
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from jantera.loader import load_mechanism
from jantera.reactor import ReactorNet
from jantera.kinetics import compute_wdot
from jantera.reactor import reactor_rhs
from diffrax import Kvaerno5, SaveAt


def warmup_jit(net, T0, P0, Y0):
    """Warmup JIT compilation before profiling."""
    print("Warming up JIT compilation...")
    res = net.advance(T0, P0, Y0, 1e-8)
    jax.block_until_ready(res)
    print("JIT warmup complete.")


def profile_with_trace(net, T0, P0, Y0, t_end, trace_dir, name):
    """Profile reactor advance with JAX trace."""
    print(f"\n--- Profiling: {name} ---")
    print(f"Trace output: {trace_dir}")
    
    # Measure baseline (no profiling overhead)
    start_baseline = time.perf_counter()
    res = net.advance(T0, P0, Y0, t_end, rtol=1e-8, atol=1e-12)
    jax.block_until_ready(res)
    baseline_time = time.perf_counter() - start_baseline
    print(f"Baseline time (no trace): {baseline_time*1e3:.1f} ms")
    
    # Measure with profiling
    jax.profiler.start_trace(trace_dir)
    start_profiled = time.perf_counter()
    res = net.advance(T0, P0, Y0, t_end, rtol=1e-8, atol=1e-12)
    jax.block_until_ready(res)
    profiled_time = time.perf_counter() - start_profiled
    jax.profiler.stop_trace()
    
    print(f"Profiled time (with trace): {profiled_time*1e3:.1f} ms")
    overhead = (profiled_time - baseline_time) / baseline_time * 100
    print(f"Tracing overhead: {overhead:.1f}%")
    
    return res


def profile_rhs_only(mech, T, P, Y, trace_dir, name):
    """Profile just the RHS function (no ODE solve)."""
    print(f"\n--- Profiling RHS only: {name} ---")
    
    state = jnp.concatenate([jnp.array([T]), Y])
    args = (P, mech)
    
    # Warmup
    for _ in range(5):
        result = reactor_rhs(0.0, state, args)
        jax.block_until_ready(result)
    
    # Baseline
    n_evals = 1000
    start = time.perf_counter()
    for _ in range(n_evals):
        result = reactor_rhs(0.0, state, args)
        jax.block_until_ready(result)
    baseline_time = time.perf_counter() - start
    print(f"Baseline: {n_evals} RHS evals in {baseline_time*1e3:.1f} ms ({baseline_time/n_evals*1e6:.1f} µs/eval)")
    
    # Profiled
    jax.profiler.start_trace(trace_dir)
    start = time.perf_counter()
    for _ in range(100):  # Fewer evals to keep trace manageable
        result = reactor_rhs(0.0, state, args)
        jax.block_until_ready(result)
    profiled_time = time.perf_counter() - start
    jax.profiler.stop_trace()
    
    print(f"Profiled: 100 RHS evals in {profiled_time*1e3:.1f} ms")


def main():
    print("=" * 60)
    print("JAX PROFILER TRACE")
    print("=" * 60)
    
    # Check if tensorboard is available
    try:
        import tensorboard
        print(f"TensorBoard version: {tensorboard.__version__}")
    except ImportError:
        print("\n[WARNING] TensorBoard not installed.")
        print("Install with: pip install tensorboard")
        print("Continuing anyway - traces will still be generated.\n")
    
    # Base trace directory
    base_trace_dir = "/tmp/jax_trace"
    os.makedirs(base_trace_dir, exist_ok=True)
    
    # Test with GRI-30
    yaml = "gri30.yaml"
    T0, P0 = 1500.0, 101325.0
    X0 = "CH4:1, O2:2, N2:7.52"
    t_end = 1e-4
    
    print(f"\nLoading {yaml}...")
    mech = load_mechanism(yaml)
    
    import cantera as ct
    sol_ct = ct.Solution(yaml)
    sol_ct.TPX = T0, P0, X0
    Y0 = jnp.array(sol_ct.Y)
    
    net = ReactorNet(mech)
    
    # Warmup JIT first
    warmup_jit(net, T0, P0, Y0)
    
    # Profile full reactor advance
    profile_with_trace(
        net, T0, P0, Y0, t_end,
        os.path.join(base_trace_dir, "gri30_full"),
        "GRI-30 full advance"
    )
    
    # Profile RHS only
    profile_rhs_only(
        mech, T0, P0, Y0,
        os.path.join(base_trace_dir, "gri30_rhs"),
        "GRI-30 RHS"
    )
    
    # Test with JP-10
    yaml = "jp10.yaml"
    X0 = "C10H16:1, O2:14, N2:52.64"
    
    print(f"\n{'='*60}")
    print(f"Loading {yaml}...")
    mech = load_mechanism(yaml)
    sol_ct = ct.Solution(yaml)
    sol_ct.TPX = T0, P0, X0
    Y0 = jnp.array(sol_ct.Y)
    
    net = ReactorNet(mech)
    warmup_jit(net, T0, P0, Y0)
    
    profile_with_trace(
        net, T0, P0, Y0, t_end,
        os.path.join(base_trace_dir, "jp10_full"),
        "JP-10 full advance"
    )
    
    profile_rhs_only(
        mech, T0, P0, Y0,
        os.path.join(base_trace_dir, "jp10_rhs"),
        "JP-10 RHS"
    )
    
    print("\n" + "=" * 60)
    print("PROFILING COMPLETE")
    print("=" * 60)
    print(f"\nTo view traces, run:")
    print(f"  tensorboard --logdir={base_trace_dir}")
    print(f"\nThen open http://localhost:6006 in your browser.")


if __name__ == "__main__":
    main()
