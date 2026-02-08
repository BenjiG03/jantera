"""
Jacobian Computation Profiling: Jantera vs Cantera.

Benchmarks Jacobian computation in isolation to understand
the cost of AD-based Jacobian vs finite difference.

Usage:
    python profile_jacobian.py
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
from jantera.reactor import reactor_rhs
from jantera.kinetics import compute_wdot


def benchmark_function(func, n_warmup=5, n_runs=100):
    """Benchmark a JAX function, returning mean and std in microseconds."""
    # Warmup
    for _ in range(n_warmup):
        result = func()
        jax.block_until_ready(result)
    
    # Timed runs
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        result = func()
        jax.block_until_ready(result)
        times.append(time.perf_counter() - start)
    
    return np.mean(times) * 1e6, np.std(times) * 1e6


def profile_jacobian_jantera(mech, T, P, Y, name):
    """Profile Jacobian computation for Jantera."""
    print(f"\n--- Jantera Jacobian: {name} ---")
    
    state = jnp.concatenate([jnp.array([T]), Y])
    args = (P, mech)
    n_state = len(state)
    
    # 1. RHS evaluation only (baseline)
    rhs_func = jax.jit(lambda: reactor_rhs(0.0, state, args))
    rhs_mean, rhs_std = benchmark_function(rhs_func)
    print(f"  RHS evaluation:    {rhs_mean:.1f} ± {rhs_std:.1f} µs")
    
    # 2. Full Jacobian via AD (jax.jacobian)
    @jax.jit
    def jacobian_ad():
        return jax.jacobian(lambda s: reactor_rhs(0.0, s, args))(state)
    
    jac_ad_mean, jac_ad_std = benchmark_function(jacobian_ad)
    print(f"  Jacobian (AD):     {jac_ad_mean:.1f} ± {jac_ad_std:.1f} µs")
    print(f"  Jacobian/RHS ratio:{jac_ad_mean/rhs_mean:.1f}x")
    
    # 3. Forward-mode Jacobian (jacfwd)
    @jax.jit
    def jacobian_fwd():
        return jax.jacfwd(lambda s: reactor_rhs(0.0, s, args))(state)
    
    jac_fwd_mean, jac_fwd_std = benchmark_function(jacobian_fwd)
    print(f"  Jacobian (fwd):    {jac_fwd_mean:.1f} ± {jac_fwd_std:.1f} µs")
    
    # 4. Reverse-mode Jacobian (jacrev)
    @jax.jit
    def jacobian_rev():
        return jax.jacrev(lambda s: reactor_rhs(0.0, s, args))(state)
    
    jac_rev_mean, jac_rev_std = benchmark_function(jacobian_rev)
    print(f"  Jacobian (rev):    {jac_rev_mean:.1f} ± {jac_rev_std:.1f} µs")
    
    # 5. JVP (Jacobian-vector product) - what implicit solver needs
    @jax.jit
    def jvp_func():
        v = jnp.ones_like(state)
        return jax.jvp(lambda s: reactor_rhs(0.0, s, args), (state,), (v,))
    
    jvp_mean, jvp_std = benchmark_function(jvp_func)
    print(f"  JVP:               {jvp_mean:.1f} ± {jvp_std:.1f} µs")
    
    # Check Jacobian shape and sparsity
    jac = jacobian_ad()
    n_nonzero = jnp.sum(jnp.abs(jac) > 1e-20)
    sparsity = 1.0 - float(n_nonzero) / (n_state * n_state)
    print(f"\n  Jacobian shape:    {jac.shape}")
    print(f"  Jacobian sparsity: {sparsity*100:.1f}% zeros")
    print(f"  Nonzero entries:   {int(n_nonzero)}/{n_state*n_state}")
    
    return {
        'rhs': rhs_mean,
        'jacobian_ad': jac_ad_mean,
        'jacobian_fwd': jac_fwd_mean,
        'jacobian_rev': jac_rev_mean,
        'jvp': jvp_mean,
        'sparsity': sparsity,
        'n_state': n_state
    }


def profile_jacobian_cantera_fd(yaml, T, P, X, name):
    """Estimate Jacobian cost for Cantera using finite differences."""
    print(f"\n--- Cantera Jacobian (FD estimate): {name} ---")
    
    sol = ct.Solution(yaml)
    sol.TPX = T, P, X
    
    n_species = sol.n_species
    n_state = n_species + 1  # T + Y_i
    
    # Time a single RHS evaluation
    reactor = ct.IdealGasConstPressureReactor(sol)
    net = ct.ReactorNet([reactor])
    
    # Cantera doesn't expose RHS timing directly, so we estimate
    # by measuring step time with very small tolerance
    sol.TPX = T, P, X
    reactor = ct.IdealGasConstPressureReactor(sol)
    net = ct.ReactorNet([reactor])
    net.rtol = 1e-14
    net.atol = 1e-18
    
    times = []
    for _ in range(10):
        sol.TPX = T, P, X
        reactor = ct.IdealGasConstPressureReactor(sol)
        net = ct.ReactorNet([reactor])
        start = time.perf_counter()
        net.step()
        times.append(time.perf_counter() - start)
    
    step_time = np.mean(times) * 1e6
    print(f"  Single step time:  {step_time:.1f} µs (includes Jacobian)")
    
    # CVODE typically uses k*n FD evaluations for Jacobian (k=1-3)
    # Estimate: FD Jacobian ~ (n+1) * RHS_eval_time
    # But CVODE optimizes this with banded/sparse structure
    estimated_fd_jacobian = step_time  # Rough estimate
    
    print(f"  State size:        {n_state}")
    print(f"  Note: Cantera CVODE uses optimized analytical/FD hybrid")
    print(f"        Direct Jacobian timing not accessible from Python")
    
    return {'step_time': step_time, 'n_state': n_state}


def compare_mechanisms():
    """Compare Jacobian costs across mechanisms."""
    print("=" * 70)
    print("JACOBIAN COMPUTATION PROFILING")
    print("=" * 70)
    
    configs = [
        ("GRI-30", "gri30.yaml", 1500.0, 101325.0, "CH4:1, O2:2, N2:7.52"),
        ("JP-10", "jp10.yaml", 1500.0, 101325.0, "C10H16:1, O2:14, N2:52.64"),
    ]
    
    results = []
    
    for name, yaml, T, P, X in configs:
        print(f"\n{'='*70}")
        print(f"Mechanism: {name}")
        print("=" * 70)
        
        # Jantera
        mech = load_mechanism(yaml)
        sol = ct.Solution(yaml)
        sol.TPX = T, P, X
        Y = jnp.array(sol.Y)
        
        jt_results = profile_jacobian_jantera(mech, T, P, Y, name)
        ct_results = profile_jacobian_cantera_fd(yaml, T, P, X, name)
        
        results.append({
            'name': name,
            'jantera': jt_results,
            'cantera': ct_results
        })
    
    # Summary table
    print("\n" + "=" * 70)
    print("JACOBIAN PROFILING SUMMARY")
    print("=" * 70)
    print(f"\n{'Mechanism':<12} {'n_state':>10} {'RHS (µs)':>12} {'Jac_AD (µs)':>14} {'Jac/RHS':>10} {'Sparsity':>10}")
    print("-" * 70)
    
    for r in results:
        jt = r['jantera']
        print(f"{r['name']:<12} {jt['n_state']:>10d} {jt['rhs']:>12.1f} {jt['jacobian_ad']:>14.1f} {jt['jacobian_ad']/jt['rhs']:>10.1f}x {jt['sparsity']*100:>9.1f}%")


def profile_scaling():
    """Profile how Jacobian time scales with mechanism size."""
    print("\n" + "=" * 70)
    print("JACOBIAN SCALING ANALYSIS")
    print("=" * 70)
    
    # This is theoretical - we only have 2 mechanisms
    # But we can analyze the O(n^2) vs O(n) scaling
    
    for yaml, name in [("gri30.yaml", "GRI-30"), ("jp10.yaml", "JP-10")]:
        mech = load_mechanism(yaml)
        sol = ct.Solution(yaml)
        sol.TPX = 1500.0, 101325.0, "H2:1, O2:0.5" if "gri" in yaml else "C10H16:1, O2:14"
        Y = jnp.array(sol.Y)
        
        n_state = len(Y) + 1
        
        # Time RHS
        state = jnp.concatenate([jnp.array([1500.0]), Y])
        args = (101325.0, mech)
        
        rhs_func = jax.jit(lambda: reactor_rhs(0.0, state, args))
        rhs_mean, _ = benchmark_function(rhs_func)
        
        # Time Jacobian
        jac_func = jax.jit(lambda: jax.jacobian(lambda s: reactor_rhs(0.0, s, args))(state))
        jac_mean, _ = benchmark_function(jac_func)
        
        print(f"\n{name}:")
        print(f"  n_state: {n_state}")
        print(f"  RHS:     {rhs_mean:.1f} µs")
        print(f"  Jacobian:{jac_mean:.1f} µs")
        print(f"  Ratio:   {jac_mean/rhs_mean:.1f}x")
        print(f"  Predicted O(n) Jacobian: {rhs_mean * n_state:.1f} µs (if linear)")
        print(f"  Predicted O(n^2) Jacobian: {rhs_mean * n_state**2 / n_state:.1f} µs (if quadratic)")


def main():
    compare_mechanisms()
    profile_scaling()
    
    print("\n" + "=" * 70)
    print("JACOBIAN PROFILING COMPLETE")
    print("=" * 70)
    print("\nKey Insights:")
    print("  - JAX AD computes dense Jacobian efficiently")
    print("  - Jacobian/RHS ratio indicates AD overhead")
    print("  - High sparsity suggests potential for sparse optimization")
    print("  - Cantera uses optimized sparse/analytical Jacobian")


if __name__ == "__main__":
    main()
