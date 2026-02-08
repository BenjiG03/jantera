"""
Gradient/Adjoint Profiling: Jantera AD vs Cantera Finite Difference.

Benchmarks the cost of computing gradients through the reactor simulation
using JAX automatic differentiation vs Cantera finite differences.

Usage:
    python profile_adjoint.py
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
from diffrax import Kvaerno5, Tsit5, SaveAt, RecursiveCheckpointAdjoint, DirectAdjoint
import diffrax


def benchmark_forward_only(net, T0, P0, Y0, t_end, n_runs=10):
    """Benchmark forward simulation only (no gradients)."""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        res = net.advance(T0, P0, Y0, t_end, rtol=1e-8, atol=1e-12)
        jax.block_until_ready(res)
        times.append(time.perf_counter() - start)
    return np.mean(times), np.std(times)


def benchmark_jantera_gradient(mech, T0, P0, Y0, t_end, n_runs=5):
    """Benchmark Jantera gradient computation using AD."""
    
    # Define loss function: final temperature
    def loss_fn(Y0_input):
        state0 = jnp.concatenate([jnp.array([T0]), Y0_input])
        term = diffrax.ODETerm(reactor_rhs)
        solver = diffrax.Tsit5()  # Use explicit for gradient stability
        stepsize_controller = diffrax.PIDController(rtol=1e-6, atol=1e-8)
        
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
        return sol.ys[-1, 0]  # Final temperature
    
    loss_fn_jit = jax.jit(loss_fn)
    grad_fn = jax.jit(jax.grad(loss_fn))
    
    # Warmup JIT
    print("  Warming up JIT (forward)...")
    _ = loss_fn_jit(Y0)
    jax.block_until_ready(_)
    
    print("  Warming up JIT (gradient)...")
    try:
        _ = grad_fn(Y0)
        jax.block_until_ready(_)
    except Exception as e:
        print(f"  Gradient warmup failed: {e}")
        return None, None, None, None
    
    # Benchmark forward
    fwd_times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        T_final = loss_fn_jit(Y0)
        jax.block_until_ready(T_final)
        fwd_times.append(time.perf_counter() - start)
    
    fwd_mean = np.mean(fwd_times)
    
    # Benchmark gradient
    grad_times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        grad = grad_fn(Y0)
        jax.block_until_ready(grad)
        grad_times.append(time.perf_counter() - start)
    
    grad_mean = np.mean(grad_times)
    grad_std = np.std(grad_times)
    
    T_final = float(loss_fn_jit(Y0))
    grad_val = np.array(grad_fn(Y0))
    
    return fwd_mean, grad_mean, grad_std, grad_val


def benchmark_cantera_fd_gradient(yaml, T0, P0, X0, t_end, n_species, eps=1e-6):
    """Benchmark Cantera gradient via finite differences."""
    
    def run_cantera(Y):
        sol = ct.Solution(yaml)
        sol.TPY = T0, P0, Y
        reactor = ct.IdealGasConstPressureReactor(sol)
        net = ct.ReactorNet([reactor])
        net.rtol, net.atol = 1e-8, 1e-12
        net.advance(t_end)
        return sol.T
    
    # Get base Y
    sol = ct.Solution(yaml)
    sol.TPX = T0, P0, X0
    Y0 = np.array(sol.Y)
    
    # Forward evaluation time
    fwd_times = []
    for _ in range(5):
        start = time.perf_counter()
        T_base = run_cantera(Y0)
        fwd_times.append(time.perf_counter() - start)
    
    fwd_mean = np.mean(fwd_times)
    
    # Finite difference gradient
    start = time.perf_counter()
    grad = np.zeros(n_species)
    
    for i in range(n_species):
        Y_plus = Y0.copy()
        Y_plus[i] += eps
        Y_plus = Y_plus / np.sum(Y_plus)  # Renormalize
        
        Y_minus = Y0.copy()
        Y_minus[i] -= eps
        Y_minus = Y_minus / np.sum(Y_minus)  # Renormalize
        
        T_plus = run_cantera(Y_plus)
        T_minus = run_cantera(Y_minus)
        
        grad[i] = (T_plus - T_minus) / (2 * eps)
    
    fd_time = time.perf_counter() - start
    
    return fwd_mean, fd_time, grad


def compare_gradients():
    """Compare gradient computation between Jantera and Cantera."""
    print("=" * 70)
    print("GRADIENT/ADJOINT PROFILING: Jantera AD vs Cantera FD")
    print("=" * 70)
    
    configs = [
        ("GRI-30", "gri30.yaml", 1500.0, 101325.0, "CH4:1, O2:2, N2:7.52", 1e-5),
        ("JP-10", "jp10.yaml", 1500.0, 101325.0, "C10H16:1, O2:14, N2:52.64", 1e-5),
    ]
    
    for name, yaml, T0, P0, X0, t_end in configs:
        print(f"\n{'='*70}")
        print(f"Mechanism: {name}")
        print(f"t_end = {t_end*1e6:.0f} µs")
        print("-" * 70)
        
        # Load mechanism
        mech = load_mechanism(yaml)
        sol = ct.Solution(yaml)
        sol.TPX = T0, P0, X0
        Y0 = jnp.array(sol.Y)
        n_species = len(Y0)
        
        print(f"  State size: {n_species + 1} (T + {n_species} species)")
        
        # Jantera AD
        print(f"\n  Jantera (AD):")
        jt_fwd, jt_grad, jt_grad_std, jt_grad_val = benchmark_jantera_gradient(
            mech, T0, P0, Y0, t_end
        )
        
        if jt_fwd is not None:
            print(f"    Forward:  {jt_fwd*1e3:.1f} ms")
            print(f"    Gradient: {jt_grad*1e3:.1f} ± {jt_grad_std*1e3:.1f} ms")
            print(f"    Grad/Fwd: {jt_grad/jt_fwd:.2f}x")
        else:
            print(f"    Gradient computation failed")
            jt_grad = None
        
        # Cantera FD
        print(f"\n  Cantera (Finite Difference):")
        ct_fwd, ct_fd_time, ct_grad_val = benchmark_cantera_fd_gradient(
            yaml, T0, P0, X0, t_end, n_species
        )
        print(f"    Forward:  {ct_fwd*1e3:.1f} ms")
        print(f"    FD Grad:  {ct_fd_time*1e3:.1f} ms ({2*n_species} forward evals)")
        print(f"    Grad/Fwd: {ct_fd_time/ct_fwd:.2f}x (= 2n)")
        
        # Compare gradient values
        if jt_grad_val is not None:
            print(f"\n  Gradient Comparison:")
            
            # Find non-zero gradients for comparison
            nonzero_mask = np.abs(ct_grad_val) > 1e-10
            if np.any(nonzero_mask):
                rel_error = np.abs(
                    (np.array(jt_grad_val)[nonzero_mask] - ct_grad_val[nonzero_mask]) 
                    / ct_grad_val[nonzero_mask]
                )
                print(f"    Max relative error: {np.max(rel_error)*100:.2f}%")
                print(f"    Mean relative error: {np.mean(rel_error)*100:.2f}%")
            else:
                print(f"    All Cantera gradients near zero - cannot compare")
        
        # Speedup
        if jt_grad is not None:
            speedup = ct_fd_time / jt_grad
            print(f"\n  → Jantera AD is {speedup:.1f}x {'faster' if speedup > 1 else 'slower'} than Cantera FD")


def profile_adjoint_modes():
    """Compare different adjoint computation modes in diffrax."""
    print("\n" + "=" * 70)
    print("ADJOINT MODE COMPARISON")
    print("=" * 70)
    
    yaml = "gri30.yaml"
    T0, P0 = 1500.0, 101325.0
    X0 = "CH4:1, O2:2, N2:7.52"
    t_end = 1e-5
    
    mech = load_mechanism(yaml)
    sol = ct.Solution(yaml)
    sol.TPX = T0, P0, X0
    Y0 = jnp.array(sol.Y)
    
    adjoint_modes = [
        ("RecursiveCheckpoint", diffrax.RecursiveCheckpointAdjoint()),
        # ("Direct", diffrax.DirectAdjoint()),  # Can be slow for stiff problems
    ]
    
    for adj_name, adjoint in adjoint_modes:
        print(f"\n  Adjoint: {adj_name}")
        
        def loss_fn(Y0_input):
            state0 = jnp.concatenate([jnp.array([T0]), Y0_input])
            term = diffrax.ODETerm(reactor_rhs)
            solver = diffrax.Tsit5()
            stepsize_controller = diffrax.PIDController(rtol=1e-6, atol=1e-8)
            
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
                adjoint=adjoint
            )
            return sol.ys[-1, 0]
        
        grad_fn = jax.jit(jax.grad(loss_fn))
        
        try:
            # Warmup
            _ = grad_fn(Y0)
            jax.block_until_ready(_)
            
            # Time
            times = []
            for _ in range(5):
                start = time.perf_counter()
                _ = grad_fn(Y0)
                jax.block_until_ready(_)
                times.append(time.perf_counter() - start)
            
            print(f"    Time: {np.mean(times)*1e3:.1f} ± {np.std(times)*1e3:.1f} ms")
        except Exception as e:
            print(f"    Failed: {e}")


def main():
    compare_gradients()
    profile_adjoint_modes()
    
    print("\n" + "=" * 70)
    print("ADJOINT PROFILING COMPLETE")
    print("=" * 70)
    print("\nKey Insights:")
    print("  - AD gradient cost scales O(1) in n_species (adjoint method)")
    print("  - FD gradient cost scales O(2n) in n_species")  
    print("  - For large mechanisms, AD has significant advantage")
    print("  - Adjoint stability requires careful dt0 tuning (see HANDOFF.md)")


if __name__ == "__main__":
    main()
