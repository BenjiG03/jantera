"""
Test impact of adjoint configuration on performance.
"""
import os
import sys
import time
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import cantera as ct
import diffrax

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from jantera.loader import load_mechanism
from jantera.reactor import reactor_rhs


def test_adjoint_overhead():
    print("=" * 60)
    print("ADJOINT OVERHEAD TEST")
    print("=" * 60)
    
    yaml_path = "gri30.yaml"
    T0, P0 = 1500.0, 101325.0
    X0 = "CH4:1, O2:2, N2:7.52"
    t_end = 1e-3
    
    mech = load_mechanism(yaml_path)
    sol_ct = ct.Solution(yaml_path)
    sol_ct.TPX = T0, P0, X0
    Y0 = jnp.array(sol_ct.Y)
    state0 = jnp.concatenate([jnp.array([T0]), Y0])
    
    term = diffrax.ODETerm(reactor_rhs)
    solver = diffrax.Kvaerno5()
    stepsize_controller = diffrax.PIDController(rtol=1e-7, atol=1e-10)
    saveat = diffrax.SaveAt(t1=True)
    
    # Test 1: With RecursiveCheckpointAdjoint (current)
    print("\nTest 1: RecursiveCheckpointAdjoint (current config)")
    
    # Warmup
    _ = diffrax.diffeqsolve(
        term, solver, t0=0.0, t1=1e-8, dt0=1e-10, y0=state0,
        args=(P0, mech), stepsize_controller=stepsize_controller,
        max_steps=100, saveat=saveat,
        adjoint=diffrax.RecursiveCheckpointAdjoint()
    )
    
    start = time.perf_counter()
    sol = diffrax.diffeqsolve(
        term, solver, t0=0.0, t1=t_end, dt0=1e-8, y0=state0,
        args=(P0, mech), stepsize_controller=stepsize_controller,
        max_steps=1000000, saveat=saveat,
        adjoint=diffrax.RecursiveCheckpointAdjoint()
    )
    jax.block_until_ready(sol)
    t1 = time.perf_counter() - start
    print(f"  Time: {t1*1e3:.1f} ms, T_final: {float(sol.ys[-1, 0]):.2f} K")
    
    # Test 2: No adjoint (default)
    print("\nTest 2: No adjoint (default)")
    
    # Warmup
    _ = diffrax.diffeqsolve(
        term, solver, t0=0.0, t1=1e-8, dt0=1e-10, y0=state0,
        args=(P0, mech), stepsize_controller=stepsize_controller,
        max_steps=100, saveat=saveat
    )
    
    start = time.perf_counter()
    sol = diffrax.diffeqsolve(
        term, solver, t0=0.0, t1=t_end, dt0=1e-8, y0=state0,
        args=(P0, mech), stepsize_controller=stepsize_controller,
        max_steps=1000000, saveat=saveat
    )
    jax.block_until_ready(sol)
    t2 = time.perf_counter() - start
    print(f"  Time: {t2*1e3:.1f} ms, T_final: {float(sol.ys[-1, 0]):.2f} K")
    
    # Test 3: DirectAdjoint
    print("\nTest 3: DirectAdjoint")
    
    # Warmup
    _ = diffrax.diffeqsolve(
        term, solver, t0=0.0, t1=1e-8, dt0=1e-10, y0=state0,
        args=(P0, mech), stepsize_controller=stepsize_controller,
        max_steps=100, saveat=saveat,
        adjoint=diffrax.DirectAdjoint()
    )
    
    start = time.perf_counter()
    sol = diffrax.diffeqsolve(
        term, solver, t0=0.0, t1=t_end, dt0=1e-8, y0=state0,
        args=(P0, mech), stepsize_controller=stepsize_controller,
        max_steps=1000000, saveat=saveat,
        adjoint=diffrax.DirectAdjoint()
    )
    jax.block_until_ready(sol)
    t3 = time.perf_counter() - start
    print(f"  Time: {t3*1e3:.1f} ms, T_final: {float(sol.ys[-1, 0]):.2f} K")
    
    # Cantera reference
    print("\n--- Cantera Reference ---")
    sol_ct.TPX = T0, P0, X0
    reac = ct.IdealGasConstPressureReactor(sol_ct)
    net_ct = ct.ReactorNet([reac])
    
    start = time.perf_counter()
    net_ct.advance(t_end)
    ct_time = time.perf_counter() - start
    print(f"  Time: {ct_time*1e3:.1f} ms, T_final: {sol_ct.T:.2f} K")


if __name__ == "__main__":
    test_adjoint_overhead()
