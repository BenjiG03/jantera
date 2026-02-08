"""
Systematic diagnosis of JP-10 nan gradients.
Traces through the computation graph to find the exact source.
"""
import os
import sys
import numpy as np
import cantera as ct
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from jantera.loader import load_mechanism
from jantera.reactor import ReactorNet, reactor_rhs
import diffrax

def diagnose_jp10():
    yaml_path = "jp10.yaml"
    T, P = 1500.0, 101325.0
    X = "C10H16:1, O2:14, N2:52.64"
    t_grad = 1e-7
    
    mech = load_mechanism(yaml_path)
    sol_ct = ct.Solution(yaml_path)
    sol_ct.TPX = T, P, X
    
    # Use non-zero baseline state
    Y_base = sol_ct.Y + 1e-4
    Y_base /= Y_base.sum()
    Y0 = jnp.array(Y_base)
    
    net = ReactorNet(mech)
    
    # Step 1: Check if RHS gradients at t=0 are finite
    print("=" * 60)
    print("Step 1: RHS gradients at t=0")
    print("=" * 60)
    
    def get_dTdt(Y):
        state = jnp.concatenate([jnp.array([T]), Y])
        return reactor_rhs(0.0, state, (P, mech))[0]
    
    grad_rhs = jax.grad(get_dTdt)(Y0)
    print(f"  Any nan in d(dTdt)/dY? {jnp.isnan(grad_rhs).any()}")
    if jnp.isnan(grad_rhs).any():
        nan_indices = jnp.where(jnp.isnan(grad_rhs))[0]
        print(f"  Nan at species indices: {nan_indices}")
        print(f"  Species: {[sol_ct.species_names[int(i)] for i in nan_indices]}")
        return # Stop here if RHS is already problematic
    else:
        print("  RHS gradients are FINITE. Proceeding to integration check.")
    
    # Step 2: Check if forward pass produces nan
    print("\n" + "=" * 60)
    print("Step 2: Forward pass integration")
    print("=" * 60)
    
    res = net.advance(T, P, Y0, t_grad, rtol=1e-10, atol=1e-14, solver=diffrax.Tsit5())
    T_final = res.ys[-1, 0]
    Y_final = res.ys[-1, 1:]
    
    print(f"  T_final: {T_final}")
    print(f"  Any nan in Y_final? {jnp.isnan(Y_final).any()}")
    if jnp.isnan(Y_final).any():
        nan_indices = jnp.where(jnp.isnan(Y_final))[0]
        print(f"  Nan at species indices: {nan_indices}")
        return
    print("  Forward pass is FINITE.")
    
    # Step 3: Check gradients via JAX (the actual failure point)
    print("\n" + "=" * 60)
    print("Step 3: JAX AD gradient of T_final w.r.t Y0")
    print("=" * 60)
    
    def get_final_T(y0):
        y_norm = y0 / jnp.sum(y0)
        res = net.advance(T, P, y_norm, t_grad, rtol=1e-10, atol=1e-14, solver=diffrax.Tsit5())
        return res.ys[-1, 0]
    
    grad_T = jax.grad(get_final_T)(Y0)
    print(f"  Any nan in d(T_final)/dY0? {jnp.isnan(grad_T).any()}")
    if jnp.isnan(grad_T).any():
        nan_indices = jnp.where(jnp.isnan(grad_T))[0]
        print(f"  Nan at species indices: {nan_indices}")
        print(f"  Species: {[sol_ct.species_names[int(i)] for i in nan_indices]}")
    else:
        print("  PASSED. All gradients are finite.")
        return
    
    # Step 4: Binary search - does the nan appear at earlier times?
    print("\n" + "=" * 60)
    print("Step 4: Binary search for nan onset time")
    print("=" * 60)
    
    for t_test in [1e-10, 1e-9, 1e-8, 1e-7]:
        def get_T_at_t(y0):
            y_norm = y0 / jnp.sum(y0)
            res = net.advance(T, P, y_norm, t_test, rtol=1e-10, atol=1e-14, solver=diffrax.Tsit5())
            return res.ys[-1, 0]
        
        grad = jax.grad(get_T_at_t)(Y0)
        has_nan = jnp.isnan(grad).any()
        print(f"  t={t_test:.0e}: nan={has_nan}")
        if has_nan:
            print(f"    -> NaN first appears at or before t={t_test:.0e}")
            break
    
    # Step 5: Check individual species sensitivities
    print("\n" + "=" * 60)
    print("Step 5: Per-species gradient trace")
    print("=" * 60)
    
    # Use the smallest t where nan appears
    def get_T_small(y0):
        y_norm = y0 / jnp.sum(y0)
        res = net.advance(T, P, y_norm, 1e-10, rtol=1e-10, atol=1e-14, solver=diffrax.Tsit5())
        return res.ys[-1, 0]
    
    grad_small = jax.grad(get_T_small)(Y0)
    if jnp.isnan(grad_small).any():
        print("  Even at t=1e-10 there are nan gradients.")
        print("  This suggests the issue is in the adjoint computation itself, not the trajectory.")
        
        # Check state after one step
        print("\n  Checking state after a single ODE step...")
        state0 = jnp.concatenate([jnp.array([T]), Y0])
        rhs0 = reactor_rhs(0.0, state0, (P, mech))
        print(f"  RHS[0] (dT/dt): {rhs0[0]}")
        print(f"  RHS has nan? {jnp.isnan(rhs0).any()}")
        
        # Check RHS gradient again for each component
        print("\n  Checking gradient of each RHS component...")
        for i in range(min(5, len(Y0))):
            def get_dYi_dt(Y):
                state = jnp.concatenate([jnp.array([T]), Y])
                return reactor_rhs(0.0, state, (P, mech))[1 + i]
            
            grad_Yi = jax.grad(get_dYi_dt)(Y0)
            has_nan_Yi = jnp.isnan(grad_Yi).any()
            print(f"    d(dY_{i}/dt)/dY: nan={has_nan_Yi}")
            if has_nan_Yi:
                nan_idx = jnp.where(jnp.isnan(grad_Yi))[0]
                print(f"      Nan at: {[sol_ct.species_names[int(j)] for j in nan_idx[:5]]}")
    else:
        print("  Gradients at t=1e-10 are finite. NaN appears during longer integration.")

if __name__ == "__main__":
    diagnose_jp10()
