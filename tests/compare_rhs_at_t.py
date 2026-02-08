"""
Compare RHS at a specific time step where divergence is occurring.
"""
import os
import sys
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import cantera as ct
from diffrax import SaveAt, Kvaerno5

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from jantera.loader import load_mechanism
from jantera.reactor import ReactorNet
from jantera.kinetics import compute_wdot
from jantera.thermo import compute_mixture_props, get_h_RT
from jantera.constants import R_GAS

def compare_at_t():
    yaml_path = "gri30.yaml"
    T0, P0 = 1500.0, 101325.0
    X0 = "CH4:1, O2:2, N2:7.52"
    t_compare = 700e-6  # 700 us - where divergence starts to be significant
    
    mech = load_mechanism(yaml_path)
    sol_ct = ct.Solution(yaml_path)
    sol_ct.TPX = T0, P0, X0
    Y0 = jnp.array(sol_ct.Y)
    
    net = ReactorNet(mech)
    
    # Advance Jantera to t_compare
    res = net.advance(T0, P0, Y0, t_compare, rtol=1e-10, atol=1e-14, solver=Kvaerno5())
    T_jt = float(res.ys[-1, 0])
    Y_jt = jnp.array(res.ys[-1, 1:])
    
    # Advance Cantera to t_compare
    sol_ct.TPX = T0, P0, X0
    reac = ct.IdealGasConstPressureReactor(sol_ct, clone=False)
    sim = ct.ReactorNet([reac])
    sim.rtol, sim.atol = 1e-10, 1e-14
    sim.advance(t_compare)
    T_ct = sol_ct.T
    Y_ct = sol_ct.Y
    
    print(f"State at t = {t_compare*1e6:.1f} us")
    print(f"=" * 60)
    print(f"T_Jantera: {T_jt:.4f} K")
    print(f"T_Cantera: {T_ct:.4f} K")
    print(f"dT: {T_jt - T_ct:.4f} K")
    
    # Compare RHS at JANTERA's state
    print(f"\n--- RHS at Jantera's State ---")
    wdot_jt, h_jt, cp_jt, rho_jt = compute_wdot(T_jt, P0, Y_jt, mech)
    
    # Compute dT/dt for Jantera
    h_RT = get_h_RT(T_jt, mech.nasa_low, mech.nasa_high, mech.nasa_T_mid)
    h_molar = h_RT * R_GAS * T_jt
    dT_dt_jt = -float(jnp.dot(wdot_jt, h_molar)) / float(cp_jt * rho_jt)
    print(f"  dT/dt: {dT_dt_jt:.4e} K/s")
    
    # Compare RHS at CANTERA's state using Cantera itself
    print(f"\n--- RHS at Cantera's State ---")
    dT_dt_ct = sol_ct.net_production_rates @ (sol_ct.partial_molar_enthalpies)
    dT_dt_ct = -dT_dt_ct / (sol_ct.cp_mass * sol_ct.density)
    print(f"  dT/dt: {dT_dt_ct:.4e} K/s")
    
    # Now compute Jantera RHS at Cantera's state for apples-to-apples
    print(f"\n--- RHS Comparison at SAME State (Cantera's) ---")
    wdot_jt_at_ct, h_jt_at_ct, cp_jt_at_ct, rho_jt_at_ct = compute_wdot(T_ct, P0, jnp.array(Y_ct), mech)
    h_RT_ct = get_h_RT(T_ct, mech.nasa_low, mech.nasa_high, mech.nasa_T_mid)
    h_molar_ct = h_RT_ct * R_GAS * T_ct
    dT_dt_jt_at_ct = -float(jnp.dot(wdot_jt_at_ct, h_molar_ct)) / float(cp_jt_at_ct * rho_jt_at_ct)
    
    print(f"  Jantera dT/dt: {dT_dt_jt_at_ct:.4e} K/s")
    print(f"  Cantera dT/dt: {dT_dt_ct:.4e} K/s")
    print(f"  Diff: {abs(dT_dt_jt_at_ct - dT_dt_ct):.4e} K/s ({abs(dT_dt_jt_at_ct - dT_dt_ct)/abs(dT_dt_ct)*100:.4f}%)")

if __name__ == "__main__":
    compare_at_t()
