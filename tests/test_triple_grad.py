"""
Refined gradient test with Jantera AD, Jantera FD, and Cantera FD.
Isolates the factor of 2 discrepancy.
"""
import os
import sys
import numpy as np
import cantera as ct
import jax
import jax.numpy as jnp
from tabulate import tabulate

jax.config.update("jax_enable_x64", True)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from jantera.loader import load_mechanism
from jantera.reactor import reactor_rhs
from jantera.kinetics import compute_wdot
from jantera.thermo import compute_mixture_props

def compare_gradients_triple():
    yaml_path = "gri30.yaml"
    T0, P0 = 1500.0, 101325.0
    X0 = "CH4:1, O2:2, N2:7.52"
    eps = 1e-8
    
    mech = load_mechanism(yaml_path)
    sol_ct = ct.Solution(yaml_path)
    sol_ct.TPX = T0, P0, X0
    # Add a small amount of every species to avoid clipping at Y=0
    Y_base = sol_ct.Y + 1e-4
    Y_base /= Y_base.sum()
    sol_ct.Y = Y_base
    Y0 = sol_ct.Y.copy()
    n2_idx = sol_ct.species_index("N2")
    h2_idx = sol_ct.species_index("H2")
    
    # Target: conc_H2
    def get_conc_h2(Y):
        _, _, rho = compute_mixture_props(T0, P0, Y, mech)
        return rho * Y[h2_idx] / mech.mol_weights[h2_idx]
    
    # 1. Jantera AD
    grad_ad = jax.grad(get_conc_h2)(jnp.array(Y0))
    jt_ad = float(grad_ad[h2_idx] - grad_ad[n2_idx])
    
    # 2. Jantera FD
    Y_p = Y0.copy(); Y_p[h2_idx] += eps; Y_p[n2_idx] -= eps
    c_p = float(get_conc_h2(jnp.array(Y_p)))
    Y_m = Y0.copy(); Y_m[h2_idx] -= eps; Y_m[n2_idx] += eps
    c_m = float(get_conc_h2(jnp.array(Y_m)))
    jt_fd = (c_p - c_m) / (2 * eps)
    
    # 3. Cantera FD
    sol_ct.Y = Y_p; sol_ct.TP = T0, P0
    ct_p = sol_ct.concentrations[h2_idx]
    sol_ct.Y = Y_m; sol_ct.TP = T0, P0
    ct_m = sol_ct.concentrations[h2_idx]
    ct_fd = (ct_p - ct_m) / (2 * eps)
    
    print(f"Jantera Params:")
    print(f"  c_p: {c_p:.10e}")
    print(f"  c_m: {c_m:.10e}")
    print(f"  delta_c: {c_p - c_m:.6e}")
    print(f"  JT_FD: {jt_fd:.8e}")
    
    print(f"\nCantera Params:")
    print(f"  ct_p: {ct_p:.10e}")
    print(f"  ct_m: {ct_m:.10e}")
    print(f"  delta_ct: {ct_p - ct_m:.6e}")
    print(f"  CT_FD: {ct_fd:.8e}")
    
    # Target: dTdt
    def get_dTdt(Y):
        state = jnp.concatenate([jnp.array([T0]), Y])
        return reactor_rhs(0.0, state, (P0, mech))[0]
    
    # 1. Jantera AD
    grad_ad_dT = jax.grad(get_dTdt)(jnp.array(Y0))
    jt_ad_dT = float(grad_ad_dT[h2_idx] - grad_ad_dT[n2_idx])
    
    # 2. Jantera FD
    c_p_dT = float(get_dTdt(jnp.array(Y_p)))
    c_m_dT = float(get_dTdt(jnp.array(Y_m)))
    jt_fd_dT = (c_p_dT - c_m_dT) / (2 * eps)
    
    # 3. Cantera FD
    sol_ct.Y = Y_p; sol_ct.TP = T0, P0
    w_p = sol_ct.net_production_rates; h_p = sol_ct.partial_molar_enthalpies
    ct_p_dT = -(w_p @ h_p) / (sol_ct.density * sol_ct.cp_mass)
    
    sol_ct.Y = Y_m; sol_ct.TP = T0, P0
    w_m = sol_ct.net_production_rates; h_m = sol_ct.partial_molar_enthalpies
    ct_m_dT = -(w_m @ h_m) / (sol_ct.density * sol_ct.cp_mass)
    
    ct_fd_dT = (ct_p_dT - ct_m_dT) / (2 * eps)
    
    print(f"\nSummary for d(dTdt)/dY_H2:")
    print(f"  Jantera AD: {jt_ad_dT:.8e}")
    print(f"  Jantera FD: {jt_fd_dT:.8e}")
    print(f"  Cantera FD: {ct_fd_dT:.8e}")
    print(f"  Ratio AD/FD: {jt_ad_dT/jt_fd_dT:.6f}")
    print(f"  Ratio JT/CT: {jt_fd_dT/ct_fd_dT:.6f}")

if __name__ == "__main__":
    compare_gradients_triple()
