"""
Final RHS gradient comparison between Jantera AD and Cantera FD.
Isolates the factor of 2 discrepancy by checking concentration gradients.
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

def compare_rhs_gradients_simplified():
    yaml_path = "gri30.yaml"
    T0, P0 = 1500.0, 101325.0
    X0 = "CH4:1, O2:2, N2:7.52"
    eps = 1e-8
    
    mech = load_mechanism(yaml_path)
    sol_ct = ct.Solution(yaml_path)
    sol_ct.TPX = T0, P0, X0
    Y0_fixed = sol_ct.Y.copy()
    n2_idx = sol_ct.species_index("N2")
    
    # Check base values
    state0 = jnp.concatenate([jnp.array([T0]), jnp.array(Y0_fixed)])
    rhs0 = reactor_rhs(0.0, state0, (P0, mech))
    jt_dTdt0 = float(rhs0[0])
    
    sol_ct.TPX = T0, P0, X0
    wdot_ct = sol_ct.net_production_rates
    h_ct = sol_ct.partial_molar_enthalpies
    ct_dTdt0 = -(wdot_ct @ h_ct) / (sol_ct.density * sol_ct.cp_mass)
    
    print(f"Base Values at t=0:")
    print(f"  Jantera dTdt: {jt_dTdt0:.6e}")
    print(f"  Cantera dTdt: {ct_dTdt0:.6e}")
    print(f"  Ratio: {jt_dTdt0/ct_dTdt0:.6f}")

    # 1. Function for concentration
    def get_conc0(Y):
        _, _, rho = compute_mixture_props(T0, P0, Y, mech)
        return rho * Y[0] / mech.mol_weights[0]
    
    print("\nCalculating Jantera AD gradients (conc[0])...")
    grad_jt_conc = jax.grad(get_conc0)(jnp.array(Y0_fixed))
    grad_jt_conc_eff = grad_jt_conc - grad_jt_conc[n2_idx]
    
    # Cantera FD for conc[0]
    print("Calculating Cantera FD gradient (conc[0])...")
    grad_ct_conc = []
    
    for i in range(sol_ct.n_species):
        if i == n2_idx:
            grad_ct_conc.append(0.0)
            continue
        # Plus
        Y_plus = Y0_fixed.copy()
        Y_plus[i] += eps; Y_plus[n2_idx] -= eps
        sol_ct.Y = Y_plus; sol_ct.TP = T0, P0
        c0_plus = sol_ct.concentrations[0]
        # Minus
        Y_minus = Y0_fixed.copy()
        Y_minus[i] -= eps; Y_minus[n2_idx] += eps
        sol_ct.Y = Y_minus; sol_ct.TP = T0, P0
        c0_minus = sol_ct.concentrations[0]
        grad_ct_conc.append((c0_plus - c0_minus) / (2 * eps))
        sol_ct.TPX = T0, P0, X0
    
    grad_ct_conc = np.array(grad_ct_conc)
    print(f"\nd(conc_{sol_ct.species_name(0)}) / dY_{sol_ct.species_name(0)}:")
    print(f"  JT: {grad_jt_conc_eff[0]:.6e}")
    print(f"  CT: {grad_ct_conc[0]:.6e}")
    print(f"  Ratio: {grad_jt_conc_eff[0]/grad_ct_conc[0]:.4f}x")

    # 2. Function for dTdt
    def get_dTdt(Y):
        state = jnp.concatenate([jnp.array([T0]), Y])
        return reactor_rhs(0.0, state, (P0, mech))[0]
    
    print("\nCalculating Jantera AD gradients (dTdt)...")
    grad_jt_dTdt = jax.grad(get_dTdt)(jnp.array(Y0_fixed))
    grad_jt_dTdt_eff = grad_jt_dTdt - grad_jt_dTdt[n2_idx]
    
    # Cantera FD for dTdt
    print("Calculating Cantera FD gradient (dTdt)...")
    grad_ct_dTdt = []
    
    for i in range(sol_ct.n_species):
        if i == n2_idx:
            grad_ct_dTdt.append(0.0)
            continue
        # Plus
        Y_plus = Y0_fixed.copy()
        Y_plus[i] += eps; Y_plus[n2_idx] -= eps
        sol_ct.Y = Y_plus; sol_ct.TP = T0, P0
        wdot_p = sol_ct.net_production_rates
        h_p = sol_ct.partial_molar_enthalpies
        dTdt_plus = -(wdot_p @ h_p) / (sol_ct.density * sol_ct.cp_mass)
        # Minus
        Y_minus = Y0_fixed.copy()
        Y_minus[i] -= eps; Y_minus[n2_idx] += eps
        sol_ct.Y = Y_minus; sol_ct.TP = T0, P0
        wdot_m = sol_ct.net_production_rates
        h_m = sol_ct.partial_molar_enthalpies
        dTdt_minus = -(wdot_m @ h_m) / (sol_ct.density * sol_ct.cp_mass)
        grad_ct_dTdt.append((dTdt_plus - dTdt_minus) / (2 * eps))
        sol_ct.TPX = T0, P0, X0
    
    grad_ct_dTdt = np.array(grad_ct_dTdt)
    
    print("\nTop 10 species by d(dTdt) / dY_j:")
    compare_idx = np.argsort(np.abs(grad_ct_dTdt))[-10:][::-1]
    table_rows = []
    for i in compare_idx:
        sp_name = sol_ct.species_names[i]
        val_jt = float(grad_jt_dTdt_eff[i])
        val_ct = grad_ct_dTdt[i]
        table_rows.append([sp_name, f"{val_jt:+.4e}", f"{val_ct:+.4e}", f"{val_jt/val_ct:.4f}x"])
    print(tabulate(table_rows, headers=["Species", "Jantera (AD)", "Cantera (FD)", "Ratio"], tablefmt="grid"))

if __name__ == "__main__":
    compare_rhs_gradients_simplified()
