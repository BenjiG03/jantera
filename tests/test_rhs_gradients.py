"""
Compare RHS gradients (Jacobian) at t=0 between JAX AD and Cantera FD.
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

def compare_rhs_gradients():
    yaml_path = "gri30.yaml"
    T0, P0 = 1500.0, 101325.0
    X0 = "CH4:1, O2:2, N2:7.52"
    
    mech = load_mechanism(yaml_path)
    sol_ct = ct.Solution(yaml_path)
    sol_ct.TPX = T0, P0, X0
    Y0 = jnp.array(sol_ct.Y)
    state0 = jnp.concatenate([jnp.array([T0]), Y0])
    
    # Objective: dT/dt (first element of RHS)
    def get_dTdt(s):
        return reactor_rhs(0.0, s, (P0, mech))[0]
    
    # 1. JAX AD Gradient
    print("Calculating JAX AD gradient of dT/dt wrt species mass fractions...")
    # Differentiate wrt species parts of state (indices 1:)
    grad_jt = jax.grad(get_dTdt)(state0)[1:]
    
    # 2. Cantera FD Gradient
    print("Calculating Cantera FD gradient of dT/dt wrt species mass fractions...")
    grad_ct = []
    eps = 1e-6
    
    for i in range(sol_ct.n_species):
        # Plus
        Y_plus = sol_ct.Y.copy()
        Y_plus[i] += eps
        sol_ct.Y = Y_plus
        sol_ct.TP = T0, P0
        wdot_plus = sol_ct.net_production_rates
        h_plus = sol_ct.partial_molar_enthalpies
        dTdt_plus = -(wdot_plus @ h_plus) / (sol_ct.density * sol_ct.cp_mass)
        
        # Minus
        Y_minus = sol_ct.Y.copy()
        # Restore Y from sol_ct.Y is safer than subtracting from Y_plus
        sol_ct.TPX = T0, P0, X0
        Y_minus = sol_ct.Y.copy()
        Y_minus[i] -= eps
        sol_ct.Y = Y_minus
        sol_ct.TP = T0, P0
        wdot_minus = sol_ct.net_production_rates
        h_minus = sol_ct.partial_molar_enthalpies
        dTdt_minus = -(wdot_minus @ h_minus) / (sol_ct.density * sol_ct.cp_mass)
        
        grad_ct.append((dTdt_plus - dTdt_minus) / (2 * eps))
        
        # Reset for next iteration
        sol_ct.TPX = T0, P0, X0
        
    grad_ct = np.array(grad_ct)
    
    # 3. Compare
    print("\nTop 10 species by d(dT/dt) / dY_i:")
    compare_idx = np.argsort(np.abs(grad_jt))[-10:][::-1]
    
    table_rows = []
    for i in compare_idx:
        sp_name = sol_ct.species_names[i]
        val_jt = float(grad_jt[i])
        val_ct = grad_ct[i]
        abs_diff = abs(val_jt - val_ct)
        rel_diff = abs_diff / (max(abs(val_ct), 1e-10))
        table_rows.append([sp_name, f"{val_jt:+.4e}", f"{val_ct:+.4e}", f"{abs_diff:.2e}", f"{rel_diff*100:.4f}%"])
        
    print(tabulate(table_rows, headers=["Species", "Jantera (AD)", "Cantera (FD)", "Abs Diff", "Rel Diff"], tablefmt="grid"))

if __name__ == "__main__":
    compare_rhs_gradients()
