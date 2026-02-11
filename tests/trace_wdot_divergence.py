import os
import sys
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import cantera as ct
from tabulate import tabulate

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from canterax.loader import load_mechanism
from canterax.solution import Solution
from canterax.kinetics import compute_wdot

def trace_divergence(mech_name, yaml_file, condition):
    # Construct absolute path to yaml file if it's gri30.yaml which might not be local
    if yaml_file == "gri30.yaml":
        # Usually Cantera finds gri30.yaml automatically, but load_mechanism might need help if it uses strict paths
        # But load_mechanism uses Cantera to parse, passing the string path.
        pass

    full_yaml_path = yaml_file
    if not os.path.exists(full_yaml_path) and yaml_file == "gri30.yaml":
        # Try to find where Cantera keeps it or just pass "gri30.yaml" assuming Cantera resolves it
        # But load_mechanism might fail if it tries to open the file directly without Cantera's search path logic
        # Let's hope load_mechanism handles "gri30.yaml" correctly (it uses ct.Solution)
        pass

    print(f"\n{'='*30}\nAnalyzing {mech_name}...\n{'='*30}")
    
    # Load
    try:
        sol_ct = ct.Solution(full_yaml_path)
        mech_jt = load_mechanism(full_yaml_path)
        sol_jt = Solution(full_yaml_path)
    except Exception as e:
        print(f"Failed to load mechanism: {e}")
        return
    
    # Parse composition to dict to be safe
    # But condition[2] is a string. Let's just pass it.
    input_str = condition[2]
    T, P = condition[0], condition[1]
    
    print(f"Setting CT state: {T} K, {P} Pa, {input_str}")
    sol_ct.TPX = T, P, input_str
    sol_jt.TPX = T, P, input_str

    # Debug State
    print(f"State: T={sol_ct.T:.1f} K, P={sol_ct.P:.1f} Pa")
    h_idx = sol_ct.species_index("H") if "H" in sol_ct.species_names else -1
    if h_idx >= 0:
        print(f"H Y: {sol_ct.Y[h_idx]:.6e} X: {sol_ct.X[h_idx]:.6e}")
        
    wdot_jt, h_jt, cp_jt, rho_jt, _ = compute_wdot(sol_jt.T, sol_jt.P, sol_jt.Y, mech_jt)
    wdot_ct = sol_ct.net_production_rates
    
    print(f"Max CT wdot: {np.max(np.abs(wdot_ct)):.6e}")
    print(f"Max JT wdot: {np.max(np.abs(wdot_jt)):.6e}")
    
    # Check Thermo first
    print(f"Thermo Check (Cp_mass):")
    cp_ct = sol_ct.cp_mass
    cp_err = abs(cp_ct - cp_jt) / cp_ct
    print(f"  CT: {cp_ct:.6e}, JT: {cp_jt:.6e}, Rel Err: {cp_err:.2e}")
    if cp_err > 1e-10:
        print("  WARNING: High thermodynamic error!")

    # Analyze wdot
    table_data = []
    headers = ["Species", "CT wdot", "JT wdot", "Abs Diff", "Rel Err"]
    
    max_rel_err = 0.0
    
    for k in range(sol_ct.n_species):
        ct_val = wdot_ct[k]
        jt_val = wdot_jt[k]
        diff = abs(ct_val - jt_val)
        
        # Avoid division by zero
        denom = max(abs(ct_val), 1e-25)
        rel_err = diff / denom
        
        if rel_err > max_rel_err:
            max_rel_err = rel_err
            
        # Filter:
        # 1. Significant relative error (> 0.1%)
        # 2. AND Absolute value is not effectively zero (e.g. > 1e-20)
        # OR
        # 3. User mentioned "lower end", so maybe show top errors even for small values
        
        if rel_err > 1e-3:
             table_data.append([
                 sol_ct.species_names[k],
                 f"{ct_val:.4e}",
                 f"{jt_val:.4e}",
                 f"{diff:.4e}",
                 f"{rel_err:.2e}"
             ])

    if table_data:
        # Sort by Rel Err (last column)
        # But Rel Err is a string now. Store as tuple then format?
        # Re-sort using raw values
        table_data.sort(key=lambda x: float(x[4]), reverse=True)
        print("\nTop Deviations (> 0.1%):")
        print(tabulate(table_data, headers=headers, tablefmt="simple"))
    else:
        print("\nNo deviations > 0.1% found.")
        
    print(f"\nMax Relative Error across all species: {max_rel_err:.2e}")

if __name__ == "__main__":
    # Test conditions from validation suite
    # print("--- 1. Testing Standard Conditions (Likely Induction/Stalled) ---")
    # trace_divergence("GRI-30 Propane (Standard)", "gri30.yaml", (1500.0, 101325.0, "C3H8:1, O2:5, N2:18.8"))
    
    # print("\n--- 2. Testing High Temperature (Active Autoignition) ---")
    # trace_divergence("GRI-30 Propane (2500K)", "gri30.yaml", (2500.0, 101325.0, "C3H8:1, O2:5, N2:18.8"))
    
    print("\n--- 3. Testing Radical Seeded (Active Kinetics) ---")
    trace_divergence("GRI-30 Propane (Seeded)", "gri30.yaml", (1500.0, 101325.0, "C3H8:1, O2:5, N2:18.8, H:0.01"))
