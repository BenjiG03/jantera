"""
Direct RHS comparison to diagnose discrepancy.
"""
import os
import sys
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import cantera as ct

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from jantera.loader import load_mechanism
from jantera.kinetics import compute_wdot, compute_Kc


def compare_rhs():
    print("=" * 60)
    print("DIRECT RHS COMPARISON")
    print("=" * 60)
    
    yaml_path = "gri30.yaml"
    T0, P0 = 1500.0, 101325.0
    X0 = "CH4:1, O2:2, N2:7.52"
    
    mech = load_mechanism(yaml_path)
    sol_ct = ct.Solution(yaml_path)
    sol_ct.TPX = T0, P0, X0
    Y0 = jnp.array(sol_ct.Y)
    
    print(f"\n--- Species Properties (CH4) ---")
    ch4_idx = sol_ct.species_index('CH4')
    print(f"  MW:  JT={float(mech.mol_weights[ch4_idx]):.4f}, CT={sol_ct.molecular_weights[ch4_idx]:.4f}")
    
    # Jantera
    wdot_jt, h_mass_jt, cp_mass_jt, rho_jt = compute_wdot(T0, P0, Y0, mech)
    
    # Cantera
    wdot_ct = sol_ct.net_production_rates  # mol/m^3/s
    rho_ct = sol_ct.density
    cp_mass_ct = sol_ct.cp_mass
    
    print(f"\n--- Mixture Properties ---")
    print(f"  rho:     JT={float(rho_jt):.6f}, CT={rho_ct:.6f}, diff={abs(float(rho_jt) - rho_ct):.2e}")
    print(f"  cp_mass: JT={float(cp_mass_jt):.6f}, CT={cp_mass_ct:.6f}, diff={abs(float(cp_mass_jt) - cp_mass_ct):.2e}")
    
    print(f"\n--- net_production_rates (wdot) ---")
    wdot_jt_np = np.array(wdot_jt)
    max_idx = np.argmax(np.abs(wdot_ct - wdot_jt_np))
    max_diff = np.max(np.abs(wdot_ct - wdot_jt_np))
    rel_err = np.max(np.abs(wdot_ct - wdot_jt_np) / (np.abs(wdot_ct) + 1e-30))
    
    print(f"  Max abs diff: {max_diff:.2e} at species {sol_ct.species_name(max_idx)}")
    print(f"  Max rel diff: {rel_err:.2e}")
    
    # Show top 5 discrepancies
    diffs = np.abs(wdot_ct - wdot_jt_np)
    top_idx = np.argsort(diffs)[::-1][:5]
    print(f"\n  Top 5 discrepancies:")
    for idx in top_idx:
        print(f"    {sol_ct.species_name(idx):10s}: CT={wdot_ct[idx]:+.4e}, JT={wdot_jt_np[idx]:+.4e}, diff={diffs[idx]:.2e}")
    
    # Kc comparison
    kc_jt = np.array(compute_Kc(T0, mech))
    kc_ct = sol_ct.equilibrium_constants  # Cantera units: kmol/m^3 based
    
    print(f"\n--- Equilibrium Constants (Kc) ---")
    max_kc_err = np.max(np.abs(kc_jt - kc_ct) / (np.abs(kc_ct) + 1e-30))
    print(f"  Max rel diff: {max_kc_err:.2e}")
    # Show top discrepancies
    top_kc = np.argsort(np.abs(kc_jt - kc_ct) / (np.abs(kc_ct) + 1e-30))[::-1][:3]
    for i in top_kc:
        print(f"    R{i}: CT={kc_ct[i]:.4e}, JT={kc_jt[i]:.4e}, rel_err={np.abs(kc_jt[i]-kc_ct[i])/(kc_ct[i]+1e-30):.2e}")

    print(f"\n--- Species Evolution (dY/dt) ---")
    dYdt_jt = np.array(wdot_jt_np * np.array(mech.mol_weights) / float(rho_jt))
    dYdt_ct = sol_ct.net_production_rates * sol_ct.molecular_weights / sol_ct.density
    
    max_dydt_err = np.max(np.abs(dYdt_jt - dYdt_ct))
    print(f"  Max abs diff: {max_dydt_err:.2e}")
    # Show top discrepancies
    top_dydt = np.argsort(np.abs(dYdt_jt - dYdt_ct))[::-1][:3]
    for i in top_dydt:
        print(f"    {sol_ct.species_name(i):10s}: CT={dYdt_ct[i]:+.4e}, JT={dYdt_jt[i]:+.4e}, diff={abs(dYdt_jt[i]-dYdt_ct[i]):.2e}")

    # Compute dT/dt
    from jantera.thermo import get_h_RT
    from jantera.constants import R_GAS
    
    h_RT = get_h_RT(T0, mech.nasa_low, mech.nasa_high, mech.nasa_T_mid)
    h_mol_jt = np.array(h_RT) * R_GAS * T0
    energy_term_jt = np.sum(h_mol_jt * wdot_jt_np)
    dTdt_jt = -energy_term_jt / (float(rho_jt) * float(cp_mass_jt))
    
    # Cantera dT/dt
    h_mol_ct = sol_ct.partial_molar_enthalpies
    energy_term_ct = np.sum(h_mol_ct * wdot_ct)
    dTdt_ct = -energy_term_ct / (rho_ct * cp_mass_ct)
    
    print(f"\n--- Temperature Rate (dT/dt) ---")
    print(f"  Jantera:  {dTdt_jt:+.4e} K/s")
    print(f"  Cantera:  {dTdt_ct:+.4e} K/s")
    print(f"  Diff:     {abs(dTdt_jt - dTdt_ct):.2e} K/s ({100*abs(dTdt_jt - dTdt_ct)/abs(dTdt_ct):.2f}%)")


if __name__ == "__main__":
    compare_rhs()
