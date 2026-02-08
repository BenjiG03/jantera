"""
Debug concentration and rate constant units.
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
from jantera.kinetics import compute_kf, compute_Kc
from jantera.thermo import compute_mixture_props


def debug_units():
    print("=" * 60)
    print("UNIT DEBUGGING")
    print("=" * 60)
    
    yaml_path = "gri30.yaml"
    T0, P0 = 1500.0, 101325.0
    X0 = "CH4:1, O2:2, N2:7.52"
    
    mech = load_mechanism(yaml_path)
    sol_ct = ct.Solution(yaml_path)
    sol_ct.TPX = T0, P0, X0
    Y0 = jnp.array(sol_ct.Y)
    
    # Jantera concentrations
    cp_mass, h_mass, rho = compute_mixture_props(T0, P0, Y0, mech)
    Y_eff = jnp.maximum(Y0, 1e-20)
    conc_jt = float(rho) * np.array(Y_eff) / np.array(mech.mol_weights)
    
    # Cantera concentrations
    conc_ct = sol_ct.concentrations  # mol/m^3
    
    print(f"\n--- Concentrations (first 10 species) ---")
    print(f"{'Species':12s} {'Jantera':>12s} {'Cantera':>12s} {'Ratio':>10s}")
    for i in range(10):
        ratio = conc_jt[i] / (conc_ct[i] + 1e-30) if conc_ct[i] > 1e-30 else 0
        print(f"{sol_ct.species_name(i):12s} {conc_jt[i]:12.4e} {conc_ct[i]:12.4e} {ratio:10.4f}")
    
    # Check major species
    ch4_idx = sol_ct.species_index('CH4')
    o2_idx = sol_ct.species_index('O2')
    n2_idx = sol_ct.species_index('N2')
    
    print(f"\n--- Major Species Concentrations ---")
    for idx, name in [(ch4_idx, 'CH4'), (o2_idx, 'O2'), (n2_idx, 'N2')]:
        ratio = conc_jt[idx] / conc_ct[idx]
        print(f"  {name}: JT={conc_jt[idx]:.4e}, CT={conc_ct[idx]:.4e}, ratio={ratio:.6f}")
    
    # Cantera uses mol/m^3, which we should match
    # Let's check density
    print(f"\n--- Density ---")
    print(f"  Jantera: {float(rho):.6f} kg/m^3")
    print(f"  Cantera: {sol_ct.density:.6f} kg/m^3")
    
    # Check forward rate constants for first few reactions
    print(f"\n--- Forward Rate Constants (kf) ---")
    print(f"(Cantera in kmol, Jantera should be in mol)")
    kf_jt = np.array(compute_kf(T0, jnp.array(conc_jt), mech))
    kf_ct = sol_ct.forward_rate_constants  # Cantera units: depends on reaction order
    
    for i in range(5):
        rxn = sol_ct.reaction(i)
        print(f"  R{i}: {rxn.equation}")
        print(f"       kf_JT = {kf_jt[i]:.4e}")
        print(f"       kf_CT = {kf_ct[i]:.4e}")
        print(f"       ratio = {kf_jt[i]/kf_ct[i]:.4f}")


if __name__ == "__main__":
    debug_units()
