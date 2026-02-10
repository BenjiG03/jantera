"""
Compare wdot at a specific intermediate state where divergence occurs.
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

from canterax.loader import load_mechanism
from canterax.kinetics import compute_wdot

def compare_wdot_at_t():
    yaml_path = "gri30.yaml"
    T0, P0 = 1500.0, 101325.0
    X0 = "CH4:1, O2:2, N2:7.52"
    t_compare = 700e-6
    
    mech = load_mechanism(yaml_path)
    sol_ct = ct.Solution(yaml_path)
    
    # Advance Cantera to t_compare to get a realistic intermediate state
    sol_ct.TPX = T0, P0, X0
    reac = ct.IdealGasConstPressureReactor(sol_ct, clone=False)
    sim = ct.ReactorNet([reac])
    sim.rtol, sim.atol = 1e-10, 1e-14
    sim.advance(t_compare)
    
    T = sol_ct.T
    Y = jnp.array(sol_ct.Y)
    
    print(f"Comparing wdot at T={T:.2f}K (t={t_compare*1e6:.1f}us)")
    print("=" * 70)
    
    # Canterax wdot
    wdot_jt, h_jt, cp_jt, rho_jt = compute_wdot(T, P0, Y, mech)
    wdot_jt = np.array(wdot_jt)
    
    # Cantera wdot
    wdot_ct = sol_ct.net_production_rates
    
    # Compare
    diffs = np.abs(wdot_jt - wdot_ct)
    rel_diffs = diffs / (np.abs(wdot_ct) + 1e-30)
    
    print(f"\nMax absolute diff: {np.max(diffs):.4e} at species {sol_ct.species_name(np.argmax(diffs))}")
    max_rel_idx = np.argmax(np.where(np.abs(wdot_ct) > 1e-10, rel_diffs, 0))
    print(f"Max relative diff (for |wdot|>1e-10): {rel_diffs[max_rel_idx]*100:.2f}% at species {sol_ct.species_name(max_rel_idx)}")
    
    # List top discrepancies
    print(f"\nTop 10 species by absolute wdot difference:")
    sorted_idx = np.argsort(diffs)[::-1]
    for i in sorted_idx[:10]:
        print(f"  {sol_ct.species_name(i):10s}: CT={wdot_ct[i]:+.4e}, JT={wdot_jt[i]:+.4e}, diff={diffs[i]:.2e}, rel={rel_diffs[i]*100:.2f}%")

if __name__ == "__main__":
    compare_wdot_at_t()
