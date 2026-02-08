"""
Compare ROP for every reaction to find the discrepancy.
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


def compare_all_reactions():
    print("=" * 60)
    print("ALL REACTIONS ROP COMPARISON")
    print("=" * 60)
    
    yaml_path = "gri30.yaml"
    T0, P0 = 1500.0, 101325.0
    X0 = "CH4:1, O2:2, N2:7.52"
    
    mech = load_mechanism(yaml_path)
    sol_ct = ct.Solution(yaml_path)
    sol_ct.TPX = T0, P0, X0
    
    # Jantera ROP
    conc = jnp.array(sol_ct.concentrations)
    kf = np.array(compute_kf(T0, conc, mech))
    pkc = np.array(compute_Kc(T0, mech))
    kr = kf / pkc
    
    # Compute ROP manually in Jantera style to inspect each step
    rop_f_jt = []
    rop_r_jt = []
    
    for i in range(mech.n_reactions):
        # Forward
        rf = kf[i]
        for sp_idx, nu in enumerate(mech.reactant_stoich[i]):
            if nu > 0:
                rf *= float(conc[sp_idx]**nu)
        rop_f_jt.append(rf)
        
        # Reverse
        rr = kr[i]
        for sp_idx, nu in enumerate(mech.product_stoich[i]):
            if nu > 0:
                rr *= float(conc[sp_idx]**nu)
        if not mech.is_reversible[i]:
            rr = 0.0
        rop_r_jt.append(rr)
        
    rop_jt = np.array(rop_f_jt) - np.array(rop_r_jt)
    
    # Cantera ROP
    rop_ct = sol_ct.net_rates_of_progress
    rop_f_ct = sol_ct.forward_rates_of_progress
    rop_r_ct = sol_ct.reverse_rates_of_progress
    
    print(f"\nComparing {mech.n_reactions} reactions...")
    
    diffs = np.abs(rop_jt - rop_ct)
    rel_diffs = diffs / (np.abs(rop_ct) + 1e-20)
    
    # Find reactions with > 1% error
    bad_idx = np.where(rel_diffs > 0.01)[0]
    
    print(f"Number of reactions with > 1% error: {len(bad_idx)}")
    
    if len(bad_idx) > 0:
        print(f"\nTop discrepancies:")
        sort_idx = np.argsort(rel_diffs)[::-1]
        for i in sort_idx[:10]:
            if rel_diffs[i] < 1e-10: continue
            rxn = sol_ct.reaction(i)
            print(f"R{i:3d}: {rxn.equation}")
            print(f"      CT: f={rop_f_ct[i]:.4e}, r={rop_r_ct[i]:.4e}, net={rop_ct[i]:.4e}")
            print(f"      JT: f={rop_f_jt[i]:.4e}, r={rop_r_jt[i]:.4e}, net={rop_jt[i]:.4e}")
            print(f"      RelErr: {rel_diffs[i]:.4e}")
    else:
        print("All reactions match ROP within 1%!")

if __name__ == "__main__":
    compare_all_reactions()
