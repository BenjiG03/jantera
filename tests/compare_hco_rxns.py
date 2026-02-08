"""
Compare kf and ROP for HCO-related reactions at the divergence state.
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

def compare_hco_reactions():
    yaml_path = "gri30.yaml"
    T0, P0 = 1500.0, 101325.0
    X0 = "CH4:1, O2:2, N2:7.52"
    t_compare = 700e-6
    
    mech = load_mechanism(yaml_path)
    sol_ct = ct.Solution(yaml_path)
    
    # Advance to t=700us
    sol_ct.TPX = T0, P0, X0
    reac = ct.IdealGasConstPressureReactor(sol_ct, clone=False)
    sim = ct.ReactorNet([reac])
    sim.rtol, sim.atol = 1e-10, 1e-14
    sim.advance(t_compare)
    
    T = sol_ct.T
    conc = jnp.array(sol_ct.concentrations)
    
    print(f"Comparing HCO reactions at T={T:.2f}K")
    print("=" * 80)
    
    # Jantera kf
    kf_jt = np.array(compute_kf(T, conc, mech))
    kc_jt = np.array(compute_Kc(T, mech))
    kr_jt = kf_jt / (kc_jt + 1e-100)
    
    # Cantera kf
    kf_ct = sol_ct.forward_rate_constants
    kr_ct = sol_ct.reverse_rate_constants
    
    # Find HCO reactions
    hco_idx = sol_ct.species_index('HCO')
    hco_rxns = []
    for i in range(sol_ct.n_reactions):
        rxn = sol_ct.reaction(i)
        if 'HCO' in rxn.equation:
            hco_rxns.append(i)
    
    print(f"\nFound {len(hco_rxns)} reactions involving HCO:")
    print("-" * 80)
    
    for i in hco_rxns:
        rxn = sol_ct.reaction(i)
        kf_rel = abs(kf_jt[i] - kf_ct[i]) / (kf_ct[i] + 1e-100) * 100
        kr_rel = abs(kr_jt[i] - kr_ct[i]) / (kr_ct[i] + 1e-100) * 100
        
        rop_f_ct = sol_ct.forward_rates_of_progress[i]
        rop_r_ct = sol_ct.reverse_rates_of_progress[i]
        rop_ct = rop_f_ct - rop_r_ct
        
        # Jantera ROP manually
        log_conc = jnp.log(jnp.maximum(conc, 1e-100))
        rop_f_jt = float(kf_jt[i] * jnp.exp(jnp.dot(mech.reactant_stoich[i], log_conc)))
        rop_r_jt = float(kr_jt[i] * jnp.exp(jnp.dot(mech.product_stoich[i], log_conc))) if mech.is_reversible[i] else 0.0
        rop_jt = rop_f_jt - rop_r_jt
        
        rop_rel = abs(rop_jt - rop_ct) / (abs(rop_ct) + 1e-100) * 100
        
        if rop_rel > 10:  # Only show significant discrepancies
            print(f"\nR{i}: {rxn.equation}")
            print(f"  Type: {rxn.reaction_type}")
            print(f"  kf: CT={kf_ct[i]:.4e}, JT={kf_jt[i]:.4e}, err={kf_rel:.2f}%")
            print(f"  kr: CT={kr_ct[i]:.4e}, JT={kr_jt[i]:.4e}, err={kr_rel:.2f}%")
            print(f"  ROP_f: CT={rop_f_ct:.4e}, JT={rop_f_jt:.4e}")
            print(f"  ROP_r: CT={rop_r_ct:.4e}, JT={rop_r_jt:.4e}")
            print(f"  ROP_net: CT={rop_ct:.4e}, JT={rop_jt:.4e}, err={rop_rel:.2f}%")

if __name__ == "__main__":
    compare_hco_reactions()
