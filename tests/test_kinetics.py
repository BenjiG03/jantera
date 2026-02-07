import os
import sys
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import cantera as ct
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from jantera.loader import load_mechanism
from jantera.kinetics import compute_wdot

def test_kinetics():
    yaml_path = "gri30.yaml"
    mech = load_mechanism(yaml_path)
    sol = ct.Solution(yaml_path)
    
    # Test conditions: Stoichiometric CH4/air at 1 atm, various temperatures
    P = 101325.0
    temperatures = [800, 1000, 1200, 1500, 2000]
    
    print(f"Comparing net_production_rates at {len(temperatures)} temperatures...")
    
    all_errs = []
    
    for T in temperatures:
        sol.TPX = T, P, 'CH4:1.0, O2:2.0, N2:7.52'
        
        # Cantera (kmol-based)
        wdot_cantera = sol.net_production_rates * 1000.0 # kmol -> mol
        kf_cantera_raw = sol.forward_rate_constants
        kc_cantera_raw = sol.equilibrium_constants
        
        # Jantera (mol-based)
        Y = jnp.array(sol.Y)
        conc = jnp.array(sol.density * sol.Y / (sol.molecular_weights / 1000.0))
        
        from jantera.kinetics import compute_kf, compute_Kc
        kf_jantera = compute_kf(T, conc, mech)
        kc_jantera = compute_Kc(T, mech)
        
        print(f"\n--- Debug T={T}K ---")
        for j in range(3):
            rxn = sol.reaction(j)
            print(f"Rxn {j}: {rxn}")
            # Scaling for comparison
            stoich_order = sum(rxn.reactants.values())
            if hasattr(rxn, 'reaction_type') and 'three-body' in rxn.reaction_type:
                n_eff = stoich_order + 1
            else:
                n_eff = stoich_order
            
            # k_jan = k_can * 1000^(1-n_eff)
            kf_can_scaled = kf_cantera_raw[j] * (1000.0**(1.0 - n_eff))
            
            dnu = sum(rxn.products.values()) - sum(rxn.reactants.values())
            # Kc_jan = Kc_can * 1000^(-dnu)
            kc_can_scaled = kc_cantera_raw[j] * (1000.0**(-dnu))
            
            print(f"  kf: Cantera(scaled)={kf_can_scaled:.3e}, Jantera={kf_jantera[j]:.3e}")
            print(f"  Kc: Cantera(scaled)={kc_can_scaled:.3e}, Jantera={kc_jantera[j]:.3e}")
        
        wdot_jantera, _, _, _ = compute_wdot(T, P, Y, mech)
        wdot_jantera = np.array(wdot_jantera)
        
        # Compare
        # Use abs error normalized by max rate if rate is large,
        # otherwise absolute error.
        max_rate = np.max(np.abs(wdot_cantera))
        abs_err = np.abs(wdot_jantera - wdot_cantera)
        rel_err = abs_err / (max_rate + 1e-20)
        
        max_rel_err = np.max(rel_err)
        all_errs.append(max_rel_err)
        
        print(f"  T={T}K: Max rel error = {max_rel_err:.2e}")
        
        # Plotting for this T
        plt.figure(figsize=(10, 6))
        plt.bar(np.arange(mech.n_species), wdot_cantera, alpha=0.5, label='Cantera')
        plt.bar(np.arange(mech.n_species), wdot_jantera, alpha=0.5, label='Jantera')
        plt.title(f"Net Production Rates at {T}K")
        plt.xlabel("Species Index")
        plt.ylabel("wdot [mol/m3/s]")
        plt.legend()
        plt.yscale('symlog', linthresh=1e-10)
        plt.savefig(f"tests/outputs/kinetics_T{T}.png")
        plt.close()

    max_overall_err = max(all_errs)
    print(f"Max overall relative error: {max_overall_err:.2e}")
    
    assert max_overall_err < 1e-6

if __name__ == "__main__":
    try:
        test_kinetics()
        print("Kinetics validation passed!")
    except Exception as e:
        print(f"Kinetics validation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
