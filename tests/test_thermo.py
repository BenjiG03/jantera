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

from canterax.loader import load_mechanism
from canterax.thermo import compute_mixture_props

def test_thermo():
    yaml_path = "gri30.yaml"
    mech = load_mechanism(yaml_path)
    sol = ct.Solution(yaml_path)
    
    n_points = 100
    np.random.seed(42)
    
    T_samples = np.random.uniform(300, 3000, n_points)
    P_samples = np.random.uniform(0.1, 50, n_points) * 101325.0
    
    # Random mole fractions
    X_samples = np.random.dirichlet(np.ones(mech.n_species), n_points)
    
    cp_canterax = []
    h_canterax = []
    rho_canterax = []
    
    cp_cantera = []
    h_cantera = []
    rho_cantera = []
    
    print(f"Running {n_points} validation points...")
    
    for i in range(n_points):
        T, P, X = T_samples[i], P_samples[i], X_samples[i]
        
        # Cantera
        sol.TPX = T, P, X
        cp_cantera.append(sol.cp_mass)
        h_cantera.append(sol.enthalpy_mass)
        rho_cantera.append(sol.density)
        
        # Canterax
        Y = jnp.array(sol.Y)
        cp, h, rho = compute_mixture_props(T, P, Y, mech)
        cp_canterax.append(float(cp))
        h_canterax.append(float(h))
        rho_canterax.append(float(rho))
        
    cp_canterax = np.array(cp_canterax)
    h_canterax = np.array(h_canterax)
    rho_canterax = np.array(rho_canterax)
    
    cp_cantera = np.array(cp_cantera)
    h_cantera = np.array(h_cantera)
    rho_cantera = np.array(rho_cantera)
    
    # Calculate errors
    def get_max_rel_error(val, ref):
        # Avoid division by zero, though h can be zero.
        # For enthalpy, use absolute error relative to a reference scale if needed, 
        # but here we'll just use max relative error for cp and rho.
        abs_err = np.abs(val - ref)
        rel_err = abs_err / (np.abs(ref) + 1e-15)
        return np.max(rel_err)

    cp_err = get_max_rel_error(cp_canterax, cp_cantera)
    h_err = np.max(np.abs(h_canterax - h_cantera)) / (np.max(np.abs(h_cantera)) + 1e-15)
    rho_err = get_max_rel_error(rho_canterax, rho_cantera)
    
    print(f"Max relative errors:")
    print(f"  Cp_mass: {cp_err:.2e}")
    print(f"  Enthalpy: {h_err:.2e}")
    print(f"  Density: {rho_err:.2e}")
    
    # Plotting
    os.makedirs("tests/outputs", exist_ok=True)
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.scatter(cp_cantera, cp_canterax, alpha=0.5)
    plt.plot([min(cp_cantera), max(cp_cantera)], [min(cp_cantera), max(cp_cantera)], 'r--')
    plt.title("Cp_mass [J/kg/K]")
    plt.xlabel("Cantera")
    plt.ylabel("Canterax")
    
    plt.subplot(1, 3, 2)
    plt.scatter(h_cantera, h_canterax, alpha=0.5)
    plt.plot([min(h_cantera), max(h_cantera)], [min(h_cantera), max(h_cantera)], 'r--')
    plt.title("Enthalpy [J/kg]")
    plt.xlabel("Cantera")
    plt.ylabel("Canterax")
    
    plt.subplot(1, 3, 3)
    plt.scatter(rho_cantera, rho_canterax, alpha=0.5)
    plt.plot([min(rho_cantera), max(rho_cantera)], [min(rho_cantera), max(rho_cantera)], 'r--')
    plt.title("Density [kg/m3]")
    plt.xlabel("Cantera")
    plt.ylabel("Canterax")
    
    plt.tight_layout()
    plt.savefig("tests/outputs/thermo_validation.png")
    print("Verification plot saved to tests/outputs/thermo_validation.png")
    
    assert cp_err < 1e-10
    assert h_err < 1e-10
    assert rho_err < 1e-10

if __name__ == "__main__":
    try:
        test_thermo()
        print("Thermodynamics validation passed!")
    except Exception as e:
        print(f"Thermodynamics validation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
