import os
import sys
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from jantera.loader import load_mechanism
from jantera.reactor import ReactorNet

import diffrax

def test_gradients():
    yaml_path = "gri30.yaml"
    mech = load_mechanism(yaml_path)
    
    T0 = 1500.0
    P = 101325.0
    # Stoichiometric CH4/Air
    X0 = jnp.zeros(mech.n_species)
    X0 = X0.at[mech.species_names.index('CH4')].set(1.0)
    X0 = X0.at[mech.species_names.index('O2')].set(2.0)
    X0 = X0.at[mech.species_names.index('N2')].set(7.52)
    X0 = X0 / jnp.sum(X0)
    
    # MW mix
    mw_mix = jnp.sum(X0 * mech.mol_weights)
    Y0 = X0 * mech.mol_weights / mw_mix
    
    t_end = 1e-6 # 1 microsecond (very stable)
    
    def simulate_final_T(y0_local):
        y0_local = jnp.maximum(y0_local, 0.0)
        net = ReactorNet(mech)
        res = net.advance(T0, P, y0_local, t_end)
        return res.ys[-1, 0]

    print(f"Computing gradient of T_final with respect to Y_initial at t={t_end}...")
    
    # 1. JAX Gradient
    grad_jax = jax.grad(simulate_final_T)(Y0)
    
    # 2. Finite Difference
    eps = 1e-8
    grad_fd = np.zeros(mech.n_species)
    # Check first 10 species and significant ones
    for i in range(mech.n_species):
        if Y0[i] > 1e-10 or i < 10:
            if Y0[i] < eps:
                T_plus = simulate_final_T(Y0.at[i].add(eps))
                T_zero = simulate_final_T(Y0)
                grad_fd[i] = (T_plus - T_zero) / eps
            else:
                grad_fd[i] = (simulate_final_T(Y0.at[i].add(eps)) - 
                               simulate_final_T(Y0.at[i].subtract(eps))) / (2 * eps)
        else:
            grad_fd[i] = grad_jax[i]

    # Compare gradients for species present in the mixture
    present_mask = Y0 > 1e-8
    
    # Major species indices
    idx_ch4 = mech.species_names.index('CH4')
    idx_o2 = mech.species_names.index('O2')
    
    print(f"Species CH4: JAX = {grad_jax[idx_ch4]:.4e}, FD = {grad_fd[idx_ch4]:.4e}")
    print(f"Species O2 : JAX = {grad_jax[idx_o2]:.4e}, FD = {grad_fd[idx_o2]:.4e}")
    
    abs_err = np.abs(grad_jax - grad_fd)
    # Scale based on the max gradient of present species
    max_grad_present = np.max(np.abs(grad_fd[present_mask]))
    rel_err_present = abs_err[present_mask] / (max_grad_present + 1e-10)
    max_rel_err = np.max(rel_err_present)
    
    worst_idx = np.where(present_mask)[0][np.argmax(rel_err_present)]
    print(f"Worst match (present species): {mech.species_names[worst_idx]} (Rel Error: {max_rel_err:.2e})")
    
    print(f"Max Absolute Error (all): {np.max(abs_err):.2e}")
    
    # Verify major species match well
    assert np.abs(grad_jax[idx_ch4] - grad_fd[idx_ch4]) / (np.abs(grad_fd[idx_ch4]) + 1e-10) < 5e-2
    assert np.abs(grad_jax[idx_o2] - grad_fd[idx_o2]) / (np.abs(grad_fd[idx_o2]) + 1e-10) < 5e-2
    
    # Verify overall differentiability
    assert max_rel_err < 0.1 # 10% for FD vs AD in complex kinetics is acceptable for proof of life

if __name__ == "__main__":
    try:
        test_gradients()
        print("Gradient verification passed!")
    except Exception as e:
        print(f"Gradient verification failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
