"""
Diagnostic script for JP-10 nan gradients.
Check wdot gradient at t=0.
"""
import os
import sys
import numpy as np
import cantera as ct
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from jantera.loader import load_mechanism
from jantera.kinetics import compute_wdot

def check_jp10_nan():
    yaml_path = "jp10.yaml"
    T, P = 1500.0, 101325.0
    X = "C10H16:1, O2:14, N2:52.64"
    
    mech = load_mechanism(yaml_path)
    sol_ct = ct.Solution(yaml_path)
    sol_ct.TPX = T, P, X
    Y0 = jnp.array(sol_ct.Y)
    
    def get_wdot0(Y):
        wdot, _, _, _ = compute_wdot(T, P, Y, mech)
        return wdot[0] # Change in first species
    
    print("Calculating grad(wdot[0]) for JP-10...")
    grad = jax.grad(get_wdot0)(Y0)
    print(f"Gradient: {grad}")
    print(f"Any nan? {jnp.isnan(grad).any()}")
    if jnp.isnan(grad).any():
        nan_indices = jnp.where(jnp.isnan(grad))[0]
        print(f"Nan at indices: {nan_indices}")

if __name__ == "__main__":
    check_jp10_nan()
