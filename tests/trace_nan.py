"""
Use JAX's nan debugging to find exact source of JP-10 gradient nan.
"""
import os
import sys

# Enable NaN debugging BEFORE importing JAX
from jax import config
config.update("jax_debug_nans", True)
config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
import cantera as ct

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from canterax.loader import load_mechanism
from canterax.reactor import ReactorNet
import diffrax

def trace_nan():
    yaml_path = "jp10.yaml"
    T, P = 1500.0, 101325.0
    X = "C10H16:1, O2:14, N2:52.64"
    t_grad = 1e-8  # This is where nan first appears
    
    mech = load_mechanism(yaml_path)
    sol_ct = ct.Solution(yaml_path)
    sol_ct.TPX = T, P, X
    
    # Use non-zero baseline state
    Y_base = sol_ct.Y + 1e-4
    Y_base /= Y_base.sum()
    Y0 = jnp.array(Y_base)
    
    net = ReactorNet(mech)
    
    def get_final_T(y0):
        y_norm = y0 / jnp.sum(y0)
        res = net.advance(T, P, y_norm, t_grad, rtol=1e-10, atol=1e-14, solver=diffrax.Tsit5())
        return res.ys[-1, 0]
    
    print("Computing gradient with jax_debug_nans=True...")
    print("JAX will raise an exception at the first nan operation.")
    print()
    
    try:
        grad_T = jax.grad(get_final_T)(Y0)
        print("No nan detected! Gradient computed successfully.")
    except FloatingPointError as e:
        print(f"NaN detected! Error: {e}")

if __name__ == "__main__":
    trace_nan()
