import jax
import jax.numpy as jnp
import numpy as np
import os

def test_precision():
    print(f"JAX x64 enabled: {jax.config.read('jax_enable_x64')}")
    
    kf = 1e15
    conc = 0.0
    
    # Current implementation logic
    def calc_wdot(floor):
        safe_conc = jnp.maximum(conc, floor)
        # Assuming nu=1 for simplicity
        rop = kf * jnp.exp(jnp.log(safe_conc))
        return rop

    print(f"Floor 1e-30: {calc_wdot(1e-30)}")
    print(f"Floor 1e-100: {calc_wdot(1e-100)}")
    print(f"Floor 0.0: {calc_wdot(0.0)}")

    # Check if XLA is doing anything funny with exp(log(x))
    @jax.jit
    def jit_wdot(floor):
        return calc_wdot(floor)
        
    print(f"JIT Floor 1e-30: {jit_wdot(1e-30)}")
    print(f"JIT Floor 0.0: {jit_wdot(0.0)}")

if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    test_precision()
