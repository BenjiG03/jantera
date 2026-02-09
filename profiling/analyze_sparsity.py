import jax
import jax.numpy as jnp
from jax.experimental import sparse
import cantera as ct
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from jantera.loader import load_mechanism
from jantera.reactor import reactor_rhs

def analyze_sparsity():
    # Use built-in gri30 or local jp10
    yaml_path = "gri30.yaml"
        
    mech = load_mechanism(yaml_path)
    n = len(mech.species_names)
    print(f"Mechanism: {yaml_path}, Species: {n}")
    
    T, P = 1500.0, 101325.0
    Y = jnp.ones(n) / n
    state = jnp.concatenate([jnp.array([T]), Y])
    args = (P, mech)
    
    # Dense Jacobian
    jac_dense = jax.jacfwd(lambda s: reactor_rhs(0.0, s, args))(state)
    
    # Analyze sparsity
    nnz = jnp.count_nonzero(jnp.abs(jac_dense) > 1e-20)
    total = jac_dense.size
    density = nnz / total
    
    print(f"NNZ: {nnz}")
    print(f"Total: {total}")
    print(f"Density: {density:.2%}")
    
    # Check sparsity pattern (how many species interact with how many others)
    # Most chemical kinetics are very sparse (5-10 participants per species)
    
if __name__ == "__main__":
    analyze_sparsity()
