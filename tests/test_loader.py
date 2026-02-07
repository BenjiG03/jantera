import os
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from jantera.loader import load_mechanism
import cantera as ct
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

def test_loader_gri30():
    # Use the gri30.yaml from the cantera installation
    # On many systems, Cantera has a shortcut to find its data
    yaml_path = "gri30.yaml"
    
    print(f"Loading {yaml_path}...")
    mech = load_mechanism(yaml_path)
    
    sol = ct.Solution(yaml_path)
    
    print(f"Species: {mech.n_species} (Cantera: {sol.n_species})")
    print(f"Reactions: {mech.n_reactions} (Cantera: {sol.n_reactions})")
    
    # Check MW
    mw_cantera = sol.molecular_weights / 1000.0 # kg/kmol -> kg/mol
    mw_error = jnp.max(jnp.abs(mech.mol_weights - mw_cantera))
    print(f"Max MW error: {mw_error:.2e}")
    
    # Check a few species names
    print(f"First 5 species: {mech.species_names[:5]}")
    
    assert mech.n_species == sol.n_species
    assert mech.n_reactions == sol.n_reactions
    assert mw_error < 1e-10

if __name__ == "__main__":
    try:
        test_loader_gri30()
        print("Loader test passed!")
    except Exception as e:
        print(f"Loader test failed: {e}")
        import traceback
        traceback.print_exc()
