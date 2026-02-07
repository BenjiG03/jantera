import equinox as eqx
import jax
import jax.numpy as jnp

class MechData(eqx.Module):
    """Frozen Equinox module holding all static mechanism data."""
    
    # Species info
    n_species: int = eqx.field(static=True)
    species_names: tuple = eqx.field(static=True)
    mol_weights: jax.Array          # (n_species,) kg/mol
    
    # Element info
    n_elements: int = eqx.field(static=True)
    element_names: tuple = eqx.field(static=True)
    element_matrix: jax.Array       # (n_elements, n_species)
    
    # NASA-7 coefficients
    # Format: (n_species, 7)
    nasa_low: jax.Array
    nasa_high: jax.Array
    nasa_T_mid: jax.Array           # (n_species,)
    nasa_T_low: jax.Array           # (n_species,)
    nasa_T_high: jax.Array          # (n_species,)
    
    # Reaction info
    n_reactions: int = eqx.field(static=True)
    reactant_stoich: jax.Array      # (n_reactions, n_species)
    product_stoich: jax.Array       # (n_reactions, n_species)
    net_stoich: jax.Array           # (n_reactions, n_species)
    
    # Arrhenius parameters (converted to mol units)
    # k = A * T^b * exp(-Ea / (R * T))
    # Ea is in kJ/mol
    A: jax.Array                    # (n_reactions,)
    b: jax.Array                    # (n_reactions,)
    Ea: jax.Array                   # (n_reactions,) kJ/mol
    
    # Three-body enhancement
    is_three_body: jax.Array        # (n_reactions,) bool
    efficiencies: jax.Array         # (n_reactions, n_species)
    
    # Falloff data
    is_falloff: jax.Array           # (n_reactions,) bool
    A_low: jax.Array                # (n_reactions,)
    b_low: jax.Array                # (n_reactions,)
    Ea_low: jax.Array               # (n_reactions,) kJ/mol
    
    # Troe parameters (F_cent calculation)
    # [alpha, T***, T*, T**]
    troe_params: jax.Array          # (n_reactions, 4)
    has_troe: jax.Array             # (n_reactions,) bool
