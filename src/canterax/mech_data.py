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
    
    # Fixed-width sparse stoichiometry
    # MAX_REACTANTS/PRODUCTS usually 3 or 4
    max_reactants: int = eqx.field(static=True)
    max_products: int = eqx.field(static=True)
    
    # (n_reactions, max_reactants)
    # Indices are padded with n_species (dummy index)
    reactants_idx: jax.Array 
    reactants_nu: jax.Array
    
    # (n_reactions, max_products)
    products_idx: jax.Array
    products_nu: jax.Array
    
    # Legacy dense matrices (kept for backward compatibility or parallel paths)
    # TODO: Remove after full transition
    reactant_stoich: jax.Array      # (n_reactions, n_species)
    product_stoich: jax.Array       # (n_reactions, n_species)
    net_stoich: jax.Array           # (n_reactions, n_species)
    
    # Arrhenius parameters (converted to mol units)
    # k = A * T^b * exp(-Ea / (R * T))
    # Ea is in J/mol (loader was using kJ/mol, but handoff says J/mol. loader.py confirmed it was J/mol internally but labeled kJ/mol in comments)
    A: jax.Array                    # (n_reactions,)
    b: jax.Array                    # (n_reactions,)
    Ea: jax.Array                   # (n_reactions,) J/mol
    
    # Three-body enhancement
    is_three_body: jax.Array        # (n_reactions,) bool
    
    # Fixed-width sparse efficiencies
    max_efficiencies: int = eqx.field(static=True)
    efficiencies_idx: jax.Array      # (n_reactions, max_efficiencies)
    efficiencies_val: jax.Array      # (n_reactions, max_efficiencies)
    default_efficiency: jax.Array    # (n_reactions,)
    
    # Legacy dense efficiencies
    efficiencies: jax.Array         # (n_reactions, n_species)
    
    # Reversibility
    is_reversible: jax.Array        # (n_reactions,) bool
    
    # Falloff data
    is_falloff: jax.Array           # (n_reactions,) bool
    A_low: jax.Array                # (n_reactions,)
    b_low: jax.Array                # (n_reactions,)
    Ea_low: jax.Array               # (n_reactions,) J/mol
    
    # Troe parameters (F_cent calculation)
    # [alpha, T***, T*, T**]
    troe_params: jax.Array          # (n_reactions, 4)
    has_troe: jax.Array             # (n_reactions,) bool
    
    # Experimental sparse representations (BCOO)
    reactant_stoich_sparse: any = None
    product_stoich_sparse: any = None
    net_stoich_sparse: any = None
    efficiencies_sparse: any = None
