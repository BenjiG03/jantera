import yaml
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import cantera as ct
from .mech_data import MechData

def load_mechanism(yaml_file: str) -> MechData:
    """Load a Cantera YAML mechanism and return a MechData object.
    
    This function uses Cantera to parse the YAML and then extracts the 
    relevant data into JAX arrays, performing unit conversions to 
    mol-based units. Compatible with Cantera 3.0+.
    """
    sol = ct.Solution(yaml_file)
    n_species = sol.n_species
    n_reactions = sol.n_reactions
    
    # 1. Species Data
    species_names = tuple(sol.species_names)
    # MW: kg/kmol -> kg/mol
    mol_weights = jnp.array(sol.molecular_weights) / 1000.0
    
    # Element Data
    element_names = tuple(sol.element_names)
    n_elements = len(element_names)
    element_matrix = np.zeros((n_elements, n_species))
    for i, spec in enumerate(sol.species()):
        for el, count in spec.composition.items():
            element_matrix[sol.element_index(el), i] = count
    
    # NASA-7 Coefficients
    nasa_low = np.zeros((n_species, 7))
    nasa_high = np.zeros((n_species, 7))
    nasa_T_mid = np.zeros(n_species)
    nasa_T_low = np.zeros(n_species)
    nasa_T_high = np.zeros(n_species)
    
    for i, spec in enumerate(sol.species()):
        poly = spec.thermo
        if not isinstance(poly, ct.NasaPoly2):
            raise ValueError(f"Species {spec.name} does not use NASA-7 (NasaPoly2) format.")
        
        coeffs = poly.coeffs
        nasa_T_mid[i] = coeffs[0]
        nasa_high[i] = coeffs[1:8]
        nasa_low[i] = coeffs[8:15]
        
        nasa_T_low[i] = poly.min_temp
        nasa_T_high[i] = poly.max_temp

    # 2. Reaction Data
    reactant_stoich = np.zeros((n_reactions, n_species))
    product_stoich = np.zeros((n_reactions, n_species))
    
    A = np.zeros(n_reactions)
    b = np.zeros(n_reactions)
    Ea = np.zeros(n_reactions) # kJ/mol
    
    is_three_body = np.zeros(n_reactions, dtype=bool)
    efficiencies = np.ones((n_reactions, n_species))
    
    is_falloff = np.zeros(n_reactions, dtype=bool)
    A_low = np.zeros(n_reactions)
    b_low = np.zeros(n_reactions)
    Ea_low = np.zeros(n_reactions) # kJ/mol
    
    troe_params = np.zeros((n_reactions, 4))
    has_troe = np.zeros(n_reactions, dtype=bool)
    
    for i, rxn in enumerate(sol.reactions()):
        # Stoichiometry
        for sp, coeff in rxn.reactants.items():
            reactant_stoich[i, sol.species_index(sp)] = coeff
        for sp, coeff in rxn.products.items():
            product_stoich[i, sol.species_index(sp)] = coeff
            
        stoich_order = sum(rxn.reactants.values())
        
        # Reaction types and parameters
        r_type = rxn.reaction_type
        rate = rxn.rate
        
        if 'three-body' in r_type:
            is_three_body[i] = True
            if rxn.third_body:
                eff_map = rxn.third_body.efficiencies
                for sp_name, eff in eff_map.items():
                    efficiencies[i, sol.species_index(sp_name)] = eff
            
            # Unit conversion: A is m^3/(kmol*s) for 2 reactants + 1 third body
            # effective order = stoich_order + 1
            n_eff = stoich_order + 1
            A[i] = rate.pre_exponential_factor * (1000.0**(1.0 - n_eff))
            b[i] = rate.temperature_exponent
            Ea[i] = rate.activation_energy / 1000.0 # J/kmol -> J/mol
            
        elif 'falloff' in r_type:
            is_falloff[i] = True
            is_three_body[i] = True # Falloff is implicitly a 3-body collision
            
            if rxn.third_body:
                eff_map = rxn.third_body.efficiencies
                for sp_name, eff in eff_map.items():
                    efficiencies[i, sol.species_index(sp_name)] = eff
            
            # High-pressure limit
            # effective order = stoich_order
            A[i] = rate.high_rate.pre_exponential_factor * (1000.0**(1.0 - stoich_order))
            b[i] = rate.high_rate.temperature_exponent
            Ea[i] = rate.high_rate.activation_energy / 1000.0
            
            # Low-pressure limit
            # effective order = stoich_order + 1
            n_eff_low = stoich_order + 1
            A_low[i] = rate.low_rate.pre_exponential_factor * (1000.0**(1.0 - n_eff_low))
            b_low[i] = rate.low_rate.temperature_exponent
            Ea_low[i] = rate.low_rate.activation_energy / 1000.0
            
            if hasattr(rate, 'falloff_coeffs') and len(rate.falloff_coeffs) > 0:
                has_troe[i] = True
                tp = rate.falloff_coeffs
                # Troe can be 3 or 4 parameters
                troe_params[i, :len(tp)] = tp
                if len(tp) == 3:
                    troe_params[i, 3] = 1e30 # Large T** effectively disables it
            else:
                # Lindemann: F_cent = 1.0
                # We can simulate this with Troe [1.0, 1e30, 1e30, 1e30]
                has_troe[i] = True
                troe_params[i] = [1.0, 1e30, 1e30, 1e30]
                
        else: # Elementary reaction
            A[i] = rate.pre_exponential_factor * (1000.0**(1.0 - stoich_order))
            b[i] = rate.temperature_exponent
            Ea[i] = rate.activation_energy / 1000.0
            
    net_stoich = product_stoich - reactant_stoich
    
    return MechData(
        n_species=n_species,
        species_names=species_names,
        mol_weights=jnp.array(mol_weights),
        n_elements=n_elements,
        element_names=element_names,
        element_matrix=jnp.array(element_matrix),
        nasa_low=jnp.array(nasa_low),
        nasa_high=jnp.array(nasa_high),
        nasa_T_mid=jnp.array(nasa_T_mid),
        nasa_T_low=jnp.array(nasa_T_low),
        nasa_T_high=jnp.array(nasa_T_high),
        n_reactions=n_reactions,
        reactant_stoich=jnp.array(reactant_stoich),
        product_stoich=jnp.array(product_stoich),
        net_stoich=jnp.array(net_stoich),
        A=jnp.array(A),
        b=jnp.array(b),
        Ea=jnp.array(Ea),
        is_three_body=jnp.array(is_three_body),
        efficiencies=jnp.array(efficiencies),
        is_falloff=jnp.array(is_falloff),
        A_low=jnp.array(A_low),
        b_low=jnp.array(b_low),
        Ea_low=jnp.array(Ea_low),
        troe_params=jnp.array(troe_params),
        has_troe=jnp.array(has_troe)
    )
