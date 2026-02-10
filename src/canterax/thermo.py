import jax
import jax.numpy as jnp
from .constants import R_GAS

@jax.jit
def get_cp_R(T, nasa_low, nasa_high, T_mid):
    """Compute non-dimensional heat capacity Cp/R for all species.
    
    T: float or array of temperatures
    nasa_low: (n_species, 7) array of coefficients for low temperature range
    nasa_high: (n_species, 7) array of coefficients for high temperature range
    T_mid: (n_species,) array of temperature boundaries
    """
    # nasa_high/low[:, 0:5] are the coefficients for Cp/R
    # Cp/R = a0 + a1*T + a2*T^2 + a3*T^3 + a4*T^4
    
    # Broadcast T to (n_species,) if it's a scalar
    T = jnp.atleast_1d(T)
    
    # Polynomial evaluation
    # mask: True if T > T_mid
    mask = T > T_mid
    
    coeffs = jnp.where(mask[:, None], nasa_high, nasa_low)
    
    cp_R = (coeffs[:, 0] + 
            coeffs[:, 1] * T + 
            coeffs[:, 2] * T**2 + 
            coeffs[:, 3] * T**3 + 
            coeffs[:, 4] * T**4)
    
    return cp_R

@jax.jit
def get_h_RT(T, nasa_low, nasa_high, T_mid):
    """Compute non-dimensional enthalpy H/RT for all species.
    
    H/RT = a0 + a1*T/2 + a2*T^2/3 + a3*T^3/4 + a4*T^4/5 + a5/T
    """
    T = jnp.atleast_1d(T)
    mask = T > T_mid
    coeffs = jnp.where(mask[:, None], nasa_high, nasa_low)
    
    h_RT = (coeffs[:, 0] + 
            coeffs[:, 1] * T / 2.0 + 
            coeffs[:, 2] * T**2 / 3.0 + 
            coeffs[:, 3] * T**3 / 4.0 + 
            coeffs[:, 4] * T**4 / 5.0 + 
            coeffs[:, 5] / T)
    
    return h_RT

@jax.jit
def get_s_R(T, nasa_low, nasa_high, T_mid):
    """Compute non-dimensional entropy S/R for all species.
    
    S/R = a0*ln(T) + a1*T + a2*T^2/2 + a3*T^3/3 + a4*T^4/4 + a6
    """
    T = jnp.atleast_1d(T)
    mask = T > T_mid
    coeffs = jnp.where(mask[:, None], nasa_high, nasa_low)
    
    s_R = (coeffs[:, 0] * jnp.log(T) + 
           coeffs[:, 1] * T + 
           coeffs[:, 2] * T**2 / 2.0 + 
           coeffs[:, 3] * T**3 / 3.0 + 
           coeffs[:, 4] * T**4 / 4.0 + 
           coeffs[:, 6])
    
    return s_R

@jax.jit
def compute_mixture_props(T, P, Y, mech):
    """Compute mixture thermodynamic properties.
    
    Returns:
        wdot_null: (n_species,) zeros (placeholder for fusion later)
        h_mass: mixture mass-weighted enthalpy (J/kg)
        cp_mass: mixture mass-weighted heat capacity (J/kg/K)
        rho: mixture density (kg/m^3)
    """
    # 1. Species non-dimensional properties
    cp_R = get_cp_R(T, mech.nasa_low, mech.nasa_high, mech.nasa_T_mid)
    h_RT = get_h_RT(T, mech.nasa_low, mech.nasa_high, mech.nasa_T_mid)
    
    # 2. Convert to dimensional (per mol)
    # Cp_mol: [J/mol/K], H_mol: [J/mol]
    cp_mol = cp_R * R_GAS
    h_mol = h_RT * R_GAS * T
    
    # 3. Mixture molecular weight
    # 1/MW_mix = sum(Y_i / MW_i)
    inv_mw_mix = jnp.sum(Y / mech.mol_weights)
    mw_mix = 1.0 / inv_mw_mix
    
    # 4. Mixture properties (mass-weighted)
    # cp_mass = sum(Y_i * cp_mol_i / MW_i)
    cp_mass = jnp.sum(Y * cp_mol / mech.mol_weights)
    h_mass = jnp.sum(Y * h_mol / mech.mol_weights)
    
    # 5. Density
    # P = rho * R_mix * T = rho * (R_gas / mw_mix) * T
    # rho = P * mw_mix / (R_gas * T)
    rho = P * mw_mix / (R_GAS * T)
    
    return cp_mass, h_mass, rho, h_mol
