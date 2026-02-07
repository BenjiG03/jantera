import jax
import jax.numpy as jnp
from .constants import R_GAS, ONE_ATM
from .thermo import get_h_RT, get_s_R

@jax.jit
def compute_kf(T, conc, mech):
    """Compute forward rate constants for all reactions.
    
    T: float, temperature [K]
    conc: (n_species,) array, concentrations [mol/m^3]
    mech: MechData object
    """
    # 1. Standard Arrhenius: k = A * T^b * exp(-Ea / (R * T))
    # Ea is in J/mol
    log_kf = jnp.log(mech.A) + mech.b * jnp.log(T) - mech.Ea / (R_GAS * T)
    kf = jnp.exp(log_kf)
    
    # 2. Three-body Enhancement
    # [M]_j = sum_i (eff_ji * conc_i)
    m_eff = jnp.dot(mech.efficiencies, conc)
    # Applied to three-body and falloff reactions
    kf = jnp.where(mech.is_three_body, kf * m_eff, kf)
    
    # 3. Falloff (Troe/Lindemann)
    # k_inf = kf (already computed)
    # k_0 = A_low * T^b_low * exp(-Ea_low / RT)
    log_k0 = jnp.log(mech.A_low) + mech.b_low * jnp.log(T) - mech.Ea_low / (R_GAS * T)
    k0 = jnp.exp(log_k0)
    
    # Pr = k0 * [M] / k_inf
    # Note: kf already contains m_eff for falloff reactions (it was marked is_three_body)
    # So we need k_inf = kf_orig, and k0_eff = k0 * m_eff
    # Pr = (k0 * m_eff) / k_inf_standard
    
    # Let's re-calculate to be clearer and avoid double multiplication for falloff
    # kf currently is A*T^b*exp(-Ea/RT) * m_eff (if three-body)
    # For falloff: k_inf = A*T^b*exp(-Ea/RT)
    k_inf = jnp.maximum(jnp.exp(jnp.log(mech.A) + mech.b * jnp.log(T) - (mech.Ea) / (R_GAS * T)), 1e-100)
    Pr = (k0 * m_eff) / k_inf
    Pr = jnp.maximum(Pr, 1e-20)
    
    # Troe blending factor F
    alpha = mech.troe_params[:, 0]
    T3 = mech.troe_params[:, 1]
    T1 = mech.troe_params[:, 2]
    T2 = mech.troe_params[:, 3]
    
    f_cent = (1.0 - alpha) * jnp.exp(-T / jnp.maximum(T3, 1e-5)) + alpha * jnp.exp(-T / jnp.maximum(T1, 1e-5)) + jnp.exp(-T2 / jnp.maximum(T, 1e-5))
    f_cent = jnp.maximum(f_cent, 1e-20)
    
    log10_Pr = jnp.log10(Pr)
    log10_Fcent = jnp.log10(f_cent)
    C = -0.4 - 0.67 * log10_Fcent
    N = 0.75 - 1.27 * log10_Fcent
    
    denom = N - 0.14 * (log10_Pr + C)
    f_exponent = 1.0 / (1.0 + ((log10_Pr + C) / jnp.where(jnp.abs(denom) > 1e-5, denom, 1e-5))**2)
    F = jnp.power(10.0, log10_Fcent * f_exponent)
    
    # If not Troe (Lindemann), F = 1.0 (handled by troe_params being zero or default)
    
    kf_falloff = k_inf * (Pr / (1.0 + Pr)) * F
    
    # Update kf for falloff reactions
    kf = jnp.where(mech.is_falloff, kf_falloff, kf)
    
    return kf

@jax.jit
def compute_Kc(T, mech):
    """Compute equilibrium constants Kc for all reactions.
    
    Kc = exp(-DG/RT) * (P_atm / RT)^Dnu
    """
    h_RT = get_h_RT(T, mech.nasa_low, mech.nasa_high, mech.nasa_T_mid)
    s_R = get_s_R(T, mech.nasa_low, mech.nasa_high, mech.nasa_T_mid)
    
    # g_RT = H/RT - S/R (Gibbs free energy normalized by RT)
    g_RT = h_RT - s_R
    
    # Delta G / RT = sum(nu_i * g_i)
    # net_stoich: (n_reactions, n_species)
    dg_RT = jnp.dot(mech.net_stoich, g_RT)
    
    # Dnu = sum(nu_i)
    dnu = jnp.sum(mech.net_stoich, axis=1)
    
    # Kc = exp(-dg_RT) * (P_atm / (R * T))^dnu
    kc = jnp.exp(-dg_RT) * (ONE_ATM / (R_GAS * T))**dnu
    
    return kc

@jax.jit
def compute_wdot(T, P, Y, mech):
    """Compute net production rates for all species.
    
    Returns: (wdot, h_mass, cp_mass, rho)
    """
    # 1. Mixture properties
    from .thermo import compute_mixture_props
    cp_mass, h_mass, rho = compute_mixture_props(T, P, Y, mech)
    
    # 2. Concentrations [mol/m^3]
    # [C]_i = rho * Y_i / MW_i
    Y_eff = jnp.maximum(Y, 1e-20)
    conc = jnp.maximum(rho, 1e-10) * Y_eff / mech.mol_weights
    conc = jnp.maximum(conc, 1e-20)
    
    # 3. Forward and reverse rate constants
    kf = compute_kf(T, conc, mech)
    kc = compute_Kc(T, mech)
    kr = kf / (kc + 1e-100)
    
    # 4. Rates of progress for each reaction
    # q = kf * prod(conc^nu_reac) - kr * prod(conc^nu_prod)
    
    # Use jnp.power for better AD stability than exp(dot(stoich, log(conc)))
    f_rop = kf * jnp.prod(jnp.power(conc, mech.reactant_stoich), axis=1)
    r_rop = kr * jnp.prod(jnp.power(conc, mech.product_stoich), axis=1)
    
    rop = f_rop - r_rop
    
    # 5. Net production rates [mol/m^3/s]
    # wdot_i = sum_j (nu_ij * q_j)
    wdot = jnp.dot(mech.net_stoich.T, rop)
    
    return wdot, h_mass, cp_mass, rho
