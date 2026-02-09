import jax
import jax.numpy as jnp
from .constants import R_GAS, ONE_ATM
from .thermo import get_h_RT, get_s_R

@jax.jit(static_argnums=(3,))
def compute_kf(T, conc, mech, use_experimental_sparse=False):
    """Compute forward rate constants for all reactions.
    
    T: float, temperature [K]
    conc: (n_species,) array, concentrations [mol/m^3]
    mech: MechData object
    """
    # 1. Standard Arrhenius: k = A * T^b * exp(-Ea / (R * T))
    log_kf = jnp.log(mech.A) + mech.b * jnp.log(T) - mech.Ea / (R_GAS * T)
    kf = jnp.exp(log_kf)
    
    # 2. Three-body Enhancement
    safe_conc = jnp.maximum(conc, 0.0)
    
    if use_experimental_sparse:
        # Experimental Sparse Path (BCOO)
        m_eff = mech.efficiencies_sparse @ safe_conc
    else:
        # Optimized "dense-sparse" approach
        sum_conc = jnp.sum(safe_conc)
        padded_conc = jnp.concatenate([safe_conc, jnp.array([0.0])])
        eff_conc = padded_conc[mech.efficiencies_idx]
        eff_diff = mech.efficiencies_val - mech.default_efficiency[:, None]
        m_eff = mech.default_efficiency * sum_conc + jnp.sum(eff_diff * eff_conc, axis=1)
        
    kf = jnp.where(mech.is_three_body, kf * m_eff, kf)
    
    # 3. Falloff (Troe/Lindemann)
    # k_inf already computed as kf_orig (before 3-body)
    k_inf = jnp.maximum(jnp.exp(log_kf), 1e-100)
    
    # k0_eff = k0 * m_eff
    log_k0 = jnp.log(mech.A_low) + mech.b_low * jnp.log(T) - mech.Ea_low / (R_GAS * T)
    k0_eff = jnp.exp(log_k0) * m_eff
    
    Pr = k0_eff / k_inf
    Pr = jnp.maximum(Pr, 1e-20)
    
    # Troe blending factor F
    alpha = mech.troe_params[:, 0]
    T3 = mech.troe_params[:, 1]
    T1 = mech.troe_params[:, 2]
    T2 = mech.troe_params[:, 3]
    
    # Mask inputs to avoid nan in non-falloff reactions
    safe_Pr = jnp.where(mech.is_falloff, Pr, 1.0)
    
    f_cent = (1.0 - alpha) * jnp.exp(-T / jnp.maximum(T3, 1e-5)) + alpha * jnp.exp(-T / jnp.maximum(T1, 1e-5)) + jnp.exp(-T2 / jnp.maximum(T, 1e-5))
    safe_f_cent = jnp.where(mech.is_falloff, f_cent, 1.0)
    
    log10_Pr = jnp.log10(safe_Pr + 1e-100)
    log10_Fcent = jnp.log10(safe_f_cent + 1e-100)
    C = -0.4 - 0.67 * log10_Fcent
    N = 0.75 - 1.27 * log10_Fcent
    
    denom = N - 0.14 * (log10_Pr + C)
    safe_denom = jnp.where(jnp.abs(denom) > 1e-10, denom, 1e-10)
    f_exponent = 1.0 / (1.0 + ((log10_Pr + C) / safe_denom)**2)
    F = jnp.power(10.0, log10_Fcent * f_exponent)
    
    kf_falloff = k_inf * (safe_Pr / (1.0 + safe_Pr)) * F
    kf = jnp.where(mech.is_falloff, kf_falloff, kf)
    
    return kf

@jax.jit
def compute_Kc(T, mech):
    """Compute equilibrium constants Kc for all reactions."""
    h_RT = get_h_RT(T, mech.nasa_low, mech.nasa_high, mech.nasa_T_mid)
    s_R = get_s_R(T, mech.nasa_low, mech.nasa_high, mech.nasa_T_mid)
    g_RT = h_RT - s_R
    
    # Pad g_RT with 0 for dummy indices
    padded_g_RT = jnp.concatenate([g_RT, jnp.array([0.0])])
    
    # Delta G / RT = sum(nu_prod * g_prod) - sum(nu_reac * g_reac)
    dg_RT_reac = jnp.sum(mech.reactants_nu * padded_g_RT[mech.reactants_idx], axis=1)
    dg_RT_prod = jnp.sum(mech.products_nu * padded_g_RT[mech.products_idx], axis=1)
    dg_RT = dg_RT_prod - dg_RT_reac
    
    # Dnu = sum(nu_prod) - sum(nu_reac)
    dnu = jnp.sum(mech.products_nu, axis=1) - jnp.sum(mech.reactants_nu, axis=1)
    
    kc = jnp.exp(-dg_RT) * (ONE_ATM / (R_GAS * T))**dnu
    return kc

@jax.jit(static_argnums=(4,))
def compute_wdot(T, P, Y, mech, use_experimental_sparse=False):
    """Compute net production rates for all species.
    
    Returns: (wdot, h_mass, cp_mass, rho)
    """
    # 1. Mixture properties
    from .thermo import compute_mixture_props
    cp_mass, h_mass, rho = compute_mixture_props(T, P, Y, mech)
    
    # 2. Concentrations [mol/m^3]
    conc = rho * Y / mech.mol_weights
    
    # 3. Forward and reverse rate constants
    kf = compute_kf(T, conc, mech, use_experimental_sparse)
    kc = compute_Kc(T, mech)
    kr = kf / (kc + 1e-100)
    
    # 4. Rates of progress
    safe_conc = jnp.maximum(conc, 1e-30)
    
    if use_experimental_sparse:
        # Experimental Sparse Path for ROP
        # f_rop = kf * exp(sum(nu * log(conc)))
        log_conc = jnp.log(safe_conc)
        f_rop = kf * jnp.exp(mech.reactant_stoich_sparse @ log_conc)
        r_rop = kr * jnp.exp(mech.product_stoich_sparse @ log_conc)
    else:
        padded_conc = jnp.concatenate([safe_conc, jnp.array([1.0])]) # Pad with 1.0 for products in power
        reac_conc = padded_conc[mech.reactants_idx]
        f_rop = kf * jnp.prod(jnp.power(reac_conc, mech.reactants_nu), axis=1)
        prod_conc = padded_conc[mech.products_idx]
        r_rop = kr * jnp.prod(jnp.power(prod_conc, mech.products_nu), axis=1)
    
    r_rop = jnp.where(mech.is_reversible, r_rop, 0.0)
    rop = f_rop - r_rop
    
    # 5. Net production rates [mol/m^3/s]
    if use_experimental_sparse:
        wdot = rop @ mech.net_stoich_sparse
    else:
        # Optimized gathering: subtract reactants, add products
        # We need to map rop to species.
        # wdot = sum(rop * products_nu) - sum(rop * reactants_nu)
        
        # Flattened indices and values for scatter_add
        def scatter_stoich(idx, nu, rop_vals, n_spec):
            # idx: (n_rxn, max_nu)
            # nu: (n_rxn, max_nu)
            # rop_vals: (n_rxn,)
            flat_idx = idx.reshape(-1)
            flat_val = (rop_vals[:, None] * nu).reshape(-1)
            # use .at[].add() which is vectorized scatter_add
            return jnp.zeros(n_spec + 1).at[flat_idx].add(flat_val)[:-1]

        wdot_prod = scatter_stoich(mech.products_idx, mech.products_nu, rop, mech.n_species)
        wdot_reac = scatter_stoich(mech.reactants_idx, mech.reactants_nu, rop, mech.n_species)
        wdot = wdot_prod - wdot_reac
        
    return wdot, h_mass, cp_mass, rho
