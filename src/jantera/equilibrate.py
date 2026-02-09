import jax
import jax.numpy as jnp
import optimistix as optx
from .thermo import get_h_RT, get_s_R
from .constants import R_GAS, ONE_ATM

def equilibrate(sol, mode='TP', rtol=1e-10):
    """Perform equilibrium calculation using KKT equations (Element Potentials).
    
    Compatible with JAX. Handles missing elements by filtering species.
    """
    if mode != 'TP':
        raise NotImplementedError("Only TP equilibrium is supported.")
    
    mech = sol.mech
    T = sol.T
    P = sol.P
    Y0 = sol.Y
    
    # 1. Element conservation constraint
    n0 = Y0 / mech.mol_weights # [mol/kg]
    b = jnp.dot(mech.element_matrix, n0)
    
    # Identify present elements (b_k > 0)
    # Use a small threshold for numerical safety
    present_elements = b > 1e-15
    
    # Identify species that can exist (all their elements are present)
    # A_ki > 0 for an element k not present -> species i cannot exist
    species_can_exist = jnp.all(jnp.where(mech.element_matrix > 0, present_elements[:, None], True), axis=0)
    
    # Filtered indices
    active_species_idx = jnp.where(species_can_exist)[0]
    active_elements_idx = jnp.where(present_elements)[0]
    
    n_active_species = len(active_species_idx)
    n_active_elements = len(active_elements_idx)
    
    if n_active_species == 0:
        return None

    # Slice data
    A_active = mech.element_matrix[active_elements_idx][:, active_species_idx]
    b_active = b[active_elements_idx]
    
    # 2. Species Gibbs energies
    h_RT = get_h_RT(T, mech.nasa_low, mech.nasa_high, mech.nasa_T_mid)
    s_R = get_s_R(T, mech.nasa_low, mech.nasa_high, mech.nasa_T_mid)
    g_standard_RT = h_RT - s_R
    log_p_ratio = jnp.log(P / ONE_ATM)
    gi_const = (g_standard_RT + log_p_ratio)[active_species_idx]
    
    # 3. KKT System
    def kkt_system(y, args):
        s = y[:n_active_species]
        lams = y[n_active_species : n_active_species + n_active_elements]
        log_n_total = y[-1]
        
        n = jnp.exp(s)
        n_total = jnp.exp(log_n_total)
        
        # eq 1: Chemical potential balance
        eq1 = gi_const + s - log_n_total - jnp.dot(A_active.T, lams)
        
        # eq 2: Element balance
        eq2 = (jnp.dot(A_active, n) - b_active) / (b_active + 1e-15)
        
        # eq 3: Total moles balance
        eq3 = (jnp.sum(n) - n_total) / n_total
        
        return jnp.concatenate([eq1, eq2, jnp.array([eq3])])

    # Initial guess for active species
    n_active0 = n0[active_species_idx]
    s0 = jnp.log(jnp.maximum(n_active0, 1e-10))
    lams0 = jnp.zeros(n_active_elements)
    log_nt0 = jnp.log(jnp.maximum(jnp.sum(n_active0), 1e-10))
    y0 = jnp.concatenate([s0, lams0, jnp.array([log_nt0])])
    
    # Solve
    solver = optx.LevenbergMarquardt(rtol=rtol, atol=rtol*1e-3)
    res = optx.least_squares(
        kkt_system,
        solver,
        y0=y0,
        max_steps=2000,
        throw=False
    )
    
    # Extract results
    s_equil = res.value[:n_active_species]
    n_equil_active = jnp.exp(s_equil)
    
    # Map back to full species array
    Y_equil = jnp.zeros(mech.n_species)
    n_full = jnp.zeros(mech.n_species)
    n_full = n_full.at[active_species_idx].set(n_equil_active)
    
    Y_equil = n_full * mech.mol_weights
    Y_equil = Y_equil / jnp.sum(Y_equil)
    
    sol.Y = Y_equil
    return res
