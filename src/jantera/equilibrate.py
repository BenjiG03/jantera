import jax
import jax.numpy as jnp
import equinox as eqx
import optimistix as optx
from .thermo import get_h_RT, get_s_R
from .constants import R_GAS, ONE_ATM

@eqx.filter_jit
def _solve_equil_core(y0, gi_const, A_active, b_active, rtol, max_steps):
    n_active_species = A_active.shape[1]
    n_active_elements = A_active.shape[0]

    def kkt_system(y, args):
        s = y[:n_active_species]
        lams = y[n_active_species : n_active_species + n_active_elements]
        log_n_total = y[-1]
        
        n = jnp.exp(s)
        n_total = jnp.exp(log_n_total)
        
        # eq 1: Chemical potential balance
        eq1 = gi_const + s - log_n_total - (A_active.T @ lams)
        
        # eq 2: Element balance
        eq2 = (A_active @ n - b_active) / (b_active + 1e-15)
        
        # eq 3: Total moles balance
        eq3 = (jnp.sum(n) - n_total) / n_total
        
        return jnp.concatenate([eq1, eq2, jnp.array([eq3])])

    solver = optx.LevenbergMarquardt(rtol=rtol, atol=rtol*1e-3)
    return optx.least_squares(
        kkt_system,
        solver,
        y0=y0,
        max_steps=max_steps,
        throw=False
    )

def equilibrate(sol, mode='TP', rtol=1e-10, max_steps=2000):
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
    
    # Filtered indices - These are dynamic but we only call JIT core on slices
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
    gi_const_all = (g_standard_RT + log_p_ratio)
    gi_const = gi_const_all[active_species_idx]
    
    # 3. Basis Selection (QR pivoting on A.T)
    # Goal: pick N_elements species that are stable and span the elements.
    # Weight A.T by exp(-g) or similar to prefer stable species?
    # Actually, just QR on A_active.T works to find a basis.
    # We'll use a simple pivot-based approach or just solve for lams using n0.
    
    # Improved initial guess for lams:
    # Use the species with highest concentrations to estimate lams
    # n_active_0: initial mol/kg
    # For species with n_i > 0, g_i + ln(n_i/ntot) approx A_i @ lams
    # We can solve this as a weighted least squares for initial lams.
    
    n_active0 = n0[active_species_idx]
    ntot0 = jnp.maximum(jnp.sum(n_active0), 1e-10)
    
    # Weight by sqrt(n) to focus on major species
    weights = jnp.sqrt(jnp.maximum(n_active0 / ntot0, 1e-6))
    rhs = (gi_const + jnp.log(jnp.maximum(n_active0 / ntot0, 1e-10))) * weights
    design_mat = A_active.T * weights[:, None]
    
    # Solve for initial lams: design_mat @ lams = rhs
    lams0, _, _, _ = jnp.linalg.lstsq(design_mat, rhs)
    
    # Initial s (log concentrations) should be consistent with these lams
    # s_i = A_i @ lams - g_i + ln(ntot)
    # But for initialization, we can just use the lams and ntot
    s0 = (A_active.T @ lams0) - gi_const + jnp.log(ntot0)
    
    log_nt0 = jnp.log(ntot0)
    y0 = jnp.concatenate([s0, lams0, jnp.array([log_nt0])])
    
    # Solve using JIT-compiled core
    res = _solve_equil_core(y0, gi_const, A_active, b_active, rtol, max_steps)
    
    # Extract results
    s_equil = res.value[:n_active_species]
    n_equil_active = jnp.exp(s_equil)
    
    # Map back to full species array
    n_full = jnp.zeros(mech.n_species).at[active_species_idx].set(n_equil_active)
    
    Y_equil = n_full * mech.mol_weights
    Y_equil = Y_equil / jnp.sum(Y_equil)
    
    sol.Y = Y_equil
    return res
