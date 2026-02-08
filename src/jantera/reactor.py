import jax
import jax.numpy as jnp
import equinox as eqx
import diffrax
from typing import NamedTuple, Callable
from .kinetics import compute_wdot
from .thermo import get_h_RT
from .constants import R_GAS
from .bdf import bdf_solve, get_sparsity_pattern
from .bdf import BDFState    
from jax.experimental import sparse
from sparsejac import jacfwd as sparse_jacfwd

@jax.jit
def reactor_rhs(t, state, args):
    """Constant-pressure adiabatic reactor RHS.
    
    State vector: [T, Y_0, Y_1, ..., Y_{n-1}]
    Args: (P, mech)
    """
    T = state[0]
    Y = state[1:]
    P, mech = args
    
    # 1. Compute production rates and mixture properties
    # wdot: [mol/m3/s], h_mass: [J/kg], cp_mass: [J/kg/K], rho: [kg/m3]
    wdot, h_mass, cp_mass, rho = compute_wdot(T, P, Y, mech)
    
    # 2. Conservation of Species
    # d(rho * Y_i) / dt = wdot_i * MW_i
    # For constant pressure, dY_i/dt = (wdot_i * MW_i) / rho
    dYdt = wdot * mech.mol_weights / rho
    
    # 3. Conservation of Energy (Constant Pressure)
    # rho * cp * dT/dt = -sum(h_i_mass * wdot_i_mass)
    # wdot_i_mass = wdot_i * MW_i [kg/m3/s]
    # h_i_mass = h_i_mol / MW_i [J/kg]
    
    # Get per-species enthalpies [J/mol]
    h_RT = get_h_RT(T, mech.nasa_low, mech.nasa_high, mech.nasa_T_mid)
    h_mol = h_RT * R_GAS * T
    
    # Energy term: sum(h_i_mol * wdot_i)
    energy_term = jnp.sum(h_mol * wdot)
    
    dTdt = -energy_term / (rho * cp_mass)
    
    return jnp.concatenate([jnp.array([dTdt]), dYdt])

class CustomSolution(NamedTuple):
    ts: jnp.ndarray
    ys: jnp.ndarray
    stats: dict

class ReactorNet(eqx.Module):
    """Reactor network solver using diffrax."""
    mech: any
    jac_fn: Callable = eqx.field(static=True)
    
    def __init__(self, mech):
        self.mech = mech
        
        # Pre-compute sparse Jacobian function
        # 1. Define Augmented RHS
        sample_y = jnp.zeros(mech.n_species + 1)
        
        def aug_rhs(z):
            t = z[0]
            P = z[1]
            state = z[2:]
            dy = reactor_rhs(t, state, (P, mech))
            return jnp.concatenate([jnp.array([1.0, 0.0]), dy])

        # 2. Compute Sparsity
        z_sample = jnp.concatenate([jnp.array([0.0, 101325.0]), sample_y + 300.0])
        z_sample = jnp.ones_like(z_sample) # Safe non-zero
        
        J_sample = jax.jacfwd(aug_rhs)(z_sample)
        aug_sparsity = sparse.BCOO.fromdense(jnp.abs(J_sample) > 0)
        
        # 3. Create Colored Function
        aug_jac_fn = sparse_jacfwd(aug_rhs, aug_sparsity)
        
        # 4. Wrap
        def jac_fn(t, y, args):
            P = args[0]
            z = jnp.concatenate([jnp.array([t, P]), y])
            J_aug = aug_jac_fn(z)
            return J_aug[2:, 2:]
            
        self.jac_fn = jac_fn

    @eqx.filter_jit
    def advance(self, T0, P, Y0, t_end, rtol=1e-7, atol=1e-10, solver=None, saveat=None):
        """Simulate combustion trajectory."""
        state0 = jnp.concatenate([jnp.array([T0]), Y0])
        args = (P, self.mech)
        
        if solver == "custom_bdf":
            final_state, converged = bdf_solve(
                reactor_rhs, 0.0, state0, t_end,
                args=args,
                jac_fn=self.jac_fn,
                rtol=rtol, atol=atol
            )
            
            # Return dummy solution object
            return CustomSolution(
                ts=jnp.array([final_state.t]),
                ys=jnp.array([final_state.y]),
                stats={
                    'n_steps': final_state.n_steps,
                    'n_fevals': final_state.n_fevals,
                    'n_jevals': final_state.n_jevals,
                    'n_lu': final_state.n_lu_decomps
                }
            )

        # Define solver
        term = diffrax.ODETerm(reactor_rhs)
        if solver is None:
            # Optimization: Use Matrix-Free GMRES
            # Avoids computing dense 54x54 Jacobian (5ms)
            # Uses JVP instead (10us)
            import optimistix
            import lineax
            
            solver = diffrax.Kvaerno5(
                root_finder=optimistix.Newton(
                    rtol=rtol, atol=atol,
                    linear_solver=lineax.GMRES(rtol=1e-3, atol=1e-6)
                )
            )
        
        # Max steps should be high for chemical kinetics
        stepsize_controller = diffrax.PIDController(rtol=rtol, atol=atol)
        
        if saveat is None:
            saveat = diffrax.SaveAt(t1=True)

        sol = diffrax.diffeqsolve(
            term,
            solver,
            t0=0.0,
            t1=t_end,
            dt0=1e-12,
            y0=state0,
            args=args,
            stepsize_controller=stepsize_controller,
            max_steps=1000000,
            saveat=saveat,
            adjoint=diffrax.RecursiveCheckpointAdjoint()
        )
        
        return sol
