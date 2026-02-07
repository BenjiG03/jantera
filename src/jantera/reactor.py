import jax
import jax.numpy as jnp
import equinox as eqx
import diffrax
from .kinetics import compute_wdot
from .thermo import get_h_RT
from .constants import R_GAS

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

class ReactorNet(eqx.Module):
    """Reactor network solver using diffrax."""
    mech: any
    
    def advance(self, T0, P, Y0, t_end, rtol=1e-7, atol=1e-10, solver=None, saveat=None):
        """Simulate combustion trajectory."""
        state0 = jnp.concatenate([jnp.array([T0]), Y0])
        
        # Define solver
        term = diffrax.ODETerm(reactor_rhs)
        if solver is None:
            solver = diffrax.Kvaerno5()
        
        # Max steps should be high for chemical kinetics
        stepsize_controller = diffrax.PIDController(rtol=rtol, atol=atol)
        
        if saveat is None:
            saveat = diffrax.SaveAt(t1=True)

        sol = diffrax.diffeqsolve(
            term,
            solver,
            t0=0.0,
            t1=t_end,
            dt0=1e-8,
            y0=state0,
            args=(P, self.mech),
            stepsize_controller=stepsize_controller,
            max_steps=1000000,
            saveat=saveat,
            adjoint=diffrax.RecursiveCheckpointAdjoint()
        )
        
        return sol
