import jax
import jax.numpy as jnp
import equinox as eqx
import diffrax
import lineax
from .kinetics import compute_wdot
from .solvers.bdf import bdf_solve

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
    # wdot: [mol/m3/s], h_mass: [J/kg], cp_mass: [J/kg/K], rho: [kg/m3], h_mol: [J/mol]
    wdot, h_mass, cp_mass, rho, h_mol = compute_wdot(T, P, Y, mech)
    
    # 2. Conservation of Species
    # dY_i/dt = (wdot_i * MW_i) / rho
    dYdt = wdot * mech.mol_weights / rho
    
    # 3. Conservation of Energy (Constant Pressure)
    # Energy term: sum(h_i_mol * wdot_i)
    energy_term = jnp.sum(h_mol * wdot)
    
    dTdt = -energy_term / (rho * cp_mass)
    
    return jnp.concatenate([jnp.array([dTdt]), dYdt])

@eqx.filter_jit
class ReactorNet(eqx.Module):
    """Reactor network solver using diffrax or custom BDF."""
    mech: any
    
    @eqx.filter_jit
    def advance(self, T0, P, Y0, t_end, rtol=1e-7, atol=1e-10, solver=None, saveat=None, max_steps=100000, dt0=1e-8, stepsize_controller=None):
        """Simulate combustion trajectory."""
        state0 = jnp.concatenate([jnp.array([T0]), Y0])
        args = (P, self.mech)
        
        if solver == "bdf":
            state = bdf_solve(reactor_rhs, 0.0, t_end, state0, args=args, rtol=rtol, atol=atol, max_steps=max_steps)
            return eqx.filter_jit(lambda s: {
                "ts": jnp.array([s.t]),
                "ys": s.y[jnp.newaxis, :],
                "stats": {
                    "num_steps": s.n_steps,
                    "n_fevals": s.n_fevals,
                    "n_jevals": s.n_jevals
                }
            })(state)
        
        term = diffrax.ODETerm(reactor_rhs)
        if solver is None:
            # Optimal configuration found in hyperparameter sweep
            solver = diffrax.Kvaerno5(
                scan_kind="lax",
                root_finder=diffrax.VeryChord(
                    rtol=rtol, 
                    atol=atol,
                    kappa=0.5, 
                    linear_solver=lineax.LU()
                )
            )
        if saveat is None:
            saveat = diffrax.SaveAt(t1=True)
        if stepsize_controller is None:
            stepsize_controller = diffrax.PIDController(rtol=rtol, atol=atol)
            
        sol = diffrax.diffeqsolve(
            term,
            solver,
            t0=0.0,
            t1=t_end,
            dt0=dt0,
            y0=state0,
            args=args,
            stepsize_controller=stepsize_controller,
            saveat=saveat,
            max_steps=max_steps
        )
        return sol
