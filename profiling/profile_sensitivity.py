import time
import jax
import jax.numpy as jnp
import equinox as eqx
import diffrax
import os
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from jantera.loader import load_mechanism
from jantera.reactor import ReactorNet
from jantera.solution import Solution

jax.config.update("jax_enable_x64", True)

def profile_sensitivity(yaml_file, condition_name):
    print(f"\n{'='*20} Profiling Sensitivity: {condition_name} {'='*20}")
    
    # Load Mechanism
    mech = load_mechanism(yaml_file)
    sol = Solution(yaml_file)
    
    # Set initial condition
    T0 = 1500.0
    P0 = 101325.0
    # Simple stoichiometric methane-air ish
    if "gri30" in yaml_file.lower():
        X_str = "CH4:1, O2:2, N2:7.52"
    elif "jp10" in yaml_file.lower():
        X_str = "C10H16:1, O2:14, N2:52.64"
    else:
        X_str = "H2:2, O2:1"
        
    sol.TPX = T0, P0, X_str
    
    # Define Gradient Function
    # Differentiate T_final w.r.t. Mechanism Parameters (A)
    # Using the same logic as validation suite
    t_end = 1e-6 # 1 us step
    
    print(f"Mechanism: {mech.n_species} species, {mech.n_reactions} reactions")
    print(f"Target: d(T_final)/d(Mech) after {t_end*1e6} us")

    @jax.jit
    def get_final_T(mech_curr):
        # Custom advance to test BacksolveAdjoint
        # We need to access reactor_rhs
        from jantera.reactor import reactor_rhs
        term = diffrax.ODETerm(reactor_rhs)
        
        y0 = jnp.concatenate([jnp.array([T0]), sol.Y])
        solver = diffrax.Kvaerno5()
        stepsize_controller = diffrax.PIDController(rtol=1e-7, atol=1e-10)
        
        sol_diff = diffrax.diffeqsolve(
            diffrax.ODETerm(reactor_rhs),
            solver,
            t0=0.0,
            t1=t_end,
            dt0=1e-12,
            y0=y0,
            args=(P0, mech_curr),
            stepsize_controller=stepsize_controller,
            max_steps=1000000,
            adjoint=diffrax.BacksolveAdjoint()
        )
        return sol_diff.ys[-1, 0]

    # Warmup / JIT Compile
    print("Compiling gradient function (JIT)...")
    start = time.time()
    grad_mech = eqx.filter_grad(get_final_T)(mech)
    jax.block_until_ready(grad_mech)
    jit_time = time.time() - start
    print(f"JIT Compile Time: {jit_time:.4f} s")
    
    # Run 1 (Warm)
    print("Running Warm Gradient (Run 1)...")
    start = time.time()
    grad_mech = eqx.filter_grad(get_final_T)(mech)
    jax.block_until_ready(grad_mech)
    warm_time_1 = time.time() - start
    print(f"Warm Run 1 Time: {warm_time_1:.4f} s")
    
    # Run 2 (Warm)
    print("Running Warm Gradient (Run 2)...")
    start = time.time()
    grad_mech = eqx.filter_grad(get_final_T)(mech)
    jax.block_until_ready(grad_mech)
    warm_time_2 = time.time() - start
    print(f"Warm Run 2 Time: {warm_time_2:.4f} s")
    
    return jit_time, warm_time_1

if __name__ == "__main__":
    # Path to GRI-30
    gri30_path = "gri30.yaml" 
    
    # Run GRI-30
    profile_sensitivity(gri30_path, "GRI-30")
    
    # Run JP-10 if exists
    jp10_path = os.path.join(os.path.dirname(__file__), "..", "jp10.yaml")
    if os.path.exists(jp10_path):
        profile_sensitivity(jp10_path, "JP-10")
