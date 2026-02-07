import os
import sys
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import cantera as ct
import matplotlib.pyplot as plt
import time

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from jantera.loader import load_mechanism
from jantera.reactor import ReactorNet

def test_reactor_trajectory():
    yaml_path = "gri30.yaml"
    mech = load_mechanism(yaml_path)
    sol_ct = ct.Solution(yaml_path)
    
    T0 = 1200.0
    P = 101325.0
    X0 = 'CH4:1.0, O2:2.0, N2:7.52'
    t_end = 1e-3 # 1ms
    
    print(f"Simulating 1ms combustion at {T0}K, 1 atm...")
    
    # --- Cantera ---
    sol_ct.TPX = T0, P, X0
    reac = ct.IdealGasConstPressureReactor(sol_ct)
    sim = ct.ReactorNet([reac])
    
    t_ct = []
    T_ct = []
    
    # Manual integration for Cantera to get points
    t = 0.0
    dt = 1e-6
    while t < t_end:
        t_ct.append(t)
        T_ct.append(reac.T)
        t = sim.step()
        
    t_ct = np.array(t_ct)
    T_ct = np.array(T_ct)
    
    # --- Jantera ---
    sol_ct.TPX = T0, P, X0
    Y0 = jnp.array(sol_ct.Y)
    
    net = ReactorNet(mech)
    
    start_jit = time.time()
    # Warmup / JIT
    net.advance(T0, P, Y0, 1e-8)
    print(f"JIT warmup took {time.time() - start_jit:.2f}s")
    
    start_sim = time.time()
    res = net.advance(T0, P, Y0, t_end)
    print(f"Jantera simulation took {time.time() - start_sim:.2f}s")
    
    # Extract results
    t_jan = res.ts
    T_jan = res.ys[:, 0]
    
    # --- Compare ---
    # Interpolate Jantera onto Cantera time grid for error calculation
    T_jan_interp = np.interp(t_ct, t_jan, T_jan)
    max_dT = np.max(np.abs(T_jan_interp - T_ct))
    
    print(f"Max T error: {max_dT:.2e} K")
    
    # Plotting
    os.makedirs("tests/outputs", exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(t_ct * 1e3, T_ct, 'b-', label='Cantera', linewidth=2)
    plt.plot(t_jan * 1e3, T_jan, 'r--', label='Jantera', linewidth=2)
    plt.title("Reactor Temperature Trajectory (CH4/Air, 1200K, 1 atm)")
    plt.xlabel("Time [ms]")
    plt.ylabel("Temperature [K]")
    plt.legend()
    plt.grid(True)
    plt.savefig("tests/outputs/reactor_trajectory.png")
    print("Verification plot saved to tests/outputs/reactor_trajectory.png")
    
    assert max_dT < 1.0 # Target: < 1K

if __name__ == "__main__":
    try:
        test_reactor_trajectory()
        print("Reactor validation passed!")
    except Exception as e:
        print(f"Reactor validation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
