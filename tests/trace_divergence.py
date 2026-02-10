"""
Compare trajectories at multiple time points to find where divergence occurs.
"""
import os
import sys
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import cantera as ct
from diffrax import SaveAt, Kvaerno5
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from canterax.loader import load_mechanism
from canterax.reactor import ReactorNet

def trace_divergence():
    yaml_path = "gri30.yaml"
    T0, P0 = 1500.0, 101325.0
    X0 = "CH4:1, O2:2, N2:7.52"
    t_end = 1e-3
    n_points = 50
    
    mech = load_mechanism(yaml_path)
    sol_ct = ct.Solution(yaml_path)
    sol_ct.TPX = T0, P0, X0
    Y0 = jnp.array(sol_ct.Y)
    
    net = ReactorNet(mech)
    
    # Canterax trajectory
    ts = jnp.linspace(0, t_end, n_points)
    saveat = SaveAt(ts=ts)
    res = net.advance(T0, P0, Y0, t_end, rtol=1e-10, atol=1e-14, solver=Kvaerno5(), saveat=saveat)
    jt_T = np.array(res.ys[:, 0])
    jt_ts = np.array(res.ts)
    
    # Cantera trajectory
    sol_ct.TPX = T0, P0, X0
    reac = ct.IdealGasConstPressureReactor(sol_ct, clone=False)
    sim = ct.ReactorNet([reac])
    sim.rtol, sim.atol = 1e-10, 1e-14
    
    ct_T = [T0]
    ct_ts = [0.0]
    for t in ts[1:]:
        sim.advance(float(t))
        ct_T.append(sol_ct.T)
        ct_ts.append(float(t))
    ct_T = np.array(ct_T)
    ct_ts = np.array(ct_ts)
    
    # Calculate divergence
    dT = jt_T - ct_T
    
    print("Time (us)  | T_Canterax | T_Cantera | dT (K)")
    print("-" * 50)
    for i in range(0, len(ts), 5):
        print(f"{ts[i]*1e6:8.2f}   | {jt_T[i]:9.2f} | {ct_T[i]:9.2f} | {dT[i]:+8.4f}")
    
    # Find first point where dT > 1K
    first_sig = np.argmax(np.abs(dT) > 1.0)
    if first_sig > 0:
        print(f"\nFirst significant divergence (> 1K) at t = {ts[first_sig]*1e6:.2f} us")
    else:
        print(f"\nNo significant divergence (> 1K) detected")
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].plot(ct_ts*1e6, ct_T, 'k-', lw=2, label='Cantera')
    axes[0].plot(jt_ts*1e6, jt_T, 'r--', lw=2, label='Canterax')
    axes[0].set_xlabel('Time (us)')
    axes[0].set_ylabel('Temperature (K)')
    axes[0].set_title('GRI-30 Trajectory')
    axes[0].legend()
    
    axes[1].plot(ts*1e6, dT, 'b-', lw=2)
    axes[1].axhline(0, color='k', ls='--')
    axes[1].set_xlabel('Time (us)')
    axes[1].set_ylabel('dT = T_JT - T_CT (K)')
    axes[1].set_title('Temperature Divergence')
    
    plt.tight_layout()
    plt.savefig("tests/outputs/gri30_divergence.png")
    plt.close()
    print("\nPlot saved to tests/outputs/gri30_divergence.png")

if __name__ == "__main__":
    trace_divergence()
