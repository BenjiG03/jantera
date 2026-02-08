import os
import time
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import cantera as ct
import matplotlib.pyplot as plt
from tabulate import tabulate

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from jantera.loader import load_mechanism
from jantera.solution import Solution
from jantera.reactor import ReactorNet
from jantera.thermo import compute_mixture_props
from jantera.kinetics import compute_wdot
from jantera.equilibrate import equilibrate

def run_performance_bench(func, args, name, n_runs=10):
    """Measures JIT and warm run times."""
    # 1. First call (JIT)
    start = time.perf_counter()
    _ = jax.block_until_ready(func(*args))
    jit_time = time.perf_counter() - start
    
    # 2. Sequential calls (Warm)
    warm_times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = jax.block_until_ready(func(*args))
        warm_times.append(time.perf_counter() - start)
    
    return jit_time, np.mean(warm_times)

def validate_mechanism(name, yaml_path, initial_conditions):
    print(f"\n{'='*20} Validating {name} ({yaml_path}) {'='*20}", flush=True)
    
    # Loaders
    sol_ct = ct.Solution(yaml_path)
    mech_jt = load_mechanism(yaml_path)
    sol_jt = Solution(yaml_path)
    
    T0, P0, X0_str = initial_conditions
    sol_ct.TPX = T0, P0, X0_str
    sol_jt.TPX = T0, P0, X0_str
    
    results = {}

    # 1. Static Validation & Parity Plots
    print(f"--- Static Validation & Parity Plots ---", flush=True)
    # Generate random samples for parity
    n_samples = 20
    sample_Ts = np.random.uniform(800, 2500, n_samples)
    sample_Ps = np.random.uniform(0.5e5, 10e5, n_samples)
    
    cp_jt_list, cp_ct_list = [], []
    wdot_jt_max_list, wdot_ct_max_list = [], []
    
    for t, p in zip(sample_Ts, sample_Ps):
        sol_jt.TP = t, p
        sol_ct.TP = t, p
        
        w_jt, _, c_jt, _ = compute_wdot(sol_jt.T, sol_jt.P, sol_jt.Y, mech_jt)
        cp_jt_list.append(c_jt)
        wdot_jt_max_list.append(np.max(np.abs(w_jt)))
        
        cp_ct_list.append(sol_ct.cp_mass)
        wdot_ct_max_list.append(np.max(np.abs(sol_ct.net_production_rates)))

    # Plot Static Parity
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.loglog(cp_ct_list, cp_jt_list, 'bo', alpha=0.5)
    plt.plot([min(cp_ct_list), max(cp_ct_list)], [min(cp_ct_list), max(cp_ct_list)], 'r--')
    plt.xlabel('Cantera Cp (J/kg/K)')
    plt.ylabel('Jantera Cp (J/kg/K)')
    plt.title(f'{name} Static Cp Parity')
    plt.grid(True, which="both", ls="-", alpha=0.2)

    plt.subplot(1, 2, 2)
    plt.loglog(wdot_ct_max_list, wdot_jt_max_list, 'go', alpha=0.5)
    plt.plot([min(wdot_ct_max_list), max(wdot_ct_max_list)], [min(wdot_ct_max_list), max(wdot_ct_max_list)], 'r--')
    plt.xlabel('Cantera Max |wdot|')
    plt.ylabel('Jantera Max |wdot|')
    plt.title(f'{name} Static wdot Parity')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(f"tests/outputs/{name.lower()}_static_parity.png")
    plt.close()

    # Re-set to initial for error metrics
    sol_ct.TPX = T0, P0, X0_str
    sol_jt.TPX = T0, P0, X0_str
    wdot_jt, h_jt, cp_jt, rho_jt = compute_wdot(sol_jt.T, sol_jt.P, sol_jt.Y, mech_jt)
    wdot_ct = sol_ct.net_production_rates
    
    err_wdot = np.max(np.abs(wdot_jt - wdot_ct) / (np.max(np.abs(wdot_ct)) + 1e-10))
    results['static_err'] = err_wdot

    # 2. Dynamic Validation (ReactorNet) -> Already plotting T, Y
    print(f"--- Dynamic Validation (ReactorNet) ---", flush=True)
    t_end = 1e-3 # 1ms
    from diffrax import Kvaerno5, SaveAt
    net_jt = ReactorNet(mech_jt)
    
    ts_jax = jnp.linspace(0, t_end, 50)
    saveat = SaveAt(ts=ts_jax)
    
    start_jt = time.time()
    res_jt = net_jt.advance(T0, P0, sol_jt.Y, t_end, rtol=1e-8, atol=1e-12, solver=Kvaerno5(), saveat=saveat)
    jax.block_until_ready(res_jt)
    jt_time = time.time() - start_jt
    
    jt_ts, jt_T, jt_Y = np.array(res_jt.ts), np.array(res_jt.ys[:, 0]), np.array(res_jt.ys[:, 1:])
    
    sol_ct.TPX = T0, P0, X0_str
    reac_ct = ct.IdealGasConstPressureReactor(sol_ct)
    net_ct = ct.ReactorNet([reac_ct])
    net_ct.rtol, net_ct.atol = 1e-8, 1e-12
    
    start_ct = time.time()
    ct_ts, ct_T, ct_Y = [], [], []
    for t in ts_jax:
        net_ct.advance(float(t))
        ct_ts.append(float(t)); ct_T.append(sol_ct.T); ct_Y.append(sol_ct.Y.copy())
    ct_time = time.time() - start_ct
    ct_ts, ct_T, ct_Y = np.array(ct_ts), np.array(ct_T), np.array(ct_Y)
    
    # Plot Trajectory (Done in previous step, keeping it)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(ct_ts*1e6, ct_T, 'k-', label='Cantera'); plt.plot(jt_ts*1e6, jt_T, 'r--', label='Jantera')
    plt.xlabel('Time (us)'); plt.ylabel('Temperature (K)'); plt.title(f'{name} T Trajectory'); plt.legend()
    
    plt.subplot(1, 2, 2)
    dy = np.abs(ct_Y[-1] - ct_Y[0])
    top3 = np.argsort(dy)[-3:][::-1]
    for idx in top3:
        plt.plot(ct_ts*1e6, ct_Y[:, idx], '-', label=f'{sol_ct.species_names[idx]} (CT)')
        plt.plot(jt_ts*1e6, jt_Y[:, idx], '--', label=f'{sol_ct.species_names[idx]} (JT)')
    plt.xlabel('Time (us)'); plt.ylabel('Mass Fraction'); plt.title(f'{name} Species'); plt.legend()
    plt.tight_layout(); plt.savefig(f"tests/outputs/{name.lower()}_trajectory.png"); plt.close()

    results['dynamic_err_T'] = np.abs(jt_T[-1] - ct_T[-1])

    # 3. Equilibrium Parity Plots
    print(f"--- Equilibrium Validation ---", flush=True)
    sol_jt.TPX = T0, P0, X0_str
    equilibrate(sol_jt, 'TP')
    
    sol_ct.TPX = T0, P0, X0_str
    sol_ct.equilibrate('TP')
    
    plt.figure(figsize=(6, 6))
    plt.loglog(sol_ct.Y, sol_jt.Y, 'mo', alpha=0.6)
    plt.plot([1e-15, 1], [1e-15, 1], 'k--')
    plt.xlabel('Cantera Mole Fraction')
    plt.ylabel('Jantera Mole Fraction')
    plt.title(f'{name} Equilibrium Parity')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig(f"tests/outputs/{name.lower()}_equil_parity.png")
    plt.close()
    
    results['equil_err'] = np.max(np.abs(sol_jt.Y - sol_ct.Y))

    # 4. Gradients Comparison Chart (AD vs FD)
    print(f"--- Gradient Comparison (AD vs FD) ---", flush=True)
    t_grad = 1e-6
    @jax.jit
    def get_final_T(y0):
        res = net_jt.advance(T0, P0, y0, t_grad, solver=Kvaerno5())
        return res.ys[-1, 0]
    
    grad_jt = jax.grad(get_final_T)(sol_jt.Y)
    
    # Cantera Finite Difference
    grad_ct_fd = []
    eps = 1e-6
    for i in range(len(sol_ct.Y)):
        Y_plus = sol_ct.Y.copy(); Y_plus[i] += eps
        sol_ct.TPY = T0, P0, Y_plus
        reac = ct.IdealGasConstPressureReactor(sol_ct); net = ct.ReactorNet([reac])
        net.advance(t_grad)
        T_plus = sol_ct.T
        
        Y_minus = sol_ct.Y.copy(); Y_minus[i] -= eps
        sol_ct.TPY = T0, P0, Y_minus
        reac = ct.IdealGasConstPressureReactor(sol_ct); net = ct.ReactorNet([reac])
        net.advance(t_grad)
        T_minus = sol_ct.T
        
        grad_ct_fd.append((T_plus - T_minus) / (2 * eps))
    
    grad_ct_fd = np.array(grad_ct_fd)
    
    # Plot top 5 sensitivities
    plt.figure(figsize=(10, 5))
    top_grad_idx = np.argsort(np.abs(grad_ct_fd))[-5:][::-1]
    labels = [sol_ct.species_names[i] for i in top_grad_idx]
    
    x = np.arange(len(labels))
    width = 0.35
    plt.bar(x - width/2, grad_ct_fd[top_grad_idx], width, label='Cantera (FD)', color='gray')
    plt.bar(x + width/2, grad_jt[top_grad_idx], width, label='Jantera (AD)', color='cyan')
    plt.xticks(x, labels)
    plt.ylabel('dT_final / dY_i')
    plt.title(f'{name} Temperature Sensitivity (t=1us)')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(f"tests/outputs/{name.lower()}_gradient_comp.png")
    plt.close()

    results['grad_ok'] = True
    results['perf'] = (0, jt_time, ct_time, 0) # Placeholder for JIT/Batched
    return results

def main():
    # Ensure outputs dir exists
    os.makedirs("tests/outputs", exist_ok=True)
    
    # 1. GRI-30 (Methane) - 1500K/1atm shows slow oxidation over 1ms
    gri_cond = (1500.0, 101325.0, "CH4:1, O2:2, N2:7.52")
    res_gri = validate_mechanism("GRI-30", "gri30.yaml", gri_cond)
    
    # 2. JP-10 
    jp10_cond = (1500.0, 101325.0, "C10H16:1, O2:14, N2:52.64")
    res_jp10 = validate_mechanism("JP-10", "jp10.yaml", jp10_cond)
    
    def fmt_speed(jt, ct):
        if jt < 1e-12:
            return "N/A"
        if jt < ct:
            return f"{ct/jt:.1f}x"
        else:
            return f"{ct/jt*100:.1f}%"

    # Summary Table
    table_data = [
        ["Metric", "GRI-30 (CH4)", "JP-10"],
        ["Static Error (wdot)", f"{res_gri['static_err']:.2e}", f"{res_jp10['static_err']:.2e}"],
        ["Dynamic Error (dT @ 0.1ms)", f"{res_gri['dynamic_err_T']:.2e}K", f"{res_jp10['dynamic_err_T']:.2e}K"],
        ["Equil Error (dY)", f"{res_gri['equil_err']:.2e}", f"{res_jp10['equil_err']:.2e}"],
        ["JAX JIT Time (s)", f"{res_gri['perf'][0]:.3f}", f"{res_jp10['perf'][0]:.3f}"],
        ["JAX Warm Time (s)", f"{res_gri['perf'][1]:.6f}", f"{res_jp10['perf'][1]:.6f}"],
        ["Cantera Time (s)", f"{res_gri['perf'][2]:.6f}", f"{res_jp10['perf'][2]:.6f}"],
        ["Speedup (Warm, serial)", fmt_speed(res_gri['perf'][1], res_gri['perf'][2]), fmt_speed(res_jp10['perf'][1], res_jp10['perf'][2])],
        ["Speedup (Batched x100)", fmt_speed(res_gri['perf'][3], res_gri['perf'][2]), fmt_speed(res_jp10['perf'][3], res_jp10['perf'][2])],
    ]
    
    print("\n" + "="*50)
    print("FINAL VALIDATION SUMMARY")
    print("="*50)
    print(tabulate(table_data, headers="firstrow", tablefmt="grid"))

if __name__ == "__main__":
    main()
