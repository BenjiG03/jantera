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
import equinox as eqx

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

def validate_mechanism(name, yaml_path, initial_conditions, skip_sensitivity=False):
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

    # 2. Dynamic Validation (ReactorNet)
    print(f"--- Dynamic Validation (ReactorNet) ---", flush=True)
    t_end = 1e-3 # 1ms
    from diffrax import Kvaerno5, SaveAt
    net_jt = ReactorNet(mech_jt)
    
    # For plotting, we use intermediate steps
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
    
    ct_ts, ct_T, ct_Y = [], [], []
    for t in ts_jax:
        net_ct.advance(float(t))
        ct_ts.append(float(t)); ct_T.append(sol_ct.T); ct_Y.append(sol_ct.Y.copy())
    ct_ts, ct_T, ct_Y = np.array(ct_ts), np.array(ct_T), np.array(ct_Y)
    
    # Plot Trajectory
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

    # 3. Equilibrium Validation
    print(f"--- Equilibrium Validation ---", flush=True)
    sol_jt.TPX = T0, P0, X0_str
    t0 = time.time()
    res_equil = equilibrate(sol_jt, 'TP')
    jt_equil_time_jit = time.time() - t0
    jt_equil_steps = res_equil.stats['num_steps'] if hasattr(res_equil, 'stats') else 0
    
    t0 = time.time()
    equilibrate(sol_jt, 'TP')
    jt_equil_time_warm = time.time() - t0
    
    sol_ct.TPX = T0, P0, X0_str
    t0 = time.time()
    sol_ct.equilibrate('TP')
    ct_equil_time = time.time() - t0
    
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
    results['equil_stats'] = {
        'jt_time_jit': jt_equil_time_jit,
        'jt_time_warm': jt_equil_time_warm,
        'jt_steps': jt_equil_steps,
        'ct_time': ct_equil_time
    }

    # 4. Sensitivity
    jt_sens_steps = 0
    ct_sens_steps = 0
    jt_sens_time_jit = 0.0
    jt_sens_time_warm = 0.0
    sens_ct_time = 0.0
    
    if not skip_sensitivity:
        print(f"--- Gradient Comparison (Reaction Sensitivities) ---", flush=True)
        t_grad = 1e-6
        sol_jt.TPX = T0, P0, X0_str
        
        @jax.jit
        def get_final_T(mech_curr, y0):
            net_curr = ReactorNet(mech_curr)
            res = net_curr.advance(T0, P0, y0, t_grad, solver=Kvaerno5())
            return res.ys[-1, 0], res.stats['num_steps']
        
        def grad_fun(mech, y):
            val, steps = get_final_T(mech, y)
            return val
            
        print(f"  Computing Jantera AD sensitivities...", end="", flush=True)
        t0 = time.time()
        grad_mech = eqx.filter_grad(grad_fun)(mech_jt, sol_jt.Y)
        jt_sens_time_jit = time.time() - t0
        
        t0 = time.time()
        grad_mech = eqx.filter_grad(grad_fun)(mech_jt, sol_jt.Y)
        jt_sens_time_warm = time.time() - t0
        
        grad_jt_norm = np.array(grad_mech.A * mech_jt.A) # This is dT/d(ln A)
        _, jt_sens_steps = get_final_T(mech_jt, sol_jt.Y)
        print(f" Done ({jt_sens_time_jit:.2f}s JIT, {jt_sens_time_warm:.4f}s Warm, {jt_sens_steps} steps)", flush=True)
        
        print(f"  Computing Cantera native sensitivities...", end="", flush=True)
        sol_ct.TPX = T0, P0, X0_str
        reac_ct = ct.IdealGasConstPressureReactor(sol_ct)
        net_ct = ct.ReactorNet([reac_ct])
        for i in range(sol_ct.n_reactions):
            reac_ct.add_sensitivity_reaction(i)
        net_ct.rtol_sensitivity = 1e-4
        net_ct.atol_sensitivity = 1e-6
        
        start_ct_sens = time.time()
        net_ct.advance(t_grad)
        sens_ct_time = time.time() - start_ct_sens
        try:
            ct_sens_steps = net_ct.get_solver_stats().get('n_steps', 0)
        except:
            ct_sens_steps = 0
        print(f" Done ({sens_ct_time:.2f}s, {ct_sens_steps} steps)", flush=True)
        
        grad_ct_norm = []
        for i in range(sol_ct.n_reactions):
            grad_ct_norm.append(net_ct.sensitivity('temperature', i))
        grad_ct_norm = np.array(grad_ct_norm)
        
        plt.figure(figsize=(12, 6))
        top_sens_idx = np.argsort(np.abs(grad_ct_norm))[-10:][::-1]
        labels = [f"R{i}: {sol_ct.reaction(i).equation}"[:30] for i in top_sens_idx]
        x = np.arange(len(labels))
        width = 0.35
        plt.bar(x - width/2, grad_ct_norm[top_sens_idx], width, label='Cantera (Native)', color='gray')
        plt.bar(x + width/2, grad_jt_norm[top_sens_idx], width, label='Jantera (AD)', color='cyan')
        plt.xticks(x, labels, rotation=45, ha='right')
        plt.ylabel('Normalized Sensitivity d(T) / d(ln A)')
        plt.title(f'{name} Reaction Sensitivity (t=1us)')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"tests/outputs/{name.lower()}_gradient_comp.png")
        plt.close()

    # Performance Benchmarks (Fair 1ms comparison)
    print(f"--- Performance Benchmarking ---", flush=True)
    jit_w, warm_w = run_performance_bench(compute_wdot, (T0, P0, sol_jt.Y, mech_jt), "compute_wdot")
    
    # Jantera 1ms Benchmark (No saveat, just t_end)
    def advance_bench(y):
        # By default advance uses saveat=t1=True (just end)
        return net_jt.advance(T0, P0, y, t_end, solver=Kvaerno5())
    
    res_jt_bench = advance_bench(sol_jt.Y)
    jt_adv_steps = res_jt_bench.stats['num_steps']
    jit_adv, warm_adv = run_performance_bench(advance_bench, (sol_jt.Y,), "advance_1ms")

    # Cantera 1ms Benchmark (measured)
    sol_ct.TPX = T0, P0, X0_str
    reac_ct_bench = ct.IdealGasConstPressureReactor(sol_ct)
    net_ct_bench = ct.ReactorNet([reac_ct_bench])
    net_ct_bench.rtol, net_ct_bench.atol = 1e-8, 1e-12
    
    # Warmup
    net_ct_bench.reinitialize() # Just in case
    net_ct_bench.advance(t_end)
    
    # Measure
    n_runs = 10
    ct_times = []
    ct_adv_steps = 0
    for _ in range(n_runs):
        sol_ct.TPX = T0, P0, X0_str
        reac_ct_bench = ct.IdealGasConstPressureReactor(sol_ct)
        net_ct_bench = ct.ReactorNet([reac_ct_bench])
        net_ct_bench.rtol, net_ct_bench.atol = 1e-8, 1e-12
        
        start = time.perf_counter()
        net_ct_bench.advance(t_end)
        ct_times.append(time.perf_counter() - start)
        try:
            # We only get stats for the last run, but it should be consistent
            ct_adv_steps = net_ct_bench.get_solver_stats()['n_steps']
        except:
            ct_adv_steps = -1 

    ct_warm_adv = np.mean(ct_times)

    results['grad_ok'] = not skip_sensitivity
    results['perf'] = {
        'jit_wdot': jit_w,
        'warm_wdot': warm_w,
        'jit_adv': jit_adv, # This is now for 1ms
        'warm_adv': warm_adv, # This is now for 1ms
        'jt_adv_steps': jt_adv_steps,
        'ct_adv_total_time': ct_warm_adv,
        'ct_adv_steps': ct_adv_steps,
        'ct_sens_time': sens_ct_time,
        'jt_sens_time_jit': jt_sens_time_jit if not skip_sensitivity else 0.0,
        'jt_sens_time_warm': jt_sens_time_warm if not skip_sensitivity else 0.0,
        'jt_sens_steps': jt_sens_steps,
        'ct_sens_steps': ct_sens_steps
    }
    return results

def main():
    os.makedirs("tests/outputs", exist_ok=True)
    
    gri_cond = (1500.0, 101325.0, "CH4:1, O2:2, N2:7.52")
    res_gri = validate_mechanism("GRI-30", "gri30.yaml", gri_cond, skip_sensitivity=False)
    
    jp10_yaml = os.path.join(os.path.dirname(__file__), "..", "jp10.yaml")
    jp10_cond = (1500.0, 101325.0, "C10H16:1, O2:14, N2:52.64")
    res_jp10 = validate_mechanism("JP-10", jp10_yaml, jp10_cond, skip_sensitivity=False)

    print("\n" + "="*25 + " DETAILED PERFORMANCE " + "="*25)
    
    # Helper to safe divide
    def time_per_step(time_s, steps):
        if steps <= 0: return "N/A"
        return f"{(time_s * 1e3 / steps):.3f}"

    detailed_table = [
        ["Phase", "Metric", "Jantera (GRI)", "Cantera (GRI)", "Jantera (JP10)", "Cantera (JP10)"],
        ["Equil", "JIT Time (ms)", f"{res_gri['equil_stats']['jt_time_jit']*1e3:.2f}", "-", f"{res_jp10['equil_stats']['jt_time_jit']*1e3:.2f}", "-"],
        ["Equil", "Warm Time (ms)", f"{res_gri['equil_stats']['jt_time_warm']*1e3:.2f}", f"{res_gri['equil_stats']['ct_time']*1e3:.2f}", f"{res_jp10['equil_stats']['jt_time_warm']*1e3:.2f}", f"{res_jp10['equil_stats']['ct_time']*1e3:.2f}"],
        ["Equil", "Steps", f"{res_gri['equil_stats']['jt_steps']}", "-", f"{res_jp10['equil_stats']['jt_steps']}", "-"],
        
        # 1ms Reactor Benchmark
        ["Adv (1ms)", "JIT Time (ms)", f"{res_gri['perf']['jit_adv']*1e3:.2f}", "-", f"{res_jp10['perf']['jit_adv']*1e3:.2f}", "-"],
        ["Adv (1ms)", "Warm Time (ms)", f"{res_gri['perf']['warm_adv']*1e3:.3f}", f"{res_gri['perf']['ct_adv_total_time']*1e3:.3f}", f"{res_jp10['perf']['warm_adv']*1e3:.3f}", f"{res_jp10['perf']['ct_adv_total_time']*1e3:.3f}"],
        ["Adv (1ms)", "Total Steps", f"{res_gri['perf']['jt_adv_steps']}", f"{res_gri['perf']['ct_adv_steps']}", f"{res_jp10['perf']['jt_adv_steps']}", f"{res_jp10['perf']['ct_adv_steps']}"],
        ["Adv (1ms)", "Time/Step (ms)", time_per_step(res_gri['perf']['warm_adv'], res_gri['perf']['jt_adv_steps']), time_per_step(res_gri['perf']['ct_adv_total_time'], res_gri['perf']['ct_adv_steps']), time_per_step(res_jp10['perf']['warm_adv'], res_jp10['perf']['jt_adv_steps']), time_per_step(res_jp10['perf']['ct_adv_total_time'], res_jp10['perf']['ct_adv_steps'])],

        ["Sens", "JIT Time (s)", f"{res_gri['perf']['jt_sens_time_jit']:.2f}", "-", f"{res_jp10['perf']['jt_sens_time_jit']:.2f}", "-"],
        ["Sens", "Warm Time (s)", f"{res_gri['perf']['jt_sens_time_warm']:.4f}", f"{res_gri['perf']['ct_sens_time']:.4f}", f"{res_jp10['perf']['jt_sens_time_warm']:.4f}", f"{res_jp10['perf']['ct_sens_time']:.4f}"],
        ["Sens", "Steps", f"{res_gri['perf']['jt_sens_steps']}", f"{res_gri['perf']['ct_sens_steps']}", f"{res_jp10['perf']['jt_sens_steps']}", f"{res_jp10['perf']['ct_sens_steps']}"],
    ]
    print(tabulate(detailed_table, headers="firstrow", tablefmt="grid"))

if __name__ == "__main__":
    main()
