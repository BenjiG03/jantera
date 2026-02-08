"""
Verify Jantera JAX gradients (AD) against Cantera Finite Differences (FD).
"""
import os
import sys
import time
import numpy as np
import cantera as ct
import jax
import jax.numpy as jnp
from tabulate import tabulate

# Ensure 64-bit precision for high-accuracy gradients
jax.config.update("jax_enable_x64", True)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from jantera.loader import load_mechanism
from jantera.reactor import ReactorNet

def compare_gradients(name, yaml_file, T0, P0, X0, t_grad=1e-7, eps=1e-8, max_species_fd=None):
    print(f"\n{'='*20} Validating Gradients: {name} {'='*20}")
    print(f"  Configuration: t_grad={t_grad:.1e}, eps={eps:.1e}")
    
    # 1. Setup
    mech = load_mechanism(yaml_file)
    net = ReactorNet(mech)
    
    sol_ct = ct.Solution(yaml_file)
    sol_ct.TPX = T0, P0, X0
    
    # Use a non-zero baseline state to avoid Cantera clipping at Y=0 during FD
    Y_base = sol_ct.Y + 1e-4
    sol_ct.Y = Y_base / Y_base.sum()
    Y0 = jnp.array(sol_ct.Y)
    
    @jax.jit
    def get_final_T(y0):
        # We must re-normalize y0 within the function to match Cantera's 
        # interpretation of mass fraction perturbations (dy_i affects all xj).
        y_norm = y0 / jnp.sum(y0)
        # Use explicit solver for tiny t_grad to avoid singular Jacobian in AD
        import diffrax
        res = net.advance(T0, P0, y_norm, t_grad, rtol=1e-10, atol=1e-14, solver=diffrax.Tsit5())
        return res.ys[-1, 0]
    
    # 2. Jantera JAX AD Gradient
    print(f"  Calculating Jantera AD gradients...", end="", flush=True)
    start = time.perf_counter()
    grad_jt = jax.grad(get_final_T)(Y0)
    jax.block_until_ready(grad_jt)
    jt_time = time.perf_counter() - start
    print(f" Done ({jt_time:.3f}s)")
    
    # 3. Cantera Finite Difference
    n_sp = len(sol_ct.Y)
    if max_species_fd is None:
        species_to_test = range(n_sp)
    else:
        # Sensitivities are usually highest for some species, pick a subset for speed if mechanism is large
        # We'll use the ones Jantera says are high
        top_indices = np.argsort(np.abs(grad_jt))[-max_species_fd:][::-1]
        species_to_test = sorted(top_indices)
        
    print(f"  Calculating Cantera FD gradients for {len(species_to_test)}/{n_sp} species...", end="", flush=True)
    grad_ct_fd = np.zeros(n_sp)
    grad_jt_fd = np.zeros(n_sp)
    start = time.perf_counter()
    
    for i in species_to_test:
        # 1. Cantera FD
        # Plus
        Y_plus = Y_base.copy()
        Y_plus[i] += eps
        sol_ct.TPY = T0, P0, Y_plus # Normalizes
        reac = ct.IdealGasConstPressureReactor(sol_ct, clone=False)
        sim = ct.ReactorNet([reac])
        sim.rtol = 1e-13; sim.atol = 1e-18 # Very tight for FD
        sim.advance(t_grad)
        T_plus_ct = sol_ct.T
        
        # Minus
        Y_minus = Y_base.copy()
        Y_minus[i] -= eps
        sol_ct.TPY = T0, P0, Y_minus
        reac = ct.IdealGasConstPressureReactor(sol_ct, clone=False)
        sim = ct.ReactorNet([reac])
        sim.rtol = 1e-13; sim.atol = 1e-18
        sim.advance(t_grad)
        T_minus_ct = sol_ct.T
        
        grad_ct_fd[i] = (T_plus_ct - T_minus_ct) / (2 * eps)
        
        # 2. Jantera FD (as a sanity check for AD)
        T_plus_jt = float(get_final_T(jnp.array(Y_plus)))
        T_minus_jt = float(get_final_T(jnp.array(Y_minus)))
        grad_jt_fd[i] = (T_plus_jt - T_minus_jt) / (2 * eps)
        
    ct_time = time.perf_counter() - start
    print(f" Done ({ct_time:.3f}s)")
    
    # 4. Analysis
    # Pick top 10 species by sensitivity for the table
    compare_idx = np.argsort(np.abs(grad_jt))[-10:][::-1]
    
    table_rows = []
    for i in compare_idx:
        sp_name = sol_ct.species_names[i]
        val_jt = float(grad_jt[i])
        val_ct = grad_ct_fd[i]
        
        abs_diff = abs(val_jt - val_ct)
        # Relative diff (careful with small numbers)
        rel_diff = abs_diff / (max(abs(val_ct), 1e-10))
        
        table_rows.append([sp_name, f"{val_jt:+.4e}", f"{val_ct:+.4e}", f"{abs_diff:.2e}", f"{rel_diff*100:.4f}%"])
    
    print("\nTop 10 species by temperature sensitivity (dT_final / dY_i):")
    print(tabulate(table_rows, headers=["Species", "Jantera (AD)", "Cantera (FD)", "Abs Diff", "Rel Diff"], tablefmt="grid"))
    
    # Overall statistics for tested species
    test_jt = np.array(grad_jt)[species_to_test]
    test_ct = grad_ct_fd[species_to_test]
    max_rel_err = np.max(np.abs(test_jt - test_ct) / (np.abs(test_ct) + 1e-10))
    
    print(f"\nSummary for {name}:")
    print(f"  Max relative error: {max_rel_err*100:.4f}%")
    print(f"  Gradient calculation speedup (AD vs FD_loop): {ct_time / (jt_time + 1e-10):.1f}x")
    
    return max_rel_err

def main():
    # 1. GRI-30
    # Condition where chemistry is active
    gri_cond = (1500.0, 101325.0, "CH4:1, O2:2, N2:7.52")
    err_gri = compare_gradients("GRI-30", "gri30.yaml", *gri_cond, t_grad=1e-7, eps=1e-8)
    
    # 2. JP-10 (Large mechanism, only test top 15 species for speed)
    jp10_cond = (1500.0, 101325.0, "C10H16:1, O2:14, N2:52.64")
    err_jp10 = compare_gradients("JP-10", "jp10.yaml", *jp10_cond, max_species_fd=15, t_grad=1e-7, eps=1e-8)
    
    print("\n" + "="*50)
    print("FINAL GRADIENT VALIDATION RESULT")
    print("="*50)
    print(f"GRI-30 Max Rel Error: {err_gri*100:.4f}%")
    print(f"JP-10 Max Rel Error:  {err_jp10*100:.4f}%")
    
    if err_gri < 1e-2 and err_jp10 < 1e-2:
        print("\n[PASS] GRADIENT VALIDATION PASSED (Threshold < 1%)")
    else:
        print("\n[FAIL] GRADIENT VALIDATION FAILED")

if __name__ == "__main__":
    main()
