import sys
import os
import time
import jax
import jax.numpy as jnp
import numpy as np

# Add src to pythonpath
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from canterax.loader import load_mechanism
from canterax.kinetics import compute_kf, compute_Kc, compute_wdot
from canterax.thermo import compute_mixture_props
from canterax.reactor import reactor_rhs

def benchmark(func, args, n_runs=1000):
    # Warm up / JIT
    jax.block_until_ready(func(*args))
    
    start = time.perf_counter()
    for _ in range(n_runs):
        res = func(*args)
        jax.block_until_ready(res)
    end = time.perf_counter()
    return (end - start) / n_runs

if __name__ == "__main__":
    # Load GRI-30
    print("Loading GRI-30...")
    mech = load_mechanism("gri30.yaml")
    print(f"Loaded mechanism with {mech.n_species} species and {mech.n_reactions} reactions.")
    
    # Set state
    T_sample = 1500.0
    P_sample = 101325.0
    Y_dict = {'CH4': 1.0, 'O2': 2.0, 'N2': 7.52}
    Y_sample_mole = jnp.zeros(mech.n_species)
    for s, v in Y_dict.items():
        if s in mech.species_names:
            idx = mech.species_names.index(s)
            Y_sample_mole = Y_sample_mole.at[idx].set(v)
    Y_sample_mole = Y_sample_mole / jnp.sum(Y_sample_mole)
    
    # Calculate mass fractions
    mw_mix = 1.0 / jnp.sum(Y_sample_mole / mech.mol_weights)
    Y_sample = Y_sample_mole * mech.mol_weights / mw_mix
    
    state = jnp.concatenate([jnp.array([T_sample]), Y_sample])
    
    # Batched performance (x100)
    batch_size = 100
    T_batch = jnp.full(batch_size, T_sample)
    P_batch = jnp.full(batch_size, P_sample)
    Y_batch = jnp.repeat(Y_sample[None, :], batch_size, axis=0)
    
    vmap_wdot = jax.vmap(compute_wdot, in_axes=(0, 0, 0, None, None))
    
    # Warm up
    vmap_wdot(T_batch, P_batch, Y_batch, mech, False)[0].block_until_ready()
    
    start = time.perf_counter()
    for _ in range(10): # Fewer runs for large batch
        vmap_wdot(T_batch, P_batch, Y_batch, mech, False)[0].block_until_ready()
    end = time.perf_counter()
    t_vmap = (end - start) / 10 / batch_size
    
    print(f"Batched (x100) wdot: {t_vmap*1e6:.2f} us per sample")
    
    # Needs conc for kf benchmark below
    _, _, rho = compute_mixture_props(T_sample, P_sample, Y_sample, mech)
    conc = rho * Y_sample / mech.mol_weights

    print("Running benchmarks (N=1000)...")
    
    # 1. Primary "Dense-Sparse" Path
    t_kf = benchmark(compute_kf, (T_sample, conc, mech, False))
    t_Kc = benchmark(compute_Kc, (T_sample, mech))
    t_wdot = benchmark(compute_wdot, (T_sample, P_sample, Y_sample, mech, False))
    t_rhs = benchmark(reactor_rhs, (0.0, state, (P_sample, mech)))
    
    # 2. Experimental Sparse Path
    try:
        t_kf_sparse = benchmark(compute_kf, (T_sample, conc, mech, True))
        t_wdot_sparse = benchmark(compute_wdot, (T_sample, P_sample, Y_sample, mech, True))
    except Exception as e:
        print(f"Experimental sparse path failed: {e}")
        t_kf_sparse, t_wdot_sparse = 0.0, 0.0

    print(f"\n{'='*60}")
    print(f"{'Function':<20} | {'Primary (us)':<15} | {'Exp. Sparse (us)':<18}")
    print(f"{'-'*20}-+-{'-'*15}-+-{'-'*18}")
    print(f"{'compute_kf':<20} | {t_kf*1e6:>15.2f} | {t_kf_sparse*1e6:>18.2f}")
    print(f"{'compute_Kc':<20} | {t_Kc*1e6:>15.2f} | {'N/A':>18}")
    print(f"{'compute_wdot':<20} | {t_wdot*1e6:>15.2f} | {t_wdot_sparse*1e6:>18.2f}")
    print(f"{'reactor_rhs':<20} | {t_rhs*1e6:>15.2f} | {'N/A':>18}")
    print(f"{'='*60}")

    # XLA HLO Inspection
    print("\n--- XLA HLO Inspection (compute_wdot) ---")
    lowered = jax.jit(compute_wdot, static_argnums=(4,)).lower(T_sample, P_sample, Y_sample, mech, False)
    hlo = lowered.as_text()
    
    if "custom-call" in hlo:
        print("[*] Found custom calls (potential kernel fusion/optimized ops)")
    if "gather" in hlo:
        print("[*] Found gather operations (expected for dense-sparse approach)")
    
    # Simple check for dot product dimensions
    dot_count = hlo.count("dot(")
    print(f"[*] Total dot() calls in HLO: {dot_count}")
    if dot_count < 5: # Arbitrary small number
         print("[*] Dot product count is low, suggesting successful sparsification.")

    # Profiler Trace
    # To run: set JAX_PROFILER_DIR and uncomment
    # import jax.profiler
    # profiler_dir = os.path.join(os.path.dirname(__file__), "trace")
    # os.makedirs(profiler_dir, exist_ok=True)
    # print(f"Capturing trace to {profiler_dir}...")
    # jax.profiler.start_trace(profiler_dir)
    # for _ in range(10):
    #     compute_wdot(T_sample, P_sample, Y_sample, mech, False).block_until_ready()
    # jax.profiler.stop_trace()
