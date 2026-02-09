import time
import cantera as ct
import numpy as np

if __name__ == "__main__":
    print("Loading GRI-30 in Cantera...")
    # Cantera solution
    gas = ct.Solution('gri30.yaml')
    T0, P0, X0_str = 1500.0, 101325.0, {'CH4': 1.0, 'O2': 2.0, 'N2': 7.52}
    gas.TPX = T0, P0, X0_str
    
    print(f"Loaded mechanism with {gas.n_species} species and {gas.n_reactions} reactions.")
    
    print("\n" + "="*30 + " EQUILIBRATE PROFILING " + "="*30)
    # Warmup
    gas.equilibrate('TP')
    gas.TPX = T0, P0, X0_str
    
    # Measure
    N_eq = 1000
    start = time.perf_counter()
    for _ in range(N_eq):
        gas.TPX = T0, P0, X0_str
        gas.equilibrate('TP')
    end = time.perf_counter()
    print(f"Equilibrate Time (Avg of {N_eq}): {(end-start)/N_eq*1000:.4f} ms")
    
    
    print("\n" + "="*30 + " REACTOR ADVANCE PROFILING " + "="*30)
    gas.TPX = T0, P0, X0_str
    reactor = ct.IdealGasConstPressureReactor(gas)
    sim = ct.ReactorNet([reactor])
    sim.rtol = 1e-8
    sim.atol = 1e-12
    
    t_end = 1e-3
    
    # Warmup
    sim.advance(t_end)
    
    # Measure
    N_adv = 100
    total_time = 0.0
    total_steps = 0
    
    for _ in range(N_adv):
        gas.TPX = T0, P0, X0_str
        reactor = ct.IdealGasConstPressureReactor(gas)
        sim = ct.ReactorNet([reactor])
        sim.rtol = 1e-8
        sim.atol = 1e-12
        
        start = time.perf_counter()
        sim.advance(t_end)
        total_time += time.perf_counter() - start
        
        # Determine steps (Cantera doesn't expose this easily for a single advance call if re-created)
        # But we can getstats from the last one
    
    try:
        # Get stats from the last run
        total_steps = sim.get_solver_stats()['n_steps']
    except:
        total_steps = -1

    avg_time = total_time / N_adv
    print(f"Advance Time (Avg of {N_adv}): {avg_time*1000:.4f} ms")
    print(f"Advance Steps: {total_steps}")
    print(f"Time per Step: {avg_time*1000/total_steps:.4f} ms/step")

    print("\n" + "="*30 + " COMPONENT PROFILING " + "="*30)
    N = 10000 
    
    # kf
    start = time.perf_counter()
    for _ in range(N):
        kf = gas.forward_rate_constants
    end = time.perf_counter()
    print(f"forward_rate_constants: {(end-start)/N*1e6:.2f} us")
    
    # Kc
    start = time.perf_counter()
    for _ in range(N):
        kc = gas.equilibrium_constants
    end = time.perf_counter()
    print(f"equilibrium_constants: {(end-start)/N*1e6:.2f} us")
    
    # wdot
    start = time.perf_counter()
    for _ in range(N):
        wd = gas.net_production_rates
    end = time.perf_counter()
    print(f"net_production_rates: {(end-start)/N*1e6:.2f} us")
