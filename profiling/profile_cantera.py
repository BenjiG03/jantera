import time
import cantera as ct
import numpy as np

if __name__ == "__main__":
    print("Loading GRI-30 in Cantera...")
    # Cantera solution
    gas = ct.Solution('gri30.yaml')
    gas.TPX = 1500.0, 101325.0, {'CH4': 1.0, 'O2': 2.0, 'N2': 7.52}
    
    print(f"Loaded mechanism with {gas.n_species} species and {gas.n_reactions} reactions.")
    
    reactor = ct.IdealGasConstPressureReactor(gas)
    sim = ct.ReactorNet([reactor])
    
    N = 10000 
    print(f"Running benchmarks (N={N})...")
    
    # kf
    start = time.time()
    for _ in range(N):
        kf = gas.forward_rate_constants
    end = time.time()
    print(f"forward_rate_constants: {(end-start)/N*1e6:.2f} us")
    
    # Kc
    start = time.time()
    for _ in range(N):
        kc = gas.equilibrium_constants
    end = time.time()
    print(f"equilibrium_constants: {(end-start)/N*1e6:.2f} us")
    
    # wdot
    start = time.time()
    for _ in range(N):
        wd = gas.net_production_rates
    end = time.time()
    print(f"net_production_rates: {(end-start)/N*1e6:.2f} us")
    
    # RHS approximate (properties + wdot)
    start = time.time()
    for _ in range(N):
        # Emulate RHS calculation components
        wdot = gas.net_production_rates
        h = gas.partial_molar_enthalpies
        cp = gas.cp_mass
        rho = gas.density
    end = time.time()
    print(f"RHS_approx: {(end-start)/N*1e6:.2f} us")

    # Measure full step
    sim.set_initial_time(0.0)
    # Force initialization
    sim.advance(1e-12)
    
    start = time.time()
    t = 1e-12
    dt = 1e-9 # Small step
    for _ in range(N):
        sim.advance(t + dt)
        t += dt
    end = time.time()
    print(f"advance(small_dt): {(end-start)/N*1e6:.2f} us")
