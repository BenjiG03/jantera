import cantera as ct
import time

def check_cantera_steps():
    yaml_path = "jp10.yaml"
    T0, P0 = 1500.0, 101325.0
    comp = "C10H16:1, O2:14, N2:52.64"
    
    sol = ct.Solution(yaml_path)
    sol.TPX = T0, P0, comp
    
    # Use explicit reactor net to try and get stats
    r = ct.IdealGasConstPressureReactor(sol)
    net = ct.ReactorNet([r])
    
    # Run to 1ms
    t_end = 1e-3
    t_now = 0.0
    
    start = time.time()
    # net.advance(t_end) 
    # To count steps manually roughly or get stats, let's step
    net.advance(t_end)
    end = time.time()
    
    print(f"Cantera Time: {end-start:.5f}s")
    
    # Try different ways to get stats depending on version
    try:
        # Newer Cantera
        # This might be 'n_steps' or similar
        # .get_solver_stats() returns a dict
        print("Stats from get_solver_stats():", net.get_solver_stats())
        # print("Step count:", net.step_count) # Sometimes available? 
    except Exception as e:
        print(f"Could not get stats: {e}") 
        
    # Another way: verify order of magnitude using step()
    # Reset
    sol.TPX = T0, P0, comp
    r = ct.IdealGasConstPressureReactor(sol)
    net = ct.ReactorNet([r])
    
    t_now = 0.0
    steps = 0
    start = time.time()
    while t_now < t_end:
        t_now = net.step()
        steps += 1
    end = time.time()
    print(f"Cantera (stepped) Time: {end-start:.5f}s, Steps: {steps}")

if __name__ == "__main__":
    check_cantera_steps()
