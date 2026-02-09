# Handoff: Optimization of Equilibrium and Reactor Advance

## Current Status (v0.3.0)

Jantera has achieved feature parity with Cantera for basic thermodynamics, kinetics, reactor simulation, and equilibrium. It excels in **sensitivity analysis** (4-6x faster than Cantera) but lags in **single-reactor simulation speed** and **equilibrium efficiency** for large mechanisms.

### 1. Equilibrium Solver (`jantera.equilibrate`)
- **Status**: Robust but slow for large mechanisms.
- **Current Implementation**: uses `optimistix.least_squares` with `LevenbergMarquardt`.
- **Performance**:
  - GRI-30: ~278ms (Warm), 33 steps.
  - JP-10: ~1784ms (Warm), 536 steps.
  - Cantera: <1ms for both.
- **Issues**:
  - The Element Potential method formulation in `equilibrate.py` requires many iterations to converge for stiff/large molecules like JP-10.
  - `max_steps` had to be increased to 2000 to ensure convergence.
- **Next Steps for Optimization**:
  - **Initialization**: Improve the initial guess for Lagrange multipliers to reduce step count.
  - **Custom Solver**: Implement a specialized dampening or line-search strategy tailored for chemical equilibrium (KKT systems) rather than generic Levenberg-Marquardt.
  - **Analytical Jacobian**: Ensure the exact Jacobian of the KKT system is being used efficiently.

### 2. Reactor Advancement (`jantera.reactor.ReactorNet`)
- **Status**: Stable and differentiable, but slower than Cantera for single-reactor trajectories.
- **Current Implementation**: Uses `diffrax.diffeqsolve` with `Kvaerno5` (Implicit Runge-Kutta).
- **Performance**:
  - GRI-30 (1ms): ~140ms vs Cantera ~6ms.
  - JP-10 (1ms): ~200ms vs Cantera ~38ms.
- **Issues**:
  - **Step Count**: `Kvaerno5` takes significantly more steps (e.g., 314 vs ~100) than Cantera's CVODE to maintain stability.
  - **Overhead**: JAX/Diffrax has higher per-step overhead than C++ CVODE.
- **Next Steps for Optimization**:
  - **Custom BDF Solver**: Implement a JAX-native Backward Differentiation Formula (BDF) solver (similar to CVODE). This is the gold standard for stiff chemical kinetics and should reduce step counts by 3-5x.
  - **Sparse Jacobian**: For large mechanisms (hundreds of species), the dense Jacobian factorization ($O(N^3)$) in the implicit solver becomes the bottleneck. Implement sparse Jacobian evaluation and linear solves.
  - **Preconditioners**: Explore preconditioners for the linear solver to speed up Newton iterations within the implicit integrator.

### 3. General
- **Sparsity**: The "dense-sparse" approach in `kinetics.py` works well for evaluating rates ($O(N)$), but true sparse matrix operations (JAX `BCOO`) are needed for the Jacobian in the ODE solver.
- **Validation**: Continue using `debug_grad_and_speed.py` and `test_validation_suite.py` to benchmark improvements.

## Artifacts
- **Validation**: `docs/wiki/Validation.md` contains the latest baselines.
- **Code**: `src/jantera/equilibrate.py`, `src/jantera/reactor.py`, `src/jantera/kinetics.py`.
