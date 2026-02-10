# Changelog

All notable changes to this project will be documented in this file.

## [0.3.0] - 2026-02-09

### Major Changes ⚠️
- **Rename**: Renamed project from `jantera` to `canterax`. Updated all import paths and package structure.
- **Validation**: Expanded validation suite with Ethane, Propane, and Hydrogen mechanisms.
- **Testing**: Added broader temperature and pressure ranges for static validation parity checks.

## [0.2.0] - 2026-02-08

### Performance 🚀
- **Sensitivity Analysis**: Achieved **10x speedup** for GRI-30 sensitivity analysis (32ms vs ~360ms) by implementing log-space Rate of Progress (ROP) calculation, which significantly accelerates JAX revers-mode differentiation.
- **Reactor Advancement**: Achieved **~2x speedup** for stiff reactor integration (GRI-30: 109ms, JP-10: 172ms) through hyperparameter tuning of `diffrax.Kvaerno5`.
- **Enthalpy Reuse**: Optimized `reactor_rhs` to reuse species enthalpies computed during rate evaluation, reducing per-step computational cost by **22%**.
- **Solver Optimization**: Switched default solver config to `Kvaerno5(scan_kind="lax", root_finder=VeryChord(kappa=0.5, linear_solver=lineax.LU()))`.

### Features ✨
- **Equilibrium Solver**: Implemented a robust Gibbs minimization solver with element-potential selection and weighted basis initialization. Matches Cantera results to within machine precision ($10^{-11}$).
- **Thermodynamics**: Enhanced `thermo` module to return efficient species enthalpy arrays (`h_mol`) for internal reuse.
- **Validation**: Added a comprehensive validation suite `tests/test_validation_suite.py` generating parity plots for static properties, equilibrium, and reactor trajectories.

### Fixes 🐛
- **Bugs**: Fixed unit conversion consistency in reactor energy conservation equation.
- **Stability**: Fixed `AutoLinearSolver` overhead by enforcing explicit LU decomposition.
- **API**: Fixed partial unpacking issues in `kinetics.py` and test suite.

## [0.1.0] - 2026-02-07

### Initial Release
- Core functionality: NASA-7 thermodynamics, Arrhenius kinetics, constant-pressure reactor.
- Initial support for `diffrax` ODE solvers.
- Cantera YAML mechanism loader.
