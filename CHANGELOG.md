# Changelog

All notable changes to Jantera will be documented in this file.

## [0.3.0] - 2026-02-08

### Added
- **Performance Benchmarking**: Comprehensive evaluation against Cantera 3.2.0 for Equilibrium, Reactor Advancement, and Sensitivity Analysis.
- **Separated Timing Metrics**: JIT compilation time is now strictly decoupled from warm execution time, providing a more accurate view of runtime performance.
- **Sensitivity Supremacy**: Demonstrated that Jantera's AD-based sensitivity analysis is **4-6x faster** than Cantera's native solver once JIT-compiled (e.g., 61ms vs 361ms for GRI-30).
- **Sparsity Architecture**: Documented "dense-sparse" approach using JAX `scatter`/`gather` for stoichiometry, ensuring linear scaling with mechanism size without dense overhead.

### Fixed
- **Equilibrium Convergence**: Switched to `LevenbergMarquardt` solver and increased `max_steps` to 2000. Resolves convergence failures for large mechanisms like JP-10 (achieves < 1e-14 error).
- **Reactor Stability for Stiff Kinetics**: Reduced initial step size (`dt0`) to `1e-12` in `ReactorNet`, resolving NaN gradients during adjoint solves for stiff JP-10 kinetics.
- **Misspelling**: Corrected `IdealGasConstPressureReactor` attribute name in test suite.

### Documentation
- Updated `README.md` and Wiki with detailed parity plots and performance tables.
- Added comprehensive `walkthrough.md` for performance breakdown.

## [0.2.0] - 2026-02-07

### Added
- **Gradient Validation**: Comprehensive AD vs FD gradient validation using `jax.grad`
  - GRI-30: 0.55% relative error vs Cantera finite differences
- **Numerical Safety**: Concentration clipping and falloff masking for stable gradients
- **Test Suite**: New `tests/test_gradients.py` for sensitivity validation

### Fixed
- **Unit System**: Corrected mol → kmol unit conversion to match Cantera
- **Irreversible Reactions**: Masked reverse rates (`kr = 0`) for non-reversible reactions
- **Three-body Efficiency**: Fixed `default_efficiency` defaulting to 0 instead of 1
- **Falloff Reactions**: Safe dummy values in Troe calculations for non-falloff reactions

### Known Issues
- **JP-10 Gradients**: NaN gradients occur during adjoint ODE solve for stiff JP-10 kinetics
  - Root cause: Explicit `Tsit5` solver's backward pass is numerically unstable for stiff systems
  - Workaround: Use implicit `Kvaerno5` solver (no gradient support yet) or reduce integration time
  - Status: Under investigation

### Performance (GRI-30 @ 1500K, t=100μs)
| Metric | Value |
|--------|-------|
| JIT Compile | 5.2 s |
| Warm Execution | 152 ms |
| Cantera | 2.9 ms |
| Serial Speedup | 0.02x (52x slower) |

> **Note**: Jantera's strength is batched/parallelized simulations via `jax.vmap`, not single-reactor performance.

## [0.1.0] - 2026-02-06

### Added
- Initial release
- NASA-7 thermodynamics
- Arrhenius, three-body, Troe falloff kinetics
- `IdealGasConstPressureReactor` via diffrax
- Gibbs equilibrium solver
- YAML mechanism loader (wraps Cantera)
