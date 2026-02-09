# Reactor Module

The `reactor` module provides ODE integration for 0D ideal gas reactors, supporting both constant pressure and constant volume configurations.

## Core Features

- **JAX Integration**: Fully differentiable reactor network simulations using `diffrax` or custom solvers.
- **Kvaerno5 Solver**: Default implicit Runge-Kutta solver (Order 5) optimized for stiff chemical kinetics.
- **Custom BDF Solver**: Experimental variable-order Backward Differentiation Formula solver for extreme stiffness.
- **Enthalpy Reuse**: Optimized RHS evaluation reuses thermodynamic properties to minimize computational overhead.

## Solvers

### Kvaerno5 (Default)
Tuned for maximum robustness and efficiency:
- **Linear Solver**: Explicit LU decomposition (`lineax.LU`) to avoid auto-detection overhead.
- **Jacobian Reuse**: Aggressive reuse starategy (`kappa=0.5`).
- **Step Controller**: PID controller with standard tolerances ($rtol=10^{-7}, atol=10^{-10}$).

### BDF (Experimental)
A custom JAX implementation of the CVODE algorithm:
- Variable order (1-5)
- Nordsieck history array
- Analytical Jacobian caching

## Usage

```python
from jantera import ReactorNet

net = ReactorNet(mech)
# Advance reactor state from t=0 to t_end
sol = net.advance(T0, P, Y0, t_end)
```

### Source Code
- [src/jantera/reactor.py](../../src/jantera/reactor.py)
