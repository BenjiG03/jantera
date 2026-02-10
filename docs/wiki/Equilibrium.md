# Equilibrium Module

The `equilibrium` module solves for the chemical equilibrium composition of a mixture at specified (T, P) or (H, P) conditions by minimizing Gibbs Free Energy.

## Core Features

- **Constrained Optimization**: Minimizes Gibbs energy subject to elemental mass constraints using Lagrange multipliers.
- **Basis Selection**: Automatically selects optimal basis species to improve solver convergence.
- **Robust Initialization**: Uses element potentials derived from major species to provide a good initial guess.
- **Penalized Objective**: Incorporates penalty terms to handle non-negativity constraints effectively.

## Implementation Details

### Element Potentials
The solver uses element potentials ($\lambda_j$) as the primary variables. The non-linear system is solved using a Levenberg-Marquardt optimizer (`optimistix.LevenbergMarquardt`).

### Verification
Validated against Cantera's `equilibrate` function for GRI-30 and JP-10 mechanisms, achieving parity with robust convergence.

### Source Code
- [src/canterax/equilibrate.py](../../src/canterax/equilibrate.py)
