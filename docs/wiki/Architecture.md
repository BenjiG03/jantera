# Architecture

## Overview

Jantera is designed around three core principles:

1. **Pure Functions**: All computational kernels are stateless JAX-traced functions
2. **Immutable Data Structures**: Mechanism data is stored in Equinox modules
3. **Composability**: Each module is independent and can be used standalone

## System Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                        User Code                            │
│  gas = Solution("gri30.yaml")                               │
│  gas.TPX = 1500, 101325, "CH4:1,O2:2"                       │
│  reactor.advance(...)                                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     Solution Class                          │
│  - Cantera-like API wrapper                                 │
│  - Manages state (T, P, Y)                                  │
│  - Delegates to pure functions                              │
└─────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│    thermo.py     │ │   kinetics.py    │ │   reactor.py     │
│  - NASA-7 polys  │ │  - Arrhenius     │ │  - ODE RHS       │
│  - get_cp_R()    │ │  - Three-body    │ │  - diffrax       │
│  - get_h_RT()    │ │  - Troe falloff  │ │  - Kvaerno5      │
│  - get_s_R()     │ │  - compute_wdot  │ │  - advance()     │
└──────────────────┘ └──────────────────┘ └──────────────────┘
          │                   │                   │
          └───────────────────┼───────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     MechData (Equinox)                      │
│  - Static JAX arrays (mol_weights, stoich, nasa_coeffs)     │
│  - Loaded once from YAML via Cantera                        │
│  - Immutable, JIT-friendly                                  │
└─────────────────────────────────────────────────────────────┘
```

## Data Flow

1. **Mechanism Loading**: `loader.py` uses Cantera to parse YAML, extracts arrays into `MechData`
2. **State Management**: `Solution` class holds mutable (T, P, Y); delegates to pure functions
3. **Computation**: Pure functions take `(T, P, Y, mech)` and return properties
4. **Integration**: `ReactorNet.advance()` calls `diffeqsolve` with the reaction RHS

## Key Design Decisions

### Why Equinox?
Equinox provides PyTree-compatible dataclasses that work with JAX transformations. `MechData` is an `eqx.Module` containing static arrays, ensuring proper tracing during JIT.

### Why Diffrax?
Diffrax provides JAX-native ODE solvers with:
- Implicit solvers (`Kvaerno5`) for stiff chemistry
- Adjoint methods for memory-efficient gradients
- Native `vmap` support for batched simulations

### Why Pure Functions?
JAX's functional paradigm requires pure functions for tracing. All core computations (`compute_wdot`, `get_h_RT`) are side-effect-free and take all inputs as arguments.
