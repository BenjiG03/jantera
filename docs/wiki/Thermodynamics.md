# Thermodynamics Module

The `thermo` module implements high-performance thermodynamic property calculations using NASA-7 polynomials.

## Core Features

- **Vectorized NASA-7 Implementation**: Calculates properties ($C_p/R$, $H/RT$, $S/R$) for all species simultaneously using JAX vectorization.
- **JIT Compilation**: All functions are `@jax.jit` compiled for execution on CPU/GPU/TPU.
- **Mixture Properties**: Efficiently computes mass-weighted mixture enthalpy, heat capacity, and density.

## Implementation Details

### NASA-7 Polynomials
Standard NASA polynomials are used:
$$ \frac{C_p}{R} = a_0 + a_1 T + a_2 T^2 + a_3 T^3 + a_4 T^4 $$
$$ \frac{H}{RT} = a_0 + \frac{a_1 T}{2} + \frac{a_2 T^2}{3} + \frac{a_3 T^3}{4} + \frac{a_4 T^4}{5} + \frac{a_5}{T} $$
$$ \frac{S}{R} = a_0 \ln T + a_1 T + \frac{a_2 T^2}{2} + \frac{a_3 T^3}{3} + \frac{a_4 T^4}{4} + a_6 $$

### Efficient Branching
Temperature range switching ($T < T_{mid}$ vs $T > T_{mid}$) is handled via `jnp.where` masking to maintain JIT compatibility without control flow divergence.

### Source Code
- [src/jantera/thermo.py](../../src/jantera/thermo.py)
