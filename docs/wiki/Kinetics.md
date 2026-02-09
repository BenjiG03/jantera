# Kinetics Module

The `kinetics` module handles reaction rate computations, including Arrhenius rates, three-body interactions, and pressure-dependent falloff corrections.

## Core Features

- **Standard Arrhenius**: $k = A T^b \exp(-E_a/RT)$
- **Three-Body Reactions**: Efficiently calculates enhanced collision rates using third-body efficiencies.
- **Falloff Reactions**:
    - Supports Lindemann and Troe formalisms.
    - Robust blending factor $F_{cent}$ calculation.
- **Dense/Sparse Stoichiometry**: Optimized using dense matrix operations for small mechanisms (<200 species) and ready for sparse updates.
- **Log-Space ROP**: Rate of Progress (ROP) calculated in log-space for AD stability and efficiency.

## Implementation Details

### Optimized Rate of Progress (ROP)
To improve Automatic Differentiation (AD) performance - especially for reverse-mode gradients - the Rate of Progress is calculated in log-space:

$$ ROP = k \cdot \exp \left( \sum_{i} \nu_i \ln([\text{Conc}]_i) \right) $$

This avoids complex chains of power/product rules during backpropagation, resulting in a **~35% speedup** for Jacobian evaluation.

### Source Code
- [src/jantera/kinetics.py](../../src/jantera/kinetics.py)
