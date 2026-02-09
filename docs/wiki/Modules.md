# Module Reference

## Overview

Jantera consists of 8 core modules:

| Module | Purpose |
|--------|---------|
| `constants.py` | Physical constants |
| `mech_data.py` | Mechanism data structure |
| `loader.py` | YAML mechanism parser |
| `thermo.py` | NASA-7 thermodynamics |
| `kinetics.py` | Reaction rate calculations |
| `reactor.py` | ODE integration |
| `solution.py` | User-facing API |
| `equilibrate.py` | Gibbs minimization |

---

## constants.py

Defines fundamental physical constants:

```python
R_GAS = 8.31446261815324  # J/(mol·K) - Universal gas constant
```

---

## mech_data.py

`MechData` is an Equinox module storing mechanism arrays:

```python
class MechData(eqx.Module):
    n_species: int              # Number of species
    n_reactions: int            # Number of reactions
    mol_weights: jnp.ndarray    # [n_species] - kg/mol
    
    # Stoichiometry
    nu_reactants: jnp.ndarray   # [n_reactions, n_species]
    nu_products: jnp.ndarray    # [n_reactions, n_species]
    
    # Arrhenius parameters
    A: jnp.ndarray              # [n_reactions] - Pre-exponential
    b: jnp.ndarray              # [n_reactions] - Temperature exponent
    Ea: jnp.ndarray             # [n_reactions] - Activation energy (J/mol)
    
    # NASA-7 coefficients
    nasa_low: jnp.ndarray       # [n_species, 7]
    nasa_high: jnp.ndarray      # [n_species, 7]
    nasa_T_mid: jnp.ndarray     # [n_species]
    
    # Three-body and falloff
    is_three_body: jnp.ndarray  # [n_reactions] - Boolean mask
    third_body_efficiencies: jnp.ndarray  # [n_reactions, n_species]
    is_falloff: jnp.ndarray     # [n_reactions]
    falloff_params: jnp.ndarray # [n_reactions, 10] - Troe parameters
```

---

## loader.py

`load_mechanism(yaml_file)` parses a Cantera YAML file and returns a `MechData` object.

**Key Steps**:
1. Load mechanism via `ct.Solution(yaml_file)`
2. Extract stoichiometry matrices
3. Extract Arrhenius parameters (converting units to SI)
4. Extract NASA-7 polynomial coefficients
5. Build three-body efficiency matrices
6. Package into `MechData`

---

## thermo.py

Implements NASA-7 polynomial thermodynamics:

### Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `get_cp_R` | `(T, low, high, T_mid) -> [n_species]` | Cp/R |
| `get_h_RT` | `(T, low, high, T_mid) -> [n_species]` | H/(RT) |
| `get_s_R` | `(T, low, high, T_mid) -> [n_species]` | S/R |
| `compute_mixture_props` | `(T, P, Y, mech) -> (cp_mass, h_mass, s_mass, rho)` | Mixture properties |

### NASA-7 Form

$$C_p/R = a_1 + a_2 T + a_3 T^2 + a_4 T^3 + a_5 T^4$$

$$H/(RT) = a_1 + \frac{a_2}{2}T + \frac{a_3}{3}T^2 + \frac{a_4}{4}T^3 + \frac{a_5}{5}T^4 + \frac{a_6}{T}$$

$$S/R = a_1 \ln T + a_2 T + \frac{a_3}{2}T^2 + \frac{a_4}{3}T^3 + \frac{a_5}{4}T^4 + a_7$$

---

## kinetics.py

Implements reaction rate calculations.

### Main Function

```python
def compute_wdot(T, P, Y, mech) -> (wdot, h_mass, cp_mass, rho, h_mol)
```

Returns:
- `wdot`: Net production rates [mol/m³/s]
- `h_mass`: Mixture enthalpy [J/kg]
- `cp_mass`: Mixture heat capacity [J/kg/K]
- `rho`: Density [kg/m³]
- `h_mol`: Species enthalpies [J/kmol] (optimization artifact)

### Rate Calculation Steps

1. **Forward Rates**: Arrhenius form $k_f = A T^b \exp(-E_a/RT)$
2. **Equilibrium Constants**: From thermodynamics $K_c = K_p (RT/P_{atm})^{\Delta\nu}$
3. **Reverse Rates**: $k_r = k_f / K_c$
4. **Three-Body**: $[M] = \sum \alpha_i [X_i]$
5. **Troe Falloff**: Lindemann-Hinshelwood with Troe blending
6. **Net Rates**: $\dot\omega_i = \sum_j \nu_{ij} q_j$

---

## reactor.py

Provides ODE integration for reactor simulations.

### ReactorNet Class

```python
class ReactorNet(eqx.Module):
    mech: MechData
    
    def advance(self, T0, P0, Y0, t_end, rtol=1e-7, atol=1e-10, solver=None):
        # Returns diffrax solution object
```

### ODE System

State vector: $\mathbf{y} = [T, Y_1, Y_2, ..., Y_n]$

$$\frac{dY_i}{dt} = \frac{\dot\omega_i M_i}{\rho}$$

$$\frac{dT}{dt} = -\frac{\sum h_i \dot\omega_i}{\rho c_p}$$

---

## solution.py

User-facing API wrapper with Cantera-like interface.

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `T` | float | Temperature [K] |
| `P` | float | Pressure [Pa] |
| `Y` | array | Mass fractions |
| `X` | array | Mole fractions |
| `density` | float | Density [kg/m³] |
| `cp_mass` | float | Cp [J/kg/K] |
| `enthalpy_mass` | float | H [J/kg] |

### State Setters

```python
gas.TP = 1500, 101325
gas.TPY = 1500, 101325, Y_array
gas.TPX = 1500, 101325, "CH4:1, O2:2"
```

---

## equilibrate.py

Gibbs minimization equilibrium solver using the Element Potential Method.

### Function

```python
def equilibrate(solution, XY='TP'):
    # Modifies solution.Y in-place to equilibrium composition
```

### Algorithm

1. Formulate Lagrangian: $L = G - \sum \lambda_e (n_e - \sum a_{ie} n_i)$
2. Solve KKT conditions using `optimistix.least_squares`
3. Iterate until element conservation and Gibbs minimum are satisfied
