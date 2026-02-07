# Jantera: Handoff Document & Future Roadmap

This document summarizes the current state of **Jantera (v0.2.0)**, highlighting technical design decisions, known limitations ("sharp edges"), and specific goals for future sessions.

---

## 🏗️ Core Architecture & Design Decisions

### **1. Functional Paradigm (JAX-First)**
- **Pure Functions**: Core computational kernels (e.g., `compute_wdot`, `get_h_RT`) are stateless and take all inputs as arguments.
- **Immutable Mechanism Data**: `MechData` is implemented as an `equinox.Module`, allowing static arrays (stoichiometry, coefficients) to be traced efficiently by JIT without causing re-compilations.

### **2. Stiff ODE Integration**
- **Solver**: Uses `diffrax.Kvaerno5` (an ESDIRK method). This was chosen over explicit solvers which fail instantly for chemical kinetics.
- **Step Tuning**: Initial time-step (`dt0=1e-8`) and tight tolerances (`rtol=1e-8`, `atol=1e-12`) are critical for stability in hydrocarbon ignition cases.

### **3. Solution API Wrapper**
- Provides a mutable state (`Solution.T`, `Solution.P`, `Solution.Y`) to mimic Cantera's usability while delegating all math to the underlying JAX pure functions.

---

## ⚠️ Sharp Edges & Precision Issues

### **1. The JP-10 Parity Gap**
- **Observation**: For GRI-30 (Methane), we have near-perfect parity ($<0.02$ K). For JP-10 (High-T ignition), a $\sim 70$ K gap develops early in the trajectory.
- **Hypothesis**: The initial $dT/dt$ is extremely high ($\sim -10^7$ K/s). Minor differences in how the Jacobian is handled or how the stiff solver transitions between time-steps can lead to divergent trajectories in these "brittle" regions of phase space.
- **Assumed Cause**: Differing adaptive stepping heuristics between `diffrax` (Jantera) and `CVODE` (Cantera).

### **2. Initial dT/dt Divergence**
- Some mechanisms show a small discrepancy in the very first step of temperature evolution. This is likely due to the extreme sensitivity of the temperature RHS ($-\frac{\sum h_i \dot\omega_i}{\rho c_p}$) to the initial $wdot$ calculation.

---

## 🐌 Performance Bottlenecks

### **1. Serial vs. Batched**
- **Serial**: Jantera is $\sim 30\text{x}$ slower than Cantera for a single reactor simulation. This is expected as JAX is optimized for throughput, not low-latency serial execution.
- **Batched**: Jantera is $\sim 3\text{x}-7\text{x}$ *faster* than Cantera when simulating 100+ reactors simultaneously due to `jax.vmap`.

### **2. Serial Optimization Gaps**
- **Memory Allocations**: JAX tracing currently regenerates many intermediate arrays per step.
- **Kernel Fusion**: Higher-level fusion of the thermodynamics and kinetics kernels into a single XLA call could significantly reduce overhead.

---

## 🛠️ Implementation Gaps (Currently Unsupported)

- **Pressure Dependencies**: Only Lindemann and Troe falloff are implemented. PLOG or SRI blending are missing.
- **Non-Ideal Gases**: Only `IdealGas` models are supported.
- **Multi-Phase**: No support for surface chemistry or liquid phases.
- **Reverse Rates**: Currently strictly derived from Equilibrium Constants; direct specification of reverse Arrhenius parameters is not implemented.

---

## 🎯 Next Steps & Goals

### **1. Absolute Trajectory Parity**
- Implement a custom Jacobian wrapper for `diffrax` to ensure identical derivatives to Cantera.
- Perform a side-by-side comparison of individual reaction rates ($q_i$) at the start of the JP-10 integration to find the specific "divergence point."
- Experiment with `diffrax.Bosh3` or `diffrax.Tsit5` for less stiff sections to see if stepping behavior stabilizes.

### **2. Serial Performance Improvement**
- Use `jax.lax.scan` for the internal solver loops if possible to reduce dispatch overhead.
- Explore `jax.checkpoint` strategically to trade recomputation for memory/dispatch speed.
- Profile the `loader.py` to ensure mechanism extraction isn't bottlenecking the warm-start time.

### **3. Extended Mechanism Support**
- Implement **PLOG** (Pressure Logarithmic) support to enable more complex aerospace mechanisms.
- Implement **Unused Species Filtering** in the loader to strip inactive species from `MechData` for faster tracing.
