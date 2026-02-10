# Canterax Wiki

Welcome to the **Canterax** documentation! This wiki provides a deep dive into the architecture, mathematical formulations, and validation of the library.

Canterax is a JAX-powered chemical kinetics library designed for **Automatic Differentiation**, **GPU acceleration**, and **batch reactor simulations**.

## 📚 Documentation Sections

| Section | Description |
| :--- | :--- |
| **[Architecture](Architecture)** | System design, static data structures, and the "dense-sparse" philosophy. |
| **[Validation](Validation)** | Parity plots and performance benchmarks against Cantera 3.2.0. |
| **[Modules](Modules)** | Overview of the core Python modules and their responsibilities. |
| **[Thermodynamics](Thermodynamics)** | Implementation of NASA-7 polynomial fits and mixture properties. |
| **[Kinetics](Kinetics)** | High-performance Arrhenius, Three-Body, and Falloff reaction rates. |
| **[Reactor](Reactor)** | Stiff ODE integration with Diffrax and reactor network models. |
| **[Equilibrium](Equilibrium)** | Gibbs minimization with element-potential selection. |

---

> [!TIP]
> To get started with the code, check out the **[Quickstart](https://github.com/BenjiG03/canterax#quickstart)** in the main README.
