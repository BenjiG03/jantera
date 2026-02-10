# JAX Latency and Performance Floor Report

## Executive Summary
Profiling confirms that the current JAX implementation encounters a "latency floor" of approximately **30 µs per step** due purely to the Python-level loop structure and XLA dispatch overhead. This intrinsic overhead is ~4x higher than Cantera's total time per step (~7 µs).

Optimization within Python/JAX can yield at most a ~2-3x speedup by reducing the calculation time (Kinetics and Linear Algebra), but cannot surpass the 30 µs floor. To match or beat Cantera on a single CPU core, moving the core integrator loop to C++ is necessary.

## Benchmark Results (N=32, JP-10 Size)

| Component | Time (µs) | Notes |
| :--- | :--- | :--- |
| **Pure JAX Loop Overhead** | 12.12 | Minimum cost of `lax.while_loop` |
| **Logic Overhead (BDF)** | 29.52 | Cost of BDF step logic (no math) |
| **Kinetics (`compute_wdot`)** | 38.39 | Cost to evaluate RHS once |
| **Linear Solve Only (`lu_solve`)** | 17.21 | Cost to back-substitute (reusing LU) |
| **Linear Factor+Solve** | 35.04 | Cost to factorize and solve |

**Observed Real Performance**: ~64 µs/step (JP-10)
**Theoretical Limit (0 math cost)**: ~30 µs/step (BDF Logic Overhead)
**Cantera Performance**: ~7 µs/step

## Analysis
The BDF solver logic (state management, error checking, step resizing) adds ~17 µs of overhead on top of the pure loop overhead (12 µs). This total ~30 µs overhead dominates when the mathematical workload (small matrix solve, small kinetics kernel) is highly optimized.

Even if we completely eliminate the cost of chemical kinetics and linear algebra (0 µs), Canterax would still be ~4x slower than Cantera on a single CPU core.

## Recommendations
1.  **Do Not Optimize Python Further for Single-Core Speed**: Further Python-level optimizations (e.g., fusing ops) will have diminishing returns and cannot overcome the 30 µs floor.
2.  **Focus on Batching (GPU)**: The 30 µs overhead is constant regardless of batch size. On a GPU solving 10,000 reactors, the amortized overhead becomes negligible (3 ns/reactor), likely creating a 100x advantage over Cantera.
3.  **Use C++ Primitives for Single-Core**: If single-core CPU speed is critical, the entire `bdf_solve` loop must be implemented as a JAX C++ primitive to bypass the XLA loop dispatch overhead.
