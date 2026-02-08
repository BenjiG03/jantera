
import jax
import jax.numpy as jnp
import time
from sparsejac import jacfwd as sparse_jacfwd

def test_sparsejac():
    N = 100
    # Create a sparse function: tridiagonal
    def f(x):
        return jnp.concatenate([
            jnp.array([x[0] - x[1]]),
            x[1:-1] - x[:-2] - x[2:],
            jnp.array([x[-1] - x[-2]])
        ])

    x0 = jnp.ones(N)
    
    # Standard JAX
    print("Benchmarking Standard JAX jacfwd...")
    jax_jfn = jax.jit(jax.jacfwd(f))
    # Warmup
    _ = jax_jfn(x0)
    t0 = time.time()
    for _ in range(100):
        J_std = jax_jfn(x0)
    print(f"Standard JAX: {(time.time()-t0)/100*1e6:.2f} us")
    
    # SparseJac
    print("Benchmarking SparseJac...")
    # 1. Compute sparsity pattern (one-off cost)
    J_sample = jax.jacfwd(f)(x0)
    from jax.experimental import sparse
    sparsity = sparse.BCOO.fromdense(jnp.abs(J_sample) > 0)
    print(f"Sparsity pattern computed. NNZ: {sparsity.nse}")
    
    sparse_jfn = jax.jit(sparse_jacfwd(f, sparsity))
    # Warmup
    J_sp = sparse_jfn(x0)
    t0 = time.time()
    for _ in range(100):
        J_sp = sparse_jfn(x0)
    print(f"SparseJac: {(time.time()-t0)/100*1e6:.2f} us")
    
    print(f"Return Type: {type(J_sp)}")
    if hasattr(J_sp, 'shape'):
        print(f"Shape: {J_sp.shape}")

if __name__ == "__main__":
    test_sparsejac()
