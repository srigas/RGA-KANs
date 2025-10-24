import jax.numpy as jnp

from jax.scipy.special import erf, i1, i1e

# Oscillatory 1D function: sin(2πx) + 3x
def f1(x):
    xval = x[:, [0]]
    return jnp.sin(2.0 * jnp.pi * xval) + 3.0*xval

# Continuous 2D function: xy
def f2(x):
    return x[:, [0]] * x[:, [1]]

# Bessel
def f3(x):
    return i1(x[:, [0]]) + jnp.exp(i1e(x[:, [1]])) + jnp.sin(x[:, [0]] * x[:, [1]])

# 3D Hartman function
def f4(x):
    α = jnp.array([1.0, 1.2, 3.0, 3.2])
    
    A = jnp.array([
        [3,   10, 30],
        [0.1, 10, 35],
        [3,   10, 30],
        [0.1, 10, 35]
    ])
    
    P = 1e-4 * jnp.array([
        [3689, 1170, 2673],
        [4699, 4387, 7470],
        [1091, 8732, 5547],
        [381, 5743, 8828]
    ])

    inner = jnp.sum(A[:, None, :] * (x[None, :, :] - P[:, None, :]) ** 2, axis=-1)
    f = -jnp.sum(α[:, None] * jnp.exp(-inner), axis=0)
    return f[:, None]


# High-dimensional Sobol g function
def f5(x):
    a = (jnp.arange(1, x.shape[1] + 1, dtype=x.dtype)-2)/2
    term = (jnp.abs(4 * x - 2) + a) / (1 + a)
    g = jnp.prod(term, axis=1)
    return g[:, None]
