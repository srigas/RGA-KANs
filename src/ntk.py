import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from flax import nnx


def _grad_flat_at_x(model, x_single):
    # scalarize output by summing over channels (gives NTK = sum_c ∂f_c/∂θ · ∂f_c/∂θ')
    def scalar_out(m):
        y = m(x_single[None, ...])           # (1, C)
        return jnp.sum(y)                    # scalar
    g = nnx.grad(scalar_out)(model)          # pytree of grads wrt model params
    g_flat, _ = ravel_pytree(g)
    return g_flat


def ntk_matrix(model, X_subset):
    # Vectorize over X_subset, broadcasting model
    @nnx.vmap(in_axes=(None, 0), out_axes=0)
    def grad_rows(m, x_single):
        return _grad_flat_at_x(m, x_single)

    J = grad_rows(model, X_subset)  # shape: (N_points, n_params)
    return J @ J.T


def stabilize_kernel(K, jitter_scale=1e-12):
    # Symmetrize + trace-scaled jitter to kill tiny negative eigs
    K = 0.5 * (K + K.T)
    eps = jitter_scale * (jnp.trace(K) / K.shape[0])
    return K + eps * jnp.eye(K.shape[0], dtype=K.dtype)


def ntk_spectrum(model, X_subset, jitter_scale=1e-12):
    K = ntk_matrix(model, X_subset)
    K = stabilize_kernel(K, jitter_scale)
    ev = jnp.linalg.eigvalsh(K)
    return jnp.sort(ev)[::-1]


# ------ PDE STUFF -------------------------------------


def _grad_flat_pde_at_x(model, pde_res_fn, x_single):
    def scalar_res(m):
        r = pde_res_fn(m, x_single[None, ...])  # (1, 1) or (1,)
        return jnp.squeeze(r)
    
    g = nnx.grad(scalar_res)(model)
    g_flat, _ = ravel_pytree(g)
    return g_flat


def _grad_flat_bc_at_xy(model, x_single, y_single):
    def scalar_res(m):
        r = m(x_single[None, ...]) - y_single[None, ...] # (1, 1) or (1,)
        return jnp.squeeze(r)
    
    g = nnx.grad(scalar_res)(model)
    g_flat, _ = ravel_pytree(g)
    return g_flat


def ntk_pde_matrix(model, pde_res_fn, X_pde):
    # Vectorize over PDE points only
    # in_axes: (model=None, pde_res_fn=None, x_single=0)
    @nnx.vmap(in_axes=(None, None, 0), out_axes=0)
    def grad_rows(m, pde_res_fn, x_single):
        return _grad_flat_pde_at_x(m, pde_res_fn, x_single)

    J = grad_rows(model, pde_res_fn, X_pde)
    return J @ J.T


def ntk_bc_matrix(model, X_bc, Y_bc):
    # Vectorize over BC points and targets
    # in_axes: (model=None, x_single=0, y_single=0)
    @nnx.vmap(in_axes=(None, 0, 0), out_axes=0)
    def grad_rows(m, x_single, y_single):
        return _grad_flat_bc_at_xy(m, x_single, y_single)

    J = grad_rows(model, X_bc, Y_bc)
    return J @ J.T


def pinntk_diag_spectra(model, pde_res_fn, X_pde, X_bc, Y_bc, jitter_scale=1e-12):
    # Compute unweighted matrices
    K_EE = stabilize_kernel(ntk_pde_matrix(model, pde_res_fn, X_pde), jitter_scale)
    K_BB = stabilize_kernel(ntk_bc_matrix(model, X_bc, Y_bc), jitter_scale)
    
    # Compute eigenvalues
    lamE = jnp.sort(jnp.linalg.eigvalsh(K_EE))[::-1]
    lamB = jnp.sort(jnp.linalg.eigvalsh(K_BB))[::-1]
    return lamE, lamB
