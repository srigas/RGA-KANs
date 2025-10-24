__all__ = [
     "_get_adam", "_get_pde_collocs", "_get_ic_collocs", "model_eval",
     "count_params", "_get_colloc_indices", "grad_norm",
     "count_rga", "count_pirate", "count_pikan",
     "generate_func_data", "func_fit_step", "func_fit_eval"
]

import jax
import jax.numpy as jnp

import numpy as np
from flax import nnx
import optax

from typing import List


def _get_adam(learning_rate: float = 1e-3, decay_steps: int = 5000, decay_rate: float = 0.9, warmup_steps: int = 5000):

    lr_schedule = optax.exponential_decay(
            init_value=learning_rate,
            transition_steps=decay_steps,
            decay_rate=decay_rate,
            staircase=False
        )
    
    if warmup_steps > 0:
        warmup = optax.linear_schedule(
            init_value=0.0,
            end_value=learning_rate,
            transition_steps=warmup_steps,
        )
    
        lr_schedule = optax.join_schedules([warmup, lr_schedule], [warmup_steps])
    
    tx = optax.adam(
        learning_rate=lr_schedule, b1=0.9, b2=0.999, eps=1e-8
    )

    return tx


def _get_pde_collocs(ranges: list[tuple[int, int]], sample_size: int):

    x1 = jnp.linspace(ranges[0][0], ranges[0][1], sample_size)
    x2 = jnp.linspace(ranges[1][0], ranges[1][1], sample_size)
    X1, X2 = jnp.meshgrid(x1, x2, indexing='ij')
    collocs_pool = jnp.stack([X1.flatten(), X2.flatten()], axis=1)

    return collocs_pool


def _get_ic_collocs(x_range: tuple[int, int], sample_size: int):
    
    t = jnp.array([0.0], dtype=float)
    x = jnp.linspace(x_range[0], x_range[1], sample_size)
    T, X = jnp.meshgrid(t, x, indexing='ij')
    ic_collocs = jnp.stack([T.flatten(), X.flatten()], axis=1)

    return ic_collocs


def model_eval(model, coords, refsol):
    
    output = model(coords).reshape(refsol.shape)
    l2err = jnp.linalg.norm(output-refsol)/jnp.linalg.norm(refsol)

    return l2err


def count_params(model):
    # Extract all parameters of type nnx.Param from the model.
    params = nnx.state(model, nnx.Param)
    
    # Flatten the tree to get individual parameter arrays.
    leaves = jax.tree_util.tree_leaves(params)
    
    # Sum the total number of elements (i.e. the product of the dimensions)
    total_params = sum(np.prod(p.shape) for p in leaves)

    return int(total_params)


def _get_colloc_indices(collocs_pool, batch_size, px, seed):
    
    collocs_key = jax.random.PRNGKey(seed)

    X_ids = jax.random.choice(key=collocs_key, a=collocs_pool.shape[0], shape=(batch_size,), replace=False, p=px)

    sorted_batch_order = jnp.argsort(collocs_pool[X_ids, 0])
    
    sorted_pool_indices = X_ids[sorted_batch_order]

    return sorted_pool_indices


@nnx.jit
def grad_norm(grads_E, grads_B, λ_E, λ_B, grad_mixing):
    
    norm_E = optax.global_norm(grads_E)
    norm_B = optax.global_norm(grads_B)
    norm_sum = norm_E + norm_B
                
    λ_E_hat = norm_sum / (norm_E + 1e-5*norm_sum)
    λ_B_hat = norm_sum / (norm_B + 1e-5*norm_sum)
                        
    λ_E_new = grad_mixing*λ_E + (1.0 - grad_mixing)*λ_E_hat
    λ_B_new = grad_mixing*λ_B + (1.0 - grad_mixing)*λ_B_hat

    return λ_E_new, λ_B_new


# ------------- Parameter Counting Utils ------------------------------

def count_rga(n_in, period_axes, n_out, n_hidden, num_blocks, D, sine_D):

    if period_axes is not None:
        d_I = n_in + 1
    else:
        d_I = n_in

    param_count = 2*n_hidden*(n_hidden*D + 1)*(num_blocks + 1) + 2*num_blocks + 2*sine_D + n_hidden*(d_I*sine_D + n_out*D + 1)

    return param_count


def count_pirate(n_in, period_axes, n_out, n_hidden, num_blocks):

    if period_axes is not None:
        d_I = n_in + 1
    else:
        d_I = n_in

    param_count = int(n_hidden*0.5*d_I) + n_hidden*n_out + n_hidden*(n_hidden + 2)*(3*num_blocks + 2) + num_blocks

    return param_count


def count_pikan(n_in, period_axes, n_out, n_hidden, num_layers, D):

    if period_axes is not None:
        d_I = n_in + 1
    else:
        d_I = n_in

    param_count = n_hidden*(d_I * D + n_out * D + (num_layers-1) * n_hidden * D) + n_hidden*num_layers + n_out

    return param_count


# ------------- Function Fitting Utils --------------------------------


def generate_func_data(function, dim, N, seed):
    key = jax.random.key(seed)
    x = jax.random.uniform(key, shape=(N,dim), minval=-1.0, maxval=1.0)

    y = function(x)

    return x, y


@nnx.jit
def func_fit_step(model, optimizer, X_train, y_train):

    def loss_fn(model):
        residual = model(X_train) - y_train
        loss = jnp.mean((residual)**2)

        return loss

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(grads)

    return loss


def func_fit_eval(model, function, dim, resolution=200):
    # Create grid
    lin = jnp.linspace(-1.0, 1.0, resolution)
    xx = jnp.meshgrid(*[lin]*dim)
    grid = jnp.stack([x.ravel() for x in xx], axis=-1)

    # Evaluate ground truth and prediction
    y_true = function(grid)
    y_pred = model(grid)

    # Compute relative L2 error
    error = jnp.linalg.norm(y_true - y_pred) / jnp.linalg.norm(y_true)

    return error
