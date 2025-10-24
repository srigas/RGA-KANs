import jax.numpy as jnp

from flax import nnx

from .layer import KANLayer
from .piratenet import PeriodEmbedder, RFFEmbedder

from typing import Union


class KAN(nnx.Module):

    def __init__(self, n_in: int, n_out: int, n_hidden: int, num_layers: int, D: int = 5,
                 init_scheme: Union[dict, None] = "default",
                 period_axes: Union[None, dict] = None, rff_std: Union[None, float] = None,
                 seed: int = 42):
        
        # Check for periodic embeddings
        if period_axes:
            self.PE = PeriodEmbedder(period_axes)
            n_in += len(period_axes.keys()) # input dimension has now changed
        else:
            self.PE = None

        # Check for RFF
        if rff_std:
            self.FE = RFFEmbedder(std = rff_std, n_in = n_in, embed_dim = n_hidden)
            n_in = n_hidden # input dimension has now changed
        else:
            self.FE = None

        # Input Layer
        self.layers = [KANLayer(n_in = n_in, n_out = n_hidden, D = D, init_scheme = init_scheme, seed = seed)]

        # Add hidden layers
        for i in range(num_layers-1):
            self.layers.append(
                KANLayer(n_in = n_hidden, n_out = n_hidden, D = D, init_scheme = init_scheme, seed = seed)
            )

        # Add output layer
        self.layers.append(
            KANLayer(n_in = n_hidden, n_out = n_out, D = D, init_scheme = init_scheme, seed = seed)
        )

    
    def __call__(self, x):

        # Apply embedders
        if self.PE:
            x = self.PE(x) # shape (batch, n_in)
            
        if self.FE:
            x = self.FE(x) # shape (batch, n_in)

        # Pass through blocks
        for layer in self.layers:
            x = layer(x) # shape (batch, n_hidden)

        return x
