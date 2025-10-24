import jax.numpy as jnp

from flax import nnx

from .layer import KANLayer, SineLayer
from .piratenet import PeriodEmbedder, RFFEmbedder

from typing import Union


class KANBlock(nnx.Module):

    def __init__(self, n_in: int, n_out: int, n_hidden: int, init_scheme: Union[dict, None] = None,
                 D: int = 5, alpha: float = 0.0, beta: float = 1.0, seed: int = 42):

        # Define the 2 layers
        self.InputLayer = KANLayer(n_in = n_in, n_out = n_hidden, D = D, init_scheme = init_scheme, seed = seed)
        self.OutputLayer = KANLayer(n_in = n_hidden, n_out = n_out, D = D, init_scheme = init_scheme, seed = seed)

        # Define alpha, beta
        self.alpha = nnx.Param(jnp.array(alpha, dtype=jnp.float32))
        self.beta = nnx.Param(jnp.array(beta, dtype=jnp.float32))


    def __call__(self, x, u, v):

        identity = x

        x = self.InputLayer(x)
        x = x * u + (1 - x) * v

        b = self.beta.value
        x = b * x + (1 - b) * identity

        x = self.OutputLayer(x)
        x = x * u + (1 - x) * v

        a = self.alpha.value
        x = a * x + (1 - a) * identity

        return x


class RGAKAN(nnx.Module):

    def __init__(self, n_in: int, n_out: int, n_hidden: int, num_blocks: int,
                 D: int = 5, init_scheme: Union[dict, None] = None,
                 alpha: float = 0.0, beta: float = 1.0, ref: Union[None, dict] = None,
                 period_axes: Union[None, dict] = None, rff_std: Union[None, float] = None,
                 sine_D: Union[None, int] = None, seed: int = 42):

        self.pi_init = True if ref is not None else False
        self.n_hidden = n_hidden
        self.D = D

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
        
        # Check for Sine-Basis Layer
        if sine_D:
            if init_scheme is not None:
                sine_scheme = {**init_scheme, 'norm_pow': 0}
            self.SineBasis = SineLayer(n_in = n_in, n_out = n_hidden, D = sine_D, init_scheme = sine_scheme, seed = seed)
        else:
            self.SineBasis = None

        # Define gates
        self.U = KANLayer(n_in = n_hidden, n_out = n_hidden, D = D, init_scheme = init_scheme, seed = seed)
        self.V = KANLayer(n_in = n_hidden, n_out = n_hidden, D = D, init_scheme = init_scheme, seed = seed)

        # Define blocks
        self.blocks = []
        for i in range(num_blocks):
        
            self.blocks.append(
                KANBlock(
                    n_in = n_hidden, n_out = n_hidden, n_hidden = n_hidden, init_scheme = init_scheme,
                    D = D, alpha = alpha, beta = beta, seed = seed
                )
            )

        # Check for physics-informed initialization
        if self.pi_init:
            C = self._pi_init(ref)
            self.OutBasis = nnx.Param(jnp.array(C))
        else:
            self.OutLayer = KANLayer(n_in = n_hidden, n_out = n_out, D = D, init_scheme = init_scheme, seed = seed)

        
    def _pi_init(self, ref):
        # We want to solve min_W(||WΦ-Y||^2)

        # Get collocation points for the spatiotemporal domain to impose initial condition
        t = ref['t'].flatten()[::10] # Downsampled temporal - shape (Nt, )
        x = ref['x'].flatten() # spatial - shape (Nx, )
        tt, xx = jnp.meshgrid(t, x, indexing="ij")

        # collocation inputs - shape (batch, 2), batch = Nt*Nx
        inputs = jnp.hstack([tt.flatten()[:, None], xx.flatten()[:, None]])

        # Get Y for inputs
        u_0 = ref['usol'][0, :] # initial condition - shape (Nx, )
        Y = jnp.tile(u_0.flatten(), (t.shape[0], 1)) # shape (Nt, Nx)
        Y = Y.flatten().reshape(-1, 1) # shape (batch, 1)
        
        # Get Φ - essentially do a full forward pass up until the final layer
        if self.PE:
            inputs = self.PE(inputs)

        if self.FE:
            inputs = self.FE(inputs)

        if self.SineBasis:
            inputs = self.SineBasis(inputs)

        u, v = self.U(inputs), self.V(inputs)

        for block in self.blocks:
            inputs = block(inputs, u, v)

        Phi = self.U.basis(inputs) # (batch, n_hidden, D)

        # Reshape to (batch, n_hidden * D)
        Phi_flat = Phi.reshape(Phi.shape[0], -1)

        # Solve least squares to get C as shape (n_hidden * D, 1)
        result, residuals, rank, s = jnp.linalg.lstsq(
                    Phi_flat, Y, rcond=None
                )

        # result.T is shaped (1, n_hidden * D), so we reshape to (1, n_hidden, D)
        C = result.T.reshape(1, self.n_hidden, self.D)

        return C

    
    def __call__(self, x):

        # Apply embedders
        if self.PE:
            x = self.PE(x)

        if self.FE:
            x = self.FE(x)

        if self.SineBasis:
            x = self.SineBasis(x)

        # Get u and v
        u = self.U(x)
        v = self.V(x)

        # Pass through blocks
        for block in self.blocks:
            x = block(x, u, v)

        # If the last layer is physics-informed
        if self.pi_init:
            C = self.OutBasis.value # (1, n_hidden, D)
            # use u (or v) as a helper to apply basis on x - helper
            B = self.U.basis(x) # (batch, n_hidden, D)
            y = jnp.einsum('bhk, ohk -> bo', B, C) # (batch, 1)
        else:
            y = self.OutLayer(x)

        return y

        