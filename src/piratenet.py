import jax.numpy as jnp

from flax import nnx

from typing import Union


class PeriodEmbedder(nnx.Module):

    def __init__(self, period_axes: dict):

        # Example: period_axes = {1: jnp.pi} (dict with key axis and value period)
        self.axes = period_axes
    
    def __call__(self, x):
        # x shape (batch, n_in)
        y = []
        
        for idx in range(x.shape[-1]):
            if idx in self.axes.keys():
                period = self.axes[idx]
                cs = jnp.cos(period * x[:, [idx]])
                ss = jnp.sin(period * x[:, [idx]])
                y.extend([cs, ss])
            else:
                y.append(x[:, [idx]])

        y = jnp.hstack(y)

        return y


class RFFEmbedder(nnx.Module):

    def __init__(self, std: float = 1.0, n_in: int = 1, embed_dim: int = 256, seed: int = 42):

        rngs = nnx.Rngs(seed)

        # Initialize kernel
        self.B = nnx.Param(nnx.initializers.normal(stddev=std)(
                            rngs.params(), (n_in, embed_dim//2), jnp.float32))
    
    def __call__(self, x):
        # x shape (batch, n_in)

        Bx = jnp.dot(x, self.B.value)
        
        y = jnp.concatenate([jnp.cos(Bx), jnp.sin(Bx)], axis=-1)

        return y


class Dense(nnx.Module):

    def __init__(self, n_in: int, n_out: int, RWF: dict = {"mean": 1.0, "std": 0.1}, seed: int = 42):

        rngs = nnx.Rngs(seed)

        # Initialize kernel via RWF - shape (n_in, n_out)
        mu, sigma = RWF["mean"], RWF["std"]

        # Glorot Initialization
        stddev = jnp.sqrt(2.0/(n_in + n_out))
        # Weight matrix with shape (n_in, n_out)
        w = nnx.initializers.normal(stddev=stddev)(
                rngs.params(), (n_in, n_out), jnp.float32
            )

        # Reparameterization towards g, v
        g = nnx.initializers.normal(stddev=sigma)(
                rngs.params(), (n_out,), jnp.float32
            )
        g += mu
        g = jnp.exp(g) # shape (n_out,)
        v = w/g # shape (n_in, n_out)

        self.g = nnx.Param(g)
        self.v = nnx.Param(v)

        # Initialize bias - shape (n_out,)
        self.bias = nnx.Param(jnp.zeros((n_out,)))

    
    def __call__(self, x):
        # Reconstruct kernel
        g, v = self.g.value, self.v.value
        kernel = g * v

        # Apply kernel and bias
        y = jnp.dot(x, kernel) + self.bias.value

        return y


class PIBlock(nnx.Module):

    def __init__(self, n_in: int, n_out: int, n_hidden: int, alpha: float, RWF: dict = {"mean": 1.0, "std": 0.1}, seed: int = 42):

        # Define 3 layers
        self.InputLayer = Dense(n_in = n_hidden, n_out = n_hidden, RWF = RWF, seed = seed)
        self.MiddleLayer = Dense(n_in = n_hidden, n_out = n_hidden, RWF = RWF, seed = seed)
        self.OutputLayer = Dense(n_in = n_hidden, n_out = n_out, RWF = RWF, seed = seed)

        # Define alpha
        self.alpha = nnx.Param(jnp.array(alpha, dtype=jnp.float32))

    
    def __call__(self, x, u, v):

        identity = x

        x = self.InputLayer(x)
        x = nnx.tanh(x)

        x = x * u + (1 - x) * v

        x = self.MiddleLayer(x)
        x = nnx.tanh(x)

        x = x * u + (1 - x) * v

        x = self.OutputLayer(x)
        x = nnx.tanh(x)

        a = self.alpha.value
        x = a * x + (1 - a) * identity

        return x
        

class PirateNet(nnx.Module):

    def __init__(self, n_in: int, n_out: int, n_hidden: int, num_blocks: int,
                 alpha: float, ref: Union[None, dict] = None,
                 period_axes: Union[None, dict] = None, rff_std: Union[None, float] = None,
                 RWF: dict = {"mean": 1.0, "std": 0.1}, seed: int = 42):
        
        self.pi_init = True if ref is not None else False
        
        if period_axes:
            # Define embedder
            self.PE = PeriodEmbedder(period_axes)
            # input dimension has now changed
            n_in += len(period_axes.keys())
        else:
            self.PE = None

        # Define embedder for random Fourier features
        if rff_std:
            self.FE = RFFEmbedder(std = rff_std, n_in = n_in, embed_dim = n_hidden)
        else:
            self.FE = None

        # Define gates
        self.U = Dense(n_in = n_hidden, n_out = n_hidden, RWF = RWF, seed = seed)
        self.V = Dense(n_in = n_hidden, n_out = n_hidden, RWF = RWF, seed = seed)
            
        # Define blocks
        self.blocks = [
            PIBlock(
                n_in = n_hidden, n_out = n_hidden, n_hidden = n_hidden,
                alpha = alpha, RWF = RWF, seed = seed)

            for i in range(num_blocks)
        ]

        if self.pi_init:
            W = self._pi_init(ref)
            self.OutKernel = nnx.Param(jnp.array(W))
        else:
            self.OutLayer = Dense(n_in = n_hidden, n_out = n_out, RWF = RWF, seed = seed)


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
        
        # Get Φ
        if self.PE:
            inputs = self.PE(inputs)
            
        Phi = self.FE(inputs)

        # Solve least squares to get W (Φ = x, Y = init_data)
        W, residuals, rank, s = jnp.linalg.lstsq(
                    Phi, Y, rcond=None
                )

        return W

    
    def __call__(self, x):

        # Apply embedders
        if self.PE:
            x = self.PE(x)

        if self.FE:
            x = self.FE(x)

        # Get u and v
        u = self.U(x)
        u = nnx.tanh(u)

        v = self.V(x)
        v = nnx.tanh(v)

        # Pass through blocks
        for block in self.blocks:
            x = block(x, u, v)

        # If the last layer is physics-informed
        if self.pi_init:
            W = self.OutKernel.value
            y = jnp.dot(x, W)
        else:
            y = self.OutLayer(x)

        return y
        