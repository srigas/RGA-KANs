import jax
import jax.numpy as jnp

from flax import nnx

from typing import Union

from jaxkan.utils.polynomials import Cb
        
        
class KANLayer(nnx.Module):
    
    def __init__(self, n_in: int = 2, n_out: int = 5, D: int = 5,
                 init_scheme: Union[dict, None] = None, seed: int = 42):
        
        max_deg = max(list(Cb.keys()))
        if D > max_deg:
            raise ValueError(f"For method 'exact', the maximum degree cannot exceed {max_deg}.")

        # Setup basic parameters
        self.n_in = n_in
        self.n_out = n_out
        self.D = D

        # Setup nnx rngs
        self.rngs = nnx.Rngs(seed)

        # Add bias
        self.bias = nnx.Param(jnp.zeros((n_out,)))

        # Initialize the trainable parameters, based on the selected initialization scheme
        c_basis = self._initialize_params(init_scheme, seed)

        self.c_basis = nnx.Param(c_basis)
            

    def basis(self, x):
        
        batch = x.shape[0]
        
        # Apply tanh activation
        x = jnp.tanh(x) # (batch, n_in)
        
        cheb = jnp.stack([Cb[i](x) for i in range(1, self.D + 1)], axis=-1)  # (batch, n_in, D)

        return cheb


    def _initialize_params(self, init_scheme, seed):

        if init_scheme is None:
            init_scheme = {"type" : "default"}

        init_type = init_scheme.get("type", "default")

        # Default initialization
        if init_type == "default":
            
            std = 1.0/jnp.sqrt(self.n_in * (self.D + 1))
            
            c_basis = nnx.initializers.normal(stddev=std)(
                self.rngs.params(), (self.n_out, self.n_in, self.D), jnp.float32
            )
            
        else:

            key = jax.random.key(seed)

            # Also get distribution type
            distrib = init_scheme.get("distribution", "uniform")

            if distrib is None:
                distrib = "uniform"

            sample_size = init_scheme.get("sample_size", 10000)

            if sample_size is None:
                sample_size = 10000

            # Generate a sample of points
            if distrib == "uniform":
                sample = jax.random.uniform(key, shape=(sample_size,), minval=-1.0, maxval=1.0)
            elif distrib == "normal":
                sample = jax.random.normal(key, shape=(sample_size,))

            # Finally get gain
            gain = init_scheme.get("gain", None)
            if gain is None:
                gain = sample.std().item()

            # Extend the sample to be able to pass through basis
            sample_ext = jnp.tile(sample[:, None], (1, self.n_in))

            norm_pow = init_scheme.get("norm_pow", 0)

            # ---------- μ⁽0⁾ₘ (⟨B_m²⟩) -------------------------------------
            B      = self.basis(sample_ext)
            mu0    = (B**2).mean(axis=(0, 1)) # (D,)
            
            # fan-in term
            denom_vec = self.n_in * mu0


            # ---------- μ⁽1⁾ₘ (⟨B'_m²⟩) -------------------------------------                   

            # Define a scalar version of the basis function
            basis_scalar = lambda x: self.basis(jnp.array([[x]]))[0, 0, :]

            jac = jax.jacrev(basis_scalar)                     # one more derivative
            mu1  = (jax.vmap(jac)(sample)**2).mean(axis=0)

            norm_vec = self.D**norm_pow 
                
            denom_vec += self.n_out * mu1 * norm_vec

            # σ_m² = 2 / ( n_in μ⁽0⁾ₘ + n_out μ⁽1⁾ₘ )
            #sigma_vec = gain * jnp.sqrt(2.0 / denom_vec)    # (D,)
            sigma_vec = gain * jnp.sqrt(1.0 / (self.D*denom_vec))    # (D,)

            # ---------- draw weights --------------------------------------
            noise    = nnx.initializers.normal(stddev=1.0)(
                         self.rngs.params(), (self.n_out, self.n_in, self.D), jnp.float32
                       )

            c_basis  = noise * sigma_vec

        return c_basis


    def __call__(self, x):
        
        batch = x.shape[0]
        
        # Calculate basis activations
        Bi = self.basis(x) # (batch, n_in, D)
        act = Bi.reshape(batch, -1) # (batch, n_in * D)

        act_w = self.c_basis.value
        
        # Calculate coefficients
        act_w = act_w.reshape(self.n_out, -1) # (n_out, n_in * D)

        y = jnp.matmul(act, act_w.T) # (batch, n_out)

        y += self.bias.value # (batch, n_out)
        
        return y


class SineLayer(nnx.Module):
    
    def __init__(self, n_in: int = 2, n_out: int = 5, D: int = 5,
                 init_scheme: Union[dict, None] = None, seed: int = 42):

        # Setup basic parameters
        self.n_in = n_in
        self.n_out = n_out
        self.D = D

        # Setup nnx rngs
        self.rngs = nnx.Rngs(seed)

        # Add bias
        self.bias = nnx.Param(jnp.zeros((n_out,)))

        # Initialize omegas from N(0,1) - shape (D, 1)
        self.omega = nnx.Param(
            nnx.initializers.normal(stddev = 1.0)(
                self.rngs.params(), (D, 1), jnp.float32)
        )

        # Initialize phases at 0 - shape (D, 1)
        self.phase = nnx.Param(jnp.zeros((D, 1)))

        # Initialize the trainable parameters, based on the selected initialization scheme
        c_basis = self._initialize_params(init_scheme, seed)

        self.c_basis = nnx.Param(c_basis)
            

    def basis(self, x):
        
        # Expand x to an extra dim for broadcasting
        x = jnp.expand_dims(x, axis=-1) # (batch, n_in, 1)
    
        # Broadcast for multiplication and addition, respectively
        omegas = self.omega.value.reshape(1, 1, self.D) # (1, 1, D)
        p = self.phase.value.reshape(1, 1, self.D) # (1, 1, D)

        # Multiply
        wx = omegas * x # (batch, n_in, D)

        # Get sine term
        s = jnp.sin(wx + p) # (batch, n_in, D)

        # Calculate mean value
        mu = jnp.exp(-0.5*(omegas**2)) * jnp.sin(p) # (1, 1, D)

        # Calculate std
        std = jnp.sqrt(0.5*(1.0 - jnp.exp(-2.0*(omegas**2))*jnp.cos(2.0*p)) - mu**2) # (1, 1, D)

        # Get basis
        eps = 1e-8 # for division stability
        B = (s + mu)/(std + eps) # (batch, n_in, D)

        return B


    def _initialize_params(self, init_scheme, seed):

        if init_scheme is None:
            init_scheme = {"type" : "default"}

        init_type = init_scheme.get("type", "default")

        # Default initialization
        if init_type == "default":
            
            std = 1.0/jnp.sqrt(self.n_in * self.D)
            
            c_basis = nnx.initializers.normal(stddev=std)(
                self.rngs.params(), (self.n_out, self.n_in, self.D), jnp.float32
            )
            
        else:

            key = jax.random.key(seed)

            # Also get distribution type
            distrib = init_scheme.get("distribution", "uniform")

            if distrib is None:
                distrib = "uniform"

            sample_size = init_scheme.get("sample_size", 10000)

            if sample_size is None:
                sample_size = 10000

            # Generate a sample of points
            if distrib == "uniform":
                sample = jax.random.uniform(key, shape=(sample_size,), minval=-1.0, maxval=1.0)
            elif distrib == "normal":
                sample = jax.random.normal(key, shape=(sample_size,))

            # Finally get gain
            gain = init_scheme.get("gain", None)
            if gain is None:
                gain = sample.std().item()

            # Extend the sample to be able to pass through basis
            sample_ext = jnp.tile(sample[:, None], (1, self.n_in))

            norm_pow = init_scheme.get("norm_pow", 0)

            # ---------- μ⁽0⁾ₘ (⟨B_m²⟩) -------------------------------------
            B      = self.basis(sample_ext)
            mu0    = (B**2).mean(axis=(0, 1)) # (D,)
            
            # fan-in term
            denom_vec = self.n_in * mu0


            # ---------- μ⁽1⁾ₘ (⟨B'_m²⟩) -------------------------------------                   

            # Define a scalar version of the basis function
            basis_scalar = lambda x: self.basis(jnp.array([[x]]))[0, 0, :]

            jac = jax.jacrev(basis_scalar)                     # one more derivative
            mu1  = (jax.vmap(jac)(sample)**2).mean(axis=0)

            norm_vec = self.D**norm_pow 
                
            denom_vec += self.n_out * mu1 * norm_vec

            # σ_m² = 2 / ( n_in μ⁽0⁾ₘ + n_out μ⁽1⁾ₘ )
            sigma_vec = gain * jnp.sqrt(2.0 / denom_vec)    # (D,)

            # ---------- draw weights --------------------------------------
            noise    = nnx.initializers.normal(stddev=1.0)(
                         self.rngs.params(), (self.n_out, self.n_in, self.D), jnp.float32
                       )

            c_basis  = noise * sigma_vec

        return c_basis


    def __call__(self, x):
        
        batch = x.shape[0]
        
        # Calculate basis activations
        Bi = self.basis(x) # (batch, n_in, D)
        act = Bi.reshape(batch, -1) # (batch, n_in * D)

        act_w = self.c_basis.value
        
        # Calculate coefficients
        act_w = act_w.reshape(self.n_out, -1) # (n_out, n_in * D)

        y = jnp.matmul(act, act_w.T) # (batch, n_out)

        y += self.bias.value # (batch, n_out)
        
        return y
