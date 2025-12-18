import jax.numpy as jnp

from jaxkan.utils.PIKAN import gradf


# Allen-Cahn Equation
def ac_res(model, collocs):
    
    # Eq. parameters
    D = jnp.array(1e-4, dtype=jnp.float32)
    c = jnp.array(5.0, dtype=jnp.float32)

    def u(x):
        y = model(x)
        return y

    u_t = gradf(u, 0, 1)
    u_xx = gradf(u, 1, 2)

    res = u_t(collocs) - D*u_xx(collocs) - c*(u(collocs)-(u(collocs)**3))

    return res


# Burgers Equation
def burgers_res(model, collocs):
    
    # Eq. parameter
    nu = jnp.array(0.01/jnp.pi, dtype=float)

    def u(x):
        y = model(x)
        return y

    # Physics Loss Terms
    u_t = gradf(u, 0, 1)
    u_x = gradf(u, 1, 1)
    u_xx = gradf(u, 1, 2)

    # Get all residuals
    res = u_t(collocs) + u(collocs)*u_x(collocs) - nu*u_xx(collocs)

    return res


# Korteweg-De Vries Equation
def kdv_res(model, collocs):
    
    # Eq. parameters
    eta = jnp.array(1.0, dtype=jnp.float32)
    mu = jnp.array(0.022, dtype=jnp.float32)

    def u(x):
        y = model(x)
        return y

    u_t = gradf(u, 0, 1)
    u_x = gradf(u, 1, 1)
    u_xxx = gradf(u, 1, 3)

    res = u_t(collocs) + eta*u(collocs)*u_x(collocs) + (mu**2)*u_xxx(collocs)

    return res


# Sine Gordon Equation
def sg_res(model, collocs):
    
    # Eq. parameters
    D = jnp.array(-1.0, dtype=jnp.float32)

    def u(x):
        y = model(x)
        return y

    u_tt = gradf(u, 0, 2)
    u_xx = gradf(u, 1, 2)

    res = u_tt(collocs) + D*u_xx(collocs) + jnp.sin(u(collocs))

    return res


# Advection Equation
def advection_res(model, collocs):
    
    #Eq. parameter
    c = jnp.array(20.0, dtype=jnp.float32)

    def u(x):
        y = model(x)
        return y

    u_t = gradf(u, 0, 1)
    u_x = gradf(u, 1, 1)

    res = u_t(collocs) + c*u_x(collocs)

    return res


# Helmholtz Equation - a_1 = 1, a_2 = 4
def helmholtz_14_res(model, collocs):
    
    # Eq. parameters
    a1 = jnp.array(1.0, dtype=jnp.float32)
    a2 = jnp.array(4.0, dtype=jnp.float32)

    def u(x):
        y = model(x)
        return y

    u_xx = gradf(u, 0, 2)
    u_yy = gradf(u, 1, 2)

    # source term - we assume k = 1.0
    factor = 1.0 - (jnp.pi**2)*(a1**2 + a2**2)
    f = factor*jnp.sin(jnp.pi*a1*collocs[:,[0]])*jnp.sin(jnp.pi*a2*collocs[:,[1]])

    res = u_xx(collocs) + u_yy(collocs) + u(collocs) - f

    return res


# Poisson Equation - ω = 1
def poisson_1_res(model, collocs):
    
    # Eq. parameters
    a1 = jnp.array(1.0, dtype=jnp.float32)
    a2 = jnp.array(1.0, dtype=jnp.float32)

    def u(x):
        y = model(x)
        return y

    u_xx = gradf(u, 0, 2)
    u_yy = gradf(u, 1, 2)

    factor = -(jnp.pi**2)*(a1**2 + a2**2)
    f = factor*jnp.sin(jnp.pi*a1*collocs[:,[0]])*jnp.sin(jnp.pi*a2*collocs[:,[1]])

    res = u_xx(collocs) + u_yy(collocs) - f

    return res


# Poisson Equation - ω = 2
def poisson_2_res(model, collocs):
    
    # Eq. parameters
    a1 = jnp.array(2.0, dtype=jnp.float32)
    a2 = jnp.array(2.0, dtype=jnp.float32)

    def u(x):
        y = model(x)
        return y

    u_xx = gradf(u, 0, 2)
    u_yy = gradf(u, 1, 2)

    factor = -(jnp.pi**2)*(a1**2 + a2**2)
    f = factor*jnp.sin(jnp.pi*a1*collocs[:,[0]])*jnp.sin(jnp.pi*a2*collocs[:,[1]])

    res = u_xx(collocs) + u_yy(collocs) - f

    return res


# Poisson Equation - ω = 4
def poisson_4_res(model, collocs):
    
    # Eq. parameters
    a1 = jnp.array(4.0, dtype=jnp.float32)
    a2 = jnp.array(4.0, dtype=jnp.float32)

    def u(x):
        y = model(x)
        return y

    u_xx = gradf(u, 0, 2)
    u_yy = gradf(u, 1, 2)

    factor = -(jnp.pi**2)*(a1**2 + a2**2)
    f = factor*jnp.sin(jnp.pi*a1*collocs[:,[0]])*jnp.sin(jnp.pi*a2*collocs[:,[1]])

    res = u_xx(collocs) + u_yy(collocs) - f

    return res


# 2D Heat Multi-scale Equation
def heat_multiscale_res(model, collocs):
    # Diffusion coefficients
    D_x = jnp.array(1.0 / (500 * jnp.pi)**2, dtype=jnp.float32)  # 1/(500π)²
    D_y = jnp.array(1.0 / (jnp.pi**2), dtype=jnp.float32)        # 1/π²
    
    def u(x):
        return model(x)
    
    # Derivatives
    u_t = gradf(u, 0, 1)   # ∂u/∂t
    u_xx = gradf(u, 1, 2)  # ∂²u/∂x²
    u_yy = gradf(u, 2, 2)  # ∂²u/∂y²
    
    # Residual: u_t - D_x * u_xx - D_y * u_yy = 0
    res = u_t(collocs) - D_x * u_xx(collocs) - D_y * u_yy(collocs)
    
    return res
    
    
# Navier-Stokes on Torus (Velocity-Vorticity formulation)
def nst_res(model, collocs, Re=100.0):

    Re = jnp.array(Re, dtype=jnp.float32)
    
    def uv(x):
        return model(x)  # shape (N, 2): [u, v]
    
    def u_func(x):
        return uv(x)[:, [0]]
    
    def v_func(x):
        return uv(x)[:, [1]]
    
    # Vorticity is derived: w = v_x - u_y
    def w_func(x):
        v_x = gradf(v_func, 1, 1)  # ∂v/∂x
        u_y = gradf(u_func, 2, 1)  # ∂u/∂y
        return v_x(x) - u_y(x)
    
    # First derivatives for vorticity transport
    w_t = gradf(w_func, 0, 1)   # ∂w/∂t
    w_x = gradf(w_func, 1, 1)   # ∂w/∂x
    w_y = gradf(w_func, 2, 1)   # ∂w/∂y
    
    # Second derivatives for vorticity diffusion
    w_xx = gradf(w_func, 1, 2)  # ∂²w/∂x²
    w_yy = gradf(w_func, 2, 2)  # ∂²w/∂y²
    
    # Velocity derivatives for continuity
    u_x = gradf(u_func, 1, 1)   # ∂u/∂x
    v_y = gradf(v_func, 2, 1)   # ∂v/∂y
    
    # Get velocity values
    u_val = u_func(collocs)
    v_val = v_func(collocs)
    
    # Vorticity transport equation:
    # w_t + u*w_x + v*w_y - (1/Re)*(w_xx + w_yy) = 0
    res_vorticity = (w_t(collocs) + u_val * w_x(collocs) + v_val * w_y(collocs) 
                     - (1.0/Re) * (w_xx(collocs) + w_yy(collocs)))
    
    # Continuity equation: u_x + v_y = 0
    res_continuity = u_x(collocs) + v_y(collocs)
    
    # Stack residuals (no vorticity definition residual - it's satisfied by construction)
    res = jnp.concatenate([res_vorticity, res_continuity], axis=1)
    
    return res


def nst_w_func(model, collocs):

    def uv(x):
        return model(x)
    
    def u_func(x):
        return uv(x)[:, [0]]
    
    def v_func(x):
        return uv(x)[:, [1]]
    
    v_x = gradf(v_func, 1, 1)  # ∂v/∂x
    u_y = gradf(u_func, 2, 1)  # ∂u/∂y
    
    return v_x(collocs) - u_y(collocs)