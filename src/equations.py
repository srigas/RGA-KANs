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
