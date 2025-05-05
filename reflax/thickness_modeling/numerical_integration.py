import jax
import jax.numpy as jnp

@jax.jit
def cumulative_trapezoid(x, y):
    """
    Implementation of the cumulative trapezoidal
    rule for numerical integration.
    """

    dx = jnp.diff(x)
    y_avg = (y[:-1] + y[1:]) / 2
    integral = jnp.cumsum(y_avg * dx)

    # start the integral at 0
    integral = jnp.insert(integral, 0, 0.0)

    return integral