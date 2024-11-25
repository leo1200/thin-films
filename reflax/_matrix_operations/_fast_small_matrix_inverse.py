import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

@jax.jit
def inverse22(
    mat: Float[Array, "2 2"]
) -> Float[Array, "2 2"]:
    m1, m2 = mat[0]
    m3, m4 = mat[1]
    inv_det = 1.0 / (m1 * m4 - m2 * m3)
    return jnp.array([[m4, -m2], [-m3, m1]]) * inv_det