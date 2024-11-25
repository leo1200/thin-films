import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from reflax._matrix_operations._block_matrix_helpers import (
    getQuadBlock,
    getS11,
    getS12,
    getS21,
    getS22
)

@jax.jit
def _redheffer_star_product(
    SA: Float[Array, "4 4"],
    SB: Float[Array, "4 4"]
) -> Float[Array, "4 4"]:
    """
    Computes the Redheffer star product of two scattering matrices.

    Args:
        SA: The first scattering matrix
        SB: The second scattering matrix

    Returns:
        The combined scattering matrix.
    """
    I = jnp.eye(2, 2)

    # Compute common terms
    D = jnp.linalg.solve(I - getS11(SB) @ getS22(SA), getS12(SA))
    F = jnp.linalg.solve(I - getS22(SA) @ getS11(SB), getS21(SB))

    # Compute combined scattering matrix
    S = getQuadBlock(
        S11 = getS11(SA) + D @ getS11(SB) @ getS21(SA),
        S12 = D @ getS12(SB),
        S21 = F @ getS21(SA),
        S22 = getS22(SB) + F @ getS22(SA) @ getS12(SB)
    )

    return S