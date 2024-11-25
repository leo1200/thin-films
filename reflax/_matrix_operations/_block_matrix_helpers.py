"""
We operate on 4 scattering matrices of size 2x2, which we store
in a 4x4 matrix. Alternatively, a dictionary with respective
entries S11, S21, S12, S22 might be used.
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

# -------------------------------------------------------------
# ======================= ↓ Getters ↓ =========================
# -------------------------------------------------------------

@jax.jit
def getS11(
    matrix: Float[Array, "4 4"]
) -> Float[Array, "2 2"]:
    return matrix[0:2, 0:2]

@jax.jit
def getS21(
    matrix: Float[Array, "4 4"]
) -> Float[Array, "2 2"]:
    return matrix[2:, 0:2]

@jax.jit
def getS12(
    matrix: Float[Array, "4 4"]
) -> Float[Array, "2 2"]:
    return matrix[0:2, 2:]

@jax.jit
def getS22(
    matrix: Float[Array, "4 4"]
) -> Float[Array, "2 2"]:
    return matrix[2:, 2:]

# -------------------------------------------------------------
# ======================= ↑ Getters ↑ =========================
# -------------------------------------------------------------

# -------------------------------------------------------------
# ======================= ↓ Setters ↓ =========================
# -------------------------------------------------------------

@jax.jit 
def setS11(
    matrix: Float[Array, "4 4"],
    S11: Float[Array, "2 2"]
) -> Float[Array, "4 4"]:
    return matrix.at[0:2, 0:2].set(S11)

@jax.jit
def setS21(
    matrix: Float[Array, "4 4"],
    S21: Float[Array, "2 2"]
) -> Float[Array, "4 4"]:
    return matrix.at[2:, 0:2].set(S21)

@jax.jit
def setS12(
    matrix: Float[Array, "4 4"],
    S12: Float[Array, "2 2"]
) -> Float[Array, "4 4"]:
    return matrix.at[0:2, 2:].set(S12)

@jax.jit
def setS22(
    matrix: Float[Array, "4 4"],
    S22: Float[Array, "2 2"]
) -> Float[Array, "4 4"]:
    return matrix.at[2:, 2:].set(S22)

# -------------------------------------------------------------
# ======================= ↑ Setters ↑ =========================
# -------------------------------------------------------------

# -------------------------------------------------------------
# ====================== ↓ Constructors ↓ =====================
# -------------------------------------------------------------

@jax.jit
def getQuadBlock(
    S11: Float[Array, "2 2"],
    S21: Float[Array, "2 2"],
    S12: Float[Array, "2 2"],
    S22: Float[Array, "2 2"]
) -> Float[Array, "4 4"]:
    quad_block = initQuadBlock()
    quad_block = setS11(quad_block, S11)
    quad_block = setS21(quad_block, S21)
    quad_block = setS12(quad_block, S12)
    quad_block = setS22(quad_block, S22)
    return quad_block

@jax.jit
def initQuadBlock() -> Float[Array, "4 4"]:
    return jnp.zeros((4,4), dtype = jnp.complex64)

# -------------------------------------------------------------
# ====================== ↑ Constructors ↑ =====================
# -------------------------------------------------------------