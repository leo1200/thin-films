import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
from functools import partial
from typing import Tuple
import jax.lax as lax

from reflax._matrix_operations import (
    _redheffer_star_product,
    getQuadBlock,
    initQuadBlock,
    inverse22,
    getS11,
    getS12,
    getS21,
    setS22,
    setS21,
    setS11,
    setS12
)

@partial(jax.jit, static_argnames=['trn0'])
def transfer_matrix_method(
    lam0: float,
    theta: float,
    phi: float,
    pte: float,
    ptm: float,
    ur1: float,
    er1: float,
    ur2: float,
    er2: float,
    trn0: int,
    UR: Float[Array, "num_layers"],
    ER: Float[Array, "num_layers"],
    L: Float[Array, "num_layers"]
) -> Tuple[float, float, float]:
    """
    Transfer Matrix Method for 1D Optical Structures.

    Args:
        lam0: Free-space wavelength.
        theta: Polar/zenith angle in radians.
        phi: Azimuthal angle in radians.
        pte: TE polarized component.
        ptm: TM polarized component.
        ur1: Relative permeability (reflection side).
        er1: Relative permittivity (reflection side).
        ur2: Relative permeability (transmission side).
        er2: Relative permittivity (transmission side).
        trn0: Decision variable for backside transmission/reflection (-1, 0, or 1).
        UR: Relative permeabilities of layers.
        ER: Relative permittivities of layers.
        L: Thicknesses of layers.

    Returns:
        Tuple[float, float, float]: Reflectance (REF), Transmittance (TRN), Conservation (CON).
    """
    I = jnp.eye(2, dtype=jnp.complex64)
    Z = jnp.zeros((2, 2), dtype=jnp.complex64)

    # Refractive indices of external regions
    nref = jnp.sqrt(ur1 * er1)
    ntrn = jnp.sqrt(ur2 * er2)

    # Calculate wave vector components
    k0 = 2 * jnp.pi / lam0
    # Compute normalized wavevector
    kinc = nref * jnp.array([jnp.sin(theta) * jnp.cos(phi), jnp.sin(theta) * jnp.sin(phi), jnp.cos(theta)])

    # Extract components
    kx, ky = kinc[0], kinc[1]

    # Calculate z-component of wave vector in reflection region
    kzref = jnp.sqrt(ur1 * er1 - kx**2 - ky**2)
    kztrn = jnp.sqrt(ur2 * er2 - kx**2 - ky**2)

    # Eigen-modes in the gap medium
    Q = jnp.array([[kx * ky, 1 + ky**2], [-1 - kx**2, -kx * ky]])
    Vg = -1j * Q

    # Initialize global scattering matrix
    SG = getQuadBlock(
        S11 = Z,
        S12 = I,
        S21 = I,
        S22 = Z
    )

    # Combine parameters for each layer
    layer_params = jnp.stack([UR, ER, L], axis=-1)
    
    def compute_layer(SG, params):
        """
        Compute the scattering matrix for a single layer and update the global scattering matrix.
        """
        ur, er, l = params  # Unpack layer-specific parameters
        
        Q = (1 / ur) * jnp.array([[kx * ky, ur * er - kx**2],
                                  [ky**2 - ur * er, -kx * ky]])
        kz = jnp.sqrt(ur * er - kx**2 - ky**2)
        OMEGA = 1j * kz * I
        V = Q @ inverse22(OMEGA)
        X = jnp.diag(jnp.exp(jnp.diag(OMEGA) * k0 * l))
        
        # Layer scattering matrix
        A = I + inverse22(V) @ Vg
        B = I - inverse22(V) @ Vg
        D = A - X @ B @ inverse22(A) @ X @ B
        
        S = initQuadBlock()
        S = setS11(S, inverse22(D) @ (X @ B @ inverse22(A) @ X @ A - B))
        S = setS12(S, inverse22(D) @ X @ (A - B @ inverse22(A) @ B))
        S = setS21(S, getS12(S))
        S = setS22(S, getS11(S))
        
        # Update global scattering matrix
        SG = _redheffer_star_product(SG, S)
        
        return SG, None  # Return updated SG and None as output

    # Run lax.scan
    SG, _ = jax.lax.scan(compute_layer, SG, layer_params)

    # Reflection region eigen-modes
    Q = (1 / ur1) * jnp.array([[kx * ky, ur1 * er1 - kx**2],
                               [ky**2 - ur1 * er1, -kx * ky]])
    OMEGA = 1j * kzref * I
    Vref = Q @ inverse22(OMEGA)

    # Reflection-side scattering matrix
    A = I + inverse22(Vg) @ Vref
    B = I - inverse22(Vg) @ Vref

    SR = getQuadBlock(
        S11 = -inverse22(A) @ B,
        S12 = 2 * inverse22(A),
        S21 = 0.5 * (A - B @ inverse22(A) @ B),
        S22 = B @ inverse22(A)
    )

    # Backside transmission/reflection
    if trn0 == 1:
        Q = (1 / ur2) * jnp.array([[kx * ky, ur2 * er2 - kx**2],
                                   [ky**2 - ur2 * er2, -kx * ky]])
        OMEGA = 1j * kztrn * I
        Vtrn = Q @ inverse22(OMEGA)
        A = I + inverse22(Vg) @ Vtrn
        B = I - inverse22(Vg) @ Vtrn

        ST = getQuadBlock(
            S11 = B @ inverse22(A),
            S12 = 0.5 * (A - B @ inverse22(A) @ B),
            S21 = 2 * inverse22(A),
            S22 = -inverse22(A) @ B
        )
    elif trn0 == 0:
        ST = getQuadBlock(
            S11 = I,
            S12 = Z,
            S21 = Z,
            S22 = I
        )
    elif trn0 == -1:
        ST = getQuadBlock(
            S11 = I,
            S12 = Z,
            S21 = Z,
            S22 = -I
        )

    # Connect global scattering matrix to external regions
    SG = _redheffer_star_product(SR, SG)
    SG = _redheffer_star_product(SG, ST)

    # Polarization vector
    n = jnp.array([0, 0, 1])

    # Define the branch for abs(theta) < 1e-6
    def branch_zero():
        return jnp.array([0.0, 1.0, 0.0])

    # Define the branch for abs(theta) >= 1e-6
    def branch_nonzero():
        return jnp.cross(n, kinc) / jnp.linalg.norm(jnp.cross(n, kinc))

    # Use lax.cond to choose between the two branches
    ate = lax.cond(abs(theta) < 1e-6, branch_zero, branch_nonzero)

    atm = jnp.cross(ate, kinc)
    atm /= jnp.linalg.norm(atm)
    
    P = pte * ate + ptm * atm
    P /= jnp.linalg.norm(P)

    # Reflected and transmitted fields
    Esrc = P[:2]

    Eref = jnp.zeros((3,), dtype = jnp.complex64)
    Etrn = jnp.zeros((3,), dtype = jnp.complex64)

    Eref = Eref.at[:2].set(getS11(SG) @ Esrc)
    Etrn = Etrn.at[:2].set(getS21(SG) @ Esrc)

    Eref = Eref.at[2].set(-(kx * Eref[0] + ky * Eref[1]) / kzref)
    Etrn = Etrn.at[2].set(-(kx * Etrn[0] + ky * Etrn[1]) / kztrn)
    
    # Calculate reflectance and transmittance
    REF = jnp.linalg.norm(Eref)**2
    TRN = jnp.linalg.norm(Etrn)**2 * jnp.real(ur1 / ur2 * kztrn / kzref)
    CON = REF + TRN

    return REF, TRN, CON