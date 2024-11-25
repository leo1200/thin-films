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
    wavelength: float,
    polar_angle: float,
    azimuthal_angle: float,
    pte: float,
    ptm: float,
    permeability_reflection: float,
    permittivity_reflection: float,
    permeability_transmission: float,
    permittivity_transmission: float,
    transmission_mode: int,
    layer_permeabilities: Float[Array, "num_layers"],
    layer_permittivities: Float[Array, "num_layers"],
    layer_thicknesses: Float[Array, "num_layers"]
) -> Tuple[float, float, float]:
    """
    Transfer Matrix Method for 1D Optical Structures.

    Args:
        wavelength: Free-space wavelength.
        polar_angle: Polar/zenith angle in radians.
        azimuthal_angle: Azimuthal angle in radians.
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
    identity22 = jnp.eye(2, dtype=jnp.complex64)
    zeros22 = jnp.zeros((2, 2), dtype=jnp.complex64)

    # Refractive indices of external regions
    nref = jnp.sqrt(permeability_reflection * permittivity_reflection)
    ntrn = jnp.sqrt(permeability_transmission * permittivity_transmission)

    # Calculate wave vector components
    k0 = 2 * jnp.pi / wavelength
    # Compute normalized wavevector
    kinc = nref * jnp.array([jnp.sin(polar_angle) * jnp.cos(azimuthal_angle), jnp.sin(polar_angle) * jnp.sin(azimuthal_angle), jnp.cos(polar_angle)])

    # Extract components
    kx, ky = kinc[0], kinc[1]

    # Calculate z-component of wave vector in reflection region
    kzref = jnp.sqrt(permeability_reflection * permittivity_reflection - kx**2 - ky**2)
    kztrn = jnp.sqrt(permeability_transmission * permittivity_transmission - kx**2 - ky**2)

    # Eigen-modes in the gap medium
    Q = jnp.array([[kx * ky, 1 + ky**2], [-1 - kx**2, -kx * ky]])
    Vg = -1j * Q

    # Initialize global scattering matrix
    SG = getQuadBlock(
        S11 = zeros22,
        S12 = identity22,
        S21 = identity22,
        S22 = zeros22
    )

    # Combine parameters for each layer
    layer_params = jnp.stack([layer_permeabilities, layer_permittivities, layer_thicknesses], axis=-1)
    
    def compute_layer(SG, params):
        """
        Compute the scattering matrix for a single layer and update the global scattering matrix.
        """
        ur, er, l = params  # Unpack layer-specific parameters
        
        Q = (1 / ur) * jnp.array([[kx * ky, ur * er - kx**2],
                                  [ky**2 - ur * er, -kx * ky]])
        kz = jnp.sqrt(ur * er - kx**2 - ky**2)
        OMEGA = 1j * kz * identity22
        V = Q @ inverse22(OMEGA)
        X = jnp.diag(jnp.exp(jnp.diag(OMEGA) * k0 * l))
        
        # Layer scattering matrix
        A = identity22 + inverse22(V) @ Vg
        B = identity22 - inverse22(V) @ Vg
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
    Q = (1 / permeability_reflection) * jnp.array([[kx * ky, permeability_reflection * permittivity_reflection - kx**2],
                               [ky**2 - permeability_reflection * permittivity_reflection, -kx * ky]])
    OMEGA = 1j * kzref * identity22
    Vref = Q @ inverse22(OMEGA)

    # Reflection-side scattering matrix
    A = identity22 + inverse22(Vg) @ Vref
    B = identity22 - inverse22(Vg) @ Vref

    SR = getQuadBlock(
        S11 = -inverse22(A) @ B,
        S12 = 2 * inverse22(A),
        S21 = 0.5 * (A - B @ inverse22(A) @ B),
        S22 = B @ inverse22(A)
    )

    # Backside transmission/reflection
    if transmission_mode == 1:
        Q = (1 / permeability_transmission) * jnp.array([[kx * ky, permeability_transmission * permittivity_transmission - kx**2],
                                   [ky**2 - permeability_transmission * permittivity_transmission, -kx * ky]])
        OMEGA = 1j * kztrn * identity22
        Vtrn = Q @ inverse22(OMEGA)
        A = identity22 + inverse22(Vg) @ Vtrn
        B = identity22 - inverse22(Vg) @ Vtrn

        ST = getQuadBlock(
            S11 = B @ inverse22(A),
            S12 = 0.5 * (A - B @ inverse22(A) @ B),
            S21 = 2 * inverse22(A),
            S22 = -inverse22(A) @ B
        )
    elif transmission_mode == 0:
        ST = getQuadBlock(
            S11 = identity22,
            S12 = zeros22,
            S21 = zeros22,
            S22 = identity22
        )
    elif transmission_mode == -1:
        ST = getQuadBlock(
            S11 = identity22,
            S12 = zeros22,
            S21 = zeros22,
            S22 = -identity22
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
    ate = lax.cond(abs(polar_angle) < 1e-6, branch_zero, branch_nonzero)

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
    TRN = jnp.linalg.norm(Etrn)**2 * jnp.real(permeability_reflection / permeability_transmission * kztrn / kzref)
    CON = REF + TRN

    return REF, TRN, CON