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
from reflax.parameter_classes.parameters import IncidentMediumParams, LayerParams, LightSourceParams, SetupParams, TransmissionMediumParams

@partial(jax.jit, static_argnames=['backside_mode'])
def transfer_matrix_method(
    setup_params: SetupParams,
    light_source_params: LightSourceParams,
    incident_medium_params: IncidentMediumParams,
    transmission_medium_params: TransmissionMediumParams,
    layer_params: LayerParams,
    backside_mode: int
) -> Tuple[float, float, float]:
    """
    Transfer Matrix Method for 1D Optical Structures.

    Args:
        setup_params: Parameters of the setup.
        optics_params: Parameters of the optics.
        layer_params: Parameters of the layers.
        backside_mode: Decision variable for backside transmission/reflection
                       (-1: reflection with phase inversion, 0: reflection, 1: transmission)

    Returns:
        Reflectance (REF), Transmittance (TRN), Conservation (CON).
    """
    type = jnp.complex128 if jax.config.jax_enable_x64 else jnp.complex64
    identity22 = jnp.eye(2, dtype = type)
    zeros22 = jnp.zeros((2, 2), dtype = type)

    # Refractive indices of external regions
    nref = jnp.sqrt(incident_medium_params.permeability_reflection * incident_medium_params.permittivity_reflection)
    ntrn = jnp.sqrt(transmission_medium_params.permeability_transmission * transmission_medium_params.permittivity_transmission)

    # Calculate wave vector components
    k0 = 2 * jnp.pi / light_source_params.wavelength
    # Compute normalized wavevector
    kinc = nref * jnp.array([jnp.sin(setup_params.polar_angle) * jnp.cos(setup_params.azimuthal_angle), jnp.sin(setup_params.polar_angle) * jnp.sin(setup_params.azimuthal_angle), jnp.cos(setup_params.polar_angle)], dtype = type)

    # Extract components
    kx, ky = kinc[0], kinc[1]

    # Calculate z-component of wave vector in reflection region
    kzref = jnp.sqrt(incident_medium_params.permeability_reflection * incident_medium_params.permittivity_reflection - kx**2 - ky**2)
    kztrn = jnp.sqrt(transmission_medium_params.permeability_transmission * transmission_medium_params.permittivity_transmission - kx**2 - ky**2)

    # Eigen-modes in the gap medium
    Q = jnp.array([[kx * ky, 1 + ky**2], [-1 - kx**2, -kx * ky]], dtype = type)
    Vg = -1j * Q

    # Initialize global scattering matrix
    SG = getQuadBlock(
        S11 = zeros22,
        S12 = identity22,
        S21 = identity22,
        S22 = zeros22
    )

    # Combine parameters for each layer
    layer_params = jnp.stack([layer_params.permeabilities, layer_params.permittivities, layer_params.thicknesses], axis=-1)
    
    def compute_layer(SG, params):
        """
        Compute the scattering matrix for a single layer and update the global scattering matrix.
        """
        ur, er, l = params  # Unpack layer-specific parameters
        
        Q = (1 / ur) * jnp.array([[kx * ky, ur * er - kx**2],
                                  [ky**2 - ur * er, -kx * ky]], dtype = type)
        kz = jnp.sqrt(ur * er - kx ** 2 - ky ** 2) # * jnp.sqrt(ur * er)
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
    Q = (1 / incident_medium_params.permeability_reflection) * jnp.array([[kx * ky, incident_medium_params.permeability_reflection * incident_medium_params.permittivity_reflection - kx**2],
                               [ky**2 - incident_medium_params.permeability_reflection * incident_medium_params.permittivity_reflection, -kx * ky]], dtype = type)
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
    if backside_mode == 1:
        Q = (1 / transmission_medium_params.permeability_transmission) * jnp.array([[kx * ky, transmission_medium_params.permeability_transmission * transmission_medium_params.permittivity_transmission - kx**2],
                                   [ky**2 - transmission_medium_params.permeability_transmission * transmission_medium_params.permittivity_transmission, -kx * ky]])
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
    elif backside_mode == 0:
        ST = getQuadBlock(
            S11 = identity22,
            S12 = zeros22,
            S21 = zeros22,
            S22 = identity22
        )
    elif backside_mode == -1:
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
        return jnp.array([0.0, 1.0, 0.0], dtype = type)

    # Define the branch for abs(theta) >= 1e-6
    def branch_nonzero():
        return jnp.cross(n, kinc) / jnp.linalg.norm(jnp.cross(n, kinc))

    # Use lax.cond to choose between the two branches
    ate = lax.cond(abs(setup_params.polar_angle) < 1e-6, branch_zero, branch_nonzero)

    atm = jnp.cross(ate, kinc)
    atm /= jnp.linalg.norm(atm)
    
    P = light_source_params.s_component * ate + light_source_params.p_component * atm
    P /= jnp.linalg.norm(P)

    # Reflected and transmitted fields
    Esrc = P[:2]

    Eref = jnp.zeros((3,), dtype = type)
    Etrn = jnp.zeros((3,), dtype = type)

    Eref = Eref.at[:2].set(getS11(SG) @ Esrc)
    Etrn = Etrn.at[:2].set(getS21(SG) @ Esrc)

    Eref = Eref.at[2].set(-(kx * Eref[0] + ky * Eref[1]) / kzref)
    Etrn = Etrn.at[2].set(-(kx * Etrn[0] + ky * Etrn[1]) / kztrn)
    
    # Calculate reflectance and transmittance
    REF = jnp.linalg.norm(Eref)**2
    TRN = jnp.linalg.norm(Etrn)**2 * jnp.real(incident_medium_params.permeability_reflection / transmission_medium_params.permeability_transmission * kztrn / kzref)
    CON = REF + TRN

    return REF, TRN, CON