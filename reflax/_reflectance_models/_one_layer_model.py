import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
from functools import partial
from typing import Tuple
import jax.lax as lax

from reflax._basic_optics._fresnel_coefficients import calculate_reflection_coeff
from reflax._basic_optics._snells_law import snell
from reflax.constants import P_POLARIZED, S_POLARIZED
from reflax.parameter_classes.parameters import IncidentMediumParams, LayerParams, LightSourceParams, SetupParams, TransmissionMediumParams

@partial(jax.jit, static_argnames=['polarization_state'])
def one_layer_model(
    setup_params: SetupParams,
    light_source_params: LightSourceParams,
    incident_medium_params: IncidentMediumParams,
    transmission_medium_params: TransmissionMediumParams,
    layer_params: LayerParams,
    polarization_state: int = S_POLARIZED,
) -> jnp.ndarray:
    """
    Calculates the reflectance of a single thin layer including multiple
    internal reflections (coherent sum).

    """

    # calculate refractive indices from relative permeability/permittivity
    n0 = jnp.sqrt(incident_medium_params.permeability_reflection * incident_medium_params.permittivity_reflection)
    n1 = jnp.sqrt(layer_params.permeabilities * layer_params.permittivities)
    n2 = jnp.sqrt(transmission_medium_params.permeability_transmission * transmission_medium_params.permittivity_transmission)

    # setup parameters
    theta_0 = setup_params.polar_angle
    wavelength = light_source_params.wavelength
    thickness = layer_params.thicknesses

    # angle inside the layer 
    # (complex if n1 or theta_0 leads to complex sin > 1)
    theta_1 = snell(n0, n1, theta_0)

    # phase calculation
    cos_theta_1 = jnp.sqrt(1.0 - (n0 / n1 * jnp.sin(theta_0))**2)
    beta = (2 * jnp.pi / wavelength) * n1 * thickness * cos_theta_1

    # Calculate Fresnel reflection coefficients at interfaces
    # r01: reflection at 0 -> 1 interface, incident angle theta_0
    r01 = calculate_reflection_coeff(n0, n1, theta_0, polarization_state)
    # r12: reflection at 1 -> 2 interface, incident angle theta_1
    r12 = calculate_reflection_coeff(n1, n2, theta_1, polarization_state)

    # Reflection coefficient for the entire layer system
    # Formula: (r01 + r12 * exp(2j * beta)) / (1 + r01 * r12 * exp(2j * beta))
    # Using r10 = -r01 property implicitly in the standard formula.
    numerator = r01 + r12 * jnp.exp(2j * beta)
    denominator = 1 + r01 * r12 * jnp.exp(2j * beta)

    # Total complex reflection amplitude
    r_total = numerator / denominator

    # Reflectance (Intensity): R = |r_total|^2
    Reflectance = jnp.abs(r_total)**2

    # Return real-valued reflectance.
    return jnp.real(Reflectance)


def get_polarization_components(polarization_state):
    """
    Returns the S and P components aka Jones vector.
    """
    if polarization_state == S_POLARIZED:
        return 1.0, 0.0
    elif polarization_state == P_POLARIZED:
        return 0.0, 1.0
    else:
        raise ValueError("Unsupported polarization state")