import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
from functools import partial
from typing import Tuple
import jax.lax as lax

from reflax._reflectance_models.basic_optics import calculate_reflection_coeff, snell
from reflax.constants import S_POLARIZED
from reflax.parameter_classes.parameters import LayerParams, OpticsParams, SetupParams

@partial(jax.jit, static_argnames=['polarization_state'])
def one_layer_model(
    setup_params: SetupParams,
    optics_params: OpticsParams,
    layer_params: LayerParams,
    polarization_state: int = S_POLARIZED,
) -> Float[Array, ""]: # Returns scalar float reflectivity
    """
    Calculates the reflectance of a single thin layer including multiple
    internal reflections (coherent sum).

    Assumes permeability and permittivity parameters are relative values (mu_r, epsilon_r).


    """
    # Calculate refractive indices (assuming relative permeability/permittivity)
    # Using complex type for generality (handles absorbing materials if mu/epsilon are complex)
    n0 = jnp.sqrt(optics_params.permeability_reflection * optics_params.permittivity_reflection)
    n1 = jnp.sqrt(layer_params.permeabilities * layer_params.permittivities)
    n2 = jnp.sqrt(optics_params.permeability_transmission * optics_params.permittivity_transmission)

    # Ensure angle is float
    theta_0 = setup_params.polar_angle
    wavelength = setup_params.wavelength
    thickness = layer_params.thicknesses

    # Angle inside the layer (complex if n1 or theta_0 leads to complex sin > 1)
    theta_1 = snell(n0, n1, theta_0)

    # Phase shift for a single pass across the layer thickness along the ray path
    # cos(theta_1) = sqrt(1 - sin^2(theta_1)) = sqrt(1 - (n0/n1 * sin(theta_0))^2)
    # n1 * cos(theta_1) = sqrt(n1^2 - n0^2 * sin^2(theta_0))
    # Corrected: Removed jnp.pi - ... from sin argument
    cos_theta_1 = jnp.sqrt(1.0 - (n0 / n1 * jnp.sin(theta_0))**2) # More direct way to get cos(theta_1)
    # Alternative for n1*cos(theta1) - careful with branch cuts of sqrt for complex n
    # n1_cos_theta1 = jnp.sqrt(n1**2 - (n0 * jnp.sin(theta_0))**2)

    # Phase change beta = k_layer * thickness * cos(theta_1) = (2*pi*n1/lambda) * d * cos(theta_1)
    # Ensure operations are done in complex numbers
    beta = (2 * jnp.pi / wavelength) * n1 * thickness * cos_theta_1
    # beta = (2 * jnp.pi / wavelength) * thickness * n1_cos_theta1 # Alternative

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

    # Return real-valued reflectance. Handle potential NaNs if inputs were problematic.
    return jnp.real(Reflectance)