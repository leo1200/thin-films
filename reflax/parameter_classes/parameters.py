from jaxtyping import Array, Float
from functools import partial
from typing import Tuple

import jax.numpy as jnp

from typing import NamedTuple

from reflax.constants import MIN_MAX_NORMALIZATION, ONE_LAYER_MODEL, S_POLARIZED

# types compatible with tracing
FloatScalar = float | Float[Array, ""]
FloatScalarList = float | Float[Array, ""] | Float[Array, "num_elements"]

class SetupParams(NamedTuple):
    """
    Parameters of the experimental setup.
    """

    #: Polar/zenith angle in radians.
    polar_angle: FloatScalar = jnp.deg2rad(25)

    #: Azimuthal angle in radians.
    azimuthal_angle: FloatScalar = jnp.deg2rad(0)

class LightSourceParams(NamedTuple):
    """
    Parameters of the light source.
    """
    
    #: Wavelength in nm.
    wavelength: FloatScalar = 632.8

    #: TE polarized component.
    s_component: FloatScalar = 1.0

    #: TM polarized component.
    p_component: FloatScalar = 0.0

class IncidentMediumParams(NamedTuple):
    """Medium the light is coming from."""

    #: Relative permeability (reflection side).
    permeability_reflection: FloatScalar = 1.0

    #: Relative permittivity (reflection side).
    permittivity_reflection: FloatScalar = 1.0

class TransmissionMediumParams(NamedTuple):
    """Medium the light enters after passing through the layers (subsrate)."""

    #: Relative permeability (transmission side).
    permeability_transmission: FloatScalar = 1.0

    #: Relative permittivity (transmission side).
    permittivity_transmission: FloatScalar = (3.8827 + 0.019626j)**2

class LayerParams(NamedTuple):
    """
    Parameters of the layer.
    """

    #: Layer thicknesses
    thicknesses: FloatScalarList = 0.0

    #: Layer permeabilities
    permeabilities: FloatScalarList = 0.0

    #: Layer permittivities
    permittivities: FloatScalarList = 0.0

class ForwardModelParams(NamedTuple):
    """
    Collection of all parameters needed for the forward model.
    """

    #: Model to use for the forward calculation.
    #: ONE_LAYER_MODEL or TRANSFER_MATRIX_METHOD.
    model: int = ONE_LAYER_MODEL

    #: Setup parameters (angle of incidence, etc.).
    setup_params: SetupParams = SetupParams()

    #: Light source parameters (wavelength, etc.).
    light_source_params: LightSourceParams = LightSourceParams()

    #: Incident medium parameters (permittivity, permeability).
    incident_medium_params: IncidentMediumParams = IncidentMediumParams()

    #: Transmission medium parameters (permittivity, permeability).
    transmission_medium_params: TransmissionMediumParams = TransmissionMediumParams()

    #: Static layer parameters (permittivities, permeabilities).
    static_layer_params: LayerParams = LayerParams()

    #: Variable layer parameters (permittivities, permeabilities).
    variable_layer_params: LayerParams = LayerParams()

    #: backside mode (0 or 1).
    backside_mode: int = 0

    #: Polarization state (only applies to ONE_LAYER_MODEL).
    #: S_POLARIZED or P_POLARIZED.
    polarization_state: int = S_POLARIZED

    #: Normalization method,
    #: either NO_NORMALIZATION or MIN_MAX_NORMALIZATION.
    normalization: int = MIN_MAX_NORMALIZATION