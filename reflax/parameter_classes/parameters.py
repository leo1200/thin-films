from jaxtyping import Array, Float
from functools import partial
from typing import Tuple

import jax.numpy as jnp

from typing import NamedTuple

from reflax.constants import S_POLARIZED

# types compatible with tracing
FloatScalar = float | Float[Array, ""]
FloatScalarList = float | Float[Array, ""] | Float[Array, "num_elements"]

class SetupParams(NamedTuple):
    """
    TODO: write class description
    """

    #: Polar/zenith angle in radians.
    polar_angle: FloatScalar = jnp.deg2rad(25)

    #: Azimuthal angle in radians.
    azimuthal_angle: FloatScalar = jnp.deg2rad(0)

class LightSourceParams(NamedTuple):
    
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
    TODO: write class description
    """

    #: Layer thicknesses
    thicknesses: FloatScalarList = 0.0

    #: Layer permeabilities
    permeabilities: FloatScalarList = 0.0

    #: Layer permittivities
    permittivities: FloatScalarList = 0.0