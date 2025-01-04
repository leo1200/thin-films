from jaxtyping import Array, Float
from functools import partial
from typing import Tuple

import jax.numpy as jnp

from typing import NamedTuple

# pytree-objects via eqx.Module
import equinox as eqx

# types compatible with tracing
FloatScalar = float | Float[Array, ""]
FloatScalarList = float | Float[Array, ""] | Float[Array, "num_elements"]

class SetupParams(NamedTuple):
    """
    TODO: write class description
    """

    #: Free-space wavelength.
    wavelength: FloatScalar = 632.8

    #: Polar/zenith angle in radians.
    polar_angle: FloatScalar = jnp.deg2rad(25)

    #: Azimuthal angle in radians.
    azimuthal_angle: FloatScalar = jnp.deg2rad(0)

    #: Polarization state.
    polstate: int = 0

class OpticsParams(NamedTuple):
    """
    TODO: write class description
    """

    #: TE polarized component.
    transverse_electric_component: FloatScalar = 1.0

    #: TM polarized component.
    transverse_magnetic_component: FloatScalar = 0.0

    #: Relative permeability (reflection side).
    permeability_reflection: FloatScalar = 1

    #: Relative permittivity (reflection side).
    permittivity_reflection: FloatScalar = 1

    #: Relative permeability (transmission side).
    permeability_transmission: FloatScalar = 1

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

class GrowthModel(NamedTuple):
    """
    TODO: write class description
    """

    #: initial thickness
    initial_thickness: FloatScalar = 0.0

    #: growth velocity
    growth_velocity: FloatScalar = 0.0

    #: growth acceleration
    growth_acceleration: FloatScalar = 0.0