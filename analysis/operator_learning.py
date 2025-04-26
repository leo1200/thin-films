"""

The problem to solve: Optimization through the simulator can be slow
as of the complex loss. For broad adoption of our method, quickly
finding the growth behavior is key. Using that the phase is related
to the growth by determining the instantaneous frequency (Hilbert
transform, ChirpGP) is often not feasible as we only have a few periods
and our signal form is very specific (one might be able to generalize
ChirpGP to different priors though).

Solution: Train a neural operator to infer the growth behavior and
tackle the details using optimization through the simulator.

Reduction to a minimal viable product: We will only consider one baseline
simulation setup, consider depositions of one hour with average growth of
800 nm/h up to 1000 nm/h. We will use a fixed time resolution.

"""

# GPU selection
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# typing
from typing import Tuple

# timing
import time as pytime

# numerics
import numpy as np
import jax.numpy as jnp

# neural nets 
from flax import nnx
import optax

# reflax (our library)
from reflax import polanalyze
from reflax.parameter_classes.parameters import (
    OpticsParams,
    SetupParams,
    LayerParams
)
from reflax.forward_model.variable_layer_size import (
    MIN_MAX_NORMALIZATION,
    ONE_LAYER_INTERNAL_REFLECTIONS,
    TRANSFER_MATRIX_METHOD
)
from reflax.forward_model.variable_layer_size import forward_model

# plotting
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# ===================== ↓ simulator setup ↓ ===================
# -------------------------------------------------------------

wavelength = 632.8
polar_angle = jnp.deg2rad(25)
azimuthal_angle = jnp.deg2rad(0)
setup_params = SetupParams(
    wavelength=wavelength,
    polar_angle=polar_angle,
    azimuthal_angle=azimuthal_angle
)

polarization_state = "Linear TE/perpendicular/s"
transverse_electric_component, transverse_magnetic_component = polanalyze(polarization_state)
permeability_reflection = 1.0
permittivity_reflection = complex(1.0, 0.0)
permeability_transmission = 1.0
permittivity_transmission = (3.8827 + 0.019626j)**2
optics_params = OpticsParams(
    permeability_reflection=permeability_reflection,
    permittivity_reflection=permittivity_reflection,
    permeability_transmission=permeability_transmission,
    permittivity_transmission=permittivity_transmission,
    transverse_electric_component=transverse_electric_component,
    transverse_magnetic_component=transverse_magnetic_component
)

backside_mode = 1

static_layer_thicknesses = jnp.array([0.0])
permeability_static_size_layers = jnp.array([permeability_transmission])
permittivity_static_size_layers = jnp.array([permittivity_transmission])
static_layer_params = LayerParams(
    permeabilities=permeability_static_size_layers,
    permittivities=permittivity_static_size_layers,
    thicknesses=static_layer_thicknesses
)

permeability_variable_layer = 1.0
permittivity_variable_layer = complex(1.57, 0.0)**2
variable_layer_params = LayerParams(
    permeabilities = permeability_variable_layer,
    permittivities = permittivity_variable_layer,
    thicknesses = None
)

# -------------------------------------------------------------
# ==================== ↑ simulator setup ↑ ====================
# -------------------------------------------------------------

# -------------------------------------------------------------
# =============== ↓ simulated data generation ↓ ===============
# -------------------------------------------------------------



# -------------------------------------------------------------
# =============== ↑ simulated data generation ↑ ===============
# -------------------------------------------------------------
