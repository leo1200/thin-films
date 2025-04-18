import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import numpy as np
import jax.numpy as jnp

import matplotlib.pyplot as plt

from reflax import polanalyze
from reflax.parameter_classes.parameters import OpticsParams, SetupParams, LayerParams
from reflax.forward_model.variable_layer_size import MIN_MAX_NORMALIZATION, MULTIPLE_LAYERS_INTERNAL_REFLECTIONS, ONE_LAYER_INTERNAL_REFLECTIONS, TRANSFER_MATRIX_METHOD, forward_model

from reflax.parameter_classes.parameters import SetupParams


# load the data
measurement = np.loadtxt("reflectance.txt", skiprows=1)
time = jnp.array(measurement[:-100, 0])
time = time / 3600  # convert to hours
reflectance = jnp.array(measurement[:-100, 1])

normalization = MIN_MAX_NORMALIZATION

if normalization == MIN_MAX_NORMALIZATION:
    reflectance = (reflectance - 0.5 * (jnp.min(reflectance) + jnp.max(reflectance))) / (0.5 * (jnp.max(reflectance) - jnp.min(reflectance)))


# plot the data
fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex = True)
axs[0].plot(time, reflectance, '-', label = "measurement", color = "black")
axs[0].set_ylabel("normalized reflectance")
# model

wavelength = 632.8
polar_angle = jnp.deg2rad(25)
azimuthal_angle = jnp.deg2rad(0)

setup_params = SetupParams(
    wavelength = wavelength,
    polar_angle = polar_angle,
    azimuthal_angle = azimuthal_angle
)

polarization_state = "Linear TE/perpendicular/s"
transverse_electric_component, transverse_magnetic_component = polanalyze(polarization_state)

permeability_reflection = 1
permittivity_reflection = 1

permeability_transmission = 1
permittivity_transmission = (3.8827 + 0.019626j)**2

optics_params = OpticsParams(
    permeability_reflection = permeability_reflection,
    permittivity_reflection = permittivity_reflection,
    permeability_transmission = permeability_transmission,
    permittivity_transmission = permittivity_transmission,
    transverse_electric_component = transverse_electric_component,
    transverse_magnetic_component = transverse_magnetic_component
)

backside_mode = 1

static_layer_thicknesses = jnp.array([0.0])
permeability_static_size_layers = jnp.array([1.0])
permittivity_static_size_layers = jnp.array([1.45704**2])

static_layer_params = LayerParams(
    permeabilities = permeability_static_size_layers,
    permittivities = permittivity_static_size_layers,
    thicknesses = static_layer_thicknesses
)

permeability_variable_layer = 1
permittivity_variable_layer = 1.57**2

variable_layer_params = LayerParams(
    permeabilities = permeability_variable_layer,
    permittivities = permittivity_variable_layer
)


layer_growing_rate = 0.2 * 3600
variable_layer_thicknesses = time * layer_growing_rate

# variable_layer_thicknesses = layer_growing_rate * (jnp.log(1 + jnp.exp(5 * time - 2.0)) - jnp.log(1 + jnp.exp(-2.0)))

reflection_coefficients = forward_model(
    model = ONE_LAYER_INTERNAL_REFLECTIONS,
    setup_params = setup_params,
    optics_params = optics_params,
    static_layer_params = static_layer_params,
    variable_layer_params = variable_layer_params,
    variable_layer_thicknesses = variable_layer_thicknesses,
    backside_mode = backside_mode,
    normalization = normalization
)

# plot the model
axs[0].plot(time, reflection_coefficients, '-', label = "model", color = "blue")

axs[0].legend()

# plot the variable layer thicknesses in the second subplot
axs[1].plot(time, variable_layer_thicknesses, '-', label = "modelled layer thickness", color = "blue")
axs[1].set_xlabel("time in hours")
axs[1].set_ylabel("layer thickness in nm")
axs[1].legend()

plt.savefig("figures/measurement.png")

