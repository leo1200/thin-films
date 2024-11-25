from functools import partial
import jax
import jax.numpy as jnp
from reflax.transfer_matrix_method import transfer_matrix_method

@partial(jax.jit, static_argnames=['backside_mode', 'final_time'])
def one_growing_layer(
    final_time,
    permeability_growing_layer,
    layer_permeabilities,
    permittivity_growing_layer,
    layer_permittivities,
    layer_thicknesses,
    growth_rate,
    wavelength,
    polar_angle,
    azimuthal_angle,
    transverse_electric_component,
    transverse_magnetic_component,
    permeability_reflection,
    permittivity_reflection,
    permeability_transmission,
    permittivity_transmission,
    backside_mode
):
    """
    Simulates the 0-th layer growing linearly at rate v.
    """

    num_points = final_time + 1

    REFtil = jnp.zeros(num_points)
    TRNtil = jnp.zeros(num_points)
    CONtil = jnp.zeros(num_points)
    calt = jnp.zeros(num_points)
    calL = jnp.zeros(num_points)

    layer_permeabilities = jnp.concatenate((jnp.array([permeability_growing_layer]), layer_permeabilities))
    layer_permittivities = jnp.concatenate((jnp.array([permittivity_growing_layer]), layer_permittivities))

    L_mat = jnp.zeros((num_points, layer_thicknesses.shape[0] + 1))
    L_mat = L_mat.at[:, 1:].set(layer_thicknesses)

    # Generate the time values that meet the condition
    t_values = jnp.arange(0, final_time + 1)

    # Update the thickness array for all selected times
    # 0-th layer grows linearly
    L_mat = L_mat.at[:, 0].set(growth_rate * t_values)
    
    # Vectorized computation using vmap for `tmm1d` across all selected times
    REFtil, TRNtil, CONtil = jax.vmap(
        lambda l: transfer_matrix_method(wavelength, polar_angle, azimuthal_angle, transverse_electric_component, transverse_magnetic_component, permeability_reflection, permittivity_reflection, permeability_transmission, permittivity_transmission, backside_mode, layer_permeabilities, layer_permittivities, l)
    )(L_mat)

    # Capture the corresponding time and thickness values
    calt = t_values
    calL = L_mat[:, 0]  # First element corresponds to the variable layer thickness

    return REFtil, TRNtil, CONtil, calt, calL