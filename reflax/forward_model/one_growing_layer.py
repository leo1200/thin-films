from functools import partial
import jax
import jax.numpy as jnp
from reflax.transfer_matrix_method import transfer_matrix_method

@partial(jax.jit, static_argnames=['trn0', 'dt'])
def one_growing_layer(dt, urv, UR, erv, ER, L, v, lam0, theta, phi, pte, ptm, ur1, er1, ur2, er2, trn0):
    """
    Simulates the 0-th layer growing linearly at rate v.
    """

    num_points = dt + 1

    REFtil = jnp.zeros(num_points)
    TRNtil = jnp.zeros(num_points)
    CONtil = jnp.zeros(num_points)
    calt = jnp.zeros(num_points)
    calL = jnp.zeros(num_points)

    UR = jnp.concatenate((jnp.array([urv]), UR))
    ER = jnp.concatenate((jnp.array([erv]), ER))

    L_mat = jnp.zeros((num_points, L.shape[0] + 1))
    L_mat = L_mat.at[:, 1:].set(L)

    # Generate the time values that meet the condition
    t_values = jnp.arange(0, dt + 1)

    # Update the thickness array for all selected times
    # 0-th layer grows linearly
    L_mat = L_mat.at[:, 0].set(v * t_values)
    
    # Vectorized computation using vmap for `tmm1d` across all selected times
    REFtil, TRNtil, CONtil = jax.vmap(
        lambda l: transfer_matrix_method(lam0, theta, phi, pte, ptm, ur1, er1, ur2, er2, trn0, UR, ER, l)
    )(L_mat)

    # Capture the corresponding time and thickness values
    calt = t_values
    calL = L_mat[:, 0]  # First element corresponds to the variable layer thickness

    return REFtil, TRNtil, CONtil, calt, calL