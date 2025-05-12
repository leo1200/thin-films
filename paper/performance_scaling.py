import os
import time
import jax
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)

# Only use GPU device 7
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from reflax import (
    ONE_LAYER_MODEL,
    TRANSFER_MATRIX_METHOD,
    forward_model,
)
from reflax.parameter_classes.parameters import (
    ForwardModelParams,
)
from reflax.thickness_modeling.function_sampling import sample_derivative_bound_gp

# Load baseline setup parameters
from baseline_setup import forward_model_params
setup_params = forward_model_params.setup_params
light_source_params = forward_model_params.light_source_params
incident_medium_params = forward_model_params.incident_medium_params
transmission_medium_params = forward_model_params.transmission_medium_params
static_layer_params = forward_model_params.static_layer_params
variable_layer_params = forward_model_params.variable_layer_params
backside_mode = forward_model_params.backside_mode
polarization_state = forward_model_params.polarization_state
normalization = forward_model_params.normalization

# Define loss function: MSE between reflectances

def reflectance_loss(model, thicknessA, thicknessB):
    rA = forward_model(
        model=model,
        setup_params=setup_params,
        light_source_params=light_source_params,
        incident_medium_params=incident_medium_params,
        transmission_medium_params=transmission_medium_params,
        static_layer_params=static_layer_params,
        variable_layer_params=variable_layer_params,
        variable_layer_thicknesses=thicknessA,
        backside_mode=backside_mode,
        polarization_state=polarization_state,
        normalization=normalization,
    )
    rB = forward_model(
        model=model,
        setup_params=setup_params,
        light_source_params=light_source_params,
        incident_medium_params=incident_medium_params,
        transmission_medium_params=transmission_medium_params,
        static_layer_params=static_layer_params,
        variable_layer_params=variable_layer_params,
        variable_layer_thicknesses=thicknessB,
        backside_mode=backside_mode,
        polarization_state=polarization_state,
        normalization=normalization,
    )
    return jnp.mean((rA - rB) ** 2)

# JIT compile the loss and its gradient

def make_funcs(model):
    loss_fn = lambda A, B: reflectance_loss(model, A, B)
    # jit-trace
    loss_jit = jax.jit(loss_fn)
    grad_jit = jax.jit(jax.grad(loss_fn, argnums=1))
    return loss_jit, grad_jit

# Benchmarking
if __name__ == '__main__':
    # Range of num_points
    num_points_list = [100, 1000, 5000, 10000, 50000, 100000]

    results = {
        'one_layer': {'forward': [], 'backward': []},
        'transfer_matrix': {'forward': [], 'backward': []},
    }

    for n in num_points_list:
        print(f"\nBenchmarking for num_points = {n}")
        # Generate sample thicknesses
        t = jnp.linspace(0, 1, n)
        keyA, keyB = jax.random.split(jax.random.PRNGKey(0))
        thicknessA, _ = sample_derivative_bound_gp(
            keyA, 1, t, 0.1, 10.0, 200.0, 1800.0,
            random_final_values=True, min_final_value=800., max_final_value=1200., convex_samples=True)
        thicknessB, _ = sample_derivative_bound_gp(
            keyB, 1, t, 0.1, 10.0, 200.0, 1800.0,
            random_final_values=True, min_final_value=800., max_final_value=1200., convex_samples=True)
        
        # squeeze to remove extra dimension
        thicknessA = jnp.squeeze(thicknessA)
        thicknessB = jnp.squeeze(thicknessB)

        for model_name, model_const in [('one_layer', ONE_LAYER_MODEL),
                                        ('transfer_matrix', TRANSFER_MATRIX_METHOD)]:
            print(f"  Model: {model_name}")
            loss_jit, grad_jit = make_funcs(model_const)

            # Warm-up
            _ = loss_jit(thicknessA, thicknessB)
            _ = grad_jit(thicknessA, thicknessB)

            # Time forward
            t0 = time.time()
            _ = loss_jit(thicknessA, thicknessB).block_until_ready()
            tf = time.time() - t0
            results[model_name]['forward'].append(tf)
            print(f"    Forward time: {tf:.4f}s")

            # Time backward
            t0 = time.time()
            _ = grad_jit(thicknessA, thicknessB).block_until_ready()
            tb = time.time() - t0
            results[model_name]['backward'].append(tb)
            print(f"    Backward time: {tb:.4f}s")

    # Display summary
    print("\nScaling Results:")
    print("num_points", num_points_list)
    for key in results:
        print(f"\n{key} model:")
        print(" forward:", results[key]['forward'])
        print(" backward:", results[key]['backward'])

    # Compare backward differentiation outputs for the largest sample size
    n = num_points_list[-1]
    t = jnp.linspace(0, 1, n)
    keyA, keyB = jax.random.split(jax.random.PRNGKey(0))
    thicknessA, _ = sample_derivative_bound_gp(
        keyA, 1, t, 0.1, 10.0, 200.0, 1800.0,
        random_final_values=True, min_final_value=800., max_final_value=1200., convex_samples=True)
    thicknessB, _ = sample_derivative_bound_gp(
        keyB, 1, t, 0.1, 10.0, 200.0, 1800.0,
        random_final_values=True, min_final_value=800., max_final_value=1200., convex_samples=True)

    _, grad_one_layer = make_funcs(ONE_LAYER_MODEL)
    _, grad_transfer = make_funcs(TRANSFER_MATRIX_METHOD)
    grad1 = grad_one_layer(thicknessA, thicknessB).block_until_ready()
    grad2 = grad_transfer(thicknessA, thicknessB).block_until_ready()

    diff_norm = jnp.linalg.norm(grad1 - grad2)
    print(f"\nGradient difference norm between models at n={n}: {diff_norm}")
