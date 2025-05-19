# # ==== GPU selection ====
# from autocvd import autocvd
# autocvd(num_gpus = 1)
# # =======================

import jax.numpy as jnp

from flax import nnx

from reflax.thickness_modeling.operator_learning import NeuralOperatorMLP, save_model, train_neural_operator

def neural_operator_training(
    training_data_path = "simulated_data/training_data.npz",
    model_save_path = "saved_models/neural_operator.pickle"
):

    # load the training data
    training_data = jnp.load(training_data_path)
    reflectances = training_data["reflectances"]
    thicknesses = training_data["thicknesses"]
    derivatives = training_data["derivatives"]
    time_points = training_data["time_points"]

    neural_operator = NeuralOperatorMLP(
        hidden_dims = [512, 512],
        num_eval_points = time_points.shape[0],
        rngs = nnx.Rngs(42),
    )

    neural_operator = train_neural_operator(
        model = neural_operator,
        reflectance_data = reflectances,
        thickness_data = thicknesses,
        learning_rate = 1e-4,
        test_set_size = 0.2,
        num_epochs = 20000,
        print_interval = 500,
        patience = 4000,
        random_seed_split = 42 
    )

    save_model(
        neural_operator,
        filepath = model_save_path,
    )