# ==== GPU selection ====
# from autocvd import autocvd

# from reflax.constants import S_POLARIZED
# autocvd(num_gpus = 1)
# only use gpu 9
import os

from reflax.forward_model.forward_model import batched_forward_model
os.environ["CUDA_VISIBLE_DEVICES"] = "9"
# =======================

import jax.numpy as jnp

from flax import nnx
from sklearn.model_selection import train_test_split

from reflax.thickness_modeling.operator_learning import NeuralOperatorMLP, load_model, save_model, train_neural_operator

import matplotlib.pyplot as plt



# load the training data
training_data = jnp.load("training_data/training_data.npz")
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
    filepath = "models/neural_operator.pickle",
)

_, reflectances_test, _, thicknesses_test = train_test_split(
    reflectances, thicknesses, test_size = 0.2, random_state = 42
)

# for a sample of the test set, predict the thickness from
# the reflectance and compare the finite differences
# of the predicted thickness with the true derivative

reflectance_sample = reflectances_test[0]
thickness_sample = thicknesses_test[0]
predicted_thickness = neural_operator(reflectance_sample)
predicted_derivative = jnp.gradient(predicted_thickness, time_points)
true_derivative = jnp.gradient(thickness_sample, time_points)

# plot the reflectance, predicted thickness, and true thickness
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))

ax1.plot(time_points, reflectance_sample, label="Reflectance")
ax1.set_title("Reflectance")
ax1.set_xlabel("Time")
ax1.set_ylabel("Reflectance")
ax1.legend()

ax2.plot(time_points, predicted_thickness, label="Predicted Thickness")
ax2.plot(time_points, thickness_sample, label="True Thickness")
ax2.set_title("Thickness")
ax2.set_xlabel("Time")
ax2.set_ylabel("Thickness")
ax2.legend()

ax3.plot(time_points, predicted_derivative, label="Predicted Derivative")
ax3.plot(time_points, true_derivative, label="True Derivative")
ax3.set_title("Derivative")
ax3.set_xlabel("Time")
ax3.set_ylabel("Derivative")
ax3.legend()

plt.tight_layout()

plt.savefig("figures/neural_operator_training.png")