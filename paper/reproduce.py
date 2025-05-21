"""
Main reproduction script.
"""

# ==== GPU selection ====
# from autocvd import autocvd
# autocvd(num_gpus = 1)
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# = Change dir to paper =
if os.getcwd() != os.path.dirname(os.path.abspath(__file__)):
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print(f"Changed working directory to {os.getcwd()}")
# =======================

# == Make figures  dir ==
if not os.path.exists("figures"):
    os.makedirs("figures")
# =======================

# from analyze_all_validation_samples import analyze_all_validation_samples
from analyze_measurement import analyze_measurement
from compare_neural_operators import neural_operator_comparison
from generate_training_data import generate_training_data
from generate_validation_data import generate_validation_data
from noise_robustness_test import test_noise_robustness
from plot_single_analysis_result import plot_single_sample_initialization_comparison
from plot_validation_analysis_results import (
    loss_analysis_plot,
    plot_validation_loss_curves,
)
from plot_validation_and_train_data import plot_validation_and_train_data
from train_neural_operator import neural_operator_training

# generate the training and validation data
print("===================================================")
print("============ Generating Training Data =============")
print("===================================================")
generate_training_data()

# generate the validation data
print("===================================================")
print("=========== Generating Validation Data ============")
print("===================================================")
generate_validation_data()

# plot the training and validation data
print("===================================================")
print("=========== Plotting Train and Val Data ===========")
print("===================================================")
plot_validation_and_train_data()  # -> Figure 2

# train the neural operator
print("===================================================")
print("============= Training Neural Operator ============")
print("===================================================")
neural_operator_training()

# analyze the validation data
# this is a bit costly, as we test 4 initializations
# on 200 samples so 800 samples with each run taking
# ~5 minutes on an A100 GPU -> ~ 3 days
# print("===================================================")
# print("=========== Running Validation Analysis ===========")
# print("===================================================")
# analyze_all_validation_samples()

# plot single validation result comparison
print("===================================================")
print("============ Plotting Single Comparison ===========")
print("===================================================")
plot_single_sample_initialization_comparison()  # -> Figure 3

# plot validation results as loss kernel density estimates
print("===================================================")
print("============== Plotting Analysis KDE ==============")
print("===================================================")
loss_analysis_plot()  # -> Figure 4

# plot the validation results
print("===================================================")
print("============== Plotting Loss Curves ===============")
print("===================================================")
plot_validation_loss_curves()  # -> Figure 5

# test robustness to noise
print("===================================================")
print("============ Testing Noise Robustness =============")
print("===================================================")
test_noise_robustness()  # -> Figure 6

# test on experimental data
print("===================================================")
print("========== Analyzing Real Lab Measurement =========")
print("===================================================")
analyze_measurement()  # -> Figure 7

# test on experimental data
print("===================================================")
print("========== Analyzing Real Lab Measurement =========")
print("===================================================")
analyze_measurement()  # -> Figure 7

# additional comparison of neural operator performance
print("===================================================")
print("====== Comparing Neural Operator Performance ======")
print("===================================================")
neural_operator_comparison()  # -> Figure 9
