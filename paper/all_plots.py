from single_optimization_comparison_plot import plot_single_sample_initialization_comparison
from plot_validation_and_train_data import plot_validation_and_train_data
from result_plotting import kde_loss_plot, plot_all_validation_results, plot_validation_loss_curves
from single_example_optimization import optimize_single_example
from reflax.constants import ONE_LAYER_MODEL, TRANSFER_MATRIX_METHOD

# plot the validation and training data
# print("🖼️: plotting validation and train data...")
plot_validation_and_train_data(figpath = "figures/train_val_data.png")

# # generate all plots of the validation results
# print("🖼️: generating plots for all validation results...")
# plot_all_validation_results()

# single sample initialization comparison
# plot_single_sample_initialization_comparison()

# plot the loss over epochs for the validation set
# print("🖼️: generating plot for the loss over epochs...")
# plot_validation_loss_curves(figpath = "figures/validation_losses_over_epochs.svg")

# # plot loss kdes comparing the different initializations
# print("🖼️: generating plot for the loss kdes...")
# kde_loss_plot(figpath = "figures/loss_kde_comparison.svg")

# example optimization with one layer model
# 14k epochs took 308s on an A100 GPU
# optimize_single_example(
#     model = ONE_LAYER_MODEL,
#     result_title = "one_layer_example",
#     loss_over_epoch_title = "one_layer_loss_over_epoch",
# )

# same example optimization with transfer matrix method
# 14k epochs took 314s on an A100 GPU
# optimize_single_example(
#     model = TRANSFER_MATRIX_METHOD,
#     result_title = "transfer_matrix_example",
#     loss_over_epoch_title = "transfer_matrix_loss_over_epoch",
# )