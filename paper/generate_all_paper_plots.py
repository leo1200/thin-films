from single_example_optimization import optimize_single_example
from reflax.constants import ONE_LAYER_MODEL, TRANSFER_MATRIX_METHOD

# example optimization with one layer model
optimize_single_example(
    model = ONE_LAYER_MODEL,
    result_title = "one_layer_example",
    loss_over_epoch_title = "one_layer_loss_over_epoch",
)

# same example optimization with transfer matrix method
optimize_single_example(
    model = TRANSFER_MATRIX_METHOD,
    result_title = "transfer_matrix_example",
    loss_over_epoch_title = "transfer_matrix_loss_over_epoch",
)