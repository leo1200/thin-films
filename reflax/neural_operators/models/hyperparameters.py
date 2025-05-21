from dataclasses import dataclass


@dataclass
class OperatorFCNNConfig:
    """
    Configuration for OperatorFCNN:
      - hidden_size: int
      - num_hidden_layers: int
      - learning_rate: float
      - regularisation_factor: float
      - activation: str (e.g. 'ReLU', 'Tanh')
    """

    # # standard values

    # hidden_size: int = 150
    # num_hidden_layers: int = 5
    # learning_rate: float = 3e-4
    # regularisation_factor: float = 0  # 0.012
    # activation: str = "ReLU"

    # # from run1, trial71

    # hidden_size: int = 253
    # num_hidden_layers: int = 4
    # learning_rate: float = 8e-4
    # regularisation_factor: float = 1.3e-6  # 0.012
    # activation: str = "ReLU"

    # from gp_test, 181

    hidden_size: int = 291
    num_hidden_layers: int = 3
    learning_rate: float = 1e-3
    regularisation_factor: float = 1.2e-6  # 0.012
    activation: str = "LeakyReLU"


@dataclass
class DirectFCNNConfig:

    # # standard values

    # hidden_size: int = 150
    # num_hidden_layers: int = 5
    # learning_rate: float = 1e-4
    # regularisation_factor: float = 0  # 0.012
    # activation: str = "ReLU"

    # # from run1, trial86

    # hidden_size: int = 137
    # num_hidden_layers: int = 4
    # learning_rate: float = 5e-3
    # regularisation_factor: float = 1.0e-6  # 0.012
    # activation: str = "ReLU"

    # from gp_test, 114

    hidden_size: int = 182
    num_hidden_layers: int = 5
    learning_rate: float = 4e-4
    regularisation_factor: float = 1.8e-7  # 0.012
    activation: str = "LeakyReLU"


@dataclass
class DeepOperatorConfig:

    # # standard values

    # hidden_width: int = 150
    # branch_layers: int = 3
    # trunk_layers: int = 3
    # latent_size: int = 64
    # learning_rate: float = 1e-4
    # regularisation_factor: float = 0  # 0.012
    # activation: str = "ReLU"

    # # from trial run1, trial39

    # activation: str = "ReLU"
    # branch_layers: int = 3
    # hidden_width: int = 199
    # latent_size: int = 120
    # learning_rate: float = 5e-4
    # regularisation_factor: float = 1.5e-6
    # trunk_layers: int = 2

    # from gp_test, 93

    hidden_width: int = 285
    branch_layers: int = 4
    trunk_layers: int = 2
    latent_size: int = 89
    learning_rate: float = 2e-4
    regularisation_factor: float = 1.2e-6
    activation: str = "ReLU"


@dataclass
class DirectFCNNPlainConfig:

    # standard values

    hidden_size: int = 512
    num_hidden_layers: int = 2
    learning_rate: float = 1e-3
    regularisation_factor: float = 0  # 0.012
    activation: str = "ReLU"
