import dataclasses
import math
from typing import Union


@dataclasses.dataclass
class Config:
    # vocab size
    vocab_size: int | None = None
    # model dimension
    model_dim: int = 192
    # number of layers
    num_layers: int = 4
    # batch size for training
    batch_size: int = 16
    # number of epochs for training
    num_epochs: int = 1
    # learning rate
    learning_rate: float = 1e-3
    # random seed
    seed: int = 42
    # number of iterations
    n_iterations: int = 10_000
    # frequency of updating metrics
    n_freq_train: int = 100
    # sequence length for training
    sequence_length: int = 64
    # convolution dimension
    conv_dim: int = 2
    # dt rank
    dt_rank: Union[int, str] = "auto"
    # state dimension
    state_dim: int = 16
    # expand factor
    expand: int = 2
    # pad vocab size multiple
    pad_vocab_size_multiple: int = 8
    # use bias
    use_bias: bool = False
    # convolution bias
    conv_bias: bool = True
    # whether to use weights and biases
    use_wandb: bool = False
    # weights and biases project
    wandb_project: str = "mambax"
    # weights and biases entity
    wandb_entity: str | None = None

    def __post_init__(self):
        self.hidden_dim = int(self.expand * self.model_dim)

        if self.dt_rank == "auto":
            self.dt_rank = math.ceil(self.model_dim / 16)

        if (
            self.vocab_size is not None
            and self.vocab_size % self.pad_vocab_size_multiple != 0
        ):
            self.vocab_size += (
                self.pad_vocab_size_multiple
                - self.vocab_size % self.pad_vocab_size_multiple
            )


def get_config():
    """Get the default hyperparameter configuration."""
    config = Config()
    return config
