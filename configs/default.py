import dataclasses
import math
from typing import Literal


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
    conv_dim: int = 4
    # dt rank
    dt_rank: int | Literal["auto"] = "auto"
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
    # RMSNorm epsilon
    norm_eps: float = 1e-5
    # whether to use weights and biases
    use_wandb: bool = False
    # weights and biases project
    wandb_project: str = "mambax"
    # weights and biases entity
    wandb_entity: str | None = None

    # Derived values are properties, not __post_init__ assignments, so that
    # command line overrides like --config.model_dim=256 propagate to them.

    @property
    def hidden_dim(self) -> int:
        return self.expand * self.model_dim

    @property
    def resolved_dt_rank(self) -> int:
        if self.dt_rank == "auto":
            return math.ceil(self.model_dim / 16)
        return self.dt_rank

    @property
    def padded_vocab_size(self) -> int:
        if self.vocab_size is None:
            raise ValueError("vocab_size is unset, set it from the dataset or checkpoint")
        multiple = self.pad_vocab_size_multiple
        return math.ceil(self.vocab_size / multiple) * multiple


def get_config():
    """Get the default hyperparameter configuration."""
    config = Config()
    return config
