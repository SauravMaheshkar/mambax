import dataclasses
import enum
import math


class Dataset(enum.StrEnum):
    TINYSTORIES = "tinystories"
    FINEWEB = "fineweb"


class Dtype(enum.StrEnum):
    FLOAT32 = "float32"
    BFLOAT16 = "bfloat16"


# Field types stick to what ml_collections can parse from the command line:
# plain scalars, Optional scalars and enums (Literal and non-Optional unions fail).
@dataclasses.dataclass
class Config:
    # vocab size
    vocab_size: int | None = None
    # model dimension
    model_dim: int = 192
    # number of layers
    num_layers: int = 4
    # sequences per optimizer step, across all devices and accumulation steps
    batch_size: int = 16
    # each step's batch is split into this many microbatches, run one after the
    # other with their gradients summed, to cap activation memory
    grad_accum_steps: int = 1
    # peak learning rate, reached after warmup_steps, then cosine-decayed to 0 at
    # n_iterations
    learning_rate: float = 1e-3
    # steps of linear warmup from 0 to learning_rate
    warmup_steps: int = 100
    # AdamW weight decay, on matrices and the embedding only (not on norms, biases,
    # A_log or D)
    weight_decay: float = 0.1
    # gradients are clipped to this global norm
    grad_clip_norm: float = 1.0
    # random seed for model initialization
    seed: int = 42
    # number of iterations
    n_iterations: int = 10_000
    # frequency of logging the train loss
    n_freq_train: int = 100
    # frequency of evaluating the val loss
    n_freq_eval: int = 1000
    # frequency of checkpointing; a run resumes from the latest checkpoint in its
    # workdir, and always checkpoints its final step
    n_freq_checkpoint: int = 1000
    # number of most recent checkpoints kept in the workdir
    max_checkpoints_to_keep: int = 2
    # val tokens per evaluation, fixed so val losses stay comparable (the FineWeb
    # speedrun uses input_pipeline.FINEWEB_SPEEDRUN_VAL_TOKENS = 10_485_760)
    val_tokens: int = 65_536
    # dataset to train on
    dataset: Dataset = Dataset.TINYSTORIES
    # TinyStories splits, `datasets` slicing like "train[:10%]" works
    train_split: str = "train"
    eval_split: str = "validation"
    # FineWeb train shards to stream, 100M tokens (200MB) each
    fineweb_train_shards: int = 1
    # sequence length for training
    sequence_length: int = 64
    # convolution dimension
    conv_dim: int = 4
    # dt rank, None for Mamba's default of ceil(model_dim / 16)
    dt_rank: int | None = None
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
    # sequence chunk length for the memory-bounded selective scan
    scan_chunk_size: int = 32
    # dtype of matmuls and convolutions; params, the residual stream, the scan and
    # the loss stay float32. bfloat16 is for accelerators, CPUs emulate it slowly.
    compute_dtype: Dtype = Dtype.FLOAT32
    # whether to use weights and biases
    use_wandb: bool = False
    # weights and biases project
    wandb_project: str = "mambax"
    # weights and biases entity
    wandb_entity: str | None = None

    # Derived values are properties, not __post_init__ assignments, so that
    # command line overrides like --config.model_dim=256 propagate to them.

    @property
    def microbatch_size(self) -> int:
        return self.batch_size // self.grad_accum_steps

    @property
    def hidden_dim(self) -> int:
        return self.expand * self.model_dim

    @property
    def resolved_dt_rank(self) -> int:
        if self.dt_rank is None:
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
