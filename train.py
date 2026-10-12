import dataclasses
import itertools
import json
import os
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from absl import logging
from flax import nnx
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Int

from configs import default
from input_pipeline import (
    TokenData,
    TokenStream,
    eval_batches,
    load_data,
    num_train_batches,
    split_microbatches,
    train_batches,
)
from model import Mamba


Batch = tuple[Int[Array, "batch seq"], Int[Array, "batch seq"]]
Microbatches = tuple[Int[Array, "accum micro seq"], Int[Array, "accum micro seq"]]

CHECKPOINT_DIR = "checkpoints"
# Fields that only change logging or checkpoint bookkeeping, so a run may resume
# with different values. Anything else changes the run itself.
_RESUMABLE_FIELDS = frozenset(
    {
        "n_freq_train",
        "n_freq_eval",
        "n_freq_checkpoint",
        "max_checkpoints_to_keep",
        # Same global batch, same math, just a different memory/time trade-off.
        "grad_accum_steps",
        "use_wandb",
        "wandb_project",
        "wandb_entity",
    }
)


def loss_fn(model: Mamba, batch: Batch) -> jax.Array:
    logits = model(batch[0])
    return optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=batch[1]
    ).mean()


@nnx.jit
def train_step(
    model: Mamba,
    optimizer: nnx.Optimizer,
    metrics: nnx.MultiMetric,
    microbatches: Microbatches,
):
    """One optimizer step on the mean gradient over the microbatches.

    The microbatches run one after another in a scan, so only one microbatch's
    activations are alive at a time. Each has the same number of tokens, so the
    mean of their gradients is the gradient of the whole batch.
    """
    graphdef, params, rest = nnx.split(model, nnx.Param, ...)

    def microbatch_loss(params: nnx.State, batch: Batch) -> jax.Array:
        return loss_fn(nnx.merge(graphdef, params, rest), batch)

    def accumulate(totals, batch: Batch):
        loss_and_grads = jax.value_and_grad(microbatch_loss)(params, batch)
        return jax.tree.map(jnp.add, totals, loss_and_grads), None

    zeros = jax.tree.map(jnp.zeros_like, (jnp.zeros(()), params))
    totals, _ = jax.lax.scan(accumulate, zeros, microbatches)
    loss, grads = jax.tree.map(lambda total: total / len(microbatches[0]), totals)
    metrics.update(loss=loss)
    optimizer.update(model, grads)


@nnx.jit
def eval_step(model: Mamba, batch: Batch) -> jax.Array:
    return loss_fn(model, batch)


def evaluate(
    model: Mamba, val: TokenStream, config: default.Config, sharding: NamedSharding
) -> float:
    """Mean loss over the first `config.val_tokens` val targets, identical every call."""
    batches = eval_batches(
        val, config.val_tokens, config.microbatch_size, config.sequence_length
    )
    losses = [eval_step(model, jax.device_put(batch, sharding)) for batch in batches]
    return float(np.mean(losses))


def make_mesh() -> Mesh:
    """A 1-D mesh over every device, for data parallelism.

    Params and optimizer state are replicated, batches are split along their
    batch axis, and XLA inserts the all-reduce that averages the gradients.
    """
    if jax.process_count() > 1:
        raise NotImplementedError("multi-host training needs per-host data loading")
    return jax.make_mesh((jax.device_count(),), ("data",), axis_types=(AxisType.Auto,))


def check_batch_layout(config: default.Config, num_devices: int) -> None:
    """Each step's batch must split evenly into microbatches, and those across devices."""
    if config.grad_accum_steps < 1 or config.batch_size % config.grad_accum_steps:
        raise ValueError(
            f"batch_size ({config.batch_size}) must be a positive multiple of "
            f"grad_accum_steps ({config.grad_accum_steps})"
        )
    if config.microbatch_size % num_devices:
        raise ValueError(
            f"batch_size / grad_accum_steps ({config.microbatch_size}) must be a "
            f"multiple of the number of devices ({num_devices})"
        )


def weight_decay_mask(params: nnx.State) -> nnx.State:
    """Decays matrices and the embedding, never norms, biases, D or A_log.

    A_log is 2-D but holds each channel's decay rates, which weight decay would
    drag toward A = -1 (mamba_ssm excludes it, D and the dt bias too).
    """
    return nnx.map_state(lambda path, p: p.ndim >= 2 and path[-1] != "A_log", params)


def make_optimizer(model: Mamba, config: default.Config) -> nnx.Optimizer:
    """AdamW with linear warmup, cosine decay to 0 and global-norm gradient clipping."""
    if not 0 <= config.warmup_steps < config.n_iterations:
        raise ValueError(
            f"warmup_steps ({config.warmup_steps}) must be in "
            f"[0, n_iterations={config.n_iterations})"
        )
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.learning_rate,
        warmup_steps=config.warmup_steps,
        decay_steps=config.n_iterations,
        end_value=0.0,
    )
    tx = optax.chain(
        optax.clip_by_global_norm(config.grad_clip_norm),
        optax.adamw(
            schedule,
            weight_decay=config.weight_decay,
            mask=weight_decay_mask,
            nesterov=True,
        ),
    )
    return nnx.Optimizer(model, tx, wrt=nnx.Param)


def check_token_budget(data: TokenData, config: default.Config) -> None:
    """Fails before training, not hours in, when the data can't cover the run."""
    available = num_train_batches(data.train, config.batch_size, config.sequence_length)
    if available < config.n_iterations:
        raise ValueError(
            f"n_iterations={config.n_iterations} needs {config.n_iterations} batches of "
            f"{config.batch_size}x{config.sequence_length} tokens, the train data has "
            f"{available}. Lower n_iterations or add data (e.g. fineweb_train_shards)."
        )
    eval_batches(
        data.val, config.val_tokens, config.microbatch_size, config.sequence_length
    )


def open_checkpoints(workdir: str, config: default.Config) -> ocp.CheckpointManager:
    """The run's checkpoints. Saves are async, and each one is written to a temp
    directory and renamed into place when complete, so a run killed mid-save
    leaves the previous checkpoint as the latest."""
    return ocp.CheckpointManager(
        os.path.join(workdir, CHECKPOINT_DIR),
        options=ocp.CheckpointManagerOptions(
            max_to_keep=config.max_checkpoints_to_keep,
            cleanup_tmp_directories=True,
        ),
    )


def save_checkpoint(
    checkpoints: ocp.CheckpointManager,
    step: int,
    model: Mamba,
    optimizer: nnx.Optimizer,
    config: default.Config,
) -> None:
    checkpoints.save(
        step,
        args=ocp.args.Composite(
            model=ocp.args.StandardSave(nnx.state(model, nnx.Param)),
            optimizer=ocp.args.StandardSave(nnx.state(optimizer)),
            config=ocp.args.JsonSave(_config_json(config)),
        ),
    )


def restore_checkpoint(
    checkpoints: ocp.CheckpointManager,
    model: Mamba,
    optimizer: nnx.Optimizer,
    config: default.Config,
) -> int:
    """Restores the latest checkpoint into `model` and `optimizer`.

    Returns its step, 0 when there is none. Refuses checkpoints of a different
    run, since resuming those would silently mix two schedules or datasets.
    """
    step = checkpoints.latest_step()
    if step is None:
        return 0
    saved = _restore_config(checkpoints, step)
    current = _config_json(config)
    changed = sorted(
        name
        for name in saved.keys() | current.keys()
        if name not in _RESUMABLE_FIELDS and saved.get(name) != current.get(name)
    )
    if changed:
        diff = ", ".join(f"{n}: {saved.get(n)!r} -> {current.get(n)!r}" for n in changed)
        raise ValueError(
            f"{checkpoints.directory} holds a run with a different config ({diff}). "
            "Resume with the same config or train in a fresh workdir."
        )
    restored = checkpoints.restore(
        step,
        args=ocp.args.Composite(
            model=ocp.args.StandardRestore(nnx.state(model, nnx.Param)),
            optimizer=ocp.args.StandardRestore(nnx.state(optimizer)),
        ),
    )
    nnx.update(model, restored["model"])
    nnx.update(optimizer, restored["optimizer"])
    return step


def load_model(workdir: str) -> Mamba:
    """The model from the latest checkpoint of the training run in `workdir`."""
    options = ocp.CheckpointManagerOptions(read_only=True)
    with ocp.CheckpointManager(
        os.path.join(workdir, CHECKPOINT_DIR), options=options
    ) as checkpoints:
        step = checkpoints.latest_step()
        if step is None:
            raise FileNotFoundError(f"no checkpoints in {workdir}")
        saved = _restore_config(checkpoints, step)
        config = dataclasses.replace(
            default.Config(**saved),
            dataset=default.Dataset(saved["dataset"]),
            compute_dtype=default.Dtype(saved["compute_dtype"]),
        )
        # Every weight gets overwritten, so skip the random init.
        model = nnx.eval_shape(lambda: Mamba.from_config(config, rngs=nnx.Rngs(0)))
        # Place the weights on this machine's devices, not the ones the run was
        # saved from (e.g. trained on 8 TPU chips, loaded on a laptop).
        replicated = NamedSharding(make_mesh(), P())
        target = jax.tree.map(
            lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=replicated),
            nnx.state(model, nnx.Param),
        )
        restored = checkpoints.restore(
            step,
            args=ocp.args.Composite(model=ocp.args.StandardRestore(target)),
        )
    nnx.update(model, restored["model"])
    return model


def _config_json(config: default.Config) -> dict[str, Any]:
    """The config as it reads back from a checkpoint, so the two compare equal."""
    return json.loads(json.dumps(dataclasses.asdict(config)))


def _restore_config(checkpoints: ocp.CheckpointManager, step: int) -> dict[str, Any]:
    args = ocp.args.Composite(config=ocp.args.JsonRestore())
    return checkpoints.restore(step, args=args)["config"]


def train_and_evaluate(config: default.Config, workdir: str) -> dict[str, list[float]]:
    """Trains on `config.dataset` and returns the logged train and val losses.

    Resumes from the latest checkpoint in `workdir` if there is one, so rerunning
    the same command after a crash or a session limit continues the run exactly
    where its last checkpoint left off.
    """
    workdir = os.path.abspath(workdir)
    mesh = make_mesh()
    check_batch_layout(config, mesh.size)
    replicated = NamedSharding(mesh, P())
    batch_sharding = NamedSharding(mesh, P("data"))
    microbatch_sharding = NamedSharding(mesh, P(None, "data"))

    if config.use_wandb:
        import wandb

        wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            config=dataclasses.asdict(config),
        )

    data = load_data(
        config.dataset,
        tinystories_train_split=config.train_split,
        tinystories_eval_split=config.eval_split,
        fineweb_train_shards=config.fineweb_train_shards,
    )
    check_token_budget(data, config)
    num_train_tokens = sum(len(shard) for shard in data.train)
    logging.info(f"Train tokens: {num_train_tokens:_}, val: {len(data.val):_}")

    config = dataclasses.replace(config, vocab_size=data.vocab_size)
    model = Mamba.from_config(config, rngs=nnx.Rngs(config.seed))
    nnx.update(model, jax.device_put(nnx.state(model), replicated))
    # Optimizer state is created like the params, so it is replicated too.
    optimizer = make_optimizer(model, config)

    logging.info(f"Total number of parameters: {model.num_params:_}")
    logging.info(f"Data parallel over {mesh.size} {jax.default_backend()} devices")
    if config.use_wandb:
        wandb.summary["num_params"] = model.num_params

    metrics = nnx.MultiMetric(loss=nnx.metrics.Average("loss"))
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    # Closing the manager waits for the async saves still in flight.
    with open_checkpoints(workdir, config) as checkpoints:
        first_step = restore_checkpoint(checkpoints, model, optimizer, config)
        if first_step:
            logging.info(f"Resuming from the checkpoint at step {first_step}")

        model.train()
        batches = train_batches(
            data.train, config.batch_size, config.sequence_length, first_batch=first_step
        )
        remaining = itertools.islice(batches, config.n_iterations - first_step)
        for step, batch in enumerate(remaining, start=first_step + 1):
            microbatches = split_microbatches(batch, config.grad_accum_steps)
            train_step(
                model,
                optimizer,
                metrics,
                jax.device_put(microbatches, microbatch_sharding),
            )

            logs = {}
            if step % config.n_freq_train == 0:
                logs["train_loss"] = float(metrics.compute()["loss"])
                metrics.reset()
            if step % config.n_freq_eval == 0:
                model.eval()
                logs["val_loss"] = evaluate(model, data.val, config, batch_sharding)
                model.train()

            for name, value in logs.items():
                history[name].append(value)
            if logs:
                summary = ", ".join(f"{k} {v:.4f}" for k, v in logs.items())
                print(f"Step {step}: {summary}")
                if config.use_wandb:
                    wandb.log(logs, step=step)

            if step % config.n_freq_checkpoint == 0 or step == config.n_iterations:
                save_checkpoint(checkpoints, step, model, optimizer, config)

    return history
