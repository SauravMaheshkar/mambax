import dataclasses
import itertools
import os

import jax
import numpy as np
import optax
from absl import logging
from flax import nnx

from configs import default
from input_pipeline import (
    TokenData,
    TokenStream,
    eval_batches,
    load_data,
    num_train_batches,
    train_batches,
)
from model import Mamba


Batch = tuple[jax.Array, jax.Array]


def loss_fn(model: Mamba, batch: Batch) -> tuple[jax.Array, jax.Array]:
    logits = model(batch[0])
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=batch[1]
    ).mean()
    return loss, logits


@nnx.jit
def train_step(
    model: Mamba,
    optimizer: nnx.Optimizer,
    metrics: nnx.MultiMetric,
    batch: Batch,
):
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model, batch)
    metrics.update(loss=loss, logits=logits, labels=batch[1])
    optimizer.update(model, grads)


@nnx.jit
def eval_step(model: Mamba, batch: Batch) -> jax.Array:
    return loss_fn(model, batch)[0]


def evaluate(model: Mamba, val: TokenStream, config: default.Config) -> float:
    """Mean loss over the first `config.val_tokens` val targets, identical every call."""
    batches = eval_batches(
        val, config.val_tokens, config.batch_size, config.sequence_length
    )
    return float(np.mean([eval_step(model, jax.device_put(batch)) for batch in batches]))


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
    eval_batches(data.val, config.val_tokens, config.batch_size, config.sequence_length)


def train_and_evaluate(config: default.Config, workdir: str) -> dict[str, list[float]]:
    """Trains on `config.dataset` and returns the logged train and val losses."""
    workdir = os.path.abspath(workdir)

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
    optimizer = make_optimizer(model, config)

    logging.info(f"Total number of parameters: {model.num_params:_}")
    if config.use_wandb:
        wandb.summary["num_params"] = model.num_params

    metrics = nnx.MultiMetric(loss=nnx.metrics.Average("loss"))
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    model.train()
    batches = train_batches(data.train, config.batch_size, config.sequence_length)
    for step, batch in enumerate(itertools.islice(batches, config.n_iterations), start=1):
        train_step(model, optimizer, metrics, jax.device_put(batch))

        logs = {}
        if step % config.n_freq_train == 0:
            logs["train_loss"] = float(metrics.compute()["loss"])
            metrics.reset()
        if step % config.n_freq_eval == 0:
            model.eval()
            logs["val_loss"] = evaluate(model, data.val, config)
            model.train()

        for name, value in logs.items():
            history[name].append(value)
        if logs:
            print(f"Step {step}: " + ", ".join(f"{k} {v:.4f}" for k, v in logs.items()))
            if config.use_wandb:
                wandb.log(logs, step=step)

    model.save(path=workdir)
    return history
