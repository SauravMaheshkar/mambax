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


def check_token_budget(data: TokenData, config: default.Config) -> None:
    """Fails before training, not hours in, when the data can't cover the run."""
    available = num_train_batches(data.train, config.batch_size, config.sequence_length)
    if available < config.n_iterations:
        raise ValueError(
            f"n_iterations={config.n_iterations} needs {config.n_iterations} batches of "
            f"{config.batch_size}x{config.sequence_length} tokens, the train data has "
            f"{available}. Lower n_iterations or train on a bigger train_split."
        )
    eval_batches(data.val, config.val_tokens, config.batch_size, config.sequence_length)


def train_and_evaluate(config: default.Config, workdir: str) -> dict[str, list[float]]:
    """Trains on `config.train_split` and returns the logged train and val losses."""
    workdir = os.path.abspath(workdir)

    if config.use_wandb:
        import wandb

        wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            config=dataclasses.asdict(config),
        )

    data = load_data(train_split=config.train_split, eval_split=config.eval_split)
    check_token_budget(data, config)
    num_train_tokens = sum(len(shard) for shard in data.train)
    logging.info(f"Train tokens: {num_train_tokens:_}, val: {len(data.val):_}")

    config = dataclasses.replace(config, vocab_size=data.vocab_size)
    model = Mamba.from_config(config, rngs=nnx.Rngs(config.seed))

    logging.info(f"Total number of parameters: {model.num_params:_}")
    if config.use_wandb:
        wandb.summary["num_params"] = model.num_params

    optimizer = nnx.Optimizer(
        model, optax.adamw(config.learning_rate, nesterov=True), wrt=nnx.Param
    )
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
