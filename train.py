import dataclasses
import os

import jax
import numpy as np
import optax
from absl import logging
from flax import nnx

from configs import default
from input_pipeline import VOCAB_SIZE, load_split, sample_batch
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


def evaluate(model: Mamba, data: np.ndarray, config: default.Config) -> float:
    """Mean loss over `n_eval_batches` windows, identical on every call."""
    rng = np.random.default_rng(config.seed)
    losses = [
        eval_step(
            model,
            jax.device_put(
                sample_batch(rng, data, config.batch_size, config.sequence_length)
            ),
        )
        for _ in range(config.n_eval_batches)
    ]
    return float(np.mean(losses))


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

    train_data = load_split(config.train_split)
    val_data = load_split(config.eval_split)
    logging.info(f"Train characters: {len(train_data):_}, val: {len(val_data):_}")

    config = dataclasses.replace(config, vocab_size=VOCAB_SIZE)
    model = Mamba.from_config(config, rngs=nnx.Rngs(0))

    logging.info(f"Total number of parameters: {model.num_params:_}")
    if config.use_wandb:
        wandb.summary["num_params"] = model.num_params

    optimizer = nnx.Optimizer(
        model, optax.adamw(config.learning_rate, nesterov=True), wrt=nnx.Param
    )
    metrics = nnx.MultiMetric(loss=nnx.metrics.Average("loss"))
    rng = np.random.default_rng(config.seed)
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    model.train()
    for step in range(1, config.n_iterations + 1):
        batch = jax.device_put(
            sample_batch(rng, train_data, config.batch_size, config.sequence_length)
        )
        train_step(model, optimizer, metrics, batch)

        logs = {}
        if step % config.n_freq_train == 0:
            logs["train_loss"] = float(metrics.compute()["loss"])
            metrics.reset()
        if step % config.n_freq_eval == 0:
            model.eval()
            logs["val_loss"] = evaluate(model, val_data, config)
            model.train()

        for name, value in logs.items():
            history[name].append(value)
        if logs:
            print(f"Step {step}: " + ", ".join(f"{k} {v:.4f}" for k, v in logs.items()))
            if config.use_wandb:
                wandb.log(logs, step=step)

    model.save(path=workdir)
    return history
