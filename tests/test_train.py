import dataclasses
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from configs.default import Config
from input_pipeline import TINYSTORIES_VOCAB_SIZE
from model import Mamba
from train import make_optimizer, train_and_evaluate, weight_decay_mask


SMALL = Config(
    train_split="validation[:5%]",
    eval_split="validation[-1%:]",
    model_dim=32,
    num_layers=2,
    sequence_length=32,
    batch_size=16,
    learning_rate=3e-3,
    warmup_steps=20,
    n_freq_train=50,
    n_freq_eval=100,
    val_tokens=4 * 16 * 32,
    n_iterations=200,
)


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> tuple[dict[str, list[float]], Path]:
    workdir = tmp_path_factory.mktemp("workdir")
    history = train_and_evaluate(SMALL, str(workdir))
    return history, workdir


def test_logs_at_configured_frequencies(run):
    history, _ = run
    assert len(history["train_loss"]) == 4
    assert len(history["val_loss"]) == 2


def test_learns_beyond_uniform_guessing(run):
    history, _ = run
    uniform = math.log(TINYSTORIES_VOCAB_SIZE)
    assert history["train_loss"][-1] < history["train_loss"][0] < uniform
    assert history["val_loss"][-1] < 0.75 * uniform


def test_saves_checkpoint(run):
    _, workdir = run
    assert (workdir / "mamba").is_dir()


@pytest.mark.parametrize(
    ("config", "match"),
    [
        (dataclasses.replace(SMALL, n_iterations=10**6), "Lower n_iterations"),
        (dataclasses.replace(SMALL, val_tokens=1000), "positive multiple"),
        (dataclasses.replace(SMALL, val_tokens=16 * 32 * 10**4), "the val stream has"),
        (dataclasses.replace(SMALL, warmup_steps=200), "warmup_steps"),
        (dataclasses.replace(SMALL, warmup_steps=-1), "warmup_steps"),
    ],
)
def test_rejects_runs_it_cannot_complete_before_training(
    tmp_path: Path, config: Config, match: str
):
    with pytest.raises(ValueError, match=match):
        train_and_evaluate(config, str(tmp_path))
    assert not (tmp_path / "mamba").exists()


@pytest.fixture
def tiny_model() -> Mamba:
    config = dataclasses.replace(SMALL, vocab_size=TINYSTORIES_VOCAB_SIZE)
    return Mamba.from_config(config, rngs=nnx.Rngs(0))


def test_weight_decay_skips_norms_biases_and_ssm_parameters(tiny_model: Mamba):
    mask = weight_decay_mask(nnx.state(tiny_model, nnx.Param))
    decayed = {path[-2:] for path, flag in mask.flat_state() if flag}
    kept = {path[-2:] for path, flag in mask.flat_state() if not flag}

    assert decayed == {
        ("embedding", "embedding"),
        ("conv1d", "kernel"),
        ("in_proj", "kernel"),
        ("x_proj", "kernel"),
        ("dt_proj", "kernel"),
        ("out_proj", "kernel"),
    }
    assert kept == {
        ("mixer", "A_log"),
        ("mixer", "D"),
        ("conv1d", "bias"),
        ("dt_proj", "bias"),
        ("norm", "scale"),
        ("norm_f", "scale"),
    }


def test_learning_rate_warms_up_from_zero(tiny_model: Mamba):
    """The first update runs at lr 0, so it moves nothing; the second does."""
    optimizer = make_optimizer(tiny_model, SMALL)
    grads = jax.tree.map(jnp.ones_like, nnx.state(tiny_model, nnx.Param))

    def snapshot() -> list[jax.Array]:
        return [p[...] for _, p in nnx.state(tiny_model, nnx.Param).flat_state()]

    initial = snapshot()
    optimizer.update(tiny_model, grads)
    after_first = snapshot()
    optimizer.update(tiny_model, grads)
    after_second = snapshot()

    for before, first, second in zip(initial, after_first, after_second):
        np.testing.assert_array_equal(first, before)
        assert not np.array_equal(second, first)
