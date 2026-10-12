import dataclasses
import math
from pathlib import Path

import pytest

from configs.default import Config
from input_pipeline import TINYSTORIES_VOCAB_SIZE
from train import train_and_evaluate


SMALL = Config(
    train_split="validation[:5%]",
    eval_split="validation[-1%:]",
    model_dim=32,
    num_layers=2,
    sequence_length=32,
    batch_size=16,
    learning_rate=3e-3,
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
    ],
)
def test_rejects_runs_the_data_cannot_cover_before_training(
    tmp_path: Path, config: Config, match: str
):
    with pytest.raises(ValueError, match=match):
        train_and_evaluate(config, str(tmp_path))
    assert not (tmp_path / "mamba").exists()
