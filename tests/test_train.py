import math
from pathlib import Path

import pytest

from configs.default import Config
from input_pipeline import VOCAB_SIZE
from train import train_and_evaluate


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> tuple[dict[str, list[float]], Path]:
    config = Config(
        train_split="validation[:5%]",
        eval_split="validation[-1%:]",
        model_dim=32,
        num_layers=2,
        sequence_length=32,
        batch_size=16,
        learning_rate=3e-3,
        n_iterations=200,
        n_freq_train=50,
        n_freq_eval=100,
        n_eval_batches=4,
    )
    workdir = tmp_path_factory.mktemp("workdir")
    return train_and_evaluate(config, str(workdir)), workdir


def test_logs_at_configured_frequencies(run):
    history, _ = run
    assert len(history["train_loss"]) == 4
    assert len(history["val_loss"]) == 2


def test_learns_beyond_uniform_guessing(run):
    history, _ = run
    uniform = math.log(VOCAB_SIZE)
    assert history["train_loss"][-1] < history["train_loss"][0] < uniform
    assert history["val_loss"][-1] < 0.75 * uniform


def test_saves_checkpoint(run):
    _, workdir = run
    assert (workdir / "mamba").is_dir()
