import dataclasses
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from configs.default import Config, Dtype
from input_pipeline import TINYSTORIES_VOCAB_SIZE, split_microbatches
from model import Mamba
from train import (
    CHECKPOINT_DIR,
    check_batch_layout,
    load_model,
    make_optimizer,
    train_and_evaluate,
    train_step,
    weight_decay_mask,
)


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
    n_freq_checkpoint=75,
    val_tokens=4 * 16 * 32,
    n_iterations=200,
)


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> tuple[dict[str, list[float]], Path]:
    workdir = tmp_path_factory.mktemp("workdir")
    history = train_and_evaluate(SMALL, str(workdir))
    return history, workdir


@pytest.fixture
def copied_run(run, tmp_path: Path) -> Path:
    """A copy of the finished run's workdir, free to be tampered with."""
    _, workdir = run
    return Path(shutil.copytree(workdir, tmp_path / "workdir"))


def checkpoint_steps(workdir: Path) -> list[int]:
    """Finalized checkpoints only: orbax renames a save into place once complete."""
    checkpoints = workdir / CHECKPOINT_DIR
    if not checkpoints.exists():
        return []
    return sorted(int(path.name) for path in checkpoints.iterdir() if path.name.isdigit())


def cli_overrides(config: Config) -> list[str]:
    """main.py flags that reproduce `config`."""
    return [
        f"--config.{field.name}={getattr(config, field.name)}"
        for field in dataclasses.fields(Config)
        if getattr(config, field.name) != field.default
    ]


def assert_same_params(a: Mamba, b: Mamba):
    for (path, x), (_, y) in zip(
        nnx.state(a, nnx.Param).flat_state(), nnx.state(b, nnx.Param).flat_state()
    ):
        np.testing.assert_array_equal(x[...], y[...], err_msg=str(path))


def test_logs_at_configured_frequencies(run):
    history, _ = run
    assert len(history["train_loss"]) == 4
    assert len(history["val_loss"]) == 2


def test_learns_beyond_uniform_guessing(run):
    history, _ = run
    uniform = math.log(TINYSTORIES_VOCAB_SIZE)
    assert history["train_loss"][-1] < history["train_loss"][0] < uniform
    assert history["val_loss"][-1] < 0.75 * uniform


def test_checkpoints_every_n_freq_checkpoint_and_the_final_step(run):
    _, workdir = run
    assert checkpoint_steps(workdir) == [150, 200]


def test_load_model_restores_the_trained_architecture(run):
    _, workdir = run
    model = load_model(str(workdir))
    assert model.num_layers == SMALL.num_layers
    assert model.model_dim == SMALL.model_dim
    assert (
        model.vocab_size
        == dataclasses.replace(SMALL, vocab_size=TINYSTORIES_VOCAB_SIZE).padded_vocab_size
    )


def test_load_model_needs_a_checkpoint(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="no checkpoints"):
        load_model(str(tmp_path))


def test_rerunning_a_finished_run_trains_nothing(copied_run: Path):
    before = load_model(str(copied_run))
    # Logging fields may change on resume.
    history = train_and_evaluate(
        dataclasses.replace(SMALL, n_freq_eval=1, n_freq_checkpoint=1), str(copied_run)
    )
    assert history == {"train_loss": [], "val_loss": []}
    assert checkpoint_steps(copied_run) == [150, 200]
    assert_same_params(load_model(str(copied_run)), before)


def test_resuming_after_a_kill_mid_save_matches_an_uninterrupted_run(
    run, copied_run: Path
):
    """A save cut short leaves an orbax temp directory, never a finalized step."""
    _, workdir = run
    checkpoints = copied_run / CHECKPOINT_DIR
    shutil.rmtree(checkpoints / "200")
    partial = checkpoints / "175.orbax-checkpoint-tmp-1760000000000000"
    partial.mkdir()
    (partial / "garbage").write_bytes(b"\x00" * 64)

    history = train_and_evaluate(SMALL, str(copied_run))

    assert len(history["val_loss"]) == 1
    assert checkpoint_steps(copied_run) == [150, 200]
    assert not partial.exists()
    assert_same_params(load_model(str(copied_run)), load_model(str(workdir)))


def test_resuming_a_killed_process_matches_an_uninterrupted_run(run, tmp_path: Path):
    """SIGKILL a real training process, then rerun: the final weights are bit-identical.

    Checkpointing every step makes saves overlap the kill, wherever it lands.
    """
    _, workdir = run
    killed = dataclasses.replace(SMALL, n_freq_checkpoint=1)
    trainer = subprocess.Popen(
        [sys.executable, "main.py", f"--workdir={tmp_path}", *cli_overrides(killed)],
        cwd=Path(__file__).parents[1],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + 300
    while not checkpoint_steps(tmp_path):
        assert trainer.poll() is None, "trainer exited before its first checkpoint"
        assert time.monotonic() < deadline, "trainer never checkpointed"
        time.sleep(0.01)
    trainer.send_signal(signal.SIGKILL)
    trainer.wait()
    assert 0 < max(checkpoint_steps(tmp_path)) < SMALL.n_iterations

    train_and_evaluate(SMALL, str(tmp_path))

    assert max(checkpoint_steps(tmp_path)) == SMALL.n_iterations
    assert_same_params(load_model(str(tmp_path)), load_model(str(workdir)))


@pytest.mark.parametrize(
    ("change", "field"),
    [
        ({"learning_rate": 1e-2}, "learning_rate"),
        ({"n_iterations": 300}, "n_iterations"),
        ({"seed": 0}, "seed"),
    ],
)
def test_refuses_to_resume_a_different_run(copied_run: Path, change: dict, field: str):
    with pytest.raises(ValueError, match=f"different config \\({field}: "):
        train_and_evaluate(dataclasses.replace(SMALL, **change), str(copied_run))


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
    assert checkpoint_steps(tmp_path) == []


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


@pytest.mark.parametrize("grad_accum_steps", [2, 4, 16])
def test_grad_accumulation_matches_a_full_batch_step(grad_accum_steps: int):
    config = dataclasses.replace(SMALL, vocab_size=TINYSTORIES_VOCAB_SIZE)
    tokens = np.random.default_rng(0).integers(
        0, TINYSTORIES_VOCAB_SIZE, (config.batch_size, config.sequence_length + 1)
    )
    batch = tokens[:, :-1], tokens[:, 1:]

    def two_steps(num_microbatches: int) -> tuple[Mamba, float]:
        # Two, since the first runs at the warmup's lr of 0.
        model = Mamba.from_config(config, rngs=nnx.Rngs(0))
        optimizer = make_optimizer(model, config)
        metrics = nnx.MultiMetric(loss=nnx.metrics.Average("loss"))
        for _ in range(2):
            microbatches = split_microbatches(batch, num_microbatches)
            train_step(model, optimizer, metrics, jax.device_put(microbatches))
        return model, float(metrics.compute()["loss"])

    full, full_loss = two_steps(1)
    accumulated, accumulated_loss = two_steps(grad_accum_steps)

    assert accumulated_loss == pytest.approx(full_loss, rel=1e-6)
    for (path, a), (_, b) in zip(
        nnx.state(accumulated, nnx.Param).flat_state(),
        nnx.state(full, nnx.Param).flat_state(),
    ):
        np.testing.assert_allclose(
            a[...], b[...], rtol=1e-5, atol=1e-6, err_msg=str(path)
        )


@pytest.mark.parametrize(
    ("batch_size", "grad_accum_steps", "num_devices", "match"),
    [
        (16, 3, 1, "multiple of grad_accum_steps"),
        (16, 0, 1, "multiple of grad_accum_steps"),
        (16, 32, 1, "multiple of grad_accum_steps"),
        (16, 2, 3, "multiple of the number of devices \\(3\\)"),
        (16, 4, 8, "multiple of the number of devices \\(8\\)"),
    ],
)
def test_rejects_batches_that_dont_split_evenly(
    batch_size: int, grad_accum_steps: int, num_devices: int, match: str
):
    config = dataclasses.replace(
        SMALL, batch_size=batch_size, grad_accum_steps=grad_accum_steps
    )
    with pytest.raises(ValueError, match=match):
        check_batch_layout(config, num_devices)


@pytest.mark.parametrize(
    ("batch_size", "grad_accum_steps", "num_devices"),
    [(16, 1, 1), (16, 2, 8), (16, 16, 1), (32, 4, 8)],
)
def test_accepts_batches_that_split_evenly(
    batch_size: int, grad_accum_steps: int, num_devices: int
):
    config = dataclasses.replace(
        SMALL, batch_size=batch_size, grad_accum_steps=grad_accum_steps
    )
    check_batch_layout(config, num_devices)


def test_data_parallel_on_8_devices_matches_one_device(tmp_path: Path):
    """8 CPU devices in a subprocess (the device count is fixed at JAX startup),
    with gradient accumulation on top, against a 1-device run of the same config."""
    short = dataclasses.replace(
        SMALL,
        n_iterations=20,
        warmup_steps=5,
        n_freq_train=10,
        n_freq_eval=10,
        n_freq_checkpoint=20,
    )
    one_device = tmp_path / "one_device"
    one_history = train_and_evaluate(short, str(one_device))

    eight_devices = tmp_path / "eight_devices"
    sharded = dataclasses.replace(short, grad_accum_steps=2)
    result = subprocess.run(
        [
            sys.executable,
            "main.py",
            f"--workdir={eight_devices}",
            "--logtostderr",
            *cli_overrides(sharded),
        ],
        cwd=Path(__file__).parents[1],
        env=os.environ | {"JAX_NUM_CPU_DEVICES": "8", "JAX_PLATFORMS": "cpu"},
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    assert "Data parallel over 8 cpu devices" in result.stderr

    val_losses = [
        float(line.rsplit(" ", 1)[1])
        for line in result.stdout.splitlines()
        if "val_loss" in line
    ]
    # main.py prints losses to 4 decimals.
    np.testing.assert_allclose(val_losses, one_history["val_loss"], atol=1e-4)
    for (path, a), (_, b) in zip(
        nnx.state(load_model(str(eight_devices)), nnx.Param).flat_state(),
        nnx.state(load_model(str(one_device)), nnx.Param).flat_state(),
    ):
        np.testing.assert_allclose(
            a[...], b[...], rtol=1e-4, atol=1e-5, err_msg=str(path)
        )

    # The 8-device checkpoint restores here too, on one device.
    history = train_and_evaluate(sharded, str(eight_devices))
    assert history == {"train_loss": [], "val_loss": []}


def test_bf16_compute_learns(tmp_path: Path):
    config = dataclasses.replace(SMALL, compute_dtype=Dtype.BFLOAT16, n_iterations=100)
    history = train_and_evaluate(config, str(tmp_path))
    uniform = math.log(TINYSTORIES_VOCAB_SIZE)
    assert history["train_loss"][-1] < history["train_loss"][0] < uniform
    assert history["val_loss"][-1] < 0.8 * uniform
