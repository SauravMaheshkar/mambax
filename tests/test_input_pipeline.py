import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import datasets
import numpy as np
import pytest

from input_pipeline import (
    DATASET,
    EOS,
    REVISION,
    UNK,
    VOCAB_SIZE,
    decode,
    encode,
    load_split,
    sample_batch,
)


VALIDATION_SLICE = "validation[:1%]"


@pytest.fixture(scope="module")
def validation_stories() -> list[str]:
    return datasets.load_dataset(
        DATASET,
        revision=REVISION,
        data_files={"validation": "data/validation-*.parquet"},
        split=VALIDATION_SLICE,
    )["text"][:]


@pytest.fixture(scope="module")
def cache_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("cache")


@pytest.fixture(scope="module")
def validation_ids(cache_dir: Path) -> np.memmap:
    return load_split(VALIDATION_SLICE, cache_dir=cache_dir)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "Once upon a time,\nthere was a cat.",
        " ~!\"#$%&'()*+,-./09:;<=>?@AZ[\\]^_`az{|}",
    ],
)
def test_ascii_round_trips(text: str):
    ids = encode(text)
    assert ids.dtype == np.uint8
    assert decode(ids) == text


@pytest.mark.parametrize("text", ["â€™", "é", "🐱", "\t", "\x00"])
def test_characters_outside_vocab_map_to_unk(text: str):
    assert np.all(encode(text) == UNK)


def test_vocab_ids_are_dense_and_distinct():
    printable = "\n" + "".join(chr(c) for c in range(32, 127))
    ids = encode(printable)
    assert sorted(ids) == list(range(2, VOCAB_SIZE))


def test_load_split_is_stories_followed_by_eos(
    validation_stories: list[str], validation_ids: np.ndarray
):
    assert validation_ids.dtype == np.uint8
    assert validation_ids.max() < VOCAB_SIZE

    boundaries = np.flatnonzero(validation_ids == EOS)
    assert len(boundaries) == len(validation_stories)
    assert boundaries[-1] == len(validation_ids) - 1

    starts = np.concatenate([[0], boundaries[:-1] + 1])
    for story, start, end in zip(validation_stories[:20], starts, boundaries):
        np.testing.assert_array_equal(validation_ids[start:end], encode(story))


def test_load_split_chunking_does_not_change_the_stream(
    validation_ids: np.ndarray, tmp_path: Path
):
    rechunked = load_split(VALIDATION_SLICE, cache_dir=tmp_path, chunk_size=7)
    np.testing.assert_array_equal(rechunked, validation_ids)


def test_load_split_is_a_read_only_memmap(validation_ids: np.ndarray):
    assert isinstance(validation_ids, np.memmap)
    assert not validation_ids.flags.writeable


def test_load_split_reuses_cache_and_leaves_no_temp_files(
    validation_ids: np.memmap, cache_dir: Path
):
    (cached,) = (cache_dir / "mambax").iterdir()
    modified = cached.stat().st_mtime_ns

    again = load_split(VALIDATION_SLICE, cache_dir=cache_dir)

    assert again.filename == validation_ids.filename
    assert cached.stat().st_mtime_ns == modified
    assert [path.name for path in (cache_dir / "mambax").iterdir()] == [cached.name]


def test_load_split_caches_each_split_separately(tmp_path: Path):
    first = load_split("validation[:10]", cache_dir=tmp_path)
    second = load_split("validation[10:20]", cache_dir=tmp_path)

    assert first.filename != second.filename
    assert (first == EOS).sum() == (second == EOS).sum() == 10
    assert len(list((tmp_path / "mambax").iterdir())) == 2


@pytest.mark.parametrize("chunk_size", [0, -1])
def test_load_split_rejects_non_positive_chunk_size(tmp_path: Path, chunk_size: int):
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        load_split(VALIDATION_SLICE, cache_dir=tmp_path, chunk_size=chunk_size)


def test_killed_encode_commits_nothing_and_next_load_recovers(tmp_path: Path):
    """SIGKILL mid-write (skipping all cleanup) must not leave a usable-looking cache."""
    encoder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "from input_pipeline import load_split; "
            f"load_split('validation', cache_dir={str(tmp_path)!r}, chunk_size=1)",
        ],
        cwd=Path(__file__).parents[1],
    )
    cache = tmp_path / "mambax"
    deadline = time.monotonic() + 120
    while not list(cache.glob("*.tmp")):
        assert encoder.poll() is None, "encoder finished before it could be killed"
        assert time.monotonic() < deadline, "encoder never started writing"
        time.sleep(0.1)
    encoder.send_signal(signal.SIGKILL)
    encoder.wait()

    assert list(cache.glob("*.u8")) == []
    assert len(list(cache.glob("*.tmp"))) == 1

    recovered = load_split("validation", cache_dir=tmp_path)

    assert (recovered == EOS).sum() == 21_990
    assert list(cache.glob("*.tmp")) == []


def test_temp_files_of_live_encoders_are_kept(tmp_path: Path):
    load_split("validation[:10]", cache_dir=tmp_path)
    (cached,) = (tmp_path / "mambax").glob("*.u8")
    cached.unlink()
    live = cached.with_name(f"{cached.name}.{os.getppid()}.tmp")
    live.touch()

    load_split("validation[:10]", cache_dir=tmp_path)

    assert live.exists()


@pytest.mark.parametrize(("batch_size", "sequence_length"), [(1, 1), (4, 16), (8, 64)])
def test_sample_batch_targets_are_inputs_shifted_by_one(
    validation_ids: np.ndarray, batch_size: int, sequence_length: int
):
    rng = np.random.default_rng(0)
    inputs, targets = sample_batch(rng, validation_ids, batch_size, sequence_length)

    assert inputs.shape == targets.shape == (batch_size, sequence_length)
    assert inputs.dtype == targets.dtype == np.int32
    np.testing.assert_array_equal(inputs[:, 1:], targets[:, :-1])
    for row_inputs, row_targets in zip(inputs, targets):
        window = np.concatenate([row_inputs, row_targets[-1:]])
        start = _find_window(validation_ids, window)
        assert start is not None


def test_sample_batch_is_deterministic_per_seed(validation_ids: np.ndarray):
    first = sample_batch(np.random.default_rng(3), validation_ids, 4, 32)
    second = sample_batch(np.random.default_rng(3), validation_ids, 4, 32)
    np.testing.assert_array_equal(first, second)


def _find_window(data: np.ndarray, window: np.ndarray) -> int | None:
    windows = np.lib.stride_tricks.sliding_window_view(data, len(window))
    matches = np.flatnonzero((windows == window).all(axis=1))
    return int(matches[0]) if len(matches) else None
