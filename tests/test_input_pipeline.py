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
    eval_batches,
    load_split,
    num_train_batches,
    train_batches,
)


def consecutive_shards(lengths: list[int]) -> list[np.ndarray]:
    """Shards holding 0, 1, 2, ... so a correct stream always has target = input + 1."""
    bounds = np.cumsum([0, *lengths])
    return [
        np.arange(start, stop, dtype=np.uint16) for start, stop in zip(bounds, bounds[1:])
    ]


SHARD_LAYOUTS = {
    "single shard": [40],
    "ragged shards": [10, 15, 15],
    "boundary on a batch edge": [12, 12, 16],
    "empty shard in the middle": [10, 0, 30],
    "one-token shards": [1] * 40,
}


@pytest.mark.parametrize("lengths", SHARD_LAYOUTS.values(), ids=SHARD_LAYOUTS.keys())
@pytest.mark.parametrize(("batch_size", "sequence_length"), [(1, 1), (2, 3), (3, 4)])
def test_train_batches_read_every_token_once_in_order(
    lengths: list[int], batch_size: int, sequence_length: int
):
    shards = consecutive_shards(lengths)
    batches = list(train_batches(shards, batch_size, sequence_length))

    assert len(batches) == num_train_batches(shards, batch_size, sequence_length)
    inputs = np.concatenate([x.ravel() for x, _ in batches])
    targets = np.concatenate([y.ravel() for _, y in batches])
    np.testing.assert_array_equal(inputs, np.arange(len(inputs)))
    np.testing.assert_array_equal(targets, inputs + 1)


def test_train_batches_shapes_and_dtype():
    for inputs, targets in train_batches(consecutive_shards([10, 15, 15]), 2, 3):
        assert inputs.shape == targets.shape == (2, 3)
        assert inputs.dtype == targets.dtype == np.int32


@pytest.mark.parametrize(
    ("total", "expected"), [(0, 0), (6, 0), (7, 1), (12, 1), (13, 2), (40, 6)]
)
def test_num_train_batches_leaves_room_for_the_last_target(total: int, expected: int):
    assert num_train_batches(consecutive_shards([total]), 2, 3) == expected


def test_eval_batches_cover_exactly_the_val_prefix():
    val = np.arange(100, dtype=np.uint16)
    batches = list(eval_batches(val, val_tokens=24, batch_size=2, sequence_length=3))

    assert len(batches) == 4
    inputs = np.concatenate([x.ravel() for x, _ in batches])
    targets = np.concatenate([y.ravel() for _, y in batches])
    np.testing.assert_array_equal(inputs, np.arange(24))
    np.testing.assert_array_equal(targets, np.arange(1, 25))


def test_eval_batches_are_identical_on_every_call():
    val = np.arange(100, dtype=np.uint16)
    first = list(eval_batches(val, 24, 2, 3))
    second = list(eval_batches(val, 24, 2, 3))
    np.testing.assert_array_equal(first, second)


def test_eval_batches_accept_the_whole_stream_minus_the_last_target():
    val = np.arange(25, dtype=np.uint16)
    assert len(list(eval_batches(val, 24, 2, 3))) == 4


@pytest.mark.parametrize(
    ("val_tokens", "match"),
    [
        (0, "positive multiple"),
        (-6, "positive multiple"),
        (7, "positive multiple"),
        (24, "the val stream has 24"),
    ],
)
def test_eval_batches_reject_bad_budgets_before_iterating(val_tokens: int, match: str):
    val = np.arange(24, dtype=np.uint16)
    with pytest.raises(ValueError, match=match):
        eval_batches(val, val_tokens, batch_size=2, sequence_length=3)


# TinyStories.

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
