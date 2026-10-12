import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import datasets
import numpy as np
import pytest
from huggingface_hub import HfApi, HfFileSystem, RepoFile

from input_pipeline import (
    FINEWEB_EOT,
    FINEWEB_HEADER_BYTES,
    FINEWEB_MAGIC,
    FINEWEB_MAX_TRAIN_SHARDS,
    FINEWEB_REPO,
    FINEWEB_REVISION,
    FINEWEB_SPEEDRUN_VAL_TOKENS,
    FINEWEB_VERSION,
    FINEWEB_VOCAB_SIZE,
    TINYSTORIES_EOS,
    TINYSTORIES_REPO,
    TINYSTORIES_REVISION,
    TINYSTORIES_UNK,
    TINYSTORIES_VOCAB_SIZE,
    decode_chars,
    encode_chars,
    eval_batches,
    load_fineweb_train,
    load_tinystories,
    num_train_batches,
    parse_fineweb_header,
    read_fineweb_shard,
    split_microbatches,
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


@pytest.mark.parametrize("lengths", SHARD_LAYOUTS.values(), ids=SHARD_LAYOUTS.keys())
def test_train_batches_from_a_first_batch_continue_the_full_pass(lengths: list[int]):
    shards = consecutive_shards(lengths)
    full = list(train_batches(shards, 2, 3))
    for first_batch in range(len(full) + 2):
        resumed = list(train_batches(shards, 2, 3, first_batch=first_batch))
        assert len(resumed) == max(len(full) - first_batch, 0)
        for (x, y), (x_full, y_full) in zip(resumed, full[first_batch:]):
            np.testing.assert_array_equal(x, x_full)
            np.testing.assert_array_equal(y, y_full)


@pytest.mark.parametrize("num_microbatches", [1, 2, 4])
def test_split_microbatches_keeps_rows_in_order(num_microbatches: int):
    (batch,) = train_batches(consecutive_shards([4 * 3 + 1]), 4, 3)

    inputs, targets = split_microbatches(batch, num_microbatches)

    assert inputs.shape == targets.shape == (num_microbatches, 4 // num_microbatches, 3)
    np.testing.assert_array_equal(inputs.reshape(4, 3), batch[0])
    np.testing.assert_array_equal(targets.reshape(4, 3), batch[1])


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
        TINYSTORIES_REPO,
        revision=TINYSTORIES_REVISION,
        data_files={"validation": "data/validation-*.parquet"},
        split=VALIDATION_SLICE,
    )["text"][:]


@pytest.fixture(scope="module")
def cache_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("cache")


@pytest.fixture(scope="module")
def validation_ids(cache_dir: Path) -> np.memmap:
    return load_tinystories(VALIDATION_SLICE, cache_dir=cache_dir)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "Once upon a time,\nthere was a cat.",
        " ~!\"#$%&'()*+,-./09:;<=>?@AZ[\\]^_`az{|}",
    ],
)
def test_ascii_round_trips(text: str):
    ids = encode_chars(text)
    assert ids.dtype == np.uint8
    assert decode_chars(ids) == text


@pytest.mark.parametrize("text", ["â€™", "é", "🐱", "\t", "\x00"])
def test_characters_outside_vocab_map_to_unk(text: str):
    assert np.all(encode_chars(text) == TINYSTORIES_UNK)


def test_vocab_ids_are_dense_and_distinct():
    printable = "\n" + "".join(chr(c) for c in range(32, 127))
    ids = encode_chars(printable)
    assert sorted(ids) == list(range(2, TINYSTORIES_VOCAB_SIZE))


def test_load_tinystories_is_stories_followed_by_eos(
    validation_stories: list[str], validation_ids: np.ndarray
):
    assert validation_ids.dtype == np.uint8
    assert validation_ids.max() < TINYSTORIES_VOCAB_SIZE

    boundaries = np.flatnonzero(validation_ids == TINYSTORIES_EOS)
    assert len(boundaries) == len(validation_stories)
    assert boundaries[-1] == len(validation_ids) - 1

    starts = np.concatenate([[0], boundaries[:-1] + 1])
    for story, start, end in zip(validation_stories[:20], starts, boundaries):
        np.testing.assert_array_equal(validation_ids[start:end], encode_chars(story))


def test_load_tinystories_chunking_does_not_change_the_stream(
    validation_ids: np.ndarray, tmp_path: Path
):
    rechunked = load_tinystories(VALIDATION_SLICE, cache_dir=tmp_path, chunk_size=7)
    np.testing.assert_array_equal(rechunked, validation_ids)


def test_load_tinystories_is_a_read_only_memmap(validation_ids: np.ndarray):
    assert isinstance(validation_ids, np.memmap)
    assert not validation_ids.flags.writeable


def test_load_tinystories_reuses_cache_and_leaves_no_temp_files(
    validation_ids: np.memmap, cache_dir: Path
):
    (cached,) = (cache_dir / "mambax").iterdir()
    modified = cached.stat().st_mtime_ns

    again = load_tinystories(VALIDATION_SLICE, cache_dir=cache_dir)

    assert again.filename == validation_ids.filename
    assert cached.stat().st_mtime_ns == modified
    assert [path.name for path in (cache_dir / "mambax").iterdir()] == [cached.name]


def test_load_tinystories_caches_each_split_separately(tmp_path: Path):
    first = load_tinystories("validation[:10]", cache_dir=tmp_path)
    second = load_tinystories("validation[10:20]", cache_dir=tmp_path)

    assert first.filename != second.filename
    assert (first == TINYSTORIES_EOS).sum() == (second == TINYSTORIES_EOS).sum() == 10
    assert len(list((tmp_path / "mambax").iterdir())) == 2


@pytest.mark.parametrize("chunk_size", [0, -1])
def test_load_tinystories_rejects_non_positive_chunk_size(
    tmp_path: Path, chunk_size: int
):
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        load_tinystories(VALIDATION_SLICE, cache_dir=tmp_path, chunk_size=chunk_size)


def test_killed_encode_commits_nothing_and_next_load_recovers(tmp_path: Path):
    """SIGKILL mid-write (skipping all cleanup) must not leave a usable-looking cache."""
    encoder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "from input_pipeline import load_tinystories; "
            f"load_tinystories('validation', cache_dir={str(tmp_path)!r}, chunk_size=1)",
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

    recovered = load_tinystories("validation", cache_dir=tmp_path)

    assert (recovered == TINYSTORIES_EOS).sum() == 21_990
    assert list(cache.glob("*.tmp")) == []


def test_temp_files_of_live_encoders_are_kept(tmp_path: Path):
    load_tinystories("validation[:10]", cache_dir=tmp_path)
    (cached,) = (tmp_path / "mambax").glob("*.u8")
    cached.unlink()
    live = cached.with_name(f"{cached.name}.{os.getppid()}.tmp")
    live.touch()

    load_tinystories("validation[:10]", cache_dir=tmp_path)

    assert live.exists()


# FineWeb.


def write_shard(
    path: Path,
    tokens: np.ndarray,
    *,
    magic: int = FINEWEB_MAGIC,
    version: int = FINEWEB_VERSION,
    claimed_tokens: int | None = None,
) -> Path:
    """Writes the modded-nanogpt layout: 256 x int32 header, then uint16 tokens."""
    header = np.zeros(256, dtype="<i4")
    header[:3] = magic, version, len(tokens) if claimed_tokens is None else claimed_tokens
    path.write_bytes(header.tobytes() + tokens.astype("<u2").tobytes())
    return path


def test_read_fineweb_shard_round_trips_the_full_uint16_range(tmp_path: Path):
    tokens = np.array(
        [0, 1, FINEWEB_EOT, FINEWEB_VOCAB_SIZE - 1, 2**16 - 1, 42], dtype=np.uint16
    )

    shard = read_fineweb_shard(write_shard(tmp_path / "shard.bin", tokens))

    assert isinstance(shard, np.memmap)
    assert shard.dtype == np.uint16
    assert not shard.flags.writeable
    np.testing.assert_array_equal(shard, tokens)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"magic": 12345}, "magic 12345"),
        ({"version": 2}, "version 2"),
        ({"claimed_tokens": 7}, "header promises"),
        ({"claimed_tokens": 5}, "header promises"),
    ],
)
def test_read_fineweb_shard_rejects_corrupt_shards(
    tmp_path: Path, overrides: dict, match: str
):
    path = write_shard(tmp_path / "shard.bin", np.arange(6, dtype=np.uint16), **overrides)
    with pytest.raises(ValueError, match=match):
        read_fineweb_shard(path)


def test_parse_fineweb_header_rejects_truncated_header():
    with pytest.raises(ValueError, match="expected 1024"):
        parse_fineweb_header(b"\x00" * 100)


@pytest.mark.parametrize("num_shards", [0, -1, FINEWEB_MAX_TRAIN_SHARDS + 1])
def test_load_fineweb_train_rejects_shard_counts_outside_the_dataset(num_shards: int):
    with pytest.raises(ValueError, match="num_shards"):
        load_fineweb_train(num_shards)


def test_real_fineweb_val_shard_matches_the_format():
    """Ranged read of the pinned Hub shard (a few KB, not the 200MB file)."""
    name = "fineweb_val_000000.bin"
    with HfFileSystem().open(
        f"datasets/{FINEWEB_REPO}/{name}",
        "rb",
        revision=FINEWEB_REVISION,
        block_size=4096,
    ) as f:
        head = f.read(FINEWEB_HEADER_BYTES + 2 * 64)
    (info,) = HfApi().get_paths_info(
        FINEWEB_REPO, [name], repo_type="dataset", revision=FINEWEB_REVISION
    )

    num_tokens = parse_fineweb_header(head)
    first_tokens = np.frombuffer(head[FINEWEB_HEADER_BYTES:], dtype="<u2")

    assert isinstance(info, RepoFile)
    assert info.size == FINEWEB_HEADER_BYTES + 2 * num_tokens
    assert FINEWEB_SPEEDRUN_VAL_TOKENS + 1 <= num_tokens
    assert first_tokens[0] == FINEWEB_EOT
    assert first_tokens.max() < FINEWEB_VOCAB_SIZE
