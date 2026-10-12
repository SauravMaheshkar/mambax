"""Datasets as token streams, and batching over them.

Every dataset reduces to token streams (1-D integer arrays, memmaps on disk):
one or more train shards and a single val stream. Training reads the train
shards as one contiguous stream, each token exactly once, the way the
modded-nanogpt speedrun does. Evaluation always covers the same val prefix, so
val losses are comparable across runs and checkpoints.

    shard 0             shard 1
    [t0 t1 ... t99]     [t100 ... ]          <- one logical stream
    |-- batch 0 --|-- batch 1 --|-- ...      <- B*L tokens each, +1 for targets

Two datasets are supported:

- TinyStories, character level. Each split is encoded once, streamed chunk by
  chunk into a uint8 file in the `datasets` cache (the train split is ~1.9GB
  of characters, more than a laptop wants in RAM):

      parquet --(datasets, memory-mapped Arrow)--> chunks of stories
              --encode--> append to <cache>/mambax/*.u8 --np.memmap--> stream

  The vocabulary is fixed (printable ASCII plus newline) instead of derived
  from the data: ~0.1% of TinyStories is non-ASCII, almost all of it mojibake
  like "â€™", which maps to UNK.

- FineWeb, GPT-2 tokenized, as used by the modded-nanogpt speedrun. Each
  `.bin` shard holds 100M tokens: a 256 x int32 little-endian header (magic,
  version, token count, then zeros) followed by the tokens as uint16. Every
  document starts with <|endoftext|>. Shards are memory-mapped straight from
  the Hugging Face cache, nothing is copied or re-encoded. Speedrun
  comparisons evaluate on the first `FINEWEB_SPEEDRUN_VAL_TOKENS` of the val
  shard.
"""

import dataclasses
import hashlib
import os
from collections.abc import Iterator, Sequence
from pathlib import Path

import datasets
import datasets.config
import numpy as np
from huggingface_hub import hf_hub_download
from jaxtyping import Int, UInt8, UInt16

from configs.default import Dataset


TokenStream = Int[np.ndarray, " tokens"]
Batch = tuple[Int[np.ndarray, "batch seq"], Int[np.ndarray, "batch seq"]]


@dataclasses.dataclass(frozen=True)
class TokenData:
    train: Sequence[TokenStream]
    val: TokenStream
    vocab_size: int


def load_data(
    dataset: Dataset,
    *,
    tinystories_train_split: str = "train",
    tinystories_eval_split: str = "validation",
    fineweb_train_shards: int = 1,
) -> TokenData:
    match dataset:
        case Dataset.TINYSTORIES:
            return TokenData(
                train=[load_tinystories(tinystories_train_split)],
                val=load_tinystories(tinystories_eval_split),
                vocab_size=TINYSTORIES_VOCAB_SIZE,
            )
        case Dataset.FINEWEB:
            return TokenData(
                train=load_fineweb_train(fineweb_train_shards),
                val=load_fineweb_val(),
                vocab_size=FINEWEB_VOCAB_SIZE,
            )


# Batching.


def num_train_batches(
    shards: Sequence[TokenStream], batch_size: int, sequence_length: int
) -> int:
    total = sum(len(shard) for shard in shards)
    return max(total - 1, 0) // (batch_size * sequence_length)


def train_batches(
    shards: Sequence[TokenStream], batch_size: int, sequence_length: int
) -> Iterator[Batch]:
    """Consecutive batches over the shards read as one stream, until it runs out.

    Batch k covers stream tokens [k*B*L, (k+1)*B*L], the extra token being the
    last target. Rows are consecutive L-token windows, so no token is skipped
    or repeated, including across shard boundaries.
    """
    offsets = np.cumsum([0, *(len(shard) for shard in shards)])
    step = batch_size * sequence_length
    for start in range(0, int(offsets[-1]) - step, step):
        window = _read(shards, offsets, start, start + step + 1)
        yield _to_batch(window, batch_size, sequence_length)


def eval_batches(
    val: TokenStream, val_tokens: int, batch_size: int, sequence_length: int
) -> Iterator[Batch]:
    """Batches covering exactly the first `val_tokens` targets of `val`, in order."""
    step = batch_size * sequence_length
    if val_tokens <= 0 or val_tokens % step:
        raise ValueError(
            f"val_tokens ({val_tokens}) must be a positive multiple of "
            f"batch_size * sequence_length ({step})"
        )
    if val_tokens + 1 > len(val):
        raise ValueError(
            f"val_tokens ({val_tokens}) needs {val_tokens + 1} val tokens, "
            f"the val stream has {len(val)}"
        )
    return (
        _to_batch(val[start : start + step + 1], batch_size, sequence_length)
        for start in range(0, val_tokens, step)
    )


def _read(
    shards: Sequence[TokenStream], offsets: np.ndarray, start: int, stop: int
) -> np.ndarray:
    """Stream tokens [start, stop), stitched across shard boundaries."""
    parts = []
    shard = int(np.searchsorted(offsets, start, side="right")) - 1
    while start < stop:
        local = start - int(offsets[shard])
        take = min(stop, int(offsets[shard + 1])) - start
        parts.append(shards[shard][local : local + take])
        start += take
        shard += 1
    return np.concatenate(parts)


def _to_batch(window: np.ndarray, batch_size: int, sequence_length: int) -> Batch:
    tokens = np.asarray(window, dtype=np.int32)
    return (
        tokens[:-1].reshape(batch_size, sequence_length),
        tokens[1:].reshape(batch_size, sequence_length),
    )


# TinyStories, character level.

TINYSTORIES_REPO = "roneneldan/TinyStories"
# Pinned so the encoded cache can never silently go stale against new data.
TINYSTORIES_REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"

TINYSTORIES_EOS = 0
TINYSTORIES_UNK = 1
_CHARS = "\n" + "".join(chr(c) for c in range(32, 127))
TINYSTORIES_VOCAB_SIZE = 2 + len(_CHARS)

# Joins stories before encoding, then maps to EOS. ASCII "end of text", not in the data.
_STORY_SEPARATOR = "\x03"

_ENCODE_TABLE = np.full(128, TINYSTORIES_UNK, dtype=np.uint8)
_ENCODE_TABLE[ord(_STORY_SEPARATOR)] = TINYSTORIES_EOS
for _id, _char in enumerate(_CHARS, start=2):
    _ENCODE_TABLE[ord(_char)] = _id

_DECODE = {TINYSTORIES_EOS: "\n\n", TINYSTORIES_UNK: "�"} | {
    i: c for i, c in enumerate(_CHARS, start=2)
}


def encode_chars(text: str) -> UInt8[np.ndarray, " chars"]:
    codepoints = np.frombuffer(text.encode("utf-32-le"), dtype=np.uint32)
    ascii_ids = _ENCODE_TABLE[np.minimum(codepoints, 127)]
    return np.where(codepoints < 128, ascii_ids, TINYSTORIES_UNK).astype(np.uint8)


def decode_chars(ids: Int[np.ndarray, " chars"]) -> str:
    return "".join(_DECODE[int(i)] for i in ids)


def load_tinystories(
    split: str, cache_dir: Path | None = None, chunk_size: int = 10_000
) -> UInt8[np.memmap, " chars"]:
    """Read-only memmap of a split as one stream of stories, each followed by EOS.

    `split` accepts `datasets` slicing, e.g. "train[:10%]". Only that split's
    parquet files are downloaded. The first call encodes and caches the split,
    later calls just map the cached file.
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    path = _tinystories_cache_path(
        split, Path(cache_dir or datasets.config.HF_DATASETS_CACHE)
    )
    if not path.exists():
        _encode_tinystories_to_file(split, path, chunk_size)
    return np.memmap(path, dtype=np.uint8, mode="r")


def _tinystories_cache_path(split: str, cache_dir: Path) -> Path:
    key = hashlib.sha256(
        f"{TINYSTORIES_REPO}@{TINYSTORIES_REVISION}:{split}".encode()
        + _ENCODE_TABLE.tobytes()
    ).hexdigest()[:16]
    return cache_dir / "mambax" / f"tinystories-{split.partition('[')[0]}-{key}.u8"


def _encode_tinystories_to_file(split: str, path: Path, chunk_size: int) -> None:
    name = split.partition("[")[0]
    stories = datasets.load_dataset(
        TINYSTORIES_REPO,
        revision=TINYSTORIES_REVISION,
        data_files={name: f"data/{name}-*.parquet"},
        split=split,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    _remove_orphaned_temp_files(path)
    # Write to a per-process temp file and rename into place: a killed run must
    # never leave a truncated file that later runs would mistake for the cache.
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        with tmp.open("wb") as f:
            for batch in stories.iter(batch_size=chunk_size):
                text = _STORY_SEPARATOR.join(batch["text"]) + _STORY_SEPARATOR
                f.write(encode_chars(text).tobytes())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _remove_orphaned_temp_files(path: Path) -> None:
    """Deletes temp files of encoders that died without cleaning up (e.g. SIGKILL)."""
    for tmp in path.parent.glob(f"{path.name}.*.tmp"):
        pid = int(tmp.suffixes[-2].lstrip("."))
        if not _is_running(pid):
            tmp.unlink(missing_ok=True)


def _is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


# FineWeb, GPT-2 tokenized shards from the modded-nanogpt speedrun.

FINEWEB_REPO = "kjj0/fineweb10B-gpt2"
# Pinned so a re-upload can't silently change what "the val set" means.
FINEWEB_REVISION = "889765ea1f903759787add96995d81171b632d0c"

FINEWEB_VOCAB_SIZE = 50257
FINEWEB_EOT = 50256
FINEWEB_SPEEDRUN_VAL_TOKENS = 10_485_760
FINEWEB_MAX_TRAIN_SHARDS = 103

FINEWEB_MAGIC = 20240520
FINEWEB_VERSION = 1
FINEWEB_HEADER_BYTES = 256 * 4


def parse_fineweb_header(header: bytes) -> int:
    """Validates a shard header and returns how many tokens follow it."""
    if len(header) < FINEWEB_HEADER_BYTES:
        raise ValueError(
            f"shard header is {len(header)} bytes, expected {FINEWEB_HEADER_BYTES}"
        )
    magic, version, num_tokens = np.frombuffer(header, dtype="<i4", count=3)
    if magic != FINEWEB_MAGIC:
        raise ValueError(f"not a FineWeb shard: magic {magic}, expected {FINEWEB_MAGIC}")
    if version != FINEWEB_VERSION:
        raise ValueError(
            f"unsupported shard version {version}, expected {FINEWEB_VERSION}"
        )
    return int(num_tokens)


def read_fineweb_shard(path: Path) -> UInt16[np.memmap, " tokens"]:
    """Read-only memmap of a shard's tokens, after checking its header and size."""
    with path.open("rb") as f:
        num_tokens = parse_fineweb_header(f.read(FINEWEB_HEADER_BYTES))
    expected_bytes = FINEWEB_HEADER_BYTES + 2 * num_tokens
    if (actual := path.stat().st_size) != expected_bytes:
        raise ValueError(
            f"{path} is {actual} bytes but its header promises {expected_bytes}"
        )
    return np.memmap(
        path, dtype="<u2", mode="r", offset=FINEWEB_HEADER_BYTES, shape=(num_tokens,)
    )


def load_fineweb_train(num_shards: int) -> list[UInt16[np.memmap, " tokens"]]:
    """The first `num_shards` train shards (100M tokens, 200MB each), downloaded once."""
    if not 1 <= num_shards <= FINEWEB_MAX_TRAIN_SHARDS:
        raise ValueError(
            f"num_shards must be in [1, {FINEWEB_MAX_TRAIN_SHARDS}], got {num_shards}"
        )
    return [
        read_fineweb_shard(_download_fineweb(f"fineweb_train_{i:06d}.bin"))
        for i in range(1, num_shards + 1)
    ]


def load_fineweb_val() -> UInt16[np.memmap, " tokens"]:
    return read_fineweb_shard(_download_fineweb("fineweb_val_000000.bin"))


def _download_fineweb(filename: str) -> Path:
    return Path(
        hf_hub_download(
            FINEWEB_REPO, filename, repo_type="dataset", revision=FINEWEB_REVISION
        )
    )
