"""Character-level TinyStories.

The train split is ~1.9GB of characters, more than a laptop wants in RAM.
Each split is encoded once, streamed chunk by chunk into a uint8 file in the
`datasets` cache, and read back as an `np.memmap`, so training only pages in
the windows it samples:

    parquet --(datasets, memory-mapped Arrow)--> chunks of stories
            --encode--> append to <cache>/mambax/*.u8 --np.memmap--> sample_batch

The vocabulary is fixed (printable ASCII plus newline) instead of derived from
the data: ~0.1% of TinyStories is non-ASCII, almost all of it mojibake like
"â€™", which maps to UNK.
"""

import hashlib
import os
from pathlib import Path

import datasets
import datasets.config
import numpy as np
from jaxtyping import Int, UInt8


DATASET = "roneneldan/TinyStories"
# Pinned so the encoded cache can never silently go stale against new data.
REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"

EOS = 0
UNK = 1
_CHARS = "\n" + "".join(chr(c) for c in range(32, 127))
VOCAB_SIZE = 2 + len(_CHARS)

# Joins stories before encoding, then maps to EOS. ASCII "end of text", not in the data.
_STORY_SEPARATOR = "\x03"

_ENCODE_TABLE = np.full(128, UNK, dtype=np.uint8)
_ENCODE_TABLE[ord(_STORY_SEPARATOR)] = EOS
for _id, _char in enumerate(_CHARS, start=2):
    _ENCODE_TABLE[ord(_char)] = _id

_DECODE = {EOS: "\n\n", UNK: "�"} | {i: c for i, c in enumerate(_CHARS, start=2)}


def encode(text: str) -> UInt8[np.ndarray, " chars"]:
    codepoints = np.frombuffer(text.encode("utf-32-le"), dtype=np.uint32)
    ascii_ids = _ENCODE_TABLE[np.minimum(codepoints, 127)]
    return np.where(codepoints < 128, ascii_ids, UNK).astype(np.uint8)


def decode(ids: Int[np.ndarray, " chars"]) -> str:
    return "".join(_DECODE[int(i)] for i in ids)


def load_split(
    split: str, cache_dir: Path | None = None, chunk_size: int = 10_000
) -> UInt8[np.memmap, " chars"]:
    """Read-only memmap of a split as one stream of stories, each followed by EOS.

    `split` accepts `datasets` slicing, e.g. "train[:10%]". Only that split's
    parquet files are downloaded. The first call encodes and caches the split,
    later calls just map the cached file.
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    path = _cache_path(split, Path(cache_dir or datasets.config.HF_DATASETS_CACHE))
    if not path.exists():
        _encode_to_file(split, path, chunk_size)
    return np.memmap(path, dtype=np.uint8, mode="r")


def _cache_path(split: str, cache_dir: Path) -> Path:
    key = hashlib.sha256(
        f"{DATASET}@{REVISION}:{split}".encode() + _ENCODE_TABLE.tobytes()
    ).hexdigest()[:16]
    return cache_dir / "mambax" / f"tinystories-{split.partition('[')[0]}-{key}.u8"


def _encode_to_file(split: str, path: Path, chunk_size: int) -> None:
    name = split.partition("[")[0]
    stories = datasets.load_dataset(
        DATASET,
        revision=REVISION,
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
                f.write(encode(text).tobytes())
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


def sample_batch(
    rng: np.random.Generator,
    data: UInt8[np.ndarray, " chars"],
    batch_size: int,
    sequence_length: int,
) -> tuple[Int[np.ndarray, "batch seq"], Int[np.ndarray, "batch seq"]]:
    """Random windows of `data` as (inputs, next-character targets)."""
    starts = rng.integers(0, len(data) - sequence_length, size=batch_size)
    windows = data[starts[:, None] + np.arange(sequence_length + 1)].astype(np.int32)
    return windows[:, :-1], windows[:, 1:]
