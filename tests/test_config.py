import dataclasses
import enum
import types
import typing

import pytest
from absl import flags
from ml_collections import config_flags

from configs.default import Config


@pytest.mark.parametrize(("vocab_size", "padded"), [(64, 64), (65, 72), (50277, 50280)])
def test_config_pads_vocab(vocab_size: int, padded: int):
    assert Config(vocab_size=vocab_size).padded_vocab_size == padded


def test_config_derived_values_follow_overrides():
    config = Config(model_dim=192)
    config.model_dim = 256
    assert config.hidden_dim == 512
    assert config.resolved_dt_rank == 16


def test_config_explicit_dt_rank_wins():
    assert Config(model_dim=256, dt_rank=8).resolved_dt_rank == 8


def test_config_rejects_unset_vocab():
    with pytest.raises(ValueError, match="vocab_size is unset"):
        Config().padded_vocab_size


def _override(field_type: type) -> tuple[str, object]:
    """A command line value for `field_type` and what it should parse to."""
    if isinstance(field_type, types.UnionType):
        (field_type,) = (t for t in typing.get_args(field_type) if t is not type(None))
    if issubclass(field_type, enum.Enum):
        member = list(field_type)[-1]
        return member.name.lower(), member
    samples: dict[type, tuple[str, object]] = {
        bool: ("true", True),
        int: ("7", 7),
        float: ("0.5", 0.5),
        str: ("override", "override"),
    }
    return samples[field_type]


@pytest.mark.parametrize("field", dataclasses.fields(Config), ids=lambda f: f.name)
def test_every_field_can_be_overridden_from_the_command_line(field: dataclasses.Field):
    """ml_collections only parses some field types (Literal and non-Optional
    unions crash it), so a field that can't be overridden fails here, not in a run."""
    raw, expected = _override(typing.get_type_hints(Config)[field.name])
    # config_flags discovers overrides from sys.argv unless given sys_argv.
    argv = ["prog", f"--config.{field.name}={raw}"]
    flag_values = flags.FlagValues()
    config_flags.DEFINE_config_file(
        "config",
        "configs/default.py",
        flag_values=flag_values,
        lock_config=True,
        sys_argv=argv,
    )

    flag_values(argv)

    assert getattr(flag_values.config, field.name) == expected
