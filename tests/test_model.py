import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from huggingface_hub import hf_hub_download
from safetensors import safe_open

from configs.default import Config
from model import Mamba, MambaBlock, dt_bias_init


PRETRAINED_REPO = "state-spaces/mamba-130m-hf"


@pytest.fixture(scope="module")
def tiny_model() -> Mamba:
    config = Config(vocab_size=64, model_dim=16, num_layers=2, state_dim=4)
    return Mamba.from_config(config, rngs=nnx.Rngs(0))


@pytest.fixture(scope="module")
def pretrained_model() -> Mamba:
    return Mamba.from_pretrained(PRETRAINED_REPO)


def sequential_scan(u, delta, A, B, C, D) -> np.ndarray:
    """Step-by-step reference for the selective scan, written straight from the paper."""
    batch, seq, hidden = u.shape
    h = np.zeros((batch, hidden, A.shape[-1]))
    ys = []
    for t in range(seq):
        dt = delta[:, t, :, None]
        h = np.exp(dt * A) * h + dt * B[:, t, None, :] * u[:, t, :, None]
        ys.append((h * C[:, t, None, :]).sum(-1) + D * u[:, t])
    return np.stack(ys, axis=1)


@pytest.mark.parametrize("seq_len", [1, 2, 7, 64])
def test_selective_scan_matches_sequential_reference(seq_len: int):
    batch, hidden, state = 2, 8, 4
    keys = jax.random.split(jax.random.key(seq_len), 6)
    u = jax.random.normal(keys[0], (batch, seq_len, hidden))
    delta = jax.nn.softplus(jax.random.normal(keys[1], (batch, seq_len, hidden)))
    A = -jnp.exp(jax.random.normal(keys[2], (hidden, state)))
    B = jax.random.normal(keys[3], (batch, seq_len, state))
    C = jax.random.normal(keys[4], (batch, seq_len, state))
    D = jax.random.normal(keys[5], (hidden,))

    actual = MambaBlock.selective_scan(u, delta, A, B, C, D)
    expected = sequential_scan(*map(np.asarray, (u, delta, A, B, C, D)))

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("position", [0, 5, 31])
def test_model_is_causal(tiny_model: Mamba, position: int):
    tokens = jax.random.randint(jax.random.key(0), (2, 32), 0, tiny_model.vocab_size)
    perturbed = tokens.at[:, position].set(
        (tokens[:, position] + 1) % tiny_model.vocab_size
    )

    logits = tiny_model(tokens)
    perturbed_logits = tiny_model(perturbed)

    np.testing.assert_array_equal(logits[:, :position], perturbed_logits[:, :position])
    assert not np.allclose(logits[:, position], perturbed_logits[:, position])


def test_conv_is_depthwise(tiny_model: Mamba):
    mixer = tiny_model.layers[0].mixer
    assert mixer.conv1d.kernel.shape == (mixer.conv_dim, 1, mixer.hidden_dim)


def test_dt_bias_init_is_log_uniform_between_bounds():
    dt_min, dt_max = 1e-3, 1e-1
    bias = dt_bias_init(dt_min, dt_max)(jax.random.key(0), (100_000,))
    log_dt = np.log(np.asarray(jax.nn.softplus(bias)))

    assert log_dt.min() >= np.log(dt_min) - 1e-4
    assert log_dt.max() <= np.log(dt_max) + 1e-4
    quartiles = np.quantile(log_dt, [0.25, 0.5, 0.75])
    expected = np.log(dt_min) + np.array([0.25, 0.5, 0.75]) * np.log(dt_max / dt_min)
    np.testing.assert_allclose(quartiles, expected, atol=0.05)


def test_dt_bias_init_clamps_to_floor():
    bias = dt_bias_init(dt_min=1e-8, dt_max=1e-6, dt_floor=1e-4)(
        jax.random.key(0), (1_000,)
    )
    np.testing.assert_allclose(jax.nn.softplus(bias), 1e-4, rtol=1e-4)


def test_dt_proj_uses_mamba_init(tiny_model: Mamba):
    dt_proj = tiny_model.layers[0].mixer.dt_proj
    assert dt_proj.bias is not None
    dt = jax.nn.softplus(dt_proj.bias[...])

    assert jnp.all((dt >= 1e-3 * (1 - 1e-4)) & (dt <= 1e-1 * (1 + 1e-4)))
    assert jnp.all(jnp.abs(dt_proj.kernel[...]) <= tiny_model.dt_rank**-0.5)


@pytest.mark.parametrize(("vocab_size", "padded"), [(64, 64), (65, 72), (50277, 50280)])
def test_config_pads_vocab(vocab_size: int, padded: int):
    assert Config(vocab_size=vocab_size).padded_vocab_size == padded


def test_config_derived_values_follow_overrides():
    config = Config(model_dim=192)
    config.model_dim = 256
    assert config.hidden_dim == 512
    assert config.resolved_dt_rank == 16


def test_config_rejects_unset_vocab():
    with pytest.raises(ValueError, match="vocab_size is unset"):
        Config().padded_vocab_size


def test_pretrained_loads_every_checkpoint_tensor(pretrained_model: Mamba):
    path = hf_hub_download(PRETRAINED_REPO, "model.safetensors")
    with safe_open(path, framework="numpy") as f:
        checkpoint_size = sum(f.get_tensor(key).size for key in f.keys())
        in_proj = f.get_tensor("backbone.layers.3.mixer.in_proj.weight")
        conv = f.get_tensor("backbone.layers.3.mixer.conv1d.weight")

    mixer = pretrained_model.layers[3].mixer
    assert pretrained_model.num_params == checkpoint_size
    np.testing.assert_array_equal(mixer.in_proj.kernel[...], in_proj.T)
    np.testing.assert_array_equal(mixer.conv1d.kernel[...], conv.transpose(2, 1, 0))


def test_pretrained_copies_in_context(pretrained_model: Mamba):
    """A correctly loaded LM repeats a random token sequence it has already seen.

    Random weights score ~0 here, and any layout or recurrence bug destroys the ability.
    """
    half = 32
    tokens = jax.random.randint(jax.random.key(0), (4, half), 1000, 20000)
    sequence = jnp.concatenate([tokens, tokens], axis=1)

    predictions = jnp.argmax(pretrained_model(sequence), axis=-1)
    accuracy = (predictions[:, half:-1] == sequence[:, half + 1 :]).mean()

    assert accuracy > 0.8
