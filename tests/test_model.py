import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from configs.default import Config
from model import Mamba, MambaBlock


@pytest.fixture(scope="module")
def tiny_model() -> Mamba:
    config = Config(vocab_size=64, model_dim=16, num_layers=2, state_dim=4)
    return Mamba.from_config(config, rngs=nnx.Rngs(0))


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


def test_mamba_from_pretrained():
    model = Mamba.from_pretrained(repo_id="state-spaces/mamba-130m")
    assert model is not None
