import json

import jax
import jax.numpy as jnp
from einops import einsum, rearrange, repeat
from flax import nnx
from flax.typing import PathParts
from huggingface_hub import hf_hub_download
from jaxtyping import Array, Float, Int
from safetensors import safe_open

from configs.default import Config


def dt_kernel_init(dt_rank: int) -> nnx.Initializer:
    """Uniform in [-dt_rank^-0.5, dt_rank^-0.5], so the projection keeps unit variance."""
    bound = dt_rank**-0.5

    def init(key: Array, shape: tuple[int, ...], dtype=jnp.float32) -> Array:
        return jax.random.uniform(key, shape, dtype, minval=-bound, maxval=bound)

    return init


def dt_bias_init(
    dt_min: float = 1e-3, dt_max: float = 1e-1, dt_floor: float = 1e-4
) -> nnx.Initializer:
    """Bias such that softplus(bias) = Δ is log-uniform in [dt_min, dt_max].

    Δ sets each channel's timescale: small Δ remembers long context, large Δ
    tracks the latest input. The default init (bias = 0) pins every channel at
    Δ = softplus(0) ≈ 0.69, so all of them start out forgetting fast.
    """

    def init(key: Array, shape: tuple[int, ...], dtype=jnp.float32) -> Array:
        log_dt = jax.random.uniform(
            key, shape, dtype, minval=jnp.log(dt_min), maxval=jnp.log(dt_max)
        )
        dt = jnp.maximum(jnp.exp(log_dt), dt_floor)
        # Inverse of softplus(x) = log(1 + e^x).
        return dt + jnp.log(-jnp.expm1(-dt))

    return init


def _scan_combine(
    left: tuple[Array, Array], right: tuple[Array, Array]
) -> tuple[Array, Array]:
    """Composes two steps of the linear recurrence h_t = a_t * h_{t-1} + b_t.

    Applying `left` then `right` is the same as one step with
    a = a_l * a_r and b = a_r * b_l + b_r, which is associative, so
    `jax.lax.associative_scan` can evaluate the whole sequence in O(log L) depth.
    """
    a_left, b_left = left
    a_right, b_right = right
    return a_left * a_right, a_right * b_left + b_right


class MambaBlock(nnx.Module):
    """Mamba block."""

    def __init__(
        self,
        *,
        model_dim: int,
        hidden_dim: int,
        conv_dim: int,
        dt_rank: int,
        state_dim: int,
        use_bias: bool = False,
        conv_bias: bool = True,
        scan_chunk_size: int = 32,
        rngs: nnx.Rngs,
    ):
        super().__init__()

        self.model_dim = model_dim
        self.hidden_dim = hidden_dim
        self.conv_dim = conv_dim
        self.dt_rank = dt_rank
        self.state_dim = state_dim
        self.use_bias = use_bias
        self.conv_bias = conv_bias
        self.scan_chunk_size = scan_chunk_size

        self.in_proj = nnx.Linear(
            in_features=model_dim,
            out_features=hidden_dim * 2,
            use_bias=use_bias,
            rngs=rngs,
        )

        # Depthwise and causal: each channel only sees its own past conv_dim - 1 steps.
        self.conv1d = nnx.Conv(
            in_features=hidden_dim,
            out_features=hidden_dim,
            use_bias=conv_bias,
            kernel_size=conv_dim,
            feature_group_count=hidden_dim,
            padding=((conv_dim - 1, 0),),
            rngs=rngs,
        )

        self.x_proj = nnx.Linear(
            in_features=hidden_dim,
            out_features=dt_rank + state_dim * 2,
            use_bias=False,
            rngs=rngs,
        )

        self.dt_proj = nnx.Linear(
            in_features=dt_rank,
            out_features=hidden_dim,
            use_bias=True,
            kernel_init=dt_kernel_init(dt_rank),
            bias_init=dt_bias_init(),
            rngs=rngs,
        )

        A = repeat(
            jnp.arange(1, state_dim + 1, dtype=jnp.float32),
            "n -> d n",
            d=hidden_dim,
        )
        self.A_log = nnx.Param(jnp.log(A))
        self.D = nnx.Param(jnp.ones(hidden_dim))

        self.out_proj = nnx.Linear(
            in_features=hidden_dim,
            out_features=model_dim,
            use_bias=use_bias,
            rngs=rngs,
        )

    @jax.named_scope("mamba_block")
    def __call__(
        self, x: Float[Array, "batch seq model_dim"]
    ) -> Float[Array, "batch seq model_dim"]:
        x, res = jnp.split(self.in_proj(x), [self.hidden_dim], axis=-1)

        x = nnx.silu(self.conv1d(x))

        y = self.ssm(x)
        y = y * nnx.silu(res)
        return self.out_proj(y)

    def ssm(
        self, x: Float[Array, "batch seq hidden_dim"]
    ) -> Float[Array, "batch seq hidden_dim"]:
        A = -jnp.exp(self.A_log[...])

        (delta, B, C) = jnp.split(
            self.x_proj(x), [self.dt_rank, self.dt_rank + self.state_dim], axis=-1
        )
        delta = nnx.softplus(self.dt_proj(delta))

        return self.selective_scan(
            x, delta, A, B, C, self.D[...], chunk_size=self.scan_chunk_size
        )

    @staticmethod
    def selective_scan(
        u: Float[Array, "batch seq hidden_dim"],
        delta: Float[Array, "batch seq hidden_dim"],
        A: Float[Array, "hidden_dim state_dim"],
        B: Float[Array, "batch seq state_dim"],
        C: Float[Array, "batch seq state_dim"],
        D: Float[Array, " hidden_dim"],
        chunk_size: int,
    ) -> Float[Array, "batch seq hidden_dim"]:
        """Runs the discretized SSM over the sequence.

        h_t = exp(Δ_t A) h_{t-1} + Δ_t B_t u_t
        y_t = C_t h_t + D u_t

        A whole-sequence scan keeps several (batch, seq, hidden, state) tensors
        alive for the backward pass, per layer. Instead the sequence is split into
        chunks: each chunk runs as an associative scan seeded with the incoming
        state, only that state crosses chunk boundaries, and chunks are recomputed
        in the backward pass. Memory is ~(batch, hidden, state) * (seq / chunk_size)
        saved states plus one chunk's worth of temporaries:

                 chunk 0           chunk 1           chunk 2
              [t_0 .. t_c-1]    [t_c .. t_2c-1]   [t_2c .. ]
                    | scan            | scan            | scan
            h=0 ----+------> h_c -----+------> h_2c ----+------> ...
                             (saved)           (saved)
        """
        batch, seq_len, _ = u.shape
        # Padded steps come last and their outputs are sliced off, so they can't
        # leak into real positions.
        pad = ((0, 0), (0, -seq_len % chunk_size), (0, 0))
        chunks = tuple(
            rearrange(jnp.pad(t, pad), "b (c l) x -> c b l x", l=chunk_size)
            for t in (u, delta, B, C)
        )

        @jax.checkpoint
        def scan_chunk(h, chunk):
            u_c, delta_c, B_c, C_c = chunk
            deltaA = jnp.exp(einsum(delta_c, A, "b l d, d n -> b l d n"))
            deltaB_u = einsum(delta_c, B_c, u_c, "b l d, b l n, b l d -> b l d n")
            # decay_t = prod of deltaA up to t, h_local_t = state if h started at 0.
            decay, h_local = jax.lax.associative_scan(
                _scan_combine, (deltaA, deltaB_u), axis=1
            )
            h_chunk = decay * h[:, None] + h_local
            return h_chunk[:, -1], einsum(h_chunk, C_c, "b l d n, b l n -> b l d")

        h0 = jnp.zeros((batch, *A.shape), u.dtype)
        _, y = jax.lax.scan(scan_chunk, h0, chunks)
        y = rearrange(y, "c b l d -> b (c l) d")[:, :seq_len]
        return y + u * D


class ResidualBlock(nnx.Module):
    def __init__(
        self,
        *,
        model_dim: int,
        hidden_dim: int,
        conv_dim: int,
        dt_rank: int,
        state_dim: int,
        use_bias: bool = False,
        conv_bias: bool = True,
        norm_eps: float = 1e-5,
        scan_chunk_size: int = 32,
        rngs: nnx.Rngs,
    ):
        super().__init__()

        self.model_dim = model_dim
        self.hidden_dim = hidden_dim
        self.conv_dim = conv_dim
        self.dt_rank = dt_rank
        self.state_dim = state_dim
        self.use_bias = use_bias
        self.conv_bias = conv_bias

        self.mixer = MambaBlock(
            model_dim=model_dim,
            hidden_dim=hidden_dim,
            conv_dim=conv_dim,
            dt_rank=dt_rank,
            state_dim=state_dim,
            use_bias=use_bias,
            conv_bias=conv_bias,
            scan_chunk_size=scan_chunk_size,
            rngs=rngs,
        )

        self.norm = nnx.RMSNorm(num_features=model_dim, epsilon=norm_eps, rngs=rngs)

    def __call__(
        self, x: Float[Array, "batch seq model_dim"]
    ) -> Float[Array, "batch seq model_dim"]:
        return self.mixer(self.norm(x)) + x


class Mamba(nnx.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        model_dim: int,
        hidden_dim: int,
        conv_dim: int,
        dt_rank: int,
        state_dim: int,
        num_layers: int,
        use_bias: bool = False,
        conv_bias: bool = True,
        norm_eps: float = 1e-5,
        scan_chunk_size: int = 32,
        rngs: nnx.Rngs,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.model_dim = model_dim
        self.hidden_dim = hidden_dim
        self.conv_dim = conv_dim
        self.dt_rank = dt_rank
        self.state_dim = state_dim
        self.num_layers = num_layers
        self.use_bias = use_bias
        self.conv_bias = conv_bias

        self.embedding = nnx.Embed(
            num_embeddings=vocab_size,
            features=model_dim,
            rngs=rngs,
        )

        self.layers = nnx.List(
            [
                ResidualBlock(
                    model_dim=model_dim,
                    hidden_dim=hidden_dim,
                    conv_dim=conv_dim,
                    dt_rank=dt_rank,
                    state_dim=state_dim,
                    use_bias=use_bias,
                    conv_bias=conv_bias,
                    norm_eps=norm_eps,
                    scan_chunk_size=scan_chunk_size,
                    rngs=rngs,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.norm_f = nnx.RMSNorm(num_features=model_dim, epsilon=norm_eps, rngs=rngs)

    @classmethod
    def from_config(cls, config: Config, *, rngs: nnx.Rngs) -> "Mamba":
        return cls(
            vocab_size=config.padded_vocab_size,
            model_dim=config.model_dim,
            hidden_dim=config.hidden_dim,
            conv_dim=config.conv_dim,
            dt_rank=config.resolved_dt_rank,
            state_dim=config.state_dim,
            num_layers=config.num_layers,
            use_bias=config.use_bias,
            conv_bias=config.conv_bias,
            norm_eps=config.norm_eps,
            scan_chunk_size=config.scan_chunk_size,
            rngs=rngs,
        )

    def __call__(
        self, x: Int[Array, "batch seq"]
    ) -> Float[Array, "batch seq vocab_size"]:
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)

        # Output projection is tied to the embedding weights.
        return self.embedding.attend(self.norm_f(x))

    @property
    def num_params(self) -> int:
        return sum(p.size for p in jax.tree.leaves(self.state))

    @property
    def state(self) -> nnx.State:
        """Splits state from the graph and returns it"""
        return nnx.split(self, nnx.Param, ...)[1]

    @property
    def state_dict(self) -> dict[str, Array]:
        """Splits state from the graph and returns it as a dictionary.

        It can be used for serialization with orbax."""
        state = self.state
        pure_dict_state = nnx.to_pure_dict(state)
        return pure_dict_state

    def save(self, path: str, **kwargs) -> None:
        """Saves the model state to a directory.

        Args:
            path: The directory path to save the model state to.
        """
        import orbax.checkpoint as ocp

        state = nnx.state(self)
        checkpointer = ocp.PyTreeCheckpointer()
        checkpointer.save(f"{path}/mamba", state, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str,
        revision: str | None = None,
        token: str | None = None,
    ) -> "Mamba":
        """Loads a `transformers`-format checkpoint, e.g. `state-spaces/mamba-130m-hf`.

        Loading is strict: any missing, unexpected or mis-shaped parameter raises.
        """

        def download(filename: str) -> str:
            return hf_hub_download(
                repo_id=repo_id, filename=filename, revision=revision, token=token
            )

        with open(download("config.json")) as f:
            hf_config = json.load(f)

        config = Config(
            vocab_size=hf_config["vocab_size"],
            model_dim=hf_config["hidden_size"],
            num_layers=hf_config["num_hidden_layers"],
            state_dim=hf_config["state_size"],
            expand=hf_config["expand"],
            conv_dim=hf_config["conv_kernel"],
            dt_rank=hf_config["time_step_rank"],
            use_bias=hf_config["use_bias"],
            conv_bias=hf_config["use_conv_bias"],
            norm_eps=hf_config["layer_norm_epsilon"],
        )

        # Every weight gets overwritten, so skip the random init.
        model = nnx.eval_shape(lambda: cls.from_config(config, rngs=nnx.Rngs(0)))
        expected = {
            path: param.shape for path, param in nnx.state(model, nnx.Param).flat_state()
        }

        with safe_open(download("model.safetensors"), framework="flax") as f:
            loaded = dict(_hf_to_nnx(key, f.get_tensor(key)) for key in f.keys())

        missing = sorted(map(str, expected.keys() - loaded.keys()))
        unexpected = sorted(map(str, loaded.keys() - expected.keys()))
        mismatched = sorted(
            f"{path}: expected {expected[path]}, got {loaded[path].shape}"
            for path in expected.keys() & loaded.keys()
            if expected[path] != loaded[path].shape
        )
        if missing or unexpected or mismatched:
            raise ValueError(
                f"Checkpoint {repo_id} does not match the model.\n"
                f"Missing: {missing}\nUnexpected: {unexpected}\nMismatched: {mismatched}"
            )

        nnx.update(model, nnx.State.from_flat_path(loaded))
        return model

    def load(self, path: str) -> "Mamba":
        """Loads the model state from a directory.

        Args:
            path: The directory path to load the model state from.
        """
        import orbax.checkpoint as ocp

        checkpointer = ocp.PyTreeCheckpointer()
        state = checkpointer.restore(f"{path}/mamba", item=nnx.state(self))
        nnx.update(self, state)
        return self


def _hf_to_nnx(key: str, tensor: Array) -> tuple[PathParts, Array]:
    """Maps a `transformers` Mamba tensor to its NNX parameter path and layout.

    torch Linear stores (out, in) and depthwise Conv1d stores (out, 1, kernel),
    while NNX expects (in, out) and (kernel, 1, out) respectively.
    """
    *module, leaf = key.removeprefix("backbone.").split(".")
    path = tuple(int(part) if part.isdigit() else part for part in module)

    if path == ("embeddings",):
        return ("embedding", "embedding"), tensor
    if leaf != "weight":
        return (*path, leaf), tensor
    if path[-1] in ("norm", "norm_f"):
        return (*path, "scale"), tensor
    if path[-1] == "conv1d":
        return (*path, "kernel"), tensor.transpose(2, 1, 0)
    return (*path, "kernel"), tensor.T
