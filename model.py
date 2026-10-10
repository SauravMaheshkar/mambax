import json

import jax
import jax.numpy as jnp
from einops import einsum, repeat
from flax import nnx
from huggingface_hub import hf_hub_download
from jaxtyping import Array, Float, Int
from safetensors import safe_open

from configs.default import Config


ParamPath = tuple[str | int, ...]


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

        return self.selective_scan(x, delta, A, B, C, self.D[...])

    @staticmethod
    def selective_scan(
        u: Float[Array, "batch seq hidden_dim"],
        delta: Float[Array, "batch seq hidden_dim"],
        A: Float[Array, "hidden_dim state_dim"],
        B: Float[Array, "batch seq state_dim"],
        C: Float[Array, "batch seq state_dim"],
        D: Float[Array, " hidden_dim"],
    ) -> Float[Array, "batch seq hidden_dim"]:
        """Runs the discretized SSM over the sequence.

        h_t = exp(Δ_t A) h_{t-1} + Δ_t B_t u_t
        y_t = C_t h_t + D u_t
        """
        deltaA = jnp.exp(einsum(delta, A, "b l d, d n -> b l d n"))
        deltaB_u = einsum(delta, B, u, "b l d, b l n, b l d -> b l d n")

        _, h = jax.lax.associative_scan(_scan_combine, (deltaA, deltaB_u), axis=1)

        y = einsum(h, C, "b l d n, b l n -> b l d")
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
                    rngs=rngs,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.norm_f = nnx.RMSNorm(num_features=model_dim, epsilon=norm_eps, rngs=rngs)

    @classmethod
    def from_config(cls, config: Config, *, rngs: nnx.Rngs) -> "Mamba":
        return cls(
            vocab_size=config.vocab_size,
            model_dim=config.model_dim,
            hidden_dim=config.hidden_dim,
            conv_dim=config.conv_dim,
            dt_rank=config.dt_rank,
            state_dim=config.state_dim,
            num_layers=config.num_layers,
            use_bias=config.use_bias,
            conv_bias=config.conv_bias,
            norm_eps=config.norm_eps,
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


def _hf_to_nnx(key: str, tensor: Array) -> tuple[ParamPath, Array]:
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
