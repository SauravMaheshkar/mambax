import json
from typing import Optional

import jax
import jax.numpy as jnp
from einops import einsum, repeat
from flax import nnx
from huggingface_hub import hf_hub_download
from jaxtyping import Array, Float, Int
from safetensors import safe_open

from configs.default import Config


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

        self.norm = nnx.RMSNorm(num_features=model_dim, rngs=rngs)

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
                    rngs=rngs,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.norm_f = nnx.RMSNorm(num_features=model_dim, rngs=rngs)

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
        token: Optional[str] = None,
    ) -> "Mamba":
        config_path = hf_hub_download(
            repo_id=repo_id, filename="config.json", repo_type="model", token=token
        )
        with open(config_path, "r") as f:
            config_data = json.load(f)

        args = Config(
            model_dim=config_data["d_model"],
            num_layers=config_data["n_layer"],
            vocab_size=config_data["vocab_size"],
        )

        ckpt_path = hf_hub_download(
            repo_id=repo_id,
            filename="model.safetensors",
            repo_type="model",
            revision="refs/pr/1",
            token=token,
        )

        with safe_open(ckpt_path, framework="flax", device="cpu") as f:
            loaded_params = {}
            for key in f.keys():
                clean_key = key.replace("backbone.", "")
                if clean_key == "embedding.weight":
                    clean_key = "embedding.embedding"
                replacements = [
                    (".conv1d.weight", ".conv1d.kernel"),
                    (".dt_proj.weight", ".dt_proj.kernel"),
                    (".in_proj.weight", ".in_proj.kernel"),
                    (".out_proj.weight", ".out_proj.kernel"),
                    (".x_proj.weight", ".x_proj.kernel"),
                    (".norm.weight", ".norm.scale"),
                    ("norm_f.weight", "norm_f.scale"),
                ]
                for old, new in replacements:
                    if clean_key.endswith(old):
                        clean_key = clean_key.replace(old, new)
                loaded_params[clean_key] = f.get_tensor(key)

        model = cls(
            vocab_size=args.vocab_size,
            model_dim=args.model_dim,
            hidden_dim=args.hidden_dim,
            conv_dim=args.conv_dim,
            dt_rank=args.dt_rank,
            state_dim=args.state_dim,
            num_layers=args.num_layers,
            use_bias=getattr(args, "use_bias", False),
            conv_bias=getattr(args, "conv_bias", True),
            rngs=nnx.Rngs(0),
        )

        # Split and update state
        graph, model_state, _ = nnx.split(model, nnx.Param, ...)
        flat_state = nnx.to_pure_dict(model_state)
        missing = []
        for k in flat_state:
            if k in loaded_params:
                flat_state[k] = loaded_params[k]
            else:
                missing.append(k)
        if missing:
            print(f"Warning: Missing parameters for keys: {missing}")

        model = nnx.merge(graph, model_state)
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

