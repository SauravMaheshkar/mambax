import json
from typing import Optional

import jax
import jax.numpy as jnp
import jaxtyping
from einops import einsum, repeat
from flax import nnx
from huggingface_hub import hf_hub_download
from safetensors import safe_open

from configs.default import Config


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

        self.conv1d = nnx.Conv(
            in_features=hidden_dim,
            out_features=hidden_dim,
            use_bias=conv_bias,
            kernel_size=conv_dim,
            # feature_group_count=hidden_dim,
            padding=conv_dim - 1,
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
            jax.numpy.arange(1, state_dim + 1).astype(jax.numpy.float32),
            "n -> d n",
            d=hidden_dim,
        )
        self.A_log = nnx.Param(jax.lax.log(A))
        self.D = nnx.Param(jax.numpy.ones(hidden_dim))

        self.out_proj = nnx.Linear(
            in_features=hidden_dim,
            out_features=model_dim,
            use_bias=use_bias,
            rngs=rngs,
        )

    @jax.named_scope("mamba_block")
    def __call__(self, x: jaxtyping.Array) -> jaxtyping.Array:
        (B, L, D) = x.shape

        x_and_res = self.in_proj(x)
        x, res = jax.numpy.split(x_and_res, [self.hidden_dim], axis=-1)

        x = self.conv1d(x)[:, :L, :]
        x = nnx.silu(x)

        y = self.ssm(x)
        y = y * nnx.silu(res)
        return self.out_proj(y)

    def ssm(self, x: jaxtyping.Array) -> jaxtyping.Array:
        (_, N) = self.A_log.shape

        A = -jax.numpy.exp(self.A_log.value)
        D = self.D.value

        x_dbl = self.x_proj(x)
        assert x_dbl.shape[-1] == self.dt_rank + 2 * self.state_dim

        (delta, B, C) = jax.numpy.split(
            x_dbl, [self.dt_rank, self.dt_rank + self.state_dim], axis=-1
        )
        delta = nnx.softplus(self.dt_proj(delta))

        y = self.selective_scan(x, delta, A, B, C, D)
        return y

    def selective_scan(
        self,
        x: jaxtyping.Array,
        delta: jaxtyping.Array,
        A: jaxtyping.Array,
        B: jaxtyping.Array,
        C: jaxtyping.Array,
        D: jaxtyping.Array,
    ) -> jaxtyping.Array:
        (b, l, d_in) = x.shape  # noqa: E741
        n = A.shape[1]

        deltaA = jax.numpy.exp(einsum(delta, A, "b l d_in, d_in n -> b l d_in n"))
        deltaB_u = einsum(delta, B, x, "b l d_in, b l n, b l d_in -> b l d_in n")

        x = jax.numpy.zeros((b, d_in, n))
        ys = []
        for i in range(l):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = einsum(x, C[:, i, :], "b d_in n, b n -> b d_in")
            ys.append(y)

        y = jax.numpy.stack(ys, axis=1)
        residual = einsum(x, D, "b d n, d -> b d")  # shape: (b, d)
        residual = jax.numpy.repeat(residual[:, None, :], l, axis=1)  # (b, l, d)
        y = y + residual

        return y


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

    def __call__(self, x: jaxtyping.Array) -> jaxtyping.Array:
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

        self.lm_head = nnx.Linear(
            in_features=model_dim,
            out_features=vocab_size,
            use_bias=False,
            rngs=rngs,
        )

    def __call__(self, x: jaxtyping.Array) -> jaxtyping.Array:
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)

        x = self.norm_f(x)
        # tie output projection to embedding weights.
        return x @ jnp.transpose(self.embedding.embedding, (1, 0))

    @property
    def num_params(self) -> int:
        return sum(p.size for p in jax.tree.leaves(self.state))

    @property
    def state(self) -> nnx.State:
        """Splits state from the graph and returns it"""
        return nnx.split(self, nnx.Param, ...)[1]

    @property
    def state_dict(self) -> dict[str, jaxtyping.Array]:
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
