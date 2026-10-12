import dataclasses
import json
import math

import jax
import jax.numpy as jnp
from einops import einsum, rearrange, repeat
from flax import nnx
from flax.typing import PathParts
from huggingface_hub import hf_hub_download
from jaxtyping import Array, Float, Int
from safetensors import safe_open

from configs.default import Block, Config


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
        dtype: jnp.dtype = jnp.float32,
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
            dtype=dtype,
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
            dtype=dtype,
            rngs=rngs,
        )

        self.x_proj = nnx.Linear(
            in_features=hidden_dim,
            out_features=dt_rank + state_dim * 2,
            use_bias=False,
            dtype=dtype,
            rngs=rngs,
        )

        # Kept in float32 whatever the compute dtype: Δ sets the decay exp(ΔA), and
        # bf16's 8 mantissa bits would visibly distort long timescales.
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
            dtype=dtype,
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
        delta = nnx.softplus(self.dt_proj(delta.astype(jnp.float32)))

        # The recurrence accumulates over the whole sequence, so it runs in float32.
        f32 = jnp.float32
        y = self.selective_scan(
            x.astype(f32),
            delta,
            A,
            B.astype(f32),
            C.astype(f32),
            self.D[...],
            chunk_size=self.scan_chunk_size,
        )
        return y.astype(x.dtype)

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


def _segment_sums(a: Float[Array, "... chunk"]) -> Float[Array, "... chunk chunk"]:
    """out[..., t, s] = a[s+1] + ... + a[t] for s <= t, and -inf above the diagonal.

    exp() of it is the decay from step s to step t within a chunk. Summing masked
    copies, instead of subtracting cumulative sums, keeps long chunks accurate.
    """
    length = a.shape[-1]
    below = jnp.tril(jnp.ones((length, length), bool), -1)
    sums = jnp.cumsum(jnp.where(below, a[..., :, None], 0.0), axis=-2)
    return jnp.where(jnp.tril(jnp.ones((length, length), bool)), sums, -jnp.inf)


class GatedRMSNorm(nnx.Module):
    """RMSNorm(y * silu(z)) per group of channels, Mamba-2's norm before out_proj."""

    def __init__(self, num_features: int, *, num_groups: int, epsilon: float, dtype):
        self.num_groups = num_groups
        self.epsilon = epsilon
        self.dtype = dtype
        self.scale = nnx.Param(jnp.ones(num_features))

    def __call__(
        self, y: Float[Array, "... hidden"], z: Float[Array, "... hidden"]
    ) -> Float[Array, "... hidden"]:
        f32 = jnp.float32
        gated = y.astype(f32) * nnx.silu(z.astype(f32))
        groups = rearrange(gated, "... (g d) -> ... g d", g=self.num_groups)
        groups = groups * jax.lax.rsqrt(
            jnp.mean(groups**2, axis=-1, keepdims=True) + self.epsilon
        )
        normed = rearrange(groups, "... g d -> ... (g d)") * self.scale[...]
        return normed.astype(self.dtype)


class Mamba2Block(nnx.Module):
    """Mamba-2 block (Dao & Gu, 2024): a scalar decay per head turns the scan into
    matmuls within chunks (SSD), which suits tensor cores and TPU MXUs.

        in_proj -> z | x B C | dt
        x B C -> causal depthwise conv -> silu
        y = SSD(x, dt, A, B, C) + D x       (heads of head_dim channels)
        out_proj(RMSNorm(y * silu(z)))
    """

    def __init__(
        self,
        *,
        model_dim: int,
        hidden_dim: int,
        conv_dim: int,
        state_dim: int,
        head_dim: int,
        num_groups: int,
        use_bias: bool = False,
        conv_bias: bool = True,
        norm_eps: float = 1e-5,
        scan_chunk_size: int = 256,
        dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        if hidden_dim % head_dim:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be a multiple of head_dim ({head_dim})"
            )
        num_heads = hidden_dim // head_dim
        if num_heads % num_groups:
            raise ValueError(
                f"num_heads ({num_heads}) must be a multiple of num_groups ({num_groups})"
            )

        self.model_dim = model_dim
        self.hidden_dim = hidden_dim
        self.conv_dim = conv_dim
        self.state_dim = state_dim
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.scan_chunk_size = scan_chunk_size

        conv_channels = hidden_dim + 2 * num_groups * state_dim
        self.in_proj = nnx.Linear(
            in_features=model_dim,
            out_features=hidden_dim + conv_channels + num_heads,
            use_bias=use_bias,
            dtype=dtype,
            rngs=rngs,
        )
        self.conv1d = nnx.Conv(
            in_features=conv_channels,
            out_features=conv_channels,
            use_bias=conv_bias,
            kernel_size=conv_dim,
            feature_group_count=conv_channels,
            padding=((conv_dim - 1, 0),),
            dtype=dtype,
            rngs=rngs,
        )
        self.dt_bias = nnx.Param(dt_bias_init()(rngs.params(), (num_heads,)))
        # Decay rates A = -exp(A_log) spread uniformly over [1, 16], as in Mamba-2.
        self.A_log = nnx.Param(
            jnp.log(jax.random.uniform(rngs.params(), (num_heads,), minval=1, maxval=16))
        )
        self.D = nnx.Param(jnp.ones(num_heads))
        self.norm = GatedRMSNorm(
            hidden_dim, num_groups=num_groups, epsilon=norm_eps, dtype=dtype
        )
        self.out_proj = nnx.Linear(
            in_features=hidden_dim,
            out_features=model_dim,
            use_bias=use_bias,
            dtype=dtype,
            rngs=rngs,
        )

    @jax.named_scope("mamba2_block")
    def __call__(
        self, u: Float[Array, "batch seq model_dim"]
    ) -> Float[Array, "batch seq model_dim"]:
        conv_channels = self.hidden_dim + 2 * self.num_groups * self.state_dim
        z, xBC, dt = jnp.split(
            self.in_proj(u), [self.hidden_dim, self.hidden_dim + conv_channels], axis=-1
        )
        xBC = nnx.silu(self.conv1d(xBC))
        x, B, C = jnp.split(
            xBC,
            [self.hidden_dim, self.hidden_dim + self.num_groups * self.state_dim],
            axis=-1,
        )

        # Like the Mamba-1 scan, the recurrence and Δ run in float32.
        f32 = jnp.float32
        x = rearrange(x, "b l (h p) -> b l h p", p=self.head_dim).astype(f32)
        B, C = (
            rearrange(t, "b l (g n) -> b l g n", g=self.num_groups).astype(f32)
            for t in (B, C)
        )
        dt = nnx.softplus(dt.astype(f32) + self.dt_bias[...])
        A = -jnp.exp(self.A_log[...])

        y = self.ssd(x, dt, A, B, C, chunk_size=self.scan_chunk_size)
        y = y + x * self.D[...][:, None]
        # The norm takes float32 y and returns the compute dtype for out_proj.
        y = rearrange(y, "b l h p -> b l (h p)")
        return self.out_proj(self.norm(y, z))

    @staticmethod
    def ssd(
        x: Float[Array, "batch seq heads head_dim"],
        dt: Float[Array, "batch seq heads"],
        A: Float[Array, " heads"],
        B: Float[Array, "batch seq groups state_dim"],
        C: Float[Array, "batch seq groups state_dim"],
        chunk_size: int,
    ) -> Float[Array, "batch seq heads head_dim"]:
        """The state space duality (SSD) form of the per-head scalar-decay SSM.

        h_t = exp(Δ_t A) h_{t-1} + Δ_t x_t B_t^T     (h: head_dim x state_dim)
        y_t = h_t C_t

        Within a chunk the recurrence unrolls into masked attention-like matmuls,
        y = (C B^T ∘ decay) (Δ x), quadratic in the chunk length. Chunks then hand
        their final state to the next, as in the Mamba-1 chunked scan, and are
        recomputed in the backward pass so memory stays ~one chunk:

                 chunk 0             chunk 1
              [matmuls in c]      [matmuls in c]
            h=0 ------+-----> h_c -----+-----> h_2c ...
        """
        batch, seq_len, heads, _ = x.shape
        groups = B.shape[2]
        # Head i reads the B and C of group i // (heads / groups).
        B, C = (jnp.repeat(t, heads // groups, axis=2) for t in (B, C))
        # Padded steps have Δ = 0: no decay and no input, so they change nothing
        # and their outputs are sliced off.
        pad = -seq_len % chunk_size
        chunks = tuple(
            rearrange(
                jnp.pad(t, ((0, 0), (0, pad)) + ((0, 0),) * (t.ndim - 2)),
                "b (c l) ... -> c b l ...",
                l=chunk_size,
            )
            for t in (x, dt, B, C)
        )

        @jax.checkpoint
        def ssd_chunk(h, chunk):
            x_c, dt_c, B_c, C_c = chunk
            log_decay = rearrange(dt_c * A, "b l h -> b h l")
            cumulative = jnp.cumsum(log_decay, axis=-1)
            x_dt = x_c * dt_c[..., None]

            # Within the chunk: y_t = sum_{s <= t} C_t.B_s decay(s -> t) Δ_s x_s.
            decay = jnp.exp(_segment_sums(log_decay))
            y = einsum(
                C_c, B_c, decay, x_dt, "b t h n, b s h n, b h t s, b s h p -> b t h p"
            )
            # From the carried state: y_t += decay(start -> t) h C_t.
            y = y + einsum(
                C_c, h, jnp.exp(cumulative), "b t h n, b h p n, b h t -> b t h p"
            )
            # Carry: decay the state through the chunk and add what it absorbed.
            to_end = jnp.exp(cumulative[..., -1:] - cumulative)
            h = h * jnp.exp(cumulative[..., -1])[..., None, None] + einsum(
                B_c, to_end, x_dt, "b s h n, b h s, b s h p -> b h p n"
            )
            return h, y

        h0 = jnp.zeros((batch, heads, x.shape[-1], B.shape[-1]), x.dtype)
        _, y = jax.lax.scan(ssd_chunk, h0, chunks)
        return rearrange(y, "c b l h p -> b (c l) h p")[:, :seq_len]


class ResidualBlock(nnx.Module):
    """Pre-norm residual around a mixer (a Mamba-1 or Mamba-2 block)."""

    def __init__(
        self,
        mixer: "MambaBlock | Mamba2Block",
        *,
        model_dim: int,
        norm_eps: float = 1e-5,
        dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        self.mixer = mixer
        self.norm = nnx.RMSNorm(
            num_features=model_dim, epsilon=norm_eps, dtype=dtype, rngs=rngs
        )

    def __call__(
        self, x: Float[Array, "batch seq model_dim"]
    ) -> Float[Array, "batch seq model_dim"]:
        # x is the float32 residual stream, so the sum promotes back to float32.
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
        block: Block = Block.MAMBA1,
        head_dim: int = 64,
        num_groups: int = 1,
        use_bias: bool = False,
        conv_bias: bool = True,
        norm_eps: float = 1e-5,
        scan_chunk_size: int = 32,
        dtype: jnp.dtype = jnp.float32,
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
        self.block = block
        self.use_bias = use_bias
        self.conv_bias = conv_bias

        self.embedding = nnx.Embed(
            num_embeddings=vocab_size,
            features=model_dim,
            dtype=dtype,
            rngs=rngs,
        )

        def mixer() -> MambaBlock | Mamba2Block:
            match block:
                case Block.MAMBA1:
                    return MambaBlock(
                        model_dim=model_dim,
                        hidden_dim=hidden_dim,
                        conv_dim=conv_dim,
                        dt_rank=dt_rank,
                        state_dim=state_dim,
                        use_bias=use_bias,
                        conv_bias=conv_bias,
                        scan_chunk_size=scan_chunk_size,
                        dtype=dtype,
                        rngs=rngs,
                    )
                case Block.MAMBA2:
                    return Mamba2Block(
                        model_dim=model_dim,
                        hidden_dim=hidden_dim,
                        conv_dim=conv_dim,
                        state_dim=state_dim,
                        head_dim=head_dim,
                        num_groups=num_groups,
                        use_bias=use_bias,
                        conv_bias=conv_bias,
                        norm_eps=norm_eps,
                        scan_chunk_size=scan_chunk_size,
                        dtype=dtype,
                        rngs=rngs,
                    )

        self.layers = nnx.List(
            [
                ResidualBlock(
                    mixer(),
                    model_dim=model_dim,
                    norm_eps=norm_eps,
                    dtype=dtype,
                    rngs=rngs,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.norm_f = nnx.RMSNorm(
            num_features=model_dim, epsilon=norm_eps, dtype=dtype, rngs=rngs
        )

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
            block=config.block,
            head_dim=config.head_dim,
            num_groups=config.num_groups,
            use_bias=config.use_bias,
            conv_bias=config.conv_bias,
            norm_eps=config.norm_eps,
            scan_chunk_size=config.scan_chunk_size,
            dtype=jnp.dtype(config.compute_dtype),
            rngs=rngs,
        )

    def __call__(
        self, x: Int[Array, "batch seq"]
    ) -> Float[Array, "batch seq vocab_size"]:
        # Params are float32 and matmuls run in the compute dtype, while the
        # residual stream, the scan and the logits stay float32 (mixed precision,
        # like Mamba's residual_in_fp32).
        x = self.embedding(x).astype(jnp.float32)
        for layer in self.layers:
            x = layer(x)

        # Output projection is tied to the embedding weights.
        return self.embedding.attend(self.norm_f(x)).astype(jnp.float32)

    @property
    def num_params(self) -> int:
        return sum(p.size for p in jax.tree.leaves(nnx.state(self, nnx.Param)))

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

        config = _config_from_hf(hf_config)

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


def _config_from_hf(hf_config: dict) -> Config:
    """Our config for a `transformers` Mamba or Mamba2 config.json."""
    shared = Config(
        vocab_size=hf_config["vocab_size"],
        model_dim=hf_config["hidden_size"],
        num_layers=hf_config["num_hidden_layers"],
        state_dim=hf_config["state_size"],
        expand=hf_config["expand"],
        conv_dim=hf_config["conv_kernel"],
        use_bias=hf_config["use_bias"],
        conv_bias=hf_config["use_conv_bias"],
        norm_eps=hf_config["layer_norm_epsilon"],
    )
    match hf_config["model_type"]:
        case "mamba":
            return dataclasses.replace(shared, dt_rank=hf_config["time_step_rank"])
        case "mamba2":
            # We don't clamp Δ, so refuse checkpoints trained with a clamp.
            if hf_config.get("time_step_limit", [0.0, math.inf]) != [0.0, math.inf]:
                raise NotImplementedError(
                    f"time_step_limit {hf_config['time_step_limit']} is not supported"
                )
            config = dataclasses.replace(
                shared,
                block=Block.MAMBA2,
                head_dim=hf_config["head_dim"],
                num_groups=hf_config["n_groups"],
                scan_chunk_size=hf_config["chunk_size"],
            )
            if config.hidden_dim // config.head_dim != hf_config["num_heads"]:
                raise ValueError(
                    f"num_heads {hf_config['num_heads']} != "
                    f"hidden_dim {config.hidden_dim} / head_dim {config.head_dim}"
                )
            return config
        case model_type:
            raise ValueError(f"unsupported model_type {model_type!r}")


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
