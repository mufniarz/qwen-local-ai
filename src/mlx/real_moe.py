"""
Real MoE — Gating Network and Expert Computation (Qwen3-MoE Formula)

The simulation (``src.moe``) picks experts with a SHA-256 hash of the
embedding. This module implements the **actual** Mixture-of-Experts
math, exactly as in the Qwen3 MoE family (real Qwen3-30B-A3B /
Qwen3-235B-A22B / Qwen3-Next-80B-A3B checkpoints, and mlx-lm's own
``Qwen3MoeSparseMoeBlock``):

    logits  = x @ W_gateᵀ          # (n, num_experts)
    probs   = softmax(logits)      # over all experts
    top-k   = select k experts     # Qwen3: k = num_experts_per_tok
    weights = topk_probs / Σ       # renormalize to sum to 1
    out     = Σ_e w_e · Expert_e(x)   # SwiGLU experts

Experts are SwiGLU MLPs, exactly as Qwen3 uses them:

    E(x) = Down( SiLU(Gate(x)) ⊙ Up(x) )

The module is engine-agnostic: routing works on NumPy arrays *and* on
MLX tensors (dispatched per input), so the same code path runs in unit
tests without MLX and against the real model on your M1 Max.

Expert backends (pick whatever the loaded model exposes):
  - a list of per-expert MLP objects with ``forward(x)`` — the classic
    unfused layout (and the default for tests via ``random_moe``);
  - a fused ``expert_fn(x, indices) -> (n, k, dim)`` — the layout
    mlx-lm 0.31+ uses (``SwitchGLU`` with all experts in one tensor).

``Qwen3MoE.from_mlx_model(model)`` extracts the **real** gate weights
(dequantized through the model's own matmul — works for every quantized
layout) and the real expert storage from a loaded model, so the routing
here is the model's actual router, not a simulation.
"""

from __future__ import annotations

from typing import Optional, Sequence


# ----------------------------------------------------------------------
# Tiny engine-dispatch layer (NumPy | MLX)
# ----------------------------------------------------------------------


def _is_mlx(a) -> bool:
    return type(a).__module__.startswith("mlx")


def _exp(a):
    if _is_mlx(a):
        import mlx.core as mx

        return mx.exp(a)
    import numpy as np

    return np.exp(a)


def _softmax(a, axis: int = -1):
    m = a.max(axis=axis, keepdims=True)
    e = _exp(a - m)
    return e / e.sum(axis=axis, keepdims=True)


def _silu(a):
    return a * (1.0 / (1.0 + _exp(-a)))


def _topk(a, k: int, axis: int = -1):
    """Top-k values + indices along ``axis`` (sorted descending)."""
    if _is_mlx(a):
        import mlx.core as mx

        values, indices = mx.topk(a, k, axis=axis)
        return values, indices

    import numpy as np

    if axis == -1:
        axis = a.ndim - 1
    # k-th largest via argpartition, then sort each row by value desc
    part = np.argpartition(-a, k - 1, axis=axis)
    idx = np.take(part, range(k), axis=axis)
    vals = np.take_along_axis(a, idx, axis=axis)
    order = np.argsort(-vals, axis=axis, kind="stable")
    idx = np.take_along_axis(idx, order, axis=axis)
    vals = np.take_along_axis(vals, order, axis=axis)
    return vals, idx


# ----------------------------------------------------------------------
# Expert (SwiGLU MLP) — the Qwen3 expert shape
# ----------------------------------------------------------------------


class ExpertMLP:
    """
    One MoE expert: SwiGLU MLP, ``Down(SiLU(Gate(x)) ⊙ Up(x))``.

    Args:
        gate_proj: (hidden, dim)
        up_proj:   (hidden, dim)
        down_proj: (dim, hidden)
    All three accept NumPy or MLX arrays; ``forward`` follows the
    engine of the input.
    """

    def __init__(self, gate_proj, up_proj, down_proj) -> None:
        self.gate_proj = gate_proj
        self.up_proj = up_proj
        self.down_proj = down_proj
        self.fire_count = 0

    def forward(self, x):
        # Weights follow the nn.Linear convention (out, in), so the
        # matmuls are against the transpose — exactly Qwen3's SwiGLU.
        h = _silu(x @ self.gate_proj.T) * (x @ self.up_proj.T)
        out = h @ self.down_proj.T
        self.fire_count += 1
        return out

    __call__ = forward

    @property
    def dim(self) -> int:
        return self.gate_proj.shape[1]

    @property
    def hidden_dim(self) -> int:
        return self.gate_proj.shape[0]

    def weights(self):
        """(gate_proj, up_proj, down_proj) — for copying/inspection."""
        return (self.gate_proj, self.up_proj, self.down_proj)

    @classmethod
    def random(
        cls,
        dim: int,
        hidden: int,
        seed: int = 0,
        backend: str = "numpy",
    ) -> "ExpertMLP":
        """A small random expert for tests (real weights, real math)."""
        if backend == "mlx":
            import mlx.core as mx

            mx.random.seed(seed)
            scale = (1.0 / dim) ** 0.5
            g = mx.random.normal((hidden, dim)) * scale
            u = mx.random.normal((hidden, dim)) * scale
            d = mx.random.normal((dim, hidden)) * scale
            return cls(g, u, d)
        import numpy as np

        rng = np.random.default_rng(seed)
        scale = (1.0 / dim) ** 0.5
        return cls(
            rng.normal(0, scale, (hidden, dim)).astype("float32"),
            rng.normal(0, scale, (hidden, dim)).astype("float32"),
            rng.normal(0, scale, (dim, hidden)).astype("float32"),
        )


# ----------------------------------------------------------------------
# The MoE block
# ----------------------------------------------------------------------


class Qwen3MoE:
    """
    Qwen3-style Mixture of Experts block.

    Args:
        gate_weight: (num_experts, dim) router matrix.
        experts:     sequence of expert MLPs with ``forward(x)``
                     (unfused layout), **or** None when using expert_fn.
        expert_fn:   fused expert callable ``(x, indices) -> (n, k, dim)``
                     — e.g. the model's own ``SwitchGLU``.
        top_k:       experts fired per token (Qwen3: 8).
    """

    def __init__(
        self,
        gate_weight,
        experts: Optional[Sequence] = None,
        expert_fn=None,
        top_k: Optional[int] = None,
    ) -> None:
        self.gate_weight = gate_weight
        self.experts = list(experts) if experts is not None else []
        self.expert_fn = expert_fn
        self.num_experts = (
            len(self.experts) if self.experts else int(gate_weight.shape[0])
        )
        self.top_k = int(top_k or 8)
        if self.top_k > self.num_experts:
            raise ValueError(
                f"top_k={self.top_k} exceeds num_experts={self.num_experts}"
            )
        self._fire_counts = [0] * self.num_experts
        self._tokens_routed = 0
        self._stacked = None  # lazily built stacked expert weights

    # ------------------------------------------------------------------
    # Routing — the learned gate, not a hash
    # ------------------------------------------------------------------

    def route(self, x):
        """
        Router: learned linear gate → softmax → top-k → renormalize.

        Args:
            x: (n_tokens, dim)

        Returns:
            (indices, weights): both (n_tokens, top_k).
            weights rows sum to 1 (Qwen3 norm_topk_prob).
        """
        logits = x @ self.gate_weight.T
        probs = _softmax(logits, axis=-1)
        vals, idx = _topk(probs, self.top_k, axis=-1)
        weights = vals / vals.sum(axis=-1, keepdims=True)
        return idx, weights

    def forward(self, x):
        """
        Full MoE forward: route, then the weighted sum of expert
        outputs.

        - Fused layout: the model's own SwitchGLU does the expert
          compute (``expert_fn(x, indices) -> (n, k, dim)``).
        - Unfused layout: a batched gather dispatch — each (token, slot)
          pair pulls its expert's weights and runs the SwiGLU matmul.
          Only uses ops supported by both NumPy and MLX (flatten,
          repeat, take, batched matmul) — no boolean/fancy indexing.
        """
        idx, weights = self.route(x)
        n = x.shape[0]

        if self.expert_fn is not None:
            y = self.expert_fn(x, _to_index_array(idx))
            out = (y * weights[..., None]).sum(axis=-2)
        else:
            out = self._forward_gather(x, idx, weights)

        if not _is_mlx(x):
            self._update_fire_counts(idx)
        self._tokens_routed += n
        return out

    __call__ = forward

    def _build_stacked(self) -> None:
        """
        Stack per-expert weights transposed into matmul-ready orientation:
            gT: (E, dim, hidden)   for x @ gT
            uT: (E, dim, hidden)   for x @ uT
            dT: (E, hidden, dim)   for h @ dT
        """
        if _is_mlx(self.experts[0].gate_proj):
            import mlx.core as mx

            stack = mx.stack
        else:
            import numpy as np

            stack = np.stack

        self._stacked = (
            stack([e.gate_proj.T for e in self.experts]),
            stack([e.up_proj.T for e in self.experts]),
            stack([e.down_proj.T for e in self.experts]),
        )

    def _forward_gather(self, x, idx, weights):
        if self._stacked is None:
            self._build_stacked()
        gT, uT, dT = self._stacked

        n, dim = x.shape
        k = self.top_k
        idx_flat = _flatten(idx)
        x_rep = _repeat0(x, k)          # (n*k, dim)
        x1 = _expand_mid(x_rep)         # (n*k, 1, dim)

        G = _take0(gT, idx_flat)        # (n*k, dim, hidden)
        U = _take0(uT, idx_flat)
        D = _take0(dT, idx_flat)        # (n*k, hidden, dim)

        h1 = _silu(x1 @ G) * (x1 @ U)   # (n*k, 1, hidden)
        y1 = h1 @ D                     # (n*k, 1, dim)
        y = _reshape(y1, (n, k, dim))
        return (y * weights[..., None]).sum(axis=-2)

    def _update_fire_counts(self, idx) -> None:
        """Per-expert fire counts (NumPy engine only; cheap there)."""
        import numpy as np

        if _is_mlx(idx):
            return
        flat = np.asarray(idx).ravel()
        for e in range(self.num_experts):
            self._fire_counts[e] += int((flat == e).sum())

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        config: dict,
        gate_weight,
        experts: Sequence,
    ) -> "Qwen3MoE":
        """Build from a real HF-style config dict (reads top-k)."""
        top_k = int(
            config.get("num_experts_per_tok")
            or config.get("moe_topk")
            or 8
        )
        return cls(gate_weight, experts, top_k=top_k)

    @classmethod
    def from_mlx_model(
        cls,
        model,
        layer_index: int = 0,
        config: Optional[dict] = None,
    ) -> Optional["Qwen3MoE"]:
        """
        Extract the **real** MoE block from a loaded MLX Qwen3-MoE model.

        The gate is dequantized through the model's own matmul path
        (identity trick — exact for any quantized layout); the experts
        are the model's own storage (per-expert list or fused SwitchGLU).

        Returns None for dense models (no MoE layers found).
        """
        from .real_engram import dequantize_linear

        layers = _find_layers(model)
        if not layers:
            return None
        mlp = _find_moe_mlp(layers[layer_index])
        if mlp is None:
            return None

        gate_w = dequantize_linear(mlp.gate, backend="mlx")
        experts = getattr(mlp, "experts", None)
        expert_fn = getattr(mlp, "switch_mlp", None)
        if experts is None and expert_fn is None:
            return None

        top_k = getattr(mlp, "top_k", None)
        if top_k is None and config:
            top_k = int(
                config.get("num_experts_per_tok")
                or config.get("moe_topk")
                or 0
            ) or None
        return cls(
            gate_weight=gate_w,
            experts=list(experts) if experts is not None else None,
            expert_fn=expert_fn,
            top_k=top_k or 8,
        )

    def to_numpy_copy(self) -> Optional["Qwen3MoE"]:
        """
        A pure-NumPy copy of this block (real weights, dequantized).
        Only available when experts are unfused; fused models return
        None (use ``forward`` on MLX directly for those).
        """
        from .real_engram import dequantize_linear, _to_numpy

        if self.expert_fn is not None or not self.experts:
            return None
        experts = []
        for expert in self.experts:
            g, u, d = _expert_weight_arrays(expert)
            experts.append(
                ExpertMLP(
                    dequantize_linear_like(g),
                    dequantize_linear_like(u),
                    dequantize_linear_like(d),
                )
            )
        return Qwen3MoE(
            _to_numpy(self.gate_weight), experts, top_k=self.top_k
        )

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def stats(self) -> dict:
        s = {
            "num_experts": self.num_experts,
            "top_k": self.top_k,
            "tokens_routed": self._tokens_routed,
            "fused_experts": self.expert_fn is not None,
        }
        if sum(self._fire_counts) > 0:
            s["most_active"] = [
                i for i, _ in sorted(
                    enumerate(self._fire_counts), key=lambda t: -t[1]
                )[: self.top_k]
            ]
        return s


# ----------------------------------------------------------------------
# Engine-agnostic array helpers
# ----------------------------------------------------------------------


def _to_index_array(idx):
    """SwitchGLU wants an integer index array (int32) for its gather."""
    if _is_mlx(idx):
        import mlx.core as mx

        return idx.astype(mx.int32)
    import numpy as np

    return idx.astype("int32")


def _flatten(a):
    if _is_mlx(a):
        import mlx.core as mx

        return mx.flatten(a)
    import numpy as np

    return a.ravel()


def _repeat0(a, k: int):
    """Repeat rows k times (token → its top_k slots)."""
    if _is_mlx(a):
        import mlx.core as mx

        return mx.repeat(a, k, axis=0)
    import numpy as np

    return np.repeat(a, k, axis=0)


def _take0(a, idx):
    """Gather rows of a (E, ...) stack by a flat index array."""
    if _is_mlx(a):
        import mlx.core as mx

        return mx.take(a, idx, axis=0)
    import numpy as np

    return np.take(a, idx, axis=0)


def _reshape(a, shape):
    if _is_mlx(a):
        import mlx.core as mx

        return mx.reshape(a, shape)
    import numpy as np

    return a.reshape(shape)


def _expand_mid(a):
    """(rows, dim) → (rows, 1, dim) for batched matmul broadcasting."""
    if _is_mlx(a):
        import mlx.core as mx

        return mx.expand_dims(a, 1)
    return a[:, None, :]


# ----------------------------------------------------------------------
# Model introspection helpers
# ----------------------------------------------------------------------


def _find_layers(model) -> Optional[list]:
    """Locate the transformer layer stack inside a loaded MLX model."""
    for attr in ("language_model", "model", "transformer"):
        sub = getattr(model, attr, None)
        if sub is not None and hasattr(sub, "layers"):
            return list(sub.layers)
    if hasattr(model, "layers"):
        return list(model.layers)
    return None


def _find_moe_mlp(layer):
    """
    Find the MoE MLP inside a decoder layer. Accepts both the classic
    unfused layout (``mlp.experts`` list + ``mlp.gate``) and mlx-lm's
    fused layout (``mlp.switch_mlp`` + ``mlp.gate``).
    """
    mlp = getattr(layer, "mlp", None)
    if mlp is None:
        return None
    if hasattr(mlp, "gate") and (
        hasattr(mlp, "experts") or hasattr(mlp, "switch_mlp")
    ):
        return mlp
    return None


def _expert_weight_arrays(expert):
    """(gate_proj, up_proj, down_proj) *modules* from an expert module."""
    g = getattr(expert, "gate_proj", None)
    u = getattr(expert, "up_proj", None)
    d = getattr(expert, "down_proj", None)
    if g is None or u is None or d is None:
        raise TypeError(
            f"Expert {type(expert).__name__} lacks gate/up/down projections"
        )
    return (g, u, d)


def dequantize_linear_like(module):
    """Dequantize a linear module to float32 NumPy (out, in)."""
    from .real_engram import dequantize_linear

    return dequantize_linear(module, backend="numpy")


def random_moe(
    dim: int = 64,
    hidden: int = 256,
    num_experts: int = 16,
    top_k: int = 4,
    seed: int = 0,
    backend: str = "numpy",
) -> Qwen3MoE:
    """A fully random (but real-math) MoE for tests/demos."""
    if backend == "mlx":
        import mlx.core as mx

        mx.random.seed(seed)
        gate = mx.random.normal((num_experts, dim)) * (1.0 / dim) ** 0.5
    else:
        import numpy as np

        rng = np.random.default_rng(seed)
        gate = rng.normal(0, (1.0 / dim) ** 0.5, (num_experts, dim)).astype(
            "float32"
        )
    experts = [
        ExpertMLP.random(dim, hidden, seed=seed + i, backend=backend)
        for i in range(num_experts)
    ]
    return Qwen3MoE(gate, experts, top_k=top_k)
