"""
MLX Model Loader — Real Inference on Apple Silicon

Wraps Apple's MLX framework (``mlx`` + ``mlx-lm``) to load *real*
Hugging Face models and run *real* inference on unified memory.

Design goals:
  - Works with any MLX-compatible repo on the HF Hub
    (``mlx-community/*``) or a local directory of weights.
  - Measures real performance: prompt-processing tok/s and
    decode tok/s, so the transcript's speed claims can be
    checked against what this machine actually does.
  - Exposes the model's ``config.json`` so the bridge can
    detect real MoE geometry (num experts, top-k, dims).
  - Degrades gracefully: importing this module never requires MLX.
    Calling ``load()`` without MLX installed raises
    ``MLXNotAvailableError`` with exact install instructions.

Install the real backend:
    python -m venv .venv && .venv/bin/pip install mlx mlx-lm

Usage:
    from src.mlx.mlx_inference import MLXModelLoader

    loader = MLXModelLoader("mlx-community/Qwen3-30B-A3B-4bit")
    loader.load()
    for tok in loader.generate("Write a solar system simulation"):
        print(tok, end="")
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional


class MLXNotAvailableError(RuntimeError):
    """MLX is not importable in the current interpreter."""

    def __init__(self) -> None:
        super().__init__(
            "MLX is not installed in this Python environment.\n"
            "  Fix: create a venv and install the real backend:\n"
            "    python3 -m venv .venv\n"
            "    .venv/bin/pip install mlx mlx-lm\n"
            "  Then run with:  .venv/bin/python -m src.mlx.mlx_bridge ...\n"
            "  (Or run without MLX: the src.* simulation still works.)"
        )


def mlx_available() -> bool:
    """True if the MLX core library can be imported."""
    try:
        import mlx.core  # type: ignore  # noqa: F401

        return True
    except Exception:
        return False


@dataclass
class GenerationMetrics:
    """Measured performance of one generation call."""

    num_tokens: int = 0
    prompt_time_s: float = 0.0      # Time to process the prompt (first token)
    decode_time_s: float = 0.0      # Time for generated tokens
    prompt_tps: float = 0.0         # Prompt tokens / s
    decode_tps: float = 0.0         # Generated tokens / s
    # Not serialized: last raw GenerationResponse (framework measurements)
    _last_response: object = None  # type: ignore[assignment]

    def as_dict(self) -> dict:
        return {
            "num_tokens": self.num_tokens,
            "prompt_time_s": round(self.prompt_time_s, 3),
            "decode_time_s": round(self.decode_time_s, 3),
            "prompt_tps": round(self.prompt_tps, 2),
            "decode_tps": round(self.decode_tps, 2),
        }


class MLXModelLoader:
    """
    Loads a real MLX model and runs real inference.

    Args:
        model: A local directory with weights, or a Hugging Face repo id
               (e.g. ``"mlx-community/Qwen3-30B-A3B-4bit"``).
        dtype: Optional MLX dtype override (e.g. ``"float16"``).
        download: Whether to fetch remote repos on load (default True).
    """

    def __init__(
        self,
        model: str,
        dtype: Optional[str] = None,
        download: bool = True,
    ) -> None:
        self.model_spec = str(model)
        self.dtype = dtype
        self.download = download

        self._model = None
        self._tokenizer = None
        self._lm = None            # mlx_lm module (imported lazily)
        self._path: Optional[Path] = None
        self.config: dict = {}
        self.last_metrics: Optional[GenerationMetrics] = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> "MLXModelLoader":
        """
        Load the model into unified memory. Idempotent.

        Raises:
            MLXNotAvailableError: if MLX is not installed.
        """
        if self._model is not None:
            return self

        try:
            import mlx.core as mx  # type: ignore  (validates install)
        except ImportError as exc:
            raise MLXNotAvailableError() from exc

        try:
            import mlx_lm  # type: ignore
        except ImportError as exc:
            raise MLXNotAvailableError() from exc

        self._lm = mlx_lm
        self._path = self._resolve_path()
        self.config = self._read_config()

        t0 = time.time()
        print(
            f"  Loading {self.model_spec} → {self._path} "
            f"(unified memory, mlx {mx.__version__})"
        )
        self._model, self._tokenizer = self._load_via_api()
        self._load_time_s = time.time() - t0
        return self

    def _load_via_api(self):
        """
        Load model + tokenizer, tolerant across mlx-lm versions:
          0.31+:  mlx_lm.load(path)               → (model, tokenizer)
          older:  mlx_lm.load_model(path)         → (model, tokenizer)
          old API: mlx_lm.utils.load_model(path)  → model only
        """
        load_kwargs = {}
        if self.dtype:
            # older mlx-lm accepted a dtype kwarg; newer one does not
            try:
                import inspect

                sig = inspect.signature(self._lm.load)
                if "model_config" in sig.parameters:
                    load_kwargs["model_config"] = {"dtype": self.dtype}
                else:
                    load_kwargs["dtype"] = self.dtype
            except Exception:
                pass

        if hasattr(self._lm, "load"):
            return self._lm.load(str(self._path), **load_kwargs)
        if hasattr(self._lm, "load_model"):
            return self._lm.load_model(str(self._path), **load_kwargs)
        model = self._lm.utils.load_model(str(self._path))
        return model, None

    def _resolve_path(self) -> Path:
        """
        Resolve the model spec to a local directory.

        Local directory with weights is used as-is; a remote repo id
        is downloaded with ``huggingface_hub`` (cached in ~/.cache/huggingface).
        """
        spec = self.model_spec
        p = Path(spec).expanduser()
        if p.is_dir():
            if self._looks_like_weights(p):
                return p
            raise FileNotFoundError(
                f"{spec} is a directory but contains no model weights "
                "(expected config.json + *.safetensors)"
            )

        # Remote repo
        if not self.download:
            raise FileNotFoundError(
                f"{spec} is not a local directory and download is disabled"
            )
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise MLXNotAvailableError() from exc

        print(f"  Downloading {spec} (Hugging Face Hub, cached)...")
        path = Path(
            snapshot_download(
                spec,
                allow_patterns=[
                    "*.safetensors", "*.json", "*.model",
                    "tokenizer*", "merges.txt", "vocab*", "*.txt",
                ],
            )
        )
        return path

    @staticmethod
    def _looks_like_weights(p: Path) -> bool:
        return (p / "config.json").exists() or any(p.glob("*.safetensors"))

    def _read_config(self) -> dict:
        """Read config.json for architecture introspection (MoE etc.)."""
        if self._path and (self._path / "config.json").exists():
            try:
                with open(self._path / "config.json") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def unload(self) -> None:
        """Release the model from unified memory."""
        try:
            import mlx.core as mx  # type: ignore

            if self._model is not None:
                mx.destroy(self._model)
        except Exception:
            pass
        self._model = None
        self._tokenizer = None

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        verbose: bool = False,
        stream: bool = True,
    ) -> Iterator[tuple[int, str]]:
        """
        Run real inference, yielding (token_index, text_chunk) pairs.

        Timing: the first yield measures prompt processing; the rest
        measure per-token decode. Metrics are stored in ``last_metrics``
        after the generator is exhausted (or closed).
        """
        if self._model is None or self._lm is None:
            if not mlx_available():
                raise MLXNotAvailableError()
            self.load()
        assert self._lm is not None and self._model is not None

        prompt_tokens = self._count_prompt_tokens(prompt)

        gen = self._stream(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            verbose=verbose,
        )

        metrics = GenerationMetrics()
        t_start = time.time()
        t_prev = t_start
        self.last_metrics = metrics

        try:
            for item in gen:
                # New API yields GenerationResponse objects;
                # old API yields (token_id, text) tuples.
                if isinstance(item, tuple):
                    token_id, text = item
                else:
                    token_id = getattr(item, "token", None)
                    text = getattr(item, "text", str(item))
                now = time.time()
                if metrics.num_tokens == 0:
                    metrics.prompt_time_s = now - t_start
                else:
                    metrics.decode_time_s += now - t_prev
                t_prev = now
                metrics.num_tokens += 1
                metrics._last_response = item
                yield (token_id, text)
        finally:
            self._finalize_metrics(metrics, prompt_tokens)

    def _stream(self, prompt, **kwargs):
        """
        Version-tolerant streaming generation source.

        mlx-lm 0.31+: sampler is a callable built via
        ``sample_utils.make_sampler(temp, top_p)``.
        Older mlx-lm: sampling params are plain kwargs.
        """
        verbose = kwargs.pop("verbose", False)
        temperature = kwargs.pop("temperature", 0.7)
        top_p = kwargs.pop("top_p", 0.9)
        repetition_penalty = kwargs.pop("repetition_penalty", 1.0)

        try:
            from mlx_lm.sample_utils import make_sampler

            sampler_kwargs = {
                "sampler": make_sampler(temp=temperature, top_p=top_p)
            }
        except Exception:
            sampler_kwargs = {
                "temperature": temperature,
                "top_p": top_p,
                "repetition_penalty": repetition_penalty,
            }

        if hasattr(self._lm, "stream_generate"):
            return self._lm.stream_generate(
                self._model, self._tokenizer, prompt, **sampler_kwargs, **kwargs
            )
        # Older mlx-lm: single generate() with stream=True
        return self._lm.generate(
            self._model, self._tokenizer, prompt,
            stream=True, verbose=verbose, **sampler_kwargs, **kwargs
        )

    def _finalize_metrics(self, metrics: GenerationMetrics, prompt_tokens: int) -> None:
        """
        Finalize throughput numbers. Prefers the framework's own
        measurements (GenerationResponse carries prompt_tps /
        generation_tps); falls back to wall-clock timing.
        """
        total = metrics.prompt_time_s + metrics.decode_time_s
        resp = getattr(metrics, "_last_response", None)

        if resp is not None:
            pt = getattr(resp, "prompt_tps", None)
            gt = getattr(resp, "generation_tps", None)
            if pt:
                metrics.prompt_tps = float(pt)
            if gt:
                metrics.decode_tps = float(gt)
            if getattr(resp, "generation_tokens", None):
                metrics.num_tokens = int(resp.generation_tokens)
            if metrics.prompt_tps == 0.0:
                pt_count = getattr(resp, "prompt_tokens", None) or prompt_tokens
                metrics.prompt_tps = (
                    pt_count / metrics.prompt_time_s
                    if metrics.prompt_time_s > 0 else 0.0
                )

        if metrics.prompt_tps == 0.0 and metrics.prompt_time_s > 0:
            metrics.prompt_tps = prompt_tokens / metrics.prompt_time_s
        if metrics.decode_tps == 0.0 and metrics.decode_time_s > 0:
            metrics.decode_tps = metrics.num_tokens / metrics.decode_time_s

    def _count_prompt_tokens(self, prompt: str) -> int:
        """Count prompt tokens via the real tokenizer (version-tolerant)."""
        try:
            out = self._tokenizer.encode(prompt)
            if isinstance(out, tuple):
                out = out[0]                      # (input_ids, mask) form
            if isinstance(out, (list, tuple)):
                if len(out) and isinstance(out[0], (list, tuple)):
                    out = out[0]                  # batched form
                return max(1, len(out))
            if isinstance(out, int):
                return max(1, out)
        except Exception:
            pass
        return max(1, len(prompt.split()))

    def generate_once(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs,
    ) -> str:
        """Convenience: full generation, returns the complete text."""
        return "".join(
            text for _, text in self.generate(
                prompt, max_tokens=max_tokens,
                temperature=temperature, **kwargs
            )
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def is_moe(self) -> bool:
        """True if the loaded model's config declares a MoE architecture."""
        arch = self.config.get("architectures") or []
        return any("Moe" in str(a) for a in arch) or (
            self.config.get("num_experts", 0) or 0
        ) > 1

    def moe_geometry(self) -> dict:
        """
        Real MoE geometry from config.json, if present:
        ``{"num_experts", "top_k", "hidden_dim", "expert_intermediate_dim"}``.
        """
        return {
            "num_experts": int(self.config.get("num_experts", 0) or 0),
            "top_k": int(
                self.config.get("num_experts_per_tok", 0)
                or self.config.get("moe_topk", 0) or 0
            ),
            "hidden_dim": int(self.config.get("hidden_size", 0) or 0),
            "expert_intermediate_dim": int(
                self.config.get("moe_intermediate_size", 0)
                or self.config.get("intermediate_size", 0) or 0
            ),
        }

    def model_info(self) -> dict:
        """Summary of what is actually loaded."""
        info = {
            "model": self.model_spec,
            "path": str(self._path) if self._path else None,
            "loaded": self.is_loaded,
            "is_moe": self.is_moe,
            "moe": self.moe_geometry() if self.is_moe else None,
            "architecture": (self.config.get("architectures") or [None])[0],
        }
        if self.last_metrics:
            info["last_generation"] = self.last_metrics.as_dict()
        return info
