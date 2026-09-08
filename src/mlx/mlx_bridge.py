"""
MLX Bridge — Real Inference, Wired to the Architecture

Ties the real components together into one object:

    ┌────────────────────────────────────────────────────────────┐
    │ QwenLocal                                                  │
    │   ├─ AppleHardware        (mlx_profiler — real detection)  │
    │   ├─ MLXModelLoader       (mlx_inference — real weights)   │
    │   ├─ Qwen3MoE             (real_moe — real gate + experts) │
    │   └─ RealPhraseBook       (real_engram — real vectors)     │
    └────────────────────────────────────────────────────────────┘

Honesty notes (kept in the code so they can't rot away):

  * "Qwen 3.8 Flash" (177B, transcript) has no public open weights,
    so QwenLocal targets *real* models — by default Qwen3-30B-A3B,
    the real MoE that fits a 64GB M1 Max. Point ``model=`` at any
    MLX repo to run something else.
  * The real Qwen3-30B-A3B checkpoint contains its MoE router and
    experts — ``self.moe`` is extracted from the *actual loaded
    model* (real gate weights, real expert modules), not simulated.
  * The real phrase book is *derived* from the model's real embedding
    matrix (see real_engram's docstring for the reasoning) and is
    reported as a mechanism demo — it is NOT injected into the
    loaded model's forward pass, because the loaded model was not
    trained with an engram table.

CLI:
    .venv/bin/python -m src.mlx.mlx_bridge \
        --model mlx-community/Qwen3-30B-A3B-4bit \
        --prompt "Explain mixture-of-experts routing in one paragraph" \
        --max-tokens 128
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

from .mlx_inference import MLXModelLoader, MLXNotAvailableError
from .mlx_profiler import (
    profile_apple_silicon,
    best_fit,
    recommend,
    summary as hardware_summary,
)
from .real_engram import RealPhraseBook, extract_embedding_matrix
from .real_moe import Qwen3MoE, _find_layers, _find_moe_mlp


class QwenLocal:
    """
    Real local inference with the full architecture wired in.

    Args:
        model: MLX model (HF repo id or local path).
        dtype: Optional MLX dtype override.
        engram_dir: Where to find / write the phrase book (.npy + index).
        phrase_tokens: Optional list of token-id phrases to build the
            phrase book from (used when ``engram_dir`` is empty).
        build_engram: Whether to (re)build the phrase book now.
    """

    def __init__(
        self,
        model: str = "mlx-community/Qwen3-30B-A3B-4bit",
        dtype: Optional[str] = None,
        engram_dir: Optional[str] = None,
        phrase_tokens: Optional[list] = None,
        build_engram: bool = True,
    ) -> None:
        self.hardware = profile_apple_silicon()
        self.loader = MLXModelLoader(model, dtype=dtype)
        self.loader.load()
        self.engram_dir = engram_dir

        # Real MoE block extracted from the actually-loaded model
        self.moe = None
        if self.loader.is_moe:
            self.moe = Qwen3MoE.from_mlx_model(
                self.loader._model, config=self.loader.config
            )

        # Real phrase book (derived from the real embedding matrix)
        self.engram = None
        if build_engram and engram_dir:
            self.engram = self._load_or_build_engram(phrase_tokens)

    # ------------------------------------------------------------------
    # Engram (real vectors)
    # ------------------------------------------------------------------

    def _load_or_build_engram(
        self, phrase_tokens: Optional[list]
    ) -> Optional[RealPhraseBook]:
        path = Path(self.engram_dir)
        if path.exists() and (path / "embeddings.npy").exists():
            print("  Phrase book: loading existing (memory-mapped)...")
            return RealPhraseBook.load(path)

        if phrase_tokens is None:
            # Derive a starter phrase book from common token n-grams
            # of a small built-in corpus — real rows, real vectors.
            phrase_tokens = self._starter_phrases()

        print(f"  Phrase book: building {len(phrase_tokens)} real rows "
              f"from the model's embedding matrix...")
        matrix = extract_embedding_matrix(self.loader._model)
        book = RealPhraseBook.from_embeddings(
            matrix, phrase_tokens, dim=matrix.shape[1]
        )
        path.mkdir(parents=True, exist_ok=True)
        book.save(path)
        return book

    def _starter_phrases(self) -> list:
        """Token-id phrases for a starter phrase book (from real tokens)."""
        try:
            corpus = (
                "the quick brown fox jumps over the lazy dog "
                "mixture of experts routing and gating network "
                "unified memory and apple silicon inference "
                "local large language model running on consumer hardware "
            ).split()
            tok = self.loader._tokenizer
            ids = []
            for word in corpus:
                for attempt in (word, word.lower(), f" {word}"):
                    enc = tok.encode(attempt)
                    if isinstance(enc, (tuple, list)) and enc and isinstance(enc[0], (tuple, list)):
                        enc = enc[0]
                    if isinstance(enc, (list, tuple)) and len(enc) > 1:
                        ids.extend(int(t) for t in enc)
                        break
                    ids.extend(int(t) for t in (enc if isinstance(enc, (list, tuple)) else []))
            phrases = []
            for i in range(len(ids) - 1):
                phrases.append([ids[i], ids[i + 1]])
            return phrases or None
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def ask(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        stream: bool = True,
    ) -> str:
        """Real generation. Prints live output + measured throughput."""
        t0 = time.time()
        text = []
        first = None
        for i, chunk in self.loader.generate(
            prompt, max_tokens=max_tokens, temperature=temperature
        ):
            if first is None:
                first = time.time()
            if stream:
                print(chunk, end="", flush=True)
            text.append(chunk)
        if stream:
            print()

        m = self.loader.last_metrics
        if m:
            print(
                f"\n  [real] {m.num_tokens} tokens | "
                f"prompt {m.prompt_tps:.1f} tok/s | "
                f"decode {m.decode_tps:.1f} tok/s | "
                f"first token {first - t0:.2f}s"
            )
        return "".join(text)

    # ------------------------------------------------------------------
    # Reporting — transcript concept vs what actually runs here
    # ------------------------------------------------------------------

    def report(self) -> str:
        lines = [
            hardware_summary(self.hardware),
            "",
            "  Real model loaded:",
            f"    {self.loader.model_info()}",
        ]
        if self.moe:
            s = self.moe.stats
            layout = "fused (model's SwitchGLU)" if s["fused_experts"] else "per-expert list"
            lines += [
                "",
                "  Real MoE (extracted from the loaded model):",
                f"    {s['num_experts']} experts, top-{s['top_k']} fire per token [{layout}]",
                f"    (transcript: 512 experts / 10 active — same pattern)",
            ]
        if self.engram:
            s = self.engram.stats
            lines += [
                "",
                "  Real phrase book (derived from real embeddings, mmap):",
                f"    {s['indexed_phrases']} phrases | "
                f"{self.engram.size_gb:.4f} GB | dim {self.engram.dim}",
                "    NOTE: demonstrated, not injected — the loaded model",
                "    was not trained with an engram table.",
            ]
        return "\n".join(lines)

    def unload(self) -> None:
        """Release the model (and phrase book) from unified memory."""
        self.loader.unload()
        self.engram = None


# ----------------------------------------------------------------------
# Verification — prove real_moe is the model's real MoE
# ----------------------------------------------------------------------


def verify_moe_reimplementation(loader: MLXModelLoader, seed: int = 0, layer_index: int = 0, tol: float = 2e-2) -> dict:
    """
    Run the model's *own* MoE block and our re-implementation (real gate
    weights + Qwen3-MoE routing formula, model's own expert storage) on
    the same random input, and compare outputs. A tiny max diff proves
    our routing reproduces the model's real MoE block.
    """
    import mlx.core as mx

    mx.random.seed(seed)
    layers = _find_layers(loader._model)
    if not layers or layer_index >= len(layers):
        return {"ok": False, "reason": "no layers found"}
    mlp = _find_moe_mlp(layers[layer_index])
    if mlp is None:
        return {"ok": False, "reason": "no MoE layer at this index (dense model?)"}

    mine = Qwen3MoE.from_mlx_model(loader._model, layer_index=layer_index,
                                   config=loader.config)
    dim = int(mine.gate_weight.shape[1])
    x = mx.random.uniform(-1.0, 1.0, (1, dim)).astype(mx.float32)

    ref = mlp(x).astype(mx.float32)      # the model's own MoE forward
    got = mine.forward(x).astype(mx.float32)

    diff = float(mx.max(mx.abs(ref - got)))
    return {
        "ok": diff < tol,
        "max_abs_diff": diff,
        "tolerance": tol,
        "hidden": dim,
        "fused_experts": mine.expert_fn is not None,
    }


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Real Qwen inference on Apple Silicon (MLX bridge)"
    )
    ap.add_argument(
        "--model",
        default=None,
        help="MLX model repo or path "
             "(default: largest real Qwen that fits this machine)",
    )
    ap.add_argument("--prompt", default="Hello! Introduce yourself in one sentence.")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--dtype", default=None, help="e.g. float16")
    ap.add_argument(
        "--engram-dir",
        default=None,
        help="Directory for the real phrase book (.npy + index.json)",
    )
    ap.add_argument(
        "--no-stream", action="store_true", help="buffer output, print at end"
    )
    ap.add_argument("--profile-only", action="store_true",
                    help="show hardware + model fit, do not load a model")
    args = ap.parse_args(argv)

    print("Qwen Local — MLX bridge (real inference on Apple Silicon)")
    print("=" * 60)

    if args.profile_only:
        print(hardware_summary())
        return 0

    model = args.model or (best_fit().repo if best_fit() else
                           "mlx-community/Qwen3-30B-A3B-4bit")

    try:
        qwen = QwenLocal(
            model=model,
            dtype=args.dtype,
            engram_dir=args.engram_dir,
            build_engram=args.engram_dir is not None,
        )
    except MLXNotAvailableError as exc:
        print(f"\n  {exc}\n")
        print("  Showing hardware profile (simulation mode) instead:\n")
        print(hardware_summary())
        return 1

    print(qwen.report())
    print("=" * 60)

    print(f"\n  You: {args.prompt}\n  Model:", end="")
    try:
        qwen.ask(
            args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            stream=not args.no_stream,
        )
    except Exception as exc:
        print(f"\n  [error] {type(exc).__name__}: {exc}")
        return 1

    qwen.unload()
    return 0


if __name__ == "__main__":
    sys.exit(main())
