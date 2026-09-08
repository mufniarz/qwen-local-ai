"""
Tests for the MLX inference bridge (src.mlx).

Designed to run in two environments:
  - system Python without MLX (simulation + numpy paths only)
  - the project venv with mlx + mlx-lm installed (full engine)

Real-model tests (downloading/loading actual checkpoints) are gated
behind the QWEN_RUN_REAL=1 environment variable so the normal suite
stays fast.
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mlx.mlx_profiler import (
    profile_apple_silicon,
    recommend,
    best_fit,
    to_memory_budget,
    fits_in_unified_memory,
    TRANSCRIPT_MODEL_177B,
    REAL_MODEL_CATALOG,
)
from mlx.mlx_inference import MLXModelLoader, MLXNotAvailableError, mlx_available
from mlx.real_engram import RealPhraseBook, extract_embedding_matrix, dequantize_linear, _to_numpy
from mlx.real_moe import Qwen3MoE, ExpertMLP, random_moe, dequantize_linear_like


# ----------------------------------------------------------------------
# Environment detection
# ----------------------------------------------------------------------


def _has_mlx() -> bool:
    try:
        import mlx.core  # type: ignore

        return True
    except Exception:
        return False


skip_no_mlx = pytest.mark.skipif(not _has_mlx(), reason="MLX not installed")


# ----------------------------------------------------------------------
# mlx_profiler
# ----------------------------------------------------------------------


class TestProfiler:
    def test_profile_returns_sane_fields(self):
        hw = profile_apple_silicon()
        assert hw.system  # always detected
        assert hw.physical_cores >= 1
        assert hw.unified_memory_gb >= 0

    def test_profile_on_apple_silicon(self):
        hw = profile_apple_silicon()
        if hw.system == "Darwin" and hw.arch in ("arm64", "aarch64"):
            assert hw.is_apple_silicon
            assert "Apple" in hw.chip
            assert hw.unified_memory_gb >= 8
            assert hw.physical_cores == hw.logical_cores or hw.logical_cores >= hw.physical_cores

    def test_177b_hypothetical_does_not_fit_64gb(self):
        assert not fits_in_unified_memory(TRANSCRIPT_MODEL_177B, 64.0)
        assert fits_in_unified_memory(TRANSCRIPT_MODEL_177B, 256.0)

    def test_recommend_ranks_and_flags(self):
        hw = profile_apple_silicon()
        if hw.unified_memory_gb < 4:
            pytest.skip("no usable memory info")
        results = dict()
        for model, fits in recommend(hw, kv_cache_gb=8.0):
            results[model.name] = fits
        assert "Qwen 3.8 Flash (transcript, hypothetical)" in results
        # small model always fits; the hypothetical 177B never fits 64GB
        assert results["Qwen3 0.6B (dense)"] is True
        if hw.unified_memory_gb < 100:
            assert results["Qwen 3.8 Flash (transcript, hypothetical)"] is False

    def test_best_fit_is_real_repo(self):
        hw = profile_apple_silicon()
        if hw.unified_memory_gb < 1:
            pytest.skip("no usable memory info")
        model = best_fit(hw)
        assert model is not None
        assert model.repo.startswith("mlx-community/")

    def test_to_memory_budget_keys(self):
        hw = profile_apple_silicon()
        budget = to_memory_budget(hw)
        for key in ("vram_total_gb", "ram_total_gb", "ram_cap_gb", "ssd_total_gb"):
            assert key in budget
            assert budget[key] >= 0

    def test_catalog_models_are_downloadable(self):
        for model in REAL_MODEL_CATALOG:
            assert model.repo.startswith("mlx-community/")
            assert model.size_gb_4bit > 0


# ----------------------------------------------------------------------
# real_moe — the Qwen3-MoE formula
# ----------------------------------------------------------------------


class TestQwen3MoERouting:
    def test_route_shapes_and_constraints(self):
        moe = random_moe(dim=32, hidden=64, num_experts=16, top_k=4, seed=1)
        x = np.random.default_rng(0).normal(size=(5, 32)).astype("float32")
        idx, w = moe.route(x)
        assert idx.shape == (5, 4)
        assert w.shape == (5, 4)
        np.testing.assert_allclose(w.sum(axis=-1), 1.0, rtol=1e-5, atol=1e-6)
        assert (idx >= 0).all() and (idx < 16).all()

    def test_route_is_deterministic(self):
        moe = random_moe(dim=32, hidden=64, num_experts=16, top_k=4, seed=2)
        x = np.random.default_rng(3).normal(size=(3, 32)).astype("float32")
        idx1, w1 = moe.route(x)
        idx2, w2 = moe.route(x)
        np.testing.assert_array_equal(idx1, idx2)
        np.testing.assert_allclose(w1, w2)

    def test_route_uses_gate_not_hash(self):
        """Routing must actually depend on the gate weights."""
        moe = random_moe(dim=32, hidden=64, num_experts=16, top_k=4, seed=4)
        x = np.random.default_rng(5).normal(size=(2, 32)).astype("float32")
        idx1, _ = moe.route(x)
        # Perturb the gate → routing must change (a hash-based fake
        # router would not depend on gate weights at all).
        moe.gate_weight = moe.gate_weight * 100.0
        idx2, _ = moe.route(x)
        assert not np.array_equal(idx1, idx2)

    def test_forward_matches_manual_formula(self):
        """
        Reference implementation of the Qwen3-MoE formula, computed
        independently from the raw weights:
            out = Σ_k (p_k / Σp) · E_k(x)
        """
        moe = random_moe(dim=32, hidden=64, num_experts=16, top_k=4, seed=6)
        x = np.random.default_rng(7).normal(size=(3, 32)).astype("float32")

        # Independent math
        logits = x @ moe.gate_weight.T
        logits = logits - logits.max(axis=-1, keepdims=True)
        e = np.exp(logits)
        probs = e / e.sum(axis=-1, keepdims=True)
        order = np.argsort(-probs, axis=-1)
        top = np.take_along_axis(probs, order[:, :4], axis=-1)
        top = top / top.sum(axis=-1, keepdims=True)
        idx = order[:, :4]

        ref = np.zeros_like(x)
        for i in range(x.shape[0]):
            for k in range(4):
                e_idx = idx[i, k]
                g, u, d = moe.experts[e_idx].weights()
                zg = x[i] @ g.T
                sig = 1.0 / (1.0 + np.exp(-zg))
                h = (zg * sig) * (x[i] @ u.T)  # SwiGLU: g·σ(g)·u
                ref[i] += top[i, k] * (h @ d.T)

        np.testing.assert_allclose(moe.forward(x), ref, rtol=1e-4, atol=1e-5)

    def test_expert_swiglu_formula(self):
        expert = ExpertMLP.random(16, 32, seed=8)
        x = np.random.default_rng(9).normal(size=(2, 16)).astype("float32")
        g, u, d = expert.weights()  # each (out, in)
        zg = x @ g.T
        sig = 1.0 / (1.0 + np.exp(-zg))
        ref = ((zg * sig) * (x @ u.T)) @ d.T  # SwiGLU
        np.testing.assert_allclose(expert.forward(x), ref, rtol=1e-5, atol=1e-6)

    def test_topk_cannot_exceed_experts(self):
        with pytest.raises(ValueError):
            Qwen3MoE(
                np.zeros((4, 8), dtype="float32"),
                [ExpertMLP.random(8, 16, seed=i) for i in range(4)],
                top_k=8,
            )

    def test_from_config_reads_top_k(self):
        experts = [ExpertMLP.random(8, 16, seed=i) for i in range(12)]
        gate = np.zeros((12, 8), dtype="float32")
        moe = Qwen3MoE.from_config(
            {"num_experts_per_tok": 6}, gate, experts
        )
        assert moe.top_k == 6

    def test_stats_track_routing(self):
        moe = random_moe(dim=32, hidden=64, num_experts=16, top_k=4, seed=10)
        x = np.random.default_rng(11).normal(size=(4, 32)).astype("float32")
        moe.forward(x)
        s = moe.stats
        assert s["tokens_routed"] == 4
        assert s["num_experts"] == 16
        assert s["top_k"] == 4
        assert s["fused_experts"] is False

    @skip_no_mlx
    def test_mlx_engine_matches_numpy(self):
        """Same weights, both engines → same outputs."""
        import mlx.core as mx

        np_moe = random_moe(dim=32, hidden=64, num_experts=16, top_k=4, seed=12)
        # Build the MLX twin from the identical weights
        gate = mx.array(np_moe.gate_weight)
        experts = [
            ExpertMLP(
                mx.array(e.gate_proj), mx.array(e.up_proj), mx.array(e.down_proj)
            )
            for e in np_moe.experts
        ]
        mlx_moe = Qwen3MoE(gate, experts, top_k=np_moe.top_k)

        x_np = np.random.default_rng(13).normal(size=(4, 32)).astype("float32")
        x_mx = mx.array(x_np)

        out_np = np_moe.forward(x_np)
        out_mx = np.asarray(mlx_moe.forward(x_mx))
        np.testing.assert_allclose(out_np, out_mx, rtol=1e-4, atol=1e-5)

        idx_np, _ = np_moe.route(x_np)
        idx_mx, _ = mlx_moe.route(x_mx)
        np.testing.assert_array_equal(idx_np, np.asarray(idx_mx))


# ----------------------------------------------------------------------
# real_engram — phrase book with real vectors
# ----------------------------------------------------------------------


def _small_vocab_matrix(vocab=50, dim=16, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, (vocab, dim)).astype("float32")


class TestRealPhraseBook:
    def test_rows_are_real_vector_math(self):
        m = _small_vocab_matrix()
        book = RealPhraseBook.from_embeddings(m, [[1, 2], [3]])
        # 2-token phrase row = mean of the two token rows * 2 (= the sum)
        np.testing.assert_allclose(
            book.lookup([1, 2]), m[[1, 2]].sum(axis=0), rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(book.lookup([3]), m[[3]].sum(axis=0))

    def test_missing_phrase_returns_zeros(self):
        m = _small_vocab_matrix(vocab=10)
        book = RealPhraseBook.from_embeddings(m, [[1, 2]])
        np.testing.assert_array_equal(book.lookup([4, 5]), np.zeros(16))

    def test_cache_hit_and_lru_eviction(self):
        m = _small_vocab_matrix()
        book = RealPhraseBook.from_embeddings(m, [[1], [2], [3]], cache_size=2)
        book.lookup([1])
        book.lookup([2])
        book.lookup([1])  # hit
        assert book.stats["cache_hits"] >= 1
        book.lookup([3])  # evicts LRU ([2] after [1] re-hit)
        s = book.stats
        assert s["cached_entries"] == 2
        assert s["indexed_phrases"] == 3

    def test_vocab_range_guard(self):
        m = _small_vocab_matrix(vocab=10)
        with pytest.raises(ValueError):
            RealPhraseBook.from_embeddings(m, [[999]])

    def test_phrase_length_guard(self):
        m = _small_vocab_matrix()
        book = RealPhraseBook(dim=m.shape[1])
        with pytest.raises(ValueError):
            book.lookup([1, 2, 3, 4])

    def test_save_load_roundtrip(self):
        m = _small_vocab_matrix()
        phrases = [[1, 2], [2, 3], [7]]
        book = RealPhraseBook.from_embeddings(m, phrases)
        with tempfile.TemporaryDirectory() as tmp:
            path = book.save(tmp)
            reloaded = RealPhraseBook.load(path)
            for p in phrases:
                np.testing.assert_allclose(
                    reloaded.lookup(p), book.lookup(p), rtol=0, atol=0
                )
        assert reloaded.dim == 16
        assert reloaded.stats["indexed_phrases"] == 3

    def test_input_embeddings_substitution(self):
        m = _small_vocab_matrix(vocab=50)
        book = RealPhraseBook.from_embeddings(m, [[1, 2], [5, 6, 7]])
        ids = [1, 2, 9, 5, 6, 7, 11]
        out = book.input_embeddings(m, ids)
        assert out.shape == (7, 16)
        # positions 0-1 covered by [1,2] phrase row
        phrase = m[[1, 2]].sum(axis=0)
        np.testing.assert_allclose(out[0], phrase, rtol=1e-5)
        np.testing.assert_allclose(out[1], phrase, rtol=1e-5)
        # position 2 is a plain token embedding
        np.testing.assert_allclose(out[2], m[9], rtol=1e-5)
        # 3-gram [5,6,7] matches before the 2-gram fallback
        tri = m[[5, 6, 7]].sum(axis=0)
        np.testing.assert_allclose(out[3], tri, rtol=1e-5)

    def test_stats_keys_match_simulation(self):
        m = _small_vocab_matrix()
        book = RealPhraseBook.from_embeddings(m, [[1], [2]])
        book.lookup([1])
        s = book.stats
        for key in ("lookups", "cache_hits", "disk_reads", "cache_hit_rate",
                    "cached_entries", "indexed_phrases"):
            assert key in s

    @skip_no_mlx
    def test_extract_embedding_matrix_from_real_model(self):
        """End-to-end: materialize the real embedding table of a loaded model."""
        import mlx.core as mx
        import mlx.nn as nn

        emb = nn.Embedding(300, 16)
        mat = extract_embedding_matrix(_wrap(emb))
        assert mat.shape == (300, 16)
        expected = np.asarray(emb(mx.array(range(300))))
        np.testing.assert_allclose(mat, expected, rtol=1e-4)


def _wrap(emb):
    import mlx.nn as nn

    class Holder(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = emb

    return Holder()


# ----------------------------------------------------------------------
# mlx_inference
# ----------------------------------------------------------------------


class TestMLXModelLoader:
    def test_availability_flag_is_bool(self):
        assert isinstance(mlx_available(), bool)

    def test_missing_mlx_raises_friendly_error(self):
        if mlx_available():
            pytest.skip("mlx is installed; error path not testable here")
        loader = MLXModelLoader("/nonexistent/model")
        with pytest.raises(MLXNotAvailableError) as excinfo:
            loader.load()
        assert "pip install mlx mlx-lm" in str(excinfo.value)

    def test_local_dir_without_weights_rejected(self):
        if not mlx_available():
            pytest.skip("mlx not installed")
        with tempfile.TemporaryDirectory() as tmp:
            loader = MLXModelLoader(tmp)
            with pytest.raises(FileNotFoundError):
                loader.load()

    @skip_no_mlx
    def test_generate_requires_loaded_model(self):
        loader = MLXModelLoader("some-repo")
        with pytest.raises(MLXNotAvailableError):
            list(loader.generate("hi", max_tokens=1))


# ----------------------------------------------------------------------
# mlx_bridge — CLI + wiring
# ----------------------------------------------------------------------


class TestBridgeCLI:
    def test_profile_only(self, capsys):
        from mlx.mlx_bridge import main

        code = main(["--profile-only"])
        out = capsys.readouterr().out
        assert code == 0
        assert "Model fit" in out
        assert "Qwen 3.8 Flash (transcript, hypothetical)" in out

    @skip_no_mlx
    def test_verify_dense_model_reports_no_moe(self, tmp_path):
        """verify_moe_reimplementation degrades gracefully on dense models."""
        from mlx.mlx_bridge import verify_moe_reimplementation

        # A tiny fake 'loader' exposing the loaded MLX model
        import mlx.core as mx
        import mlx.nn as nn

        class FakeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_tokens = nn.Embedding(50, 8)
                self.layers = []

        class FakeLoader:
            def __init__(self):
                self._model = FakeModel()
                self.config = {}

        result = verify_moe_reimplementation(FakeLoader())
        assert result["ok"] is False
        assert "no MoE" in result["reason"]


# ----------------------------------------------------------------------
# Real-model integration (opt-in: QWEN_RUN_REAL=1)
# ----------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("QWEN_RUN_REAL") != "1" or not _has_mlx(),
    reason="set QWEN_RUN_REAL=1 (and install mlx) to run real-model tests",
)
class TestRealModel:
    MODEL = "mlx-community/Qwen3-0.6B-4bit"

    def test_bridge_end_to_end(self, tmp_path):
        from mlx.mlx_bridge import QwenLocal

        qwen = QwenLocal(
            model=self.MODEL,
            engram_dir=str(tmp_path / "engram"),
        )
        assert qwen.loader.is_loaded
        assert qwen.engram is not None
        assert qwen.engram.stats["indexed_phrases"] > 0

        text = qwen.ask("Say hello.", max_tokens=16, stream=False)
        assert len(text) > 0
        metrics = qwen.loader.last_metrics.as_dict()
        assert metrics["num_tokens"] > 0
        qwen.unload()
        assert not qwen.loader.is_loaded
