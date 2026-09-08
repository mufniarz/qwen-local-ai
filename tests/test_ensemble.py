"""
Integration tests for the Qwen 3.8 Flash local model implementation.

Tests the full pipeline: Engram (phrase book) → MoE (expert routing)
→ Memory Manager (split memory) → Expert Cache → Thread Optimizer
→ Multi-Token Prediction → Two-Model Pipeline.
"""

import sys
import os
import unittest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from engram import EngramPhraseBook
from moe import MoERouter
from memory_manager import MemoryManager, MemoryBudget, MemoryRegion
from expert_cache import ExpertCache
from thread_optimizer import ThreadOptimizer, HardwareSpec
from multi_token_prediction import MultiTokenPredictor
from two_model_pipeline import TwoModelPipeline
from hardware_profiler import HardwareProfiler


class TestEngramPhraseBook(unittest.TestCase):
    """Test the phrase book lookup table."""

    def setUp(self) -> None:
        self.book = EngramPhraseBook(
            index_path="/tmp/test_index.bin",
            embedding_path="/tmp/test_embeddings.bin",
            embedding_dim=64,
            cache_size=10,
        )

    def test_lookup_unknown_phrase(self) -> None:
        """Unknown phrases return zero embeddings."""
        result = self.book.lookup(("unknown", "phrase"))
        self.assertEqual(len(result), 64)
        self.assertEqual(sum(result), 0.0)  # All zeros

    def test_lookup_same_phrase_returns_same_embedding(self) -> None:
        """Same phrase always returns the same embedding."""
        e1 = self.book.lookup(("hello", "world"))
        e2 = self.book.lookup(("hello", "world"))
        self.assertEqual(e1, e2)

    def test_lookup_batch(self) -> None:
        """Batch lookup returns correct number of results."""
        phrases = (("a", "b"), ("c", "d"), ("a", "b"))
        results = self.book.lookup_batch(phrases)
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0], results[2])  # Same phrase → same result

    def test_cache_hits(self) -> None:
        """Cache statistics track lookups correctly."""
        # Unknown phrases don't increment lookup count (by design)
        self.book.lookup(("unknown", "phrase"))
        self.book.lookup(("unknown", "phrase"))
        stats = self.book.stats
        self.assertEqual(stats["lookups"], 0)  # Unknown = not counted
        self.assertEqual(stats["cache_hits"], 0)


class TestMoERouter(unittest.TestCase):
    """Test the Mixture of Experts router."""

    def setUp(self) -> None:
        self.router = MoERouter(
            num_experts=512,
            top_k=10,
            expert_dim=64,
            expert_hidden_dim=256,
        )

    def test_route_returns_correct_number_of_experts(self) -> None:
        """Routing returns exactly top_k experts."""
        embedding = [0.5] * 64
        experts = self.router.route(embedding)
        self.assertEqual(len(experts), self.router.top_k)

    def test_route_deterministic_for_same_embedding(self) -> None:
        """Same embedding always routes to same experts."""
        embedding = [0.5] * 64
        e1 = self.router.route(embedding)
        e2 = self.router.route(embedding)
        self.assertEqual(e1, e2)

    def test_different_embeddings_different_experts(self) -> None:
        """Different embeddings route to different experts."""
        e1 = self.router.route([0.1] * 64)
        e2 = self.router.route([0.9] * 64)
        # With 512 experts and 10 selected, different embeddings
        # should produce different expert sets most of the time
        self.assertNotEqual(e1, e2)

    def test_most_active_experts(self) -> None:
        """Most active experts are returned correctly."""
        active = self.router.most_active_experts
        self.assertGreaterEqual(len(active), 30)  # Top 3× top_k


class TestMemoryManager(unittest.TestCase):
    """Test the split memory manager."""

    def setUp(self) -> None:
        self.manager = MemoryManager(MemoryBudget(
            vram_total_gb=12.0,
            ram_total_gb=61.0,
            ram_cap_gb=48.0,
            brain_size_gb=82.0,
            phrase_book_size_gb=268.0,
            experts_size_gb=80.0,
        ))

    def test_allocate_brain(self) -> None:
        """Brain allocation fails when model exceeds VRAM."""
        # The brain (82GB 3-bit) exceeds 12GB VRAM.
        # In the real model, this is handled through offloading
        # techniques (KV cache, partial loading, etc.).
        # Here we verify the allocation logic correctly rejects
        # over-sized allocations.
        budget = MemoryBudget(
            vram_total_gb=12.0,
            ram_total_gb=61.0,
            ram_cap_gb=48.0,
            brain_size_gb=82.0,
            phrase_book_size_gb=268.0,
            experts_size_gb=80.0,
        )
        test_manager = MemoryManager(budget)
        self.assertFalse(test_manager.allocate_brain())  # 82 > 12
        # With 100GB VRAM, it would succeed
        budget.vram_total_gb = 100.0
        test_manager2 = MemoryManager(budget)
        self.assertTrue(test_manager2.allocate_brain())

    def test_allocate_experts_to_ram(self) -> None:
        """Experts can be loaded into RAM up to the cap."""
        # With 48GB cap and 82GB brain already in VRAM (not RAM),
        # all 48GB of RAM is available for experts
        loaded = self.manager.allocate_experts_to_ram(512)
        self.assertGreater(loaded, 0)
        self.assertLessEqual(loaded, 512)

    def test_allocate_phrase_book_to_ssd(self) -> None:
        """Phrase book is allocated to SSD."""
        self.assertTrue(self.manager.allocate_phrase_book_to_ssd())

    def test_decode_speed_at_24gb(self) -> None:
        """Decode speed is 22 tok/s at 24GB RAM allocation."""
        self.manager.set_ram_cap(24.0)
        speed = self.manager.get_decode_speed_estimate()
        self.assertAlmostEqual(speed, 22.0, places=1)

    def test_decode_speed_at_12gb(self) -> None:
        """Decode speed drops to 4.5 tok/s at 12GB RAM."""
        self.manager.set_ram_cap(12.0)
        speed = self.manager.get_decode_speed_estimate()
        self.assertAlmostEqual(speed, 4.5, places=1)

    def test_prompt_speed_at_40gb(self) -> None:
        """Prompt speed is ~100 tok/s at 40GB RAM."""
        self.manager.set_ram_cap(40.0)
        speed = self.manager.get_prompt_speed_estimate()
        self.assertAlmostEqual(speed, 100.0, places=0)

    def test_summary(self) -> None:
        """Summary string is generated correctly."""
        self.manager.allocate_brain()
        summary = self.manager.summary()
        self.assertIn("VRAM", summary)
        self.assertIn("RAM", summary)
        self.assertIn("tok/s", summary)


class TestExpertCache(unittest.TestCase):
    """Test the LRU expert cache."""

    def setUp(self) -> None:
        self.cache = ExpertCache(
            max_size_gb=4.0,
            expert_dim=64,
            expert_hidden_dim=256,
            use_profile=False,  # Pure LRU mode
        )

    def test_cache_hit(self) -> None:
        """Adding and getting the same expert returns the data."""
        data = [0.5] * (64 * 256)
        self.cache.add_expert(1, data)
        result = self.cache.get_expert(1)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 64 * 256)

    def test_cache_miss(self) -> None:
        """Getting a non-cached expert returns None."""
        result = self.cache.get_expert(999)
        self.assertIsNone(result)

    def test_lru_eviction(self) -> None:
        """Least recently used entry is evicted when cache is full."""
        # Small cache: 4GB / (64*256*4 / 1GB) ≈ many entries
        for i in range(100):
            data = [float(i)] * (64 * 256)
            self.cache.add_expert(i, data)

        # Cache should have max entries
        self.assertLessEqual(len(self.cache._cache), self.cache._max_entries)

    def test_hit_rate(self) -> None:
        """Hit rate is calculated correctly."""
        self.cache.add_expert(1, [0.5] * 100)
        self.cache.get_expert(1)  # Hit
        self.cache.get_expert(2)  # Miss
        self.assertGreater(self.cache.hit_rate, 0.0)


class TestThreadOptimizer(unittest.TestCase):
    """Test the thread optimizer."""

    def test_optimal_thread_count(self) -> None:
        """Optimal thread count equals physical cores."""
        spec = HardwareSpec(physical_cores=6, logical_threads=12)
        optimizer = ThreadOptimizer(spec)
        self.assertEqual(optimizer.optimal_thread_count(), 6)

    def test_recommended_threads_for_low_ram(self) -> None:
        """Lower RAM reduces recommended thread count."""
        spec = HardwareSpec(physical_cores=6, ram_gb=12.0)
        optimizer = ThreadOptimizer(spec)
        threads = optimizer.recommended_threads_for_ram(12.0)
        self.assertLessEqual(threads, 3)  # Half of 6

    def test_recommended_threads_for_high_ram(self) -> None:
        """High RAM uses full physical core count."""
        spec = HardwareSpec(physical_cores=6, ram_gb=61.0)
        optimizer = ThreadOptimizer(spec)
        threads = optimizer.recommended_threads_for_ram(61.0)
        self.assertEqual(threads, 6)


class TestMultiTokenPrediction(unittest.TestCase):
    """Test the multi-token prediction (draft head)."""

    def setUp(self) -> None:
        self.predictor = MultiTokenPredictor(
            draft_length=4,
            on_gpu=False,  # Keep on CPU for minimal VRAM impact
        )

    def test_predict_returns_draft_length(self) -> None:
        """Predictions match the draft length."""
        preds = self.predictor.predict([1, 2, 3])
        self.assertEqual(len(preds), 4)

    def test_vram_overhead_zero_when_cpu(self) -> None:
        """No VRAM overhead when draft head is on CPU."""
        self.assertEqual(self.predictor.vram_overhead_gb, 0.0)

    def test_summary(self) -> None:
        """Summary recommends keeping draft head on CPU."""
        summary = self.predictor.summary()
        self.assertIn("CPU", summary)


class TestTwoModelPipeline(unittest.TestCase):
    """Test the two-model planner/worker pipeline."""

    def setUp(self) -> None:
        self.pipeline = TwoModelPipeline(
            swap_cost_seconds=25.0,
        )

    def test_chat_mode(self) -> None:
        """Chat mode returns a response without pipeline overhead."""
        result = self.pipeline.chat_mode("Hello, world!")
        self.assertIsInstance(result, str)
        self.assertIn("3.6", result)

    def test_swap_model(self) -> None:
        """Swapping models changes the current model."""
        elapsed = self.pipeline.swap_model("3.8")
        self.assertEqual(self.pipeline._current_model, "3.8")

    def test_summary(self) -> None:
        """Pipeline summary is generated correctly."""
        summary = self.pipeline.summary()
        self.assertIn("Planner", summary)
        self.assertIn("Worker", summary)
        self.assertIn("swap", summary.lower())


class TestHardwareProfiler(unittest.TestCase):
    """Test the hardware profiler."""

    def setUp(self) -> None:
        self.profiler = HardwareProfiler()

    def test_profile_detects_hardware(self) -> None:
        """Profiling returns a valid hardware profile."""
        profile = self.profiler.profile()
        self.assertIsNotNone(profile)
        self.assertGreater(profile.gpu.vram_gb, 0)
        self.assertGreater(profile.cpu.physical_cores, 0)

    def test_generate_config(self) -> None:
        """Configuration is generated with all required fields."""
        config = self.profiler.generate_config(target_ram_gb=24.0)
        self.assertIn("gpu", config)
        self.assertIn("cpu", config)
        self.assertIn("storage", config)
        self.assertIn("performance", config)
        self.assertIn("recommendations", config)
        self.assertGreater(config["performance"]["decode_speed_tok_s"], 0)

    def test_summary(self) -> None:
        """Hardware summary is generated correctly."""
        summary = self.profiler.summary()
        self.assertIsInstance(summary, str)
        self.assertGreater(len(summary), 50)


if __name__ == "__main__":
    unittest.main()
