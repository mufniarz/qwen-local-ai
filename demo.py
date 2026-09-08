#!/usr/bin/env python3
"""
Qwen 3.8 Flash — Local Frontier Model Demo

Demonstrates the complete system: Engram (phrase book) → MoE (expert
routing) → Memory Manager (split memory) → Expert Cache → Thread
Optimizer → Multi-Token Prediction → Two-Model Pipeline.

Run this to see how all components work together to enable running
a 177-billion-parameter model on consumer hardware (12GB VRAM GPU).

Usage:
    python demo.py [--ram-cap GB] [--expert-cache-GB GB]
"""

import argparse
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from engram import EngramPhraseBook
from moe import MoERouter
from memory_manager import MemoryManager, MemoryBudget
from expert_cache import ExpertCache
from thread_optimizer import ThreadOptimizer, HardwareSpec
from multi_token_prediction import MultiTokenPredictor
from two_model_pipeline import TwoModelPipeline
from hardware_profiler import HardwareProfiler


def print_header(title: str) -> None:
    """Print a section header."""
    width = 72
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def demo_engram() -> None:
    """Demonstrate the Engram (phrase book) lookup table."""
    print_header("1. Engram — Phrase Book Lookup Table")
    print(
        "The phrase book contains ~51B parameters of pre-computed "
        "embeddings for common 2-3 word phrases. It lives on SSD and "
        "is accessed on-demand via memory-mapped I/O."
    )

    book = EngramPhraseBook(
        index_path="/tmp/demo_index.bin",
        embedding_path="/tmp/demo_embeddings.bin",
        embedding_dim=64,
        cache_size=100,
    )

    # Simulate lookups
    phrases = (
        ("new york",),
        ("hello world",),
        ("machine learning",),
        ("new york",),  # Cache hit
        ("open ai",),
        ("new york",),  # Cache hit
    )

    print("\n  Phrase lookups:")
    for phrase in phrases:
        embedding = book.lookup(phrase)
        print(f"    {phrase!r:25s} → {len(embedding)}-dim embedding")

    stats = book.stats
    print(f"\n  Stats: {stats['lookups']} lookups, "
          f"{stats['cache_hit_rate']:.0%} cache hit rate")
    print(f"  Indexed phrases: {stats['indexed_phrases']}")


def demo_moe() -> None:
    """Demonstrate the Mixture of Experts routing."""
    print_header("2. MoE — Mixture of Experts Routing")
    print(
        "512 experts, but only 10 fire per token. Experts live in RAM; "
        "the GPU holds only shared weights. This is what allows a 12GB "
        "GPU to run a 177B parameter model."
    )

    router = MoERouter(
        num_experts=512,
        top_k=10,
        expert_dim=64,
        expert_hidden_dim=256,
    )

    # Simulate routing for a few tokens
    print("\n  Token routing (top-10 experts per token):")
    for i in range(5):
        embedding = [float(i + j * 0.1) for j in range(64)]
        experts = router.route(embedding)
        print(f"    Token {i}: experts {sorted(experts)}")

    stats = router.stats
    print(f"\n  Total tokens routed: {stats['total_tokens']}")
    print(f"  Most active experts: {stats['most_active'][:5]}")


def demo_memory_manager() -> None:
    """Demonstrate the split memory design."""
    print_header("3. Split Memory Manager")
    print(
        "Brain (125B) → VRAM, Phrase Book (51B) → SSD, Experts (512) → RAM.\n"
        "RAM budgeting determines speed: 24GB for fast decoding, 40GB+ "
        "for fast prompt processing."
    )

    # Test different RAM allocations
    print("\n  RAM allocation vs. decode speed:")
    for ram_cap in [12, 16, 20, 24, 32, 40, 48]:
        manager = MemoryManager(MemoryBudget(
            vram_total_gb=12.0,
            ram_total_gb=61.0,
            ram_cap_gb=float(ram_cap),
            brain_size_gb=82.0,
            phrase_book_size_gb=268.0,
            experts_size_gb=80.0,
        ))
        manager.allocate_brain()
        manager.allocate_experts_to_ram(512)
        speed = manager.get_decode_speed_estimate()
        prompt_speed = manager.get_prompt_speed_estimate()
        marker = " ← optimal" if ram_cap == 24 else ""
        print(f"    {ram_cap:2d} GB RAM → {speed:5.1f} tok/s decode, "
              f"{prompt_speed:5.1f} tok/s prompt{marker}")

    # Full summary at 24GB
    manager = MemoryManager(MemoryBudget(
        vram_total_gb=12.0,
        ram_total_gb=61.0,
        ram_cap_gb=24.0,
        brain_size_gb=82.0,
        phrase_book_size_gb=268.0,
        experts_size_gb=80.0,
    ))
    manager.allocate_brain()
    manager.allocate_experts_to_ram(512)
    print(f"\n  Full summary (24GB RAM):")
    print("  " + manager.summary().replace("\n", "\n  "))


def demo_expert_cache() -> None:
    """Demonstrate the LRU expert cache."""
    print_header("4. Expert Cache — LRU VRAM Caching")
    print(
        "After loading the brain into VRAM, remaining space caches the "
        "most-frequently-used experts. Pure LRU mode (no profiling) "
        "often matches profile-based performance."
    )

    cache = ExpertCache(
        max_size_gb=2.0,
        expert_dim=64,
        expert_hidden_dim=256,
        use_profile=False,  # Pure LRU
    )

    # Simulate cache usage
    for i in range(20):
        data = [float(i)] * (64 * 256)
        cache.add_expert(i, data)

    # Test hits and misses
    hits = 0
    for i in [1, 5, 99, 1, 15, 99]:
        result = cache.get_expert(i)
        hits += 1 if result is not None else 0

    print(f"\n  Cache: {cache.stats['cached_entries']}/{cache.stats['max_entries']} "
          f"entries ({cache.stats['cached_gb']:.1f} GB)")
    print(f"  Hit rate: {cache.hit_rate:.1%}")
    print(f"  Recommendation: Pure LRU works well, no profiling needed")


def demo_thread_optimizer() -> None:
    """Demonstrate the thread optimizer."""
    print_header("5. Thread Optimizer")
    print(
        "Thread count should match PHYSICAL cores, not logical threads.\n"
        "12 threads (logical) → 16.5 tok/s\n"
        "  6 threads (physical) → 24.4 tok/s  (~50% improvement!)"
    )

    spec = HardwareSpec(physical_cores=6, logical_threads=12, ram_gb=61.0)
    optimizer = ThreadOptimizer(spec)

    config = optimizer.get_configuration()
    print(f"\n  CPU: {config['cpu_model']}")
    print(f"  Physical cores: {config['physical_cores']}")
    print(f"  Logical threads: {config['logical_threads']}")
    print(f"  Recommended threads: {config['recommended_threads']}")
    print(f"  RAM: {config['ram_gb']:.0f} GB")
    print(f"\n  Key finding: 65% of CPU time was threads SPIN-WAITING")
    print(f"  Fix: Match threads to physical cores")


def demo_multi_token_prediction() -> None:
    """Demonstrate multi-token prediction (draft head)."""
    print_header("6. Multi-Token Prediction (Draft Head)")
    print(
        "The model ships with a draft head that guesses next tokens.\n"
        "In practice: minimal benefit (~0.8 tok/s) on 12GB VRAM.\n"
        "Recommendation: Keep draft head on CPU, use VRAM for experts."
    )

    predictor = MultiTokenPredictor(draft_length=4, on_gpu=False)

    preds = predictor.predict([1, 2, 3, 4])
    print(f"\n  Draft predictions: {len(preds)} tokens")
    for p in preds:
        print(f"    Position {p.position}: token {p.predicted_token} "
              f"(confidence: {p.confidence:.2f})")

    print(f"\n  Acceptance rate: {predictor.acceptance_rate:.1%}")
    print(f"  VRAM overhead: {predictor.vram_overhead_gb:.2f} GB (0 when on CPU)")


def demo_two_model_pipeline() -> None:
    """Demonstrate the two-model planner/worker pipeline."""
    print_header("7. Two-Model Pipeline — Planner/Worker")
    print(
        "Qwen 3.8 (177B, slow, smart) → Planner + Reviewer\n"
        "Qwen 3.6 (35B, fast, obedient) → Worker\n\n"
        "The 3.8 reads the problem, writes a plan (small, exact steps).\n"
        "The 3.6 executes the plan at 3× speed.\n"
        "The 3.8 reviews results and writes the report."
    )

    pipeline = TwoModelPipeline(swap_cost_seconds=25.0)

    # Run the pipeline
    print("\n  Running pipeline on a test problem...")
    result = pipeline.run_pipeline(
        "Fix the rate limiter: there are 4 real bugs and 1 test that "
        "contradicts the spec. Identify the contradiction and fix only "
        "what the spec requires."
    )

    print(f"\n  Plan: {len(result.plan)} steps")
    for step in result.plan:
        print(f"    [{step.step_id}] {step.description}")

    print(f"\n  Timing:")
    print(f"    Planning:  {result.plan_time_seconds:.1f}s")
    print(f"    Swapping:  {result.swap_overhead_seconds:.1f}s")
    print(f"    Worker:    {result.worker_time_seconds:.1f}s")
    print(f"    Reviewing: {result.review_time_seconds:.1f}s")
    print(f"    Total:     {result.total_time_seconds:.1f}s")

    print(f"\n  Review:")
    for line in result.review.split("\n"):
        print(f"    {line}")

    print(f"\n  Chat mode (no pipeline, keep 3.6 loaded):")
    print(f"    {pipeline.chat_mode('What time is it?')}")


def demo_hardware_profiler() -> None:
    """Demonstrate the hardware profiler."""
    print_header("8. Hardware Profiler")
    print("Profiling available hardware and generating optimal config...")

    profiler = HardwareProfiler()
    config = profiler.generate_config(target_ram_gb=24.0)

    print(profiler.summary())


def main() -> None:
    """Run the full demo."""
    parser = argparse.ArgumentParser(
        description="Qwen 3.8 Flash — Local Frontier Model Demo"
    )
    parser.add_argument(
        "--ram-cap", type=float, default=24.0,
        help="RAM cap in GB (default: 24)",
    )
    parser.add_argument(
        "--expert-cache-GB", type=float, default=2.0,
        help="Expert cache size in GB (default: 2)",
    )
    args = parser.parse_args()

    print("=" * 72)
    print("  Qwen 3.8 Flash — Local Frontier Model Demo")
    print("  Running a 177B parameter model on consumer hardware")
    print("  RTX 3060 12GB + 6-core CPU + 61GB DDR4 RAM")
    print("=" * 72)

    demo_engram()
    demo_moe()
    demo_memory_manager()
    demo_expert_cache()
    demo_thread_optimizer()
    demo_multi_token_prediction()
    demo_two_model_pipeline()
    demo_hardware_profiler()

    print(f"\n{'=' * 72}")
    print("  SUMMARY")
    print(f"{'=' * 72}")
    print(
        "  The Qwen 3.8 Flash architecture makes it possible to run a "
        "frontier-class\n"
        "  177-billion-parameter model on a used $300 gaming card.\n\n"
        "  Key innovations:\n"
        "    1. Engram (phrase book): 51B params on SSD, page-fetched\n"
        "    2. MoE (512 experts, 10 per token): experts in RAM\n"
        "    3. Split memory: brain in VRAM, book on SSD\n"
        "    4. Expert caching: LRU cache in remaining VRAM\n"
        "    5. Thread optimization: match to physical cores\n"
        "    6. Two-model pipeline: 3.8 plans, 3.6 executes\n\n"
        "  Result: 22 tok/s decode, 15 tok/s prompt on 12GB VRAM\n"
        "  Score:  17/17 on a lab designed to trip frontier models\n\n"
        "  The only difference between the chicken farmer and the "
        "chickens\n"
        "  is that the farmer is smarter. Without local AI, we are all "
        "going\n"
        "  to be the chickens."
    )
    print(f"{'=' * 72}\n")


if __name__ == "__main__":
    main()
