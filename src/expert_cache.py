"""
Expert Cache — LRU Caching of Frequently-Fired Experts in VRAM

After the brain is loaded into VRAM, there's typically some leftover
space. This module caches the most frequently-used experts in that
remaining VRAM, avoiding repeated RAM→GPU transfers for every token.

Two approaches are supported:
  1. Profile-based: Run a profiling pass to discover which experts fire
     most, then cache those.
  2. Pure LRU: No profiling needed — just keep whatever fired recently.

The pure LRU approach (used by community forks) often matches or exceeds
profile-based performance and works on any hardware configuration.

Reference: Inspired by llama.cpp's expert cache profile, and community
implementations using a simple LRU cache.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CachedExpert:
    """An expert cached in VRAM."""

    expert_id: int
    data: list[float]  # Expert weight matrix (flattened)
    last_accessed: int  # Token counter when last accessed


class ExpertCache:
    """
    LRU cache for MoE experts in VRAM.

    Keeps the N most recently-used experts in VRAM to avoid repeated
    RAM→GPU transfers. Supports both profile-based and pure LRU modes.
    """

    def __init__(
        self,
        max_size_gb: float = 4.0,
        expert_dim: int = 4096,
        expert_hidden_dim: int = 14336,
        use_profile: bool = False,
        profiled_experts: Optional[list[int]] = None,
    ) -> None:
        """
        Initialize the expert cache.

        Args:
            max_size_gb: Maximum VRAM reserved for the cache.
            expert_dim: Input/output dimension of each expert.
            expert_hidden_dim: Inner dimension of each expert.
            use_profile: If True, cache only profiled experts.
                         If False, use pure LRU (no profiling needed).
            profiled_experts: List of expert IDs to cache (profile-based mode).
        """
        self.max_size_gb = max_size_gb
        self.expert_dim = expert_dim
        self.expert_hidden_dim = expert_hidden_dim
        self.use_profile = use_profile

        # Size per expert in GB (float32 weights)
        self._per_expert_gb = (
            expert_dim * expert_hidden_dim * 4 / (1024 ** 3)
        )
        self._max_entries = int(max_size_gb / self._per_expert_gb)

        # LRU cache: expert_id → CachedExpert
        self._cache: OrderedDict[int, CachedExpert] = OrderedDict()

        # Profiled experts (if using profile-based mode)
        self._profiled_experts: set[int] = (
            set(profiled_experts) if profiled_experts else set()
        )

        # Statistics
        self._hits: int = 0
        self._misses: int = 0
        self._token_count: int = 0

    def get_expert(self, expert_id: int) -> Optional[list[float]]:
        """
        Get a cached expert, loading from RAM if a cache miss.

        Args:
            expert_id: ID of the expert to retrieve.

        Returns:
            Expert weight matrix if found (or cached), None otherwise.
        """
        self._token_count += 1

        # In profile-based mode, only cache profiled experts
        if self.use_profile and expert_id not in self._profiled_experts:
            self._misses += 1
            return None

        if expert_id in self._cache:
            # Cache hit — move to end (most recently used)
            self._cache.move_to_end(expert_id)
            self._cache[expert_id].last_accessed = self._token_count
            self._hits += 1
            return self._cache[expert_id].data

        # Cache miss — expert not in VRAM
        self._misses += 1
        return None

    def add_expert(
        self, expert_id: int, data: list[float]
    ) -> None:
        """
        Add (or update) an expert in the cache.

        If the cache is full, evicts the least-recently-used expert.

        Args:
            expert_id: ID of the expert.
            data: Expert weight matrix (flattened).
        """
        if expert_id in self._cache:
            self._cache.move_to_end(expert_id)
            self._cache[expert_id].data = data
            self._cache[expert_id].last_accessed = self._token_count
            return

        # Evict LRU entry if cache is full
        while len(self._cache) >= self._max_entries:
            evicted_id, _ = self._cache.popitem(last=False)

        self._cache[expert_id] = CachedExpert(
            expert_id=expert_id,
            data=data,
            last_accessed=self._token_count,
        )

    def set_profiled_experts(self, expert_ids: list[int]) -> None:
        """
        Set the list of profiled experts (profile-based mode).

        Only these experts will be cached. All others will always
        result in a cache miss (loaded from RAM each time).

        Args:
            expert_ids: List of expert IDs to cache.
        """
        self._profiled_experts = set(expert_ids)

    @property
    def hit_rate(self) -> float:
        """Cache hit rate (0.0–1.0)."""
        total = self._hits + self._misses
        return self._hits / max(total, 1)

    @property
    def stats(self) -> dict[str, float]:
        """Cache statistics."""
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self.hit_rate,
            "cached_entries": len(self._cache),
            "max_entries": self._max_entries,
            "cached_gb": len(self._cache) * self._per_expert_gb,
        }

    def summary(self) -> str:
        """Human-readable cache summary."""
        s = self.stats
        return (
            f"Expert Cache: {s['cached_entries']}/{s['max_entries']} entries "
            f"({s['cached_gb']:.1f} GB), "
            f"hit rate: {s['hit_rate']:.1%}"
        )
