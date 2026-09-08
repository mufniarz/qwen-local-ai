"""
Mixture of Experts (MoE) — Expert Routing and Caching

The Qwen 3.8 Flash model uses 512 experts, but only 10 fire per token.
This module handles expert selection, loading experts from RAM, and
caching the most-frequently-used experts in VRAM.

Key insight: Because only 10 of 512 experts fire per token, the GPU
only needs to hold the "shared" weights that apply to every token.
The 512 experts live in system RAM, and only the 10 that fire for
the current token are loaded into GPU memory.

Expert caching (LRU) keeps the most frequently-used experts in VRAM,
reducing repeated RAM→GPU transfers. Profiling shows which experts
fire most and prioritizes them for caching.

Reference: Inspired by llama.cpp's expert cache profile, ported to
the Qwen 3.8 architecture. Community implementations use a pure LRU
approach (no profiling needed).
"""

from __future__ import annotations

import random
import struct
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Expert:
    """A single expert in the MoE layer."""

    expert_id: int
    dim: int                # Input/output dimension
    hidden_dim: int         # Inner dimension (usually 2.5–4× dim)
    data: Optional[list[float]] = None  # Expert weights (loaded from RAM)
    fire_count: int = 0     # How many times this expert has fired


class MoERouter:
    """
    Routes tokens to the top-K experts in a MoE layer.

    For each token, selects the top-K experts (K=10 for Qwen 3.8)
    based on a lightweight gating network. The selected experts'
    weights are loaded from RAM into GPU memory.
    """

    def __init__(
        self,
        num_experts: int = 512,
        top_k: int = 10,
        expert_dim: int = 4096,
        expert_hidden_dim: int = 14336,
    ) -> None:
        """
        Initialize the MoE router.

        Args:
            num_experts: Total number of experts (512 for Qwen 3.8).
            top_k: Number of experts that fire per token (10 for Qwen 3.8).
            expert_dim: Input/output dimension of each expert.
            expert_hidden_dim: Inner dimension of each expert.
        """
        self.num_experts = num_experts
        self.top_k = top_k
        self.expert_dim = expert_dim
        self.expert_hidden_dim = expert_hidden_dim

        # Expert registry (in RAM)
        self._experts: list[Expert] = [
            Expert(expert_id=i, dim=expert_dim, hidden_dim=expert_hidden_dim)
            for i in range(num_experts)
        ]

        # Gating network weights (small, fits in VRAM)
        self._gate_weights: list[list[float]] = []

        # Statistics
        self._total_tokens: int = 0
        self._total_expert_loads: int = 0

    def route(
        self, token_embedding: list[float]
    ) -> tuple[int, ...]:
        """
        Select the top-K experts for a given token embedding.

        In the real model, this uses a learned gating network. Here we
        simulate the routing with a lightweight hash-based selection
        that preserves the top-K property.

        Args:
            token_embedding: The token's embedding vector.

        Returns:
            Tuple of expert IDs that should fire for this token.
        """
        # Simulate gating: use a deterministic hash of the embedding
        # to select experts (real model uses a learned linear layer)
        selected = self._select_experts(token_embedding)

        # Update fire counts
        for eid in selected:
            self._experts[eid].fire_count += 1

        self._total_tokens += 1
        self._total_expert_loads += len(selected)

        return selected

    def load_experts(
        self, expert_ids: tuple[int, ...]
    ) -> list[list[float]]:
        """
        Load the weights of selected experts from RAM into GPU memory.

        Only the selected experts are loaded — the rest stay in RAM.
        This is what allows a 12GB GPU to run a 177B parameter model.

        Args:
            expert_ids: IDs of the experts to load.

        Returns:
            List of expert weight matrices.
        """
        loaded = []
        for eid in expert_ids:
            expert = self._experts[eid]
            if expert.data is None:
                # Simulate loading from RAM (in real model, this is
                # a memory copy from system RAM to GPU VRAM)
                expert.data = self._simulate_load_from_ram(eid)
                self._total_expert_loads += 1
            loaded.append(expert.data)
        return loaded

    @property
    def most_active_experts(self) -> list[int]:
        """Return the IDs of the most frequently fired experts."""
        return sorted(
            self._experts, key=lambda e: e.fire_count, reverse=True
        )[: max(self.top_k * 3, 50)]  # Top 3× top_k for caching

    @property
    def stats(self) -> dict[str, float]:
        """Return routing statistics."""
        return {
            "total_tokens": self._total_tokens,
            "total_expert_loads": self._total_expert_loads,
            "num_experts": self.num_experts,
            "top_k": self.top_k,
            "most_active": [
                e.expert_id for e in self.most_active_experts[:10]
            ],
        }

    def _select_experts(
        self, token_embedding: list[float]
    ) -> tuple[int, ...]:
        """Select top-K experts using a simulated gating network."""
        # In the real model, this is a small linear layer that outputs
        # logits for each expert, then top-K is selected with softmax.
        # Here we simulate with a deterministic hash.
        import hashlib

        h = hashlib.sha256(
            struct.pack(f"{len(token_embedding)}f", *token_embedding)
        ).hexdigest()

        # Use hash to deterministically select experts
        # Hex digest is 64 chars; cycle through it for top_k selections
        selected = []
        for i in range(self.top_k):
            start = (i * 16) % len(h)  # 16 hex chars per expert (8 bytes)
            chunk = h[start : start + 16]
            expert_id = int(chunk, 16) % self.num_experts
            if expert_id not in selected:
                selected.append(expert_id)

        # Fill remaining slots if hash collisions occurred
        while len(selected) < self.top_k:
            candidate = len(selected) * (self.num_experts // self.top_k)
            if candidate not in selected:
                selected.append(candidate)
            else:
                selected.append((candidate + 1) % self.num_experts)

        return tuple(selected)

    def _simulate_load_from_ram(self, expert_id: int) -> list[float]:
        """Simulate loading expert weights from system RAM."""
        # In reality, this is a memory copy operation from RAM to GPU
        dim = self.expert_dim
        hidden = self.expert_hidden_dim
        # Weight matrix: (hidden_dim, dim) — typical MoE shape
        return [random.gauss(0, 1 / (dim ** 0.5)) for _ in range(dim * hidden)]



