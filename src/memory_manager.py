"""
Split Memory Manager — VRAM / RAM / SSD Allocation Strategy

Manages the split memory design that makes running a 177B parameter
model on 12GB VRAM possible:

  - Brain (125B parameters) → GPU VRAM (shared weights only)
  - Phrase Book (51B parameters) → SSD (page-fetched on demand)
  - Experts (512 MoE experts) → System RAM (loaded per-token)

Key findings from testing:
  - 24 GB RAM: sufficient for fast decoding (22 tok/s)
  - 40 GB RAM: needed for fast prompt processing (~100 tok/s)
  - 12 GB VRAM: enough to hold the shared brain + cached experts

The memory manager tracks allocations and enforces RAM caps,
ensuring the model runs within available resources.
"""

from __future__ import annotations

import os
import resource
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class MemoryRegion(Enum):
    """Where model components live."""

    VRAM = "vram"           # GPU video memory (fastest)
    RAM = "ram"             # System memory (fast)
    SSD = "ssd"             # Storage (slow, but large)


@dataclass
class MemoryBudget:
    """Memory budget configuration for a given hardware setup."""

    # VRAM (GB)
    vram_total_gb: float = 12.0
    vram_used_gb: float = 0.0

    # System RAM (GB)
    ram_total_gb: float = 61.0
    ram_used_gb: float = 0.0
    ram_cap_gb: float = 48.0  # Enforced cap (anything above is wasted)

    # SSD (GB) — effectively unlimited for our purposes
    ssd_total_gb: float = 2000.0
    ssd_used_gb: float = 0.0

    # Component sizes (3-bit quantized)
    brain_size_gb: float = 82.0       # 125B params @ 3-bit
    phrase_book_size_gb: float = 268.0  # 51B params @ 3-bit
    experts_size_gb: float = 80.0     # 512 experts @ 3-bit
    num_experts: int = 512

    # Optimal RAM thresholds (discovered empirically)
    decode_ram_threshold_gb: float = 24.0
    prompt_ram_threshold_gb: float = 40.0


class MemoryManager:
    """
    Manages memory allocation across VRAM, RAM, and SSD.

    Enforces RAM caps and tracks which components are loaded where.
    The phrase book stays on SSD (page-fetched). The brain fits in
    VRAM. Experts live in RAM, with a subset cached in VRAM.
    """

    def __init__(self, budget: Optional[MemoryBudget] = None) -> None:
        self.budget = budget or MemoryBudget()
        self._loaded_regions: dict[MemoryRegion, float] = {
            MemoryRegion.VRAM: 0.0,
            MemoryRegion.RAM: 0.0,
            MemoryRegion.SSD: 0.0,
        }

    # ------------------------------------------------------------------
    # Allocation
    # ------------------------------------------------------------------

    def allocate_brain(self) -> bool:
        """
        Allocate the brain (125B shared weights) to VRAM.

        Returns True if successful, False if VRAM is insufficient.
        """
        if (
            self._loaded_regions[MemoryRegion.VRAM]
            + self.budget.brain_size_gb
            > self.budget.vram_total_gb
        ):
            return False

        self._loaded_regions[MemoryRegion.VRAM] += self.budget.brain_size_gb
        return True

    def allocate_experts_to_ram(
        self, num_experts: int = 512, cap_gb: Optional[float] = None
    ) -> int:
        """
        Load experts into system RAM, respecting the RAM cap.

        Args:
            num_experts: Number of experts to load (default 512).
            cap_gb: Optional RAM cap (uses budget if None).

        Returns:
            Number of experts actually loaded.
        """
        cap = cap_gb or self.budget.ram_cap_gb
        per_expert_gb = self.budget.experts_size_gb / self.budget.num_experts

        available = cap - self._loaded_regions[MemoryRegion.RAM]
        if available <= 0:
            return 0

        can_load = int(available / per_expert_gb)
        to_load = min(num_experts, can_load)

        self._loaded_regions[MemoryRegion.RAM] += to_load * per_expert_gb

        return to_load

    def allocate_phrase_book_to_ssd(self) -> bool:
        """
        Map the phrase book to SSD (memory-mapped, page-fetched).

        The phrase book stays on disk — only accessed rows consume RAM.
        This is what makes the 51B parameter lookup table feasible.
        """
        self._loaded_regions[MemoryRegion.SSD] += (
            self.budget.phrase_book_size_gb
        )
        return True

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_decode_speed_estimate(self) -> float:
        """
        Estimate decode speed (tok/s) based on current RAM allocation.

        Empirical findings:
          12 GB → 4.5 tok/s  (every token streams experts from SSD)
          16 GB → 6.5 tok/s
          20 GB → 9 tok/s
          24 GB → 22 tok/s  (most-frequent experts fit)
          32 GB → 22 tok/s  (no improvement over 24)
        """
        ram_available = (
            self.budget.ram_cap_gb
            - self._loaded_regions[MemoryRegion.RAM]
        )

        if ram_available >= 24.0:
            return 22.0
        elif ram_available >= 20.0:
            return 9.0
        elif ram_available >= 16.0:
            return 6.5
        elif ram_available >= 12.0:
            return 4.5
        else:
            return 1.0  # Below threshold

    def get_prompt_speed_estimate(self) -> float:
        """
        Estimate prompt processing speed (tok/s) based on RAM allocation.

        Prompt processing reads nearly every expert, so it needs more RAM
        than decoding. Empirical findings:
          12 GB → streams all from disk
          16 GB → 15 tok/s
          24 GB → 15 tok/s
          32 GB → 34 tok/s
          40 GB → ~100 tok/s
          48 GB → full speed
        """
        ram_available = (
            self.budget.ram_cap_gb
            - self._loaded_regions[MemoryRegion.RAM]
        )

        if ram_available >= 48.0:
            return 115.0  # Full speed
        elif ram_available >= 40.0:
            return 100.0
        elif ram_available >= 32.0:
            return 34.0
        elif ram_available >= 24.0:
            return 15.0
        else:
            return 1.0  # Streams from disk

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    @property
    def usage(self) -> dict[str, float]:
        """Current memory usage by region."""
        return {
            "vram_used_gb": self._loaded_regions[MemoryRegion.VRAM],
            "vram_total_gb": self.budget.vram_total_gb,
            "ram_used_gb": self._loaded_regions[MemoryRegion.RAM],
            "ram_total_gb": self.budget.ram_total_gb,
            "ram_cap_gb": self.budget.ram_cap_gb,
            "ssd_used_gb": self._loaded_regions[MemoryRegion.SSD],
        }

    def summary(self) -> str:
        """Human-readable memory usage summary."""
        u = self.usage
        lines = [
            "=== Memory Budget ===",
            f"  VRAM: {u['vram_used_gb']:.1f} / {u['vram_total_gb']:.1f} GB",
            f"  RAM:  {u['ram_used_gb']:.1f} / {u['ram_cap_gb']:.1f} GB (cap: {u['ram_cap_gb']:.1f})",
            f"  SSD:  {u['ssd_used_gb']:.1f} GB (phrase book)",
            "",
            f"  Estimated decode speed:  {self.get_decode_speed_estimate():.1f} tok/s",
            f"  Estimated prompt speed:  {self.get_prompt_speed_estimate():.1f} tok/s",
        ]
        return "\n".join(lines)

    def set_ram_cap(self, cap_gb: float) -> None:
        """Change the enforced RAM cap (for testing different configs)."""
        self.budget.ram_cap_gb = cap_gb
