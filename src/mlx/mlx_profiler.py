"""
MLX Hardware Profiler — Real Apple Silicon Detection

Detects the actual hardware running this code (Apple Silicon chip,
unified memory, core counts, Metal device) and maps it onto:

  1. The real model catalog — which real Qwen/DeepSeek models actually
     fit in the available unified memory, with an honest fit verdict.
  2. The simulation layer — translates the unified-memory reality into
     a ``MemoryBudget`` for the ``src.memory_manager`` simulation, since
     on Apple Silicon the VRAM/RAM split collapses into one pool.

The transcript's hardware (RTX 3060 + DDR4) is a discrete-GPU setup:
12GB VRAM + system RAM + SSD. On an M1 Max there is a single unified
memory pool (this machine: 64GB) shared by CPU and Metal GPU, so the
"brain in VRAM, experts in RAM, phrase book on SSD" split becomes:

    unified memory → model weights (quantized) + KV cache + OS
    SSD           → phrase book (engram), still page-fetched

``recommend()`` returns honest verdicts — including that the 177B
hypothetical "Qwen 3.8 Flash" does NOT fit in 64GB — and points at
real, downloadable models that implement the same architecture.

Works on any OS; on non-Apple hardware it still profiles what it can.
"""

from __future__ import annotations

import platform
import subprocess
from dataclasses import dataclass, field
from typing import Optional

# ----------------------------------------------------------------------
# Hardware data
# ----------------------------------------------------------------------


@dataclass
class AppleHardware:
    """Profiled hardware of the current machine."""

    is_apple_silicon: bool = False
    chip: str = ""                # e.g. "Apple M1 Max"
    machine: str = ""             # e.g. "MacBookPro18,2"
    system: str = ""              # e.g. "Darwin" / "Linux"
    arch: str = ""                # e.g. "arm64"
    physical_cores: int = 0
    logical_cores: int = 0
    unified_memory_gb: float = 0.0
    metal: bool = False           # MLX/Metal backend available (importable)
    mlx_version: str = ""


@dataclass
class RealModel:
    """A real, downloadable model from the Hugging Face MLX community."""

    name: str                     # Human name, e.g. "Qwen3 30B-A3B"
    repo: str                     # HF repo id, e.g. "mlx-community/Qwen3-30B-A3B-4bit"
    total_params_b: float         # Total parameters (billions)
    active_params_b: float        # Active parameters per token (MoE)
    size_gb_4bit: float           # Approx weight size on disk / in memory (4-bit)
    is_moe: bool = False
    num_experts: int = 0
    top_k: int = 0
    has_engram: bool = False      # Ships with a real conditional-memory (engram) table
    notes: str = ""


#: The real-model catalog. Sizes are approximate (4-bit MLX quants).
#: Qwen3-Next-80B-A3B is the closest *real* match to the transcript's
#: architecture: hybrid linear attention + MoE, 3B active per token.
REAL_MODEL_CATALOG: list[RealModel] = [
    RealModel(
        name="Qwen3 0.6B (dense)",
        repo="mlx-community/Qwen3-0.6B-4bit",
        total_params_b=0.6, active_params_b=0.6, size_gb_4bit=0.5,
        notes="Tiny dense model — ideal for validating the full bridge end-to-end.",
    ),
    RealModel(
        name="Qwen3 30B-A3B (MoE)",
        repo="mlx-community/Qwen3-30B-A3B-4bit",
        total_params_b=30, active_params_b=3.3, size_gb_4bit=19.0,
        is_moe=True, num_experts=128, top_k=8,
        notes="Real MoE: 128 experts, 8 fire per token, 3.3B active. "
              "The transcript's '512 experts / 10 active' pattern at real scale.",
    ),
    RealModel(
        name="Qwen3 Next 80B-A3B (hybrid + MoE)",
        repo="mlx-community/Qwen3-Next-80B-A3B-4bit",
        total_params_b=80, active_params_b=3.0, size_gb_4bit=48.0,
        is_moe=True, num_experts=512, top_k=10,
        notes="Closest real analog of the transcript design: 512 experts, "
              "10 active per token, hybrid gated-linear attention. "
              "Fits 64GB but leaves little headroom for long contexts.",
    ),
    RealModel(
        name="Qwen3 235B-A22B (MoE)",
        repo="mlx-community/Qwen3-235B-A22B-4bit",
        total_params_b=235, active_params_b=22.0, size_gb_4bit=131.0,
        is_moe=True, num_experts=128, top_k=8,
        notes="Largest mainstream Qwen MoE. Needs ~150GB+ unified memory.",
    ),
    RealModel(
        name="DeepSeek V3.2 Exp (MoE + engram)",
        repo="mlx-community/DeepSeek-V3.2-Exp-4bit",
        total_params_b=685, active_params_b=37.0, size_gb_4bit=400.0,
        is_moe=True, num_experts=256, top_k=8, has_engram=True,
        notes="Real conditional-memory (engram) table inside the checkpoint. "
              "The reference for the 'phrase book' concept at production scale.",
    ),
]

#: The hypothetical 177B model from the transcript — included so the
#: profiler can report an honest "does not fit" verdict.
TRANSCRIPT_MODEL_177B = RealModel(
    name="Qwen 3.8 Flash (transcript, hypothetical)",
    repo="(no public open weights)",
    total_params_b=177, active_params_b=125, size_gb_4bit=160.0,
    is_moe=True, num_experts=512, top_k=10, has_engram=True,
    notes="125B brain + 51B engram phrase book + 512 experts (10 active). "
          "~82GB at 3-bit for the brain alone. No downloadable weights exist; "
          "the mlx bridge targets the real models above until/if it is released.",
)


# ----------------------------------------------------------------------
# Detection
# ----------------------------------------------------------------------


def _sysctl(key: str) -> str:
    """Read a sysctl value (macOS/BSD). Returns '' on failure or other OSes."""
    try:
        return subprocess.run(
            ["sysctl", "-n", key], capture_output=True, text=True, timeout=5
        ).stdout.strip()
    except Exception:
        return ""


def profile_apple_silicon() -> AppleHardware:
    """
    Profile the machine this code is running on.

    Detects Apple Silicon via ``platform`` + ``sysctl``, checks unified
    memory, and probes whether the MLX/Metal backend is importable.
    Never raises — fields default to safe values so the bridge works
    (in degraded mode) on machines without MLX.
    """
    hw = AppleHardware(
        system=platform.system(),
        arch=platform.machine(),
        machine=_sysctl("hw.model") or platform.node(),
    )

    # Apple Silicon: arm64 hardware. (Darwin + arm64 implies Apple Silicon;
    # Linux arm64 can be real Apple Silicon too — report it, but Metal no.)
    is_macos = hw.system == "Darwin"
    is_arm = hw.arch in ("arm64", "aarch64")
    hw.is_apple_silicon = is_macos and is_arm

    chip = _sysctl("machdep.cpu.brand_string")
    hw.chip = chip or platform.processor() or "unknown"

    mem_bytes = _sysctl("hw.memsize")
    if mem_bytes:
        hw.unified_memory_gb = int(mem_bytes) / (1024 ** 3)
    else:
        # Non-macOS fallback: /proc/meminfo
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        hw.unified_memory_gb = (
                            int(line.split()[1]) * 1024 / (1024 ** 3)
                        )
                        break
        except Exception:
            pass

    phys = _sysctl("hw.physicalcpu")
    logic = _sysctl("hw.logicalcpu")
    if phys:
        hw.physical_cores = int(phys)
    if logic:
        hw.logical_cores = int(logic)
    if hw.physical_cores == 0:
        hw.physical_cores = platform.cpu_count() or 0
        hw.logical_cores = hw.physical_cores

    # Probe the MLX / Metal backend (optional dependency)
    try:
        import mlx.core as mx  # type: ignore

        hw.metal = mx.default_device() is not None
        try:
            hw.mlx_version = str(getattr(mx, "__version__", ""))
        except Exception:
            pass
    except Exception:
        hw.metal = False

    return hw


# ----------------------------------------------------------------------
# Model fit + recommendations
# ----------------------------------------------------------------------


def fits_in_unified_memory(
    model: RealModel,
    unified_gb: float,
    kv_cache_gb: float = 8.0,
    os_reserve_gb: float = 6.0,
) -> bool:
    """
    Does ``model`` fit, leaving room for KV cache and the OS?

    ``kv_cache_gb`` is a working budget for activations + KV cache;
    ``os_reserve_gb`` keeps the machine usable while the model loads.
    """
    return model.size_gb_4bit <= max(
        0.0, unified_gb - kv_cache_gb - os_reserve_gb
    )


def recommend(
    hw: Optional[AppleHardware] = None,
    kv_cache_gb: float = 8.0,
) -> list[tuple[RealModel, bool]]:
    """
    Rank the real-model catalog against this machine's unified memory.

    Returns (model, fits) pairs, largest first, with an honest verdict
    for each — including the transcript's hypothetical 177B model.
    """
    hw = hw or profile_apple_silicon()
    candidates = REAL_MODEL_CATALOG + [TRANSCRIPT_MODEL_177B]
    ranked = sorted(candidates, key=lambda m: m.total_params_b, reverse=True)
    return [(m, fits_in_unified_memory(m, hw.unified_memory_gb, kv_cache_gb))
            for m in ranked]


def best_fit(hw: Optional[AppleHardware] = None) -> Optional[RealModel]:
    """The largest real (downloadable) model this machine can run."""
    hw = hw or profile_apple_silicon()
    for model, fits in recommend(hw):
        if fits and model.repo != "(no public open weights)":
            return model
    return None


# ----------------------------------------------------------------------
# Bridge to the simulation layer
# ----------------------------------------------------------------------


def to_memory_budget(hw: Optional[AppleHardware] = None) -> dict:
    """
    Translate the real unified-memory profile into the keyword arguments
    of ``src.memory_manager.MemoryBudget``.

    On Apple Silicon the VRAM and RAM regions are the same physical pool;
    we still report both (VRAM = what the Metal GPU can address ≈ total,
    RAM = the same pool, with the OS reserve carved out of the cap).
    """
    hw = hw or profile_apple_silicon()
    total = hw.unified_memory_gb or 16.0
    os_reserve = min(8.0, max(2.0, total * 0.1))
    model_reserve = min(total * 0.75, 48.0)  # weights + KV cache budget

    return {
        "vram_total_gb": total,
        "ram_total_gb": total,
        "ram_cap_gb": max(0.0, total - os_reserve - model_reserve),
        "ssd_total_gb": 2000.0,
        "num_experts": 512,
    }


def summary(hw: Optional[AppleHardware] = None) -> str:
    """Human-readable hardware + model-fit summary for display."""
    hw = hw or profile_apple_silicon()
    lines = [
        f"  Chip:            {hw.chip or 'unknown'} ({hw.machine})",
        f"  Architecture:    {hw.arch}"
        + ("  [Apple Silicon]" if hw.is_apple_silicon else ""),
        f"  Cores:           {hw.physical_cores} physical / {hw.logical_cores} logical",
        f"  Unified memory:  {hw.unified_memory_gb:.0f} GB"
        if hw.unified_memory_gb else "  Unified memory:  unknown",
        f"  Metal / MLX:     {'yes (' + hw.mlx_version + ')' if hw.metal else 'not installed (simulation only)'}",
        "",
        "  Model fit (4-bit weights, 8GB KV cache, 6GB OS reserve):",
    ]
    for model, fits in recommend(hw):
        mark = "OK  " if fits else "TOO BIG"
        suffix = f"  [{model.num_experts} experts, top-{model.top_k}]" if model.is_moe else ""
        engram = "  [engram]" if model.has_engram else ""
        size = f"{model.size_gb_4bit:.0f}" if model.size_gb_4bit >= 10 else f"{model.size_gb_4bit:.1f}"
        lines.append(
            f"    {mark} {model.name:40s} {size:>6s} GB{suffix}{engram}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    print("MLX Hardware Profiler — current machine")
    print("=" * 60)
    print(summary())
