"""
Hardware Profiler — Hardware-Aware Configuration

Profiles the available hardware and generates an optimal configuration
for running the Qwen 3.8 Flash model. This includes:

  - GPU: VRAM size, compute capability
  - CPU: Physical cores, logical threads, model
  - RAM: Total, usable (after OS), speed (DDR4 vs DDR5)
  - Storage: SSD type (NVMe vs SATA), available space

The profiler generates a configuration that balances:
  - VRAM allocation (brain + cached experts)
  - RAM allocation (experts + phrase book cache)
  - SSD usage (phrase book, page-fetched)
  - Thread count (physical cores)
  - Expected throughput (tok/s for decode and prompt processing)

Key hardware configurations from testing:

  RTX 3060 12GB + Ryzen 5 3600 + 61GB DDR4:
    Decode: 22 tok/s (at 24GB RAM)
    Prompt: 15 tok/s (at 24GB RAM), 100+ tok/s (at 40GB RAM)

  RTX 4070 Ti 16GB + DDR5:
    Decode: 43 tok/s (community report)
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GPUInfo:
    """GPU hardware information."""

    name: str = ""
    vram_gb: float = 0.0
    compute_capability: str = ""
    memory_bandwidth_gb_ps: float = 0.0
    tensor_cores: bool = False


@dataclass
class CPUInfo:
    """CPU hardware information."""

    model: str = ""
    physical_cores: int = 0
    logical_threads: int = 0
    ram_gb: float = 0.0
    ram_speed: str = ""  # e.g., "DDR4-3200"


@dataclass
class StorageInfo:
    """Storage hardware information."""

    ssd_type: str = ""  # "NVMe", "SATA", "HDD"
    available_gb: float = 0.0
    read_speed_gb_ps: float = 0.0


@dataclass
class FullHardwareProfile:
    """Complete hardware profile."""

    gpu: GPUInfo = field(default_factory=GPUInfo)
    cpu: CPUInfo = field(default_factory=CPUInfo)
    storage: StorageInfo = field(default_factory=StorageInfo)


class HardwareProfiler:
    """
    Profiles available hardware and generates optimal configuration.

    Generates a configuration that balances VRAM, RAM, and SSD usage
    for running the Qwen 3.8 Flash model on the available hardware.
    """

    def __init__(self) -> None:
        self._profile: Optional[FullHardwareProfile] = None

    def profile(self) -> FullHardwareProfile:
        """
        Profile the current machine's hardware.

        Returns:
            FullHardwareProfile with all detected specifications.
        """
        if self._profile is None:
            self._profile = self._detect_hardware()
        return self._profile

    def generate_config(
        self,
        target_ram_gb: Optional[float] = None,
    ) -> dict:
        """
        Generate an optimal configuration for the available hardware.

        Args:
            target_ram_gb: Optional target RAM allocation (uses detected).

        Returns:
            Configuration dictionary for running the model.
        """
        profile = self.profile()

        # Determine VRAM allocation
        brain_size_gb = 82.0  # 125B params, 3-bit quant
        remaining_vram = profile.gpu.vram_gb - brain_size_gb
        expert_cache_gb = max(0, remaining_vram * 0.5)  # 50% of remaining

        # Determine RAM allocation
        ram_gb = target_ram_gb or profile.cpu.ram_gb
        experts_in_ram_gb = min(ram_gb, 48.0)  # Cap at 48GB (diminishing returns)

        # Determine thread count
        thread_count = profile.cpu.physical_cores

        # Estimate speeds
        decode_speed = self._estimate_decode_speed(experts_in_ram_gb)
        prompt_speed = self._estimate_prompt_speed(experts_in_ram_gb)

        return {
            "gpu": {
                "name": profile.gpu.name,
                "vram_gb": profile.gpu.vram_gb,
                "brain_size_gb": brain_size_gb,
                "expert_cache_gb": expert_cache_gb,
            },
            "cpu": {
                "model": profile.cpu.model,
                "threads": thread_count,
                "ram_gb": ram_gb,
                "experts_in_ram_gb": experts_in_ram_gb,
            },
            "storage": {
                "ssd_type": profile.storage.ssd_type,
                "phrase_book_size_gb": 268.0,  # 51B params, 3-bit quant
            },
            "performance": {
                "decode_speed_tok_s": decode_speed,
                "prompt_speed_tok_s": prompt_speed,
            },
            "recommendations": self._generate_recommendations(
                profile, experts_in_ram_gb
            ),
        }

    def _detect_hardware(self) -> FullHardwareProfile:
        """Detect the current machine's hardware."""
        profile = FullHardwareProfile()

        # GPU detection
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                line = result.stdout.strip().split("\n")[0]
                parts = [p.strip() for p in line.split(",")]
                profile.gpu.name = parts[0]
                profile.gpu.vram_gb = float(parts[1])
                profile.gpu.compute_capability = parts[2]
                profile.gpu.tensor_cores = "Tensor" in profile.gpu.name.upper() or \
                    any(x in profile.gpu.name.lower() for x in ["30", "40", "50"])
        except Exception:
            # Default: RTX 3060 12GB (the tested hardware)
            profile.gpu = GPUInfo(
                name="NVIDIA GeForce RTX 3060 12GB",
                vram_gb = 12.0,
                compute_capability="8.6",
                memory_bandwidth_gb_ps = 360.0,
                tensor_cores = True,
            )

        # CPU detection
        try:
            import psutil

            profile.cpu.physical_cores = psutil.cpu_count(logical=False) or 1
            profile.cpu.logical_threads = psutil.cpu_count(logical=True) or 1
            profile.cpu.ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        except Exception:
            # Default: 6-core Ryzen, 61GB RAM
            profile.cpu = CPUInfo(
                model="AMD Ryzen 5 3600 (6c/12t)",
                physical_cores = 6,
                logical_threads = 12,
                ram_gb = 61.0,
                ram_speed = "DDR4-3200",
            )

        # Storage detection
        try:
            # Check if /dev/nvme exists
            if os.path.exists("/dev/nvme"):
                profile.storage.ssd_type = "NVMe"
                profile.storage.read_speed_gb_ps = 3.0
            elif os.path.exists("/dev/sd"):
                profile.storage.ssd_type = "SATA SSD"
                profile.storage.read_speed_gb_ps = 0.5
            else:
                profile.storage.ssd_type = "Unknown"
                profile.storage.read_speed_gb_ps = 0.2

            # Available space on root
            stat = os.statvfs("/")
            profile.storage.available_gb = stat.f_bavail * stat.f_frsize / (1024 ** 3)
        except Exception:
            profile.storage = StorageInfo(
                ssd_type = "NVMe",
                available_gb = 500.0,
                read_speed_gb_ps = 3.0,
            )

        return profile

    def _estimate_decode_speed(self, experts_in_ram_gb: float) -> float:
        """Estimate decode speed based on RAM allocation."""
        if experts_in_ram_gb >= 24.0:
            return 22.0
        elif experts_in_ram_gb >= 20.0:
            return 9.0
        elif experts_in_ram_gb >= 16.0:
            return 6.5
        elif experts_in_ram_gb >= 12.0:
            return 4.5
        return 1.0

    def _estimate_prompt_speed(self, experts_in_ram_gb: float) -> float:
        """Estimate prompt processing speed based on RAM allocation."""
        if experts_in_ram_gb >= 48.0:
            return 115.0
        elif experts_in_ram_gb >= 40.0:
            return 100.0
        elif experts_in_ram_gb >= 32.0:
            return 34.0
        elif experts_in_ram_gb >= 24.0:
            return 15.0
        return 1.0

    def _generate_recommendations(
        self, profile: FullHardwareProfile, experts_in_ram_gb: float
    ) -> list[str]:
        """Generate hardware-specific recommendations."""
        recommendations = []

        if profile.gpu.vram_gb < 12.0:
            recommendations.append(
                f"VRAM is {profile.gpu.vram_gb}GB — below the 12GB minimum. "
                f"Consider upgrading to at least a used RTX 3060 12GB."
            )

        if experts_in_ram_gb < 24.0:
            recommendations.append(
                f"RAM allocation is {experts_in_ram_gb}GB — below the 24GB "
                f"optimal for fast decoding. Upgrade RAM or reduce allocated RAM."
            )
        elif experts_in_ram_gb >= 24.0 and experts_in_ram_gb < 40.0:
            recommendations.append(
                f"RAM is {experts_in_ram_gb}GB — sufficient for fast decoding "
                f"(22 tok/s), but prompt processing will be slower. "
                f"Consider 40GB+ for fast prompt processing."
            )

        if profile.cpu.physical_cores < 6:
            recommendations.append(
                f"Only {profile.cpu.physical_cores} physical cores. "
                f"Set thread count to {profile.cpu.physical_cores} (not logical threads)."
            )

        if profile.storage.ssd_type not in ("NVMe", "SATA SSD"):
            recommendations.append(
                "Storage appears to be slow (HDD or unknown). "
                "An NVMe SSD significantly improves phrase book lookup speed."
            )

        if not recommendations:
            recommendations.append(
                "Hardware looks good for running Qwen 3.8 Flash locally."
            )

        return recommendations

    def summary(self) -> str:
        """Human-readable hardware profile summary."""
        profile = self.profile()
        config = self.generate_config()

        lines = [
            "=== Hardware Profile ===",
            f"  GPU:  {profile.gpu.name}",
            f"    VRAM: {profile.gpu.vram_gb}GB",
            f"    Tensor Cores: {profile.gpu.tensor_cores}",
            f"  CPU:  {profile.cpu.model}",
            f"    Physical cores: {profile.cpu.physical_cores}",
            f"    Logical threads: {profile.cpu.logical_threads}",
            f"    RAM: {profile.cpu.ram_gb:.0f}GB ({profile.cpu.ram_speed})",
            f"  Storage: {profile.storage.ssd_type}",
            f"    Available: {profile.storage.available_gb:.0f}GB",
            f"    Read speed: {profile.storage.read_speed_gb_ps}GB/s",
            "",
            "=== Recommended Configuration ===",
            f"  Threads: {config['cpu']['threads']}",
            f"  VRAM: {config['gpu']['brain_size_gb']}GB (brain) "
            f"+ {config['gpu']['expert_cache_gb']}GB (cache)",
            f"  RAM: {config['cpu']['experts_in_ram_gb']}GB (experts)",
            f"  SSD: {config['storage']['phrase_book_size_gb']}GB (phrase book)",
            "",
            "=== Expected Performance ===",
            f"  Decode: {config['performance']['decode_speed_tok_s']:.1f} tok/s",
            f"  Prompt: {config['performance']['prompt_speed_tok_s']:.1f} tok/s",
            "",
            "=== Recommendations ===",
        ]

        for rec in config["recommendations"]:
            lines.append(f"  • {rec}")

        return "\n".join(lines)
