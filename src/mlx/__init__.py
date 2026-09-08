"""
MLX Inference Bridge — Real inference for Apple Silicon

This package provides a real inference bridge that replaces the simulation
with actual MLX-based model loading and forward passes. It turns the
architectural simulation into something that runs real models on your M1 Max.

Components:
  - mlx_inference: Core MLX model loading, forward pass, and KV cache management
  - real_engram: Real phrase book lookup using actual embedding matrices
  - real_moe: Real MoE gating network with actual expert computation
  - mlx_bridge: Integration glue between the simulation layer and real MLX inference
  - mlx_profiler: M1 Max hardware profiler (Metal GPU detection, unified memory)

Usage:
    from src.mlx.mlx_inference import MLXModelLoader
    loader = MLXLoader("/path/to/qwen-model")
    result = loader.generate("hello world", max_tokens=100)
"""

__all__ = [
    "mlx_inference",
    "real_engram",
    "real_moe",
    "mlx_bridge",
    "mlx_profiler",
]

__version__ = "0.2.0"
