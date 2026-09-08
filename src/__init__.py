"""
Qwen 3.8 Flash — Local Frontier Model Implementation

A reference implementation of the architectural innovations that enable running
a 177-billion-parameter frontier-class model on consumer hardware (12GB VRAM GPU
with modest system RAM).

Components:
  - engram: Phrase book lookup table (51B parameters on SSD)
  - moe: Mixture of Experts routing (512 experts, 10 per token)
  - memory_manager: Split memory design (brain in VRAM, book on SSD)
  - expert_cache: LRU VRAM expert cache
  - thread_optimizer: Physical-core thread mapping
  - multi_token_prediction: Draft head speculative decoding
  - two_model_pipeline: Planner/Worker pipeline
  - hardware_profiler: Hardware-aware configuration
  - mlx: Real MLX inference bridge for Apple Silicon (NEW)
    - mlx_inference: Core MLX model loading and forward pass
    - real_engram: Real phrase book with actual embeddings
    - real_moe: Real MoE gating network and expert computation
    - mlx_bridge: Integration glue between simulation and real inference
    - mlx_profiler: M1 Max hardware profiler
"""

__version__ = "0.2.0"
