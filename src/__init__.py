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
"""

__version__ = "0.1.0"
