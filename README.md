# Qwen 3.8 Flash — Local Frontier Model Implementation

A reference implementation of the architectural innovations that enable running
a **177-billion-parameter frontier-class model** on consumer hardware (12GB VRAM GPU
with modest system RAM).

## Main Points from the Transcript

### 1. The Architecture: "Brain + Phrase Book" (Engram)

- **Brain**: 125B standard neural network parameters — the "thinking" part
- **Phrase Book (Engram)**: 51B lookup-table parameters — pre-computed embeddings
  for common 2–3 word phrases, stored on SSD, fetched on-demand
- Total: **177B parameters**, but only the 125B brain needs to fit in VRAM

### 2. Split Memory Design

| Component | Size | Location |
|-----------|------|----------|
| Brain (shared weights) | ~82 GB (3-bit quant) | GPU VRAM (12GB holds what runs on every token) |
| Phrase Book (Engram) | ~268 GB (3-bit quant) | SSD, page-fetched as needed |
| Experts (MoE) | ~80 GB (3-bit quant) | System RAM |

The phrase book stays on disk — only the rows (phrase entries) actually accessed
are read. The GPU never copies the whole file.

### 3. Mixture of Experts (MoE)

- **512 experts**, but only **10 fire per token**
- Experts live in system RAM; GPU holds only the shared "every-token" weights
- Expert caching (LRU) keeps the most-frequently-used experts in remaining VRAM

### 4. Thread Optimization

- Stock llama.cpp: **16.5 tok/s** with 12 threads (logical cores)
- Problem: 65% of CPU time was threads spinning, waiting for each other
- Fix: **6 threads** (one per physical core) → **24.4 tok/s** (~50% improvement)

### 5. RAM Budgeting

| System RAM | Decode Speed | Prompt Processing |
|------------|-------------|-------------------|
| 12 GB | 4.5 tok/s | Streams all experts from disk |
| 16 GB | 6.5 tok/s | 15 tok/s |
| 20 GB | 9 tok/s | 34 tok/s |
| **24 GB** | **22 tok/s** | 15 tok/s |
| 32 GB | 22 tok/s | 34 tok/s |
| **40 GB** | 22 tok/s | **~100 tok/s** |
| 48 GB | 22 tok/s | Full speed |

- **24 GB**: sufficient for fast answer generation
- **40 GB+**: needed for fast prompt processing (reading large codebases)

### 6. Two-Model Pipeline

A practical inference pattern discovered in testing:

```
┌─────────────────────────────────────────────┐
│  Qwen 3.8 (177B, slow, smart)               │
│  Role: Planner + Reviewer                    │
│  - Reads the problem                         │
│  - Writes a plan (small, exact steps)        │
│  - Reviews results, writes report            │
└──────────────────┬──────────────────────────┘
                   │ swap (25s cost)
┌──────────────────▼──────────────────────────┐
│  Qwen 3.6 (35B, fast, obedient)             │
│  Role: Worker                               │
│  - Executes the plan at 3× speed            │
│  - Does what it's told, doesn't second-guess│
└─────────────────────────────────────────────┘
```

### 7. Performance Results

| Model | Hardware | Speed | Lab Score |
|-------|----------|-------|-----------|
| Claude Opus 5 | Cloud | Minutes/task | 17/17 |
| **Qwen 3.8 Flash** | **RTX 3060 12GB** | **~24 tok/s** | **17/17** |
| Qwen 3.6 (35B) | RTX 3060 12GB | ~70 tok/s | 14/17 |

Qwen 3.8 on a **used $300 gaming card** matches Claude Opus 5 on a custom-built
lab designed to trip it up.

### 8. Key Takeaways

1. **Yes, you can run frontier-class models locally** on consumer hardware
2. The **phrase book (Engram)** is what makes it possible — 51B params of
   lookups that cost nothing to compute
3. **RAM is the new bottleneck**, not GPU VRAM
4. **Thread count matters more than you'd think** — match to physical cores
5. **Green tests don't mean correct code** — always read the code
6. **Local AI is becoming a necessity**, not a luxury, for privacy and control

## Repository Structure

```
qwen-local-ai/
├── README.md                          # This file
├── src/
│   ├── __init__.py
│   ├── engram.py                      # Phrase book lookup table
│   ├── moe.py                         # Mixture of Experts routing
│   ├── memory_manager.py              # Split memory / RAM budgeting
│   ├── expert_cache.py                # LRU VRAM expert cache
│   ├── thread_optimizer.py            # Physical-core thread mapping
│   ├── multi_token_prediction.py      # Draft head speculative decoding
│   ├── two_model_pipeline.py          # Planner/Worker pipeline
│   └── hardware_profiler.py           # Hardware-aware configuration
├── tests/
│   ├── __init__.py
│   ├── test_ensemble.py               # Integration tests
│   └── test_performance.py            # Performance benchmarks
└── docs/
    └── architecture.md                # Detailed design docs
```

## Requirements

- Python 3.10+
- PyTorch (for any GPU-accelerated components)
- numpy, scipy (numerical operations)

## License

MIT — use this to build local AI. We don't want a handful of cloud companies
holding all the power.

> *"The only difference between the chicken farmer and the chickens in his coop
> is that the chicken farmer is smarter than the chickens. And that's why he's
> the one controlling them. Without local AI, we are all going to be the
> chickens."*
