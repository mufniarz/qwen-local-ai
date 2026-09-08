"""
Real Engram — Phrase Book With Actual Embedding Vectors

The simulation (``src.engram``) fakes phrase embeddings with hashes and
Gaussian noise. This module implements the *same* phrase-book mechanism
with *real* vectors:

  - Phrase rows are computed from the **actual embedding matrix of a
    loaded model** (real vector arithmetic: each phrase row is the
    combination of its tokens' true embeddings).
  - Rows persist to a NumPy ``.npy`` file and are loaded with
    ``mmap_mode="r"`` — the OS pages rows in from disk on access,
    which is the real version of the transcript's "page-fetched from
    SSD" design.
  - An in-process LRU cache mirrors ``EngramPhraseBook``'s semantics
    (hit/miss/disk-read statistics).

What this is, honestly: in the real DeepSeek/Qwen engram design the
phrase table is *trained into the checkpoint* and the model's forward
pass reads it. No downloadable checkpoint currently ships a Qwen
engram table, so here the phrase rows are *derived* from the real
token-embedding matrix at load time (or a supplied corpus) and served
via the same lookup machinery. That makes the lookup path real
(actual vectors, actual paging, actual LRU) without pretending a
hypothetical model was loaded.

Storage layout (``engram_dir``):
    embeddings.npy   (N, dim) float32 — memory-mapped phrase rows
    index.json       {"dim": D, "phrases": [[tok, ...], ...]}
"""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Sequence

import numpy as np


def _to_numpy(a) -> np.ndarray:
    """
    Convert an array-like (NumPy or MLX tensor, possibly quantized) to a
    float32 NumPy array.

    Handles MLX ``QuantizedTensor`` (group-wise quantized weights, as
    produced by mlx-community 4-bit repos) by dequantizing:
        value = scale * (code - 2^(bits-1)) / 2^(bits-1)
    """
    if isinstance(a, np.ndarray):
        return a.astype(np.float32)

    # MLX plain tensor
    try:
        return np.asarray(a).astype(np.float32)
    except (TypeError, ValueError):
        pass

    # MLX quantized tensor: .data (codes), .scales, .groups, .bits
    if hasattr(a, "data") and hasattr(a, "scales"):
        data = np.asarray(a.data).astype(np.int32)
        scales = np.asarray(a.scales).astype(np.float32)
        bits = int(getattr(a, "bits", 4))
        groups = int(getattr(a, "groups", 0)) or 0
        # data: (rows, cols // groups, groups)
        deq = (data - (1 << (bits - 1))) / (1 << (bits - 1)) * scales[:, :, None]
        rows = data.shape[0]
        cols = data.shape[1] * groups if groups else data.shape[1]
        return deq.reshape(rows, cols).astype(np.float32)

    raise TypeError(
        f"Cannot convert {type(a).__name__} to float32 NumPy array "
        "(not a recognized tensor/quantized-tensor layout)"
    )


class RealPhraseBook:
    """
    Phrase book with real embedding rows.

    Mirrors the ``EngramPhraseBook`` API (``lookup`` / ``lookup_batch`` /
    ``prefetch`` / ``stats``) so the two are drop-in comparable, but
    every vector it returns is a real combination of real model
    embeddings.
    """

    def __init__(
        self,
        dim: int,
        cache_size: int = 1024,
        combine: str = "mean",
    ) -> None:
        self.dim = int(dim)
        self.cache_size = int(cache_size)
        self.combine = combine  # "mean" or "sum"

        # phrase key (tuple of token ids) → row index
        self._index: "OrderedDict[tuple, int]" = OrderedDict()
        # Real rows: (N, dim) float32 — either a full array or a memmap
        self._rows: Optional[np.ndarray] = None
        self._cache: "OrderedDict[tuple, np.ndarray]" = OrderedDict()

        self._lookups = 0
        self._cache_hits = 0
        self._misses = 0
        self._disk_reads = 0

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_embeddings(
        cls,
        embed_matrix,
        phrases: Sequence[Sequence],
        dim: Optional[int] = None,
        cache_size: int = 1024,
        combine: str = "mean",
    ) -> "RealPhraseBook":
        """
        Build a phrase book from a real token-embedding matrix.

        Args:
            embed_matrix: (vocab, dim) array — e.g. the real
                ``embed_tokens.weight`` of a loaded model.
            phrases: iterable of phrases, each a sequence of 1–3 token ids.
            dim: embedding dim (inferred from embed_matrix if None).
        """
        matrix = _to_numpy(embed_matrix)
        dim = int(dim or matrix.shape[1])
        book = cls(dim=dim, cache_size=cache_size, combine=combine)

        for phrase in phrases:
            ids = tuple(int(t) for t in phrase)
            if not (1 <= len(ids) <= 3):
                raise ValueError(f"Phrases must be 1–3 tokens, got {len(ids)}")
            if any(t >= matrix.shape[0] for t in ids):
                raise ValueError(
                    f"Token id {max(ids)} out of vocab range {matrix.shape[0]}"
                )
            vecs = matrix[list(ids)]
            row = vecs.mean(axis=0) if combine == "mean" else vecs.sum(axis=0)
            if combine == "mean" and len(ids) > 1:
                # keep per-token magnitude scale like a learned row would
                row = row * len(ids)
            book._add(ids, row)
        return book

    def _add(self, key: tuple, row: np.ndarray) -> None:
        self._index[key] = len(self._index)
        if self._rows is None:
            self._rows = row[None, :]
        else:
            self._rows = np.vstack([self._rows, row[None, :]])

    # ------------------------------------------------------------------
    # Persistence (real mmap I/O)
    # ------------------------------------------------------------------

    def save(self, path) -> Path:
        """Write embeddings.npy + index.json into ``path``."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if self._rows is None:
            raise RuntimeError("Phrase book is empty — nothing to save")
        np.save(path / "embeddings.npy", self._rows.astype(np.float32))
        with open(path / "index.json", "w") as f:
            json.dump(
                {
                    "dim": self.dim,
                    "combine": self.combine,
                    "phrases": [list(k) for k in self._index],
                },
                f,
            )
        return path

    @classmethod
    def load(cls, path, cache_size: int = 1024) -> "RealPhraseBook":
        """
        Load a saved phrase book. Embeddings are memory-mapped
        (``mmap_mode="r"``) so only touched rows cost real memory —
        the transcript's SSD page-fetch design.
        """
        path = Path(path)
        with open(path / "index.json") as f:
            meta = json.load(f)
        book = cls(dim=meta["dim"], cache_size=cache_size, combine=meta.get("combine", "mean"))
        book._rows = np.load(path / "embeddings.npy", mmap_mode="r")
        for i, phrase in enumerate(meta["phrases"]):
            book._index[tuple(int(t) for t in phrase)] = i
        return book

    # ------------------------------------------------------------------
    # Lookup (mirrors src.engram API)
    # ------------------------------------------------------------------

    def lookup(self, token_ids: Sequence[int]) -> np.ndarray:
        """
        Look up the real phrase row for 1–3 tokens.

        Returns a zero vector for phrases not in the book (the model
        would simply use the token embeddings instead — see
        ``input_embeddings``).
        """
        key = tuple(int(t) for t in token_ids)
        if not (1 <= len(key) <= 3):
            raise ValueError(f"Phrase must be 1–3 tokens, got {len(key)}")

        self._lookups += 1

        if key in self._cache:
            self._cache_hits += 1
            self._cache.move_to_end(key)
            return self._cache[key]

        self._misses += 1
        row_idx = self._index.get(key)
        if row_idx is None:
            # Not in the book — the model would use plain token embeddings
            return np.zeros(self.dim, dtype=np.float32)

        # Real disk access (page fetch on memmap first touch)
        self._disk_reads += 1
        row = np.asarray(self._rows[row_idx]).copy()  # materialize the page

        self._cache[key] = row
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)  # evict LRU
        return row

    def lookup_batch(self, phrases: Sequence[Sequence[int]]) -> list:
        return [self.lookup(p) for p in phrases]

    def prefetch(self, phrases: Sequence[Sequence[int]]) -> None:
        for phrase in phrases:
            self.lookup(phrase)

    def row_for(self, token_ids: Sequence[int]) -> Optional[np.ndarray]:
        """Real row if the phrase is indexed, else None."""
        key = tuple(int(t) for t in token_ids)
        row_idx = self._index.get(key)
        if row_idx is None:
            return None
        return np.asarray(self._rows[row_idx])

    def input_embeddings(self, embed_matrix, token_ids: Sequence[int]) -> np.ndarray:
        """
        Produce the input embedding row sequence for a token stream,
        substituting the phrase row for every position covered by an
        indexed 2–3-gram (greedy left-to-right, longest match first).

        This is the real engram injection point: positions that form a
        known phrase get the pre-computed phrase vector instead of the
        plain token embedding.
        """
        matrix = (
            embed_matrix
            if isinstance(embed_matrix, np.ndarray)
            else _to_numpy(embed_matrix)
        )
        ids = [int(t) for t in token_ids]
        out = np.zeros((len(ids), self.dim), dtype=np.float32)
        i = 0
        while i < len(ids):
            matched = False
            for span in (3, 2):
                if i + span <= len(ids):
                    key = tuple(ids[i : i + span])
                    if key in self._index:
                        row = self.lookup(key)
                        out[i : i + span] = row
                        i += span
                        matched = True
                        break
            if not matched:
                out[i] = matrix[ids[i]]
                i += 1
        return out

    # ------------------------------------------------------------------
    # Stats (same keys as the simulation for comparison)
    # ------------------------------------------------------------------

    @property
    def size_gb(self) -> float:
        n = len(self._index) if self._rows is None else self._rows.shape[0]
        return n * self.dim * 4 / (1024 ** 3)

    @property
    def stats(self) -> dict:
        total = max(self._lookups, 1)
        return {
            "lookups": self._lookups,
            "cache_hits": self._cache_hits,
            "disk_reads": self._disk_reads,
            "cache_misses": self._misses,
            "cache_hit_rate": self._cache_hits / total,
            "cached_entries": len(self._cache),
            "indexed_phrases": len(self._index),
        }


# ----------------------------------------------------------------------
# Extraction from a loaded MLX model
# ----------------------------------------------------------------------


def extract_embedding_matrix(model) -> np.ndarray:
    """
    Find the real token-embedding table inside a loaded MLX model and
    return it as a float32 NumPy array of shape (vocab, dim).

    The robust way is to *call the model's own embedding layer* on every
    vocabulary id: this exercises the model's real (de)quantization path
    and works for every storage layout — plain float, group-quantized,
    packed uint32, mxfp4/nvfp4 — without us having to reverse-engineer
    any particular checkpoint's weight packing.
    """
    import mlx.core as mx

    targets = ("embed_tokens", "token_embedding", "wte", "wte_emb")
    for name, mod in model.named_modules():
        tail = name.split(".")[-1]
        if tail not in targets or not hasattr(mod, "weight"):
            continue
        w = mod.weight
        if w.ndim != 2 or w.shape[0] < 100:
            continue
        vocab = int(w.shape[0])
        mat = mod(mx.arange(vocab)).astype(mx.float32)
        return np.asarray(mat)

    raise RuntimeError(
        "Could not find an embedding table in the model "
        "(looked for embed_tokens/token_embedding/wte)"
    )


def dequantize_linear(module, backend: str = "numpy"):
    """
    Recover the real (dequantized) weight matrix of any linear-like
    module — plain, group-quantized, packed, mxfp4/nvfp4 — as a float32
    array of shape (out, in).

    Trick: call the module with the identity matrix. The model's own
    matmul runs its real (de)quantization, and multiplying by the
    identity extracts each weight exactly (no accumulation error).

    For quantized modules (QuantizedLinear) that lack .in_features and
    .out_features, dimensions are extracted from the weight tensor.

    Args:
        module: a module with ``weight`` + ``in_features``/``out_features``
        backend: "numpy" (default) or "mlx" for the return type.
    """
    import mlx.core as mx

    # Identity matrix trick works for both plain Linear and QuantizedLinear:
    # calling the module with identity exercises its real (de)quantization path
    # and extracting the output gives the exact weight matrix.
    if hasattr(module, "out_features"):
        out_dim = int(module.out_features)
        in_dim = int(module.in_features)
    elif hasattr(module, "weight"):
        # QuantizedLinear: .weight is quantized codes; infer shape from it.
        w_q = module.weight
        if w_q.ndim == 2:
            out_dim, packed_dim = w_q.shape
            bits = int(getattr(module, "bits", 4))
            group_size = int(getattr(module, "group_size", 64))
            elems_per_word = 32 // bits  # 8 for 4-bit
            num_groups = packed_dim // elems_per_word
            in_dim = num_groups * group_size
        else:
            raise RuntimeError(
                f"Cannot extract dimensions from {type(module).__name__} "
                f"(weight has {w_q.ndim} dims)"
            )
    else:
        raise RuntimeError(
            f"Cannot extract dimensions from {type(module).__name__} "
            "(no .out_features, no .weight)"
        )

    ident = mx.eye(in_dim).astype(mx.float32)
    y = module(ident).astype(mx.float32)  # (in, out) == Wᵀ
    bias = getattr(module, "bias", None)
    if bias is not None:
        y = y - bias.astype(mx.float32)[None, :]
    w = y.T
    if backend == "mlx":
        return w
    return np.asarray(w)
