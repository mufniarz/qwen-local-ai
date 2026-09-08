"""
Engram — Phrase Book Lookup Table

The core innovation that makes running a 177B parameter model on 12GB VRAM
possible. Instead of storing all parameters in GPU memory, 51B parameters
are stored as a phrase book on SSD and fetched on-demand.

Key idea: Every language model has a word embedding table (the "dictionary").
The Engram adds a "phrase book" — entries for 2–3 word phrases (hundreds of
millions of them). When the model knows the last 2–3 words, it can look up
their pre-computed embedding instead of computing it through layers.

A lookup costs nothing — no multiplication. And since the model knows which
phrases it'll need ahead of time, the lookups can happen in parallel with
layer 1 computation.

Reference: Inspired by DeepSeek's Engram paper (January 2025), deployed at
frontier scale by Qwen as an open-weight experimental preview.
"""

from __future__ import annotations

import mmap
import os
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class PhraseEntry:
    """A single entry in the phrase book."""

    phrase_id: int          # Hash of the phrase tokens
    embedding_offset: int   # Byte offset into the embedding file
    embedding_dim: int      # Dimension of the embedding vector
    usage_count: int = 0    # How many times this phrase has been looked up


class EngramPhraseBook:
    """
    On-disk phrase book for a Qwen 3.8 Flash model.

    The phrase book contains ~51 billion parameters worth of pre-computed
    embeddings for common 2–3 word phrases. It lives on SSD and is accessed
    via memory-mapped I/O — only the rows actually needed are read.

    Memory layout (simplified):
        [index file]  →  phrase_id → (embedding_offset, embedding_dim)
        [embedding file]  →  raw embedding vectors, accessed by offset

    The index is small enough to fit in RAM. The embeddings are memory-mapped
    from disk, so the OS handles paging — only touched pages consume RAM.
    """

    # Index format: (phrase_id: uint64, embedding_offset: uint64,
    #                embedding_dim: uint32, usage_count: uint32)
    INDEX_ENTRY_SIZE = 8 + 8 + 4 + 4  # 24 bytes per index entry

    def __init__(
        self,
        index_path: str | Path,
        embedding_path: str | Path,
        embedding_dim: int = 4096,
        cache_size: int = 1024,
    ) -> None:
        """
        Initialize the phrase book.

        Args:
            index_path: Path to the index file (small, fits in RAM).
            embedding_path: Path to the memory-mapped embedding file (large, on SSD).
            embedding_dim: Dimension of each embedding vector (e.g., 4096).
            cache_size: Number of embeddings to keep in the in-RAM LRU cache.
        """
        self.index_path = Path(index_path)
        self.embedding_path = Path(embedding_path)
        self.embedding_dim = embedding_dim
        self.cache_size = cache_size

        # In-RAM index: phrase_id → PhraseEntry
        self._index: dict[int, PhraseEntry] = {}

        # In-RAM LRU cache for recently-accessed embeddings
        self._cache: dict[int, list[float]] = {}
        self._cache_order: list[int] = []  # LRU order (most recent at end)

        # Memory-mapped embedding file (opened lazily)
        self._mmapped_file: Optional[mmap.mmap] = None

        # Statistics
        self._lookup_count: int = 0
        self._cache_hit_count: int = 0
        self._disk_read_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def lookup(self, phrase_tokens: tuple[str, ...]) -> list[float]:
        """
        Look up the embedding for a phrase (1–3 tokens).

        Args:
            phrase_tokens: 1, 2, or 3 tokens forming the phrase.

        Returns:
            The embedding vector as a list of floats.
        """
        if len(phrase_tokens) < 1 or len(phrase_tokens) > 3:
            raise ValueError(
                f"Phrase must be 1–3 tokens, got {len(phrase_tokens)}"
            )

        phrase_id = self._hash_phrase(phrase_tokens)

        # Check LRU cache first
        if phrase_id in self._cache:
            self._cache_hit_count += 1
            # Move to end (most recently used)
            self._cache_order.remove(phrase_id)
            self._cache_order.append(phrase_id)
            self._lookup_count += 1
            return self._cache[phrase_id]

        # Check index
        entry = self._index.get(phrase_id)
        if entry is None:
            # Phrase not in book — return zero embedding (fallback)
            return [0.0] * self.embedding_dim

        # Read from memory-mapped file
        embedding = self._read_embedding(entry.embedding_offset, entry.embedding_dim)

        # Update usage count
        entry.usage_count += 1

        # Insert into LRU cache
        self._evict_cache_if_full()
        self._cache[phrase_id] = embedding
        self._cache_order.append(phrase_id)

        self._lookup_count += 1
        self._disk_read_count += 1

        return embedding

    def lookup_batch(
        self, phrases: tuple[tuple[str, ...], ...]
    ) -> list[list[float]]:
        """
        Look up embeddings for multiple phrases in batch.

        The model knows which phrases it'll need before computing the next
        layer, so lookups can be batched and prefetch while layer 1 computes.

        Args:
            phrases: Tuple of phrase tuples (each 1–3 tokens).

        Returns:
            List of embedding vectors.
        """
        return [self.lookup(phrase) for phrase in phrases]

    def prefetch(self, phrases: tuple[tuple[str, ...], ...]) -> None:
        """
        Prefetch embeddings into the LRU cache.

        Called when the model knows which phrases it'll need in the next
        layer, while the current layer is still computing. This hides
        disk latency.

        Args:
            phrases: Phrases to prefetch.
        """
        for phrase in phrases:
            self.lookup(phrase)  # reuse lookup logic (cache-aware)

    @property
    def stats(self) -> dict[str, float]:
        """Return lookup statistics."""
        total = max(self._lookup_count, 1)
        return {
            "lookups": self._lookup_count,
            "cache_hits": self._cache_hit_count,
            "disk_reads": self._disk_read_count,
            "cache_hit_rate": self._cache_hit_count / total,
            "cached_entries": len(self._cache),
            "indexed_phrases": len(self._index),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _hash_phrase(phrase_tokens: tuple[str, ...]) -> int:
        """Hash a phrase tuple to a stable phrase_id."""
        import hashlib

        h = hashlib.sha256(":".join(phrase_tokens).encode()).hexdigest()
        return int(h[:16], 16)

    def _read_embedding(
        self, offset: int, dim: int
    ) -> list[float]:
        """Read a single embedding from the memory-mapped file."""
        if self._mmapped_file is None:
            fd = os.open(
                str(self.embedding_path), os.O_RDONLY | os.O_BINARY
            )
            self._mmapped_file = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
            os.close(fd)

        byte_size = dim * 4  # float32 = 4 bytes
        data = self._mmapped_file[offset : offset + byte_size]
        return list(struct.unpack(f"{dim}f", data))

    def _evict_cache_if_full(self) -> None:
        """Evict the least-recently-used entry if cache is full."""
        if len(self._cache) >= self.cache_size and self._cache_order:
            lru_id = self._cache_order.pop(0)
            del self._cache[lru_id]

    def load_index(self, index_path: Optional[str | Path] = None) -> None:
        """
        Load the index file into RAM.

        The index is small (a few MB for hundreds of millions of phrases)
        and fits entirely in RAM.

        Args:
            index_path: Optional path to the index file.
        """
        path = Path(index_path or self.index_path)
        if not path.exists():
            return

        with open(path, "rb") as f:
            while True:
                raw = f.read(self.INDEX_ENTRY_SIZE)
                if len(raw) < self.INDEX_ENTRY_SIZE:
                    break
                phrase_id, emb_offset, emb_dim, usage = struct.unpack(
                    "QQII", raw
                )
                self._index[phrase_id] = PhraseEntry(
                    phrase_id=phrase_id,
                    embedding_offset=emb_offset,
                    embedding_dim=emb_dim,
                    usage_count=usage,
                )

    def save_index(self, index_path: Optional[str | Path] = None) -> None:
        """Save the index (with updated usage counts) to disk."""
        path = Path(index_path or self.index_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            for entry in self._index.values():
                f.write(struct.pack(
                    "QQII",
                    entry.phrase_id,
                    entry.embedding_offset,
                    entry.embedding_dim,
                    entry.usage_count,
                ))
