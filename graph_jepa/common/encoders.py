"""Text encoders for graph featurization.

Used by the edge-plausibility classifier and the ablation refiners to turn node
text into vectors. Prefers ``sentence-transformers`` (BGE-M3) when installed;
otherwise falls back to a deterministic hashing encoder so the whole pipeline
runs offline with no model downloads (lower quality — logged loudly).
"""

from __future__ import annotations

import hashlib
from typing import List, Sequence

import numpy as np

from .io_utils import LOG


class HashingEncoder:
    """Deterministic, dependency-free fallback encoder.

    Hashes character 3-grams into a fixed-width L2-normalised vector. Captures
    surface lexical overlap (enough for a working baseline); not semantic.
    """

    def __init__(self, dim: int = 256):
        self.dim = dim
        self.name = f"hashing-{dim}"

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        text = (text or "").lower()
        grams = [text[i:i + 3] for i in range(max(len(text) - 2, 1))] or [text]
        for g in grams:
            h = int(hashlib.md5(g.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return np.vstack([self._vec(t) for t in texts]) if texts else np.zeros((0, self.dim))


class SentenceTransformerEncoder:
    """BGE-M3 (or any sentence-transformers model) wrapper."""

    def __init__(self, model: str, cache_dir: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model, cache_folder=cache_dir)
        self.dim = self.model.get_sentence_embedding_dimension()
        self.name = model

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim))
        return np.asarray(
            self.model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False),
            dtype=np.float32,
        )


def build_encoder(cfg) -> object:
    """Construct the encoder named in config (``encoder.backend``)."""
    backend = cfg.get("encoder.backend", "auto")
    model = cfg.get("encoder.model", "BAAI/bge-m3")
    cache = str(cfg.path("encoder.cache_dir", ".cache/graph_jepa/encoder"))
    dim = int(cfg.get("encoder.hashing_dim", 256))

    if backend in ("auto", "sentence-transformers"):
        try:
            enc = SentenceTransformerEncoder(model, cache)
            LOG.info("encoder: sentence-transformers (%s, dim=%d)", enc.name, enc.dim)
            return enc
        except Exception as exc:  # noqa: BLE001 — any import/download failure -> fallback
            if backend == "sentence-transformers":
                raise
            LOG.warning("sentence-transformers unavailable (%s); using hashing encoder", exc)

    enc = HashingEncoder(dim)
    LOG.warning("encoder: HASHING fallback (dim=%d) — surface-lexical only, not semantic", dim)
    return enc


def node_embeddings(encoder, graph) -> dict:
    """Map ``node_id -> embedding`` for a graph, encoding ``"TYPE: name"``."""
    nodes = graph.nodes
    keys = [f"{n.type}: {n.name}" for n in nodes]
    mat = encoder.encode(keys)
    return {n.id: mat[i] for i, n in enumerate(nodes)}
