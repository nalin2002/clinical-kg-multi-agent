"""Node encoders mapping ``(type, text)`` pairs into a shared vector space.

Two interchangeable implementations:

* :class:`BgeNodeEncoder` - frozen BGE-M3 (via ``FlagEmbedding``) of the string
  ``"{TYPE}: {text}"``. Same encoder family used by the pipeline's ``dump_graph``
  entity resolution, so MIMIC and pipeline graphs land in the *same* space.
  Embeddings are cached on disk keyed by a hash of ``(type, text)`` so repeated
  runs (and train/score sharing the same nodes) avoid recomputation.

* :class:`MockEncoder` - deterministic pseudo-random vector derived from a hash
  of ``"{TYPE}: {text}"``. No downloads, no model. Lets dev / CI run the full
  pipeline without the BGE-M3 weights.

Both expose the same surface::

    enc.dim                       # -> int
    enc.encode([(type, text), ...])  # -> np.ndarray [N, dim], L2-normalised
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

def _key_text(node_type: str, text: str) -> str:
    return f"{node_type}: {text}"


def _hash_key(node_type: str, text: str) -> str:
    return hashlib.sha1(_key_text(node_type, text).encode("utf-8")).hexdigest()


def _l2_normalise(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


class MockEncoder:
    """Deterministic, dependency-light encoder for dev/CI.

    The vector for a node is sampled from a Gaussian seeded by the hash of
    ``"{TYPE}: {text}"``, then L2-normalised. Identical inputs always yield
    identical vectors, so train/score stay consistent.
    """

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _one(self, node_type: str, text: str) -> np.ndarray:
        h = hashlib.sha256(_key_text(node_type, text).encode("utf-8")).digest()
        seed = int.from_bytes(h[:8], "little", signed=False)
        rng = np.random.default_rng(seed)
        return rng.standard_normal(self.dim).astype(np.float32)

    def encode(self, keys: Sequence[Tuple[str, str]]) -> np.ndarray:
        if not keys:
            return np.zeros((0, self.dim), dtype=np.float32)
        mat = np.stack([self._one(t, x) for t, x in keys], axis=0)
        return _l2_normalise(mat)


class BgeNodeEncoder:
    """Frozen BGE-M3 encoder with an on-disk per-key cache.

    Parameters
    ----------
    model_name : HuggingFace id for the BGE-M3 model.
    cache_dir  : directory holding ``<sha1>.npy`` per ``(type, text)`` key.
    device     : passed through to FlagEmbedding ("cpu" / "cuda").
    """

    DIM = 1024  # BGE-M3 dense embedding dimension

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        cache_dir: str | Path = ".cache/graph_jepa/bge",
        device: str | None = None,
    ):
        self.dim = self.DIM
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self._model = None  # lazily loaded on first cache miss

    def _ensure_model(self):
        if self._model is None:
            # Imported lazily so the package imports without FlagEmbedding/torch.
            from FlagEmbedding import BGEM3FlagModel

            self._model = BGEM3FlagModel(self.model_name, use_fp16=False)

    def _cache_path(self, node_type: str, text: str) -> Path:
        return self.cache_dir / f"{_hash_key(node_type, text)}.npy"

    def _encode_uncached(self, keys: List[Tuple[str, str]]) -> np.ndarray:
        self._ensure_model()
        texts = [_key_text(t, x) for t, x in keys]
        out = self._model.encode(texts, return_dense=True, return_sparse=False,
                                 return_colbert_vecs=False)
        return np.asarray(out["dense_vecs"], dtype=np.float32)

    def encode(self, keys: Sequence[Tuple[str, str]]) -> np.ndarray:
        keys = list(keys)
        if not keys:
            return np.zeros((0, self.dim), dtype=np.float32)

        vectors: List[np.ndarray | None] = [None] * len(keys)
        misses: List[int] = []
        for i, (t, x) in enumerate(keys):
            p = self._cache_path(t, x)
            if p.exists():
                vectors[i] = np.load(p)
            else:
                misses.append(i)

        if misses:
            miss_keys = [keys[i] for i in misses]
            computed = self._encode_uncached(miss_keys)
            for j, i in enumerate(misses):
                vec = computed[j].astype(np.float32)
                vectors[i] = vec
                np.save(self._cache_path(*keys[i]), vec)

        mat = np.stack([v for v in vectors], axis=0).astype(np.float32)
        return _l2_normalise(mat)


class SapBertNodeEncoder:
    """SapBERT encoder for biomedical entity similarity.

    Uses ``cambridgeltl/SapBERT-from-PubMedBERT-fulltext`` (768-dim), which is
    specifically trained on UMLS synonym pairs. Produces much better clinical
    concept similarity than general-purpose encoders like BGE-M3.

    Same interface as :class:`BgeNodeEncoder`: ``dim``, ``encode(keys)``.

    Parameters
    ----------
    model_name : HuggingFace id for the SapBERT model.
    cache_dir  : directory holding ``<sha1>.npy`` per ``(type, text)`` key.
    device     : "cpu" / "cuda" / "cuda:0" etc.
    """

    DIM = 768

    def __init__(
        self,
        model_name: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        cache_dir: str | Path = ".cache/graph_jepa/sapbert",
        device: str | None = None,
    ):
        self.dim = self.DIM
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self._tokenizer = None
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name)
            dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._model = self._model.to(dev).eval()

    def _cache_path(self, node_type: str, text: str) -> Path:
        return self.cache_dir / f"{_hash_key(node_type, text)}.npy"

    def _encode_uncached(self, keys: List[Tuple[str, str]]) -> np.ndarray:
        self._ensure_model()
        texts = [_key_text(t, x) for t, x in keys]
        dev = next(self._model.parameters()).device
        toks = self._tokenizer(
            texts, padding=True, truncation=True, max_length=64,
            return_tensors="pt",
        ).to(dev)
        with torch.no_grad():
            out = self._model(**toks)
        # CLS pooling
        return out.last_hidden_state[:, 0, :].cpu().numpy().astype(np.float32)

    def encode(self, keys: Sequence[Tuple[str, str]]) -> np.ndarray:
        keys = list(keys)
        if not keys:
            return np.zeros((0, self.dim), dtype=np.float32)

        vectors: List[np.ndarray | None] = [None] * len(keys)
        misses: List[int] = []
        for i, (t, x) in enumerate(keys):
            p = self._cache_path(t, x)
            if p.exists():
                vectors[i] = np.load(p)
            else:
                misses.append(i)

        if misses:
            miss_keys = [keys[i] for i in misses]
            computed = self._encode_uncached(miss_keys)
            for j, i in enumerate(misses):
                vec = computed[j].astype(np.float32)
                vectors[i] = vec
                np.save(self._cache_path(*keys[i]), vec)

        mat = np.stack([v for v in vectors], axis=0).astype(np.float32)
        return _l2_normalise(mat)


def build_encoder(kind: str, *, mock_dim: int = 256, **kwargs):
    """Factory: ``kind`` is ``"mock"``, ``"bge"``, or ``"sapbert"``."""
    if kind == "mock":
        return MockEncoder(dim=mock_dim)
    if kind == "bge":
        return BgeNodeEncoder(**kwargs)
    if kind == "sapbert":
        return SapBertNodeEncoder(**kwargs)
    raise ValueError(f"unknown encoder kind: {kind!r}")
