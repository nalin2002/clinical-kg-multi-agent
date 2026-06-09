"""JEPA-style masked-edge relation predictor (ABLATION / prototype).

A relation classifier head that predicts a hidden edge's RELATION label from the
embeddings of its endpoints plus a context vector summarising the *visible*
(unmasked) graph::

    input  = [emb(src), emb(tgt), context]
    context = mean of node embeddings of the visible graph
    output  = softmax over RELATION_TYPES

torch is imported lazily so the rest of the repo runs without it.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from ...common.encoders import node_embeddings
from ...common.graph_schema import RELATION_TYPES, Graph


def _require_torch():
    try:
        import torch  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "pip install torch — needed for this ablation (masked_edge_prediction)"
        ) from exc
    import torch

    return torch


_REL_INDEX: Dict[str, int] = {r: i for i, r in enumerate(RELATION_TYPES)}


def relation_to_index(relation: str) -> Optional[int]:
    """Index of a relation in RELATION_TYPES, or None if out of vocabulary."""
    return _REL_INDEX.get((relation or "").upper())


def context_vector(emb: Dict[str, np.ndarray], emb_dim: int) -> np.ndarray:
    """Mean of the visible graph's node embeddings ``(emb_dim,)``."""
    if emb:
        return np.vstack(list(emb.values())).astype(np.float32).mean(axis=0)
    return np.zeros(emb_dim, dtype=np.float32)


def edge_input(
    emb: Dict[str, np.ndarray],
    src_id: str,
    tgt_id: str,
    context: np.ndarray,
    emb_dim: int,
) -> Optional[np.ndarray]:
    """Build ``[emb(src), emb(tgt), context]`` for one edge, or None if an
    endpoint is missing an embedding."""
    if src_id not in emb or tgt_id not in emb:
        return None
    return np.concatenate([emb[src_id], emb[tgt_id], context]).astype(np.float32)


def input_dim(emb_dim: int) -> int:
    """Dimensionality of :func:`edge_input` (src + tgt + context)."""
    return 3 * emb_dim


def visible_node_embeddings(encoder, graph: Graph) -> Dict[str, np.ndarray]:
    """Node embeddings for the (visible) graph, keyed by node id."""
    return node_embeddings(encoder, graph) if graph.nodes else {}


def build_model(in_dim: int, hidden_dim: int, n_relations: int):
    """A 2-layer MLP relation classifier (logits over RELATION_TYPES)."""
    torch = _require_torch()
    import torch.nn as nn

    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, n_relations),
    )


def predict_relation_probs(
    model,
    emb: Dict[str, np.ndarray],
    src_id: str,
    tgt_id: str,
    context: np.ndarray,
    emb_dim: int,
) -> Optional[np.ndarray]:
    """Softmax probabilities over RELATION_TYPES for one (src, tgt) pair.

    Returns None if either endpoint lacks an embedding.
    """
    torch = _require_torch()
    x = edge_input(emb, src_id, tgt_id, context, emb_dim)
    if x is None:
        return None
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(x).float().unsqueeze(0))
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        return probs.numpy()
