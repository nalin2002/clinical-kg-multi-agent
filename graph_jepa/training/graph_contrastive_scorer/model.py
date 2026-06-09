"""Graph-level plausibility scorer (ABLATION).

A small torch model that pools a graph's node and edge features into a single
graph vector and emits a scalar plausibility score. It is trained with a
margin-ranking objective so a real (silver) graph scores higher than a
corrupted copy of it.

torch is imported lazily so the rest of the repo runs without it.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from ...common.graph_schema import RELATION_TYPES, Graph
from ...common.encoders import node_embeddings


def _require_torch():
    try:
        import torch  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "pip install torch — needed for this ablation (graph_contrastive_scorer)"
        ) from exc
    import torch

    return torch


# --------------------------------------------------------------------------- #
# Featurization (numpy, torch-free)
# --------------------------------------------------------------------------- #
_REL_INDEX: Dict[str, int] = {r: i for i, r in enumerate(RELATION_TYPES)}


def _relation_onehot(relation: str) -> np.ndarray:
    vec = np.zeros(len(RELATION_TYPES), dtype=np.float32)
    idx = _REL_INDEX.get((relation or "").upper())
    if idx is not None:
        vec[idx] = 1.0
    return vec


def graph_features(encoder, graph: Graph, emb_dim: int) -> Dict[str, np.ndarray]:
    """Pool a graph into a flat feature vector.

    Returns a dict with:
      * ``node_mean``  — mean of node embeddings ``(emb_dim,)``
      * ``edge_mean``  — mean of edge features ``(2*emb_dim + n_relations,)``
    Empty node/edge sets pool to zeros so empty graphs are handled gracefully.

    The concatenation ``[node_mean, edge_mean]`` is the model input.
    """
    n_rel = len(RELATION_TYPES)
    emb = node_embeddings(encoder, graph) if graph.nodes else {}

    if emb:
        node_mat = np.vstack(list(emb.values())).astype(np.float32)
        node_mean = node_mat.mean(axis=0)
    else:
        node_mean = np.zeros(emb_dim, dtype=np.float32)

    edge_feats: List[np.ndarray] = []
    zero = np.zeros(emb_dim, dtype=np.float32)
    for e in graph.edges:
        if e.source not in emb or e.target not in emb:
            continue
        edge_feats.append(
            np.concatenate([emb[e.source], emb[e.target], _relation_onehot(e.relation)])
        )
    if edge_feats:
        edge_mean = np.vstack(edge_feats).astype(np.float32).mean(axis=0)
    else:
        edge_mean = np.concatenate([zero, zero, np.zeros(n_rel, dtype=np.float32)])

    return {"node_mean": node_mean, "edge_mean": edge_mean}


def graph_vector(encoder, graph: Graph, emb_dim: int) -> np.ndarray:
    """Flat input vector ``[node_mean, edge_mean]`` for one graph."""
    f = graph_features(encoder, graph, emb_dim)
    return np.concatenate([f["node_mean"], f["edge_mean"]]).astype(np.float32)


def input_dim(emb_dim: int) -> int:
    """Dimensionality of :func:`graph_vector` for a given embedding size."""
    n_rel = len(RELATION_TYPES)
    return emb_dim + (2 * emb_dim + n_rel)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def build_model(in_dim: int, hidden_dim: int):
    """A 2-layer MLP mapping a pooled graph vector to a scalar score."""
    torch = _require_torch()
    import torch.nn as nn

    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1),
    )


def score_graph(model, encoder, graph: Graph, emb_dim: int) -> float:
    """Run the model on one graph and return its scalar plausibility score."""
    torch = _require_torch()
    vec = graph_vector(encoder, graph, emb_dim)
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(vec).float().unsqueeze(0)
        return float(model(x).squeeze().item())
