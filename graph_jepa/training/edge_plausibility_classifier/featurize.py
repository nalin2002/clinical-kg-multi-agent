"""Triple featurization for the edge-plausibility classifier.

A candidate edge ``(source_node, relation, target_node, context)`` is turned
into a fixed-width feature vector:

    [ emb(src) , emb(tgt) , emb(src)-emb(tgt) , emb(src)*emb(tgt) ,
      onehot(src_type) , onehot(tgt_type) , onehot(relation) ]

The embedding captures node semantics; the element-wise difference/product give
the classifier relational signal; the one-hots inject the typed schema. The
"context" is the source/target node text (already encoded) — a full-graph
context vector is intentionally omitted to keep the default model light and
fast for a short deadline (the masked-edge ablation models richer context).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from graph_jepa.common.encoders import node_embeddings
from graph_jepa.common.graph_schema import (
    NODE_TYPES,
    RELATION_TYPES,
    Edge,
    Graph,
)

_NODE_IDX = {t: i for i, t in enumerate(NODE_TYPES)}
_REL_IDX = {r: i for i, r in enumerate(RELATION_TYPES)}


def _onehot(value: str, table: Dict[str, int]) -> np.ndarray:
    v = np.zeros(len(table), dtype=np.float32)
    if value in table:
        v[table[value]] = 1.0
    return v


class EdgeFeaturizer:
    """Stateless featurizer bound to an encoder. Embeddings are cached per graph."""

    def __init__(self, encoder):
        self.encoder = encoder

    def graph_embeddings(self, graph: Graph) -> Dict[str, np.ndarray]:
        return node_embeddings(self.encoder, graph)

    def feature(self, graph: Graph, edge: Edge, emb: Dict[str, np.ndarray] | None = None) -> np.ndarray | None:
        emb = emb if emb is not None else self.graph_embeddings(graph)
        idx = graph.node_index()
        s, t = idx.get(edge.source), idx.get(edge.target)
        if s is None or t is None or edge.source not in emb or edge.target not in emb:
            return None
        es, et = emb[edge.source], emb[edge.target]
        return np.concatenate([
            es, et, es - et, es * et,
            _onehot(s.type, _NODE_IDX),
            _onehot(t.type, _NODE_IDX),
            _onehot(edge.relation, _REL_IDX),
        ]).astype(np.float32)

    def feature_dim(self, emb_dim: int) -> int:
        return 4 * emb_dim + 2 * len(NODE_TYPES) + len(RELATION_TYPES)


def build_dataset(
    featurizer: EdgeFeaturizer,
    graphs: Dict[str, Graph],
    negatives: Dict[str, List[Tuple[Edge, str]]],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Assemble (X, y, patient_ids) from positive edges + sampled negatives.

    ``negatives[pid]`` is a list of ``(edge, strategy)`` for that patient's
    graph (see :func:`graph_utils.sample_negatives`).
    """
    X: List[np.ndarray] = []
    y: List[int] = []
    pids: List[str] = []
    for pid, g in graphs.items():
        emb = featurizer.graph_embeddings(g)
        for e in g.edges:                       # positives
            f = featurizer.feature(g, e, emb)
            if f is not None:
                X.append(f); y.append(1); pids.append(pid)
        for neg, _strat in negatives.get(pid, []):   # negatives
            f = featurizer.feature(g, neg, emb)
            if f is not None:
                X.append(f); y.append(0); pids.append(pid)
    if not X:
        return np.zeros((0, 1), dtype=np.float32), np.zeros((0,), dtype=np.int64), []
    return np.vstack(X), np.asarray(y, dtype=np.int64), pids
