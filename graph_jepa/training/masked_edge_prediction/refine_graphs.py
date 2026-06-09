"""Refine LLM graphs by masked-edge relation recovery / completion (ABLATION).

Using the trained masked-edge predictor, each LLM graph is refined in two
phases, where the *context* is always the mean of the graph's node embeddings:

1. **Relabel / drop existing edges.** For each edge, predict relation
   probabilities from its endpoints + context. If the predicted probability of
   the CURRENT relation is below ``completion_threshold``, relabel the edge to
   the argmax relation — unless even the argmax probability is below the
   threshold, in which case the edge is dropped.
2. **Add high-confidence missing edges.** Enumerate type-plausible node pairs
   with :func:`graph_utils.fully_connected_candidates` and add any whose
   predicted relation probability exceeds a high threshold.

Run::

    python -m graph_jepa.training.masked_edge_prediction.refine_graphs --config graph_jepa/config.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np

from ...common.encoders import build_encoder
from ...common.graph_schema import RELATION_TYPES, Edge, Graph
from ...common.graph_utils import fully_connected_candidates
from ...common.io_utils import (
    Config,
    LOG,
    ensure_dir,
    load_config,
    load_graphs,
    save_graph,
    setup_logging,
)
from .model import (
    build_model,
    context_vector,
    predict_relation_probs,
    relation_to_index,
    visible_node_embeddings,
)

_METHOD = "masked_edge_prediction"
# Probability a *new* (currently absent) edge must clear to be added.
_ADD_THRESHOLD = 0.9


def _require_torch():
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "pip install torch — needed for this ablation (masked_edge_prediction)"
        ) from exc
    import torch

    return torch


def load_predictor(cfg: Config):
    """Load the trained predictor checkpoint; return ``(model, emb_dim)``."""
    torch = _require_torch()
    ckpt_path = cfg.path("paths.training_outputs") / _METHOD / "model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"no trained predictor at {ckpt_path} — run train.py first"
        )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_model(int(ckpt["in_dim"]), int(ckpt["hidden_dim"]), int(ckpt["n_relations"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, int(ckpt["emb_dim"])


def refine_graph(
    model,
    encoder,
    emb_dim: int,
    graph: Graph,
    completion_threshold: float,
) -> Graph:
    """Relabel/drop existing edges then add high-confidence missing edges."""
    if not graph.nodes:
        return Graph(graph.patient_id, f"refined:{_METHOD}", [], [], dict(graph.extra))

    emb = visible_node_embeddings(encoder, graph)
    ctx = context_vector(emb, emb_dim)

    new_edges: List[Edge] = []
    n_relabel = n_drop = 0
    for e in graph.edges:
        probs = predict_relation_probs(model, emb, e.source, e.target, ctx, emb_dim)
        if probs is None:
            # Endpoint missing an embedding (e.g. dangling); keep edge unchanged.
            new_edges.append(e)
            continue
        cur_idx = relation_to_index(e.relation)
        cur_prob = float(probs[cur_idx]) if cur_idx is not None else 0.0
        if cur_prob >= completion_threshold:
            new_edges.append(e)
            continue
        best_idx = int(np.argmax(probs))
        best_prob = float(probs[best_idx])
        if best_prob < completion_threshold:
            n_drop += 1
            continue  # drop low-confidence edge
        new_edges.append(
            Edge(e.source, e.target, RELATION_TYPES[best_idx], e.evidence, best_prob)
        )
        n_relabel += 1

    existing = {(e.source, e.target, e.relation) for e in new_edges}
    n_add = 0
    for cand in fully_connected_candidates(graph):
        if (cand.source, cand.target, cand.relation) in existing:
            continue
        probs = predict_relation_probs(model, emb, cand.source, cand.target, ctx, emb_dim)
        if probs is None:
            continue
        rel_idx = relation_to_index(cand.relation)
        if rel_idx is None:
            continue
        p = float(probs[rel_idx])
        if p > _ADD_THRESHOLD:
            new_edges.append(Edge(cand.source, cand.target, cand.relation, "", p))
            existing.add((cand.source, cand.target, cand.relation))
            n_add += 1

    LOG.info(
        "  %s: relabel=%d drop=%d add=%d",
        graph.patient_id,
        n_relabel,
        n_drop,
        n_add,
    )
    return Graph(
        graph.patient_id, f"refined:{_METHOD}", list(graph.nodes), new_edges, dict(graph.extra)
    )


def refine_all(cfg: Config) -> Path:
    """Refine every LLM graph and write results under refined_graphs/<method>."""
    tcfg = cfg.get("training.masked_edge_prediction", {}) or {}
    completion_threshold = float(tcfg.get("completion_threshold", 0.5))

    llm_graphs = load_graphs(cfg.path("paths.llm_graphs"))
    if not llm_graphs:
        LOG.warning("no LLM graphs under %s — nothing to refine", cfg.path("paths.llm_graphs"))

    model, emb_dim = load_predictor(cfg)
    encoder = build_encoder(cfg)

    out_dir = ensure_dir(cfg.path("paths.refined_graphs") / _METHOD)
    for pid in sorted(llm_graphs):
        refined = refine_graph(model, encoder, emb_dim, llm_graphs[pid], completion_threshold)
        save_graph(out_dir, refined)
    LOG.info("refined %d graph(s) -> %s", len(llm_graphs), out_dir)
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="graph_jepa/config.yaml")
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    refine_all(cfg)


if __name__ == "__main__":
    main()
