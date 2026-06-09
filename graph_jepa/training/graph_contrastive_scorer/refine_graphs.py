"""Refine LLM graphs with the graph-level contrastive scorer (ABLATION).

Greedy hill-climbing: starting from each LLM graph, candidate single edits are
proposed (drop an existing edge, or relabel an edge to a type-plausible
relation via :func:`graph_utils.candidate_relations_for`). An edit is kept only
if it strictly INCREASES the whole-graph plausibility score. Up to
``edits_per_graph`` edits are applied per graph.

Run::

    python -m graph_jepa.training.graph_contrastive_scorer.refine_graphs --config graph_jepa/config.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

from ...common.encoders import build_encoder
from ...common.graph_schema import Edge, Graph
from ...common.graph_utils import candidate_relations_for
from ...common.io_utils import (
    Config,
    LOG,
    ensure_dir,
    load_config,
    load_graphs,
    save_graph,
    setup_logging,
)
from .model import build_model, input_dim, score_graph

_METHOD = "graph_contrastive_scorer"


def _require_torch():
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "pip install torch — needed for this ablation (graph_contrastive_scorer)"
        ) from exc
    import torch

    return torch


def load_scorer(cfg: Config):
    """Load the trained scorer checkpoint; return ``(model, emb_dim)``."""
    torch = _require_torch()
    ckpt_path = cfg.path("paths.training_outputs") / _METHOD / "model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"no trained scorer at {ckpt_path} — run train.py first"
        )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_model(int(ckpt["in_dim"]), int(ckpt["hidden_dim"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, int(ckpt["emb_dim"])


def _candidate_edits(graph: Graph) -> List[tuple]:
    """Enumerate single-edit proposals as ``(kind, edge_index, new_edge|None)``.

    * ``("drop", i, None)``         — delete edge i.
    * ``("relabel", i, new_edge)``  — replace edge i with a type-plausible relabel.
    """
    idx = graph.node_index()
    edits: List[tuple] = []
    for i, e in enumerate(graph.edges):
        edits.append(("drop", i, None))
        src, tgt = idx.get(e.source), idx.get(e.target)
        if not src or not tgt:
            continue
        for rel in candidate_relations_for(src, tgt):
            if rel == e.relation:
                continue
            new_edge = Edge(e.source, e.target, rel, e.evidence, e.confidence)
            edits.append(("relabel", i, new_edge))
    return edits


def _apply_edit(graph: Graph, edit: tuple) -> Graph:
    """Return a new graph with one edit applied (inputs untouched)."""
    kind, i, new_edge = edit
    edges = list(graph.edges)
    if kind == "drop":
        edges.pop(i)
    elif kind == "relabel":
        edges[i] = new_edge
    return Graph(graph.patient_id, graph.source, list(graph.nodes), edges, dict(graph.extra))


def refine_graph(
    model,
    encoder,
    emb_dim: int,
    graph: Graph,
    edits_per_graph: int,
) -> Graph:
    """Greedily apply up to ``edits_per_graph`` score-increasing single edits."""
    current = Graph(
        graph.patient_id, f"refined:{_METHOD}", list(graph.nodes), list(graph.edges), dict(graph.extra)
    )
    if not current.edges:
        return current

    applied = 0
    while applied < edits_per_graph:
        base_score = score_graph(model, encoder, current, emb_dim)
        best_gain = 0.0
        best_graph: Optional[Graph] = None
        for edit in _candidate_edits(current):
            cand = _apply_edit(current, edit)
            gain = score_graph(model, encoder, cand, emb_dim) - base_score
            if gain > best_gain:
                best_gain = gain
                best_graph = cand
        if best_graph is None:
            break  # no edit improves the score
        current = best_graph
        applied += 1
        if not current.edges:
            break
    LOG.info("  %s: applied %d edit(s)", graph.patient_id, applied)
    return current


def refine_all(cfg: Config) -> Path:
    """Refine every LLM graph and write results under refined_graphs/<method>."""
    tcfg = cfg.get("training.graph_contrastive_scorer", {}) or {}
    edits_per_graph = int(tcfg.get("edits_per_graph", 50))

    llm_graphs = load_graphs(cfg.path("paths.llm_graphs"))
    if not llm_graphs:
        LOG.warning("no LLM graphs under %s — nothing to refine", cfg.path("paths.llm_graphs"))

    model, emb_dim = load_scorer(cfg)
    encoder = build_encoder(cfg)

    out_dir = ensure_dir(cfg.path("paths.refined_graphs") / _METHOD)
    for pid in sorted(llm_graphs):
        refined = refine_graph(model, encoder, emb_dim, llm_graphs[pid], edits_per_graph)
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
