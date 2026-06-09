"""Refine LLM graphs with the trained edge-plausibility classifier.

The world model acts here as a graph CRITIC/REFINER (not a future-state
predictor): it scores each edge and (a) prunes edges below the tuned threshold
and (b) optionally relabels an edge to its highest-scoring type-plausible
relation.

Two graph sources are supported:
* ``--mode refine``         — refine the existing LLM (EIR) graphs.
* ``--mode fully_connected`` — start from the type-plausible fully-connected
  candidate set over the LLM graph's nodes and keep only edges the classifier
  scores above threshold (the FullyConnected+WM ablation: can the world model
  recover relations from nodes alone?).

Every refined graph is tagged with its patient split (train/val/test) taken from
the trained model's saved split, so downstream evaluation can report held-out
(test) numbers — see this module's README and the Q&A note in the main README.

Run::

    python -m graph_jepa.training.edge_plausibility_classifier.refine_graphs \
        --config graph_jepa/config.yaml [--mode refine|fully_connected]
"""

from __future__ import annotations

import argparse
from typing import Dict

import numpy as np

from graph_jepa.common.encoders import build_encoder
from graph_jepa.common.graph_schema import Edge, Graph
from graph_jepa.common.graph_utils import (
    candidate_relations_for,
    fully_connected_candidates,
)
from graph_jepa.common.io_utils import (
    LOG,
    ensure_dir,
    load_config,
    load_graphs,
    read_json,
    save_graph,
    setup_logging,
    write_json,
)
from graph_jepa.training.edge_plausibility_classifier.featurize import EdgeFeaturizer
from graph_jepa.training.edge_plausibility_classifier.model import TrainedModel

METHOD = "edge_plausibility_classifier"


def _split_lookup(cfg) -> Dict[str, str]:
    """Map patient_id -> split from the trained model's metrics.json."""
    metrics_path = cfg.path("paths.training_outputs") / METHOD / "metrics.json"
    if not metrics_path.exists():
        return {}
    split = read_json(metrics_path).get("split", {})
    return {pid: name for name, ids in split.items() for pid in ids}


def _score_edges(featurizer, model, graph: Graph, edges):
    """Return per-edge plausibility probabilities aligned with ``edges``."""
    emb = featurizer.graph_embeddings(graph)
    feats, keep = [], []
    for i, e in enumerate(edges):
        f = featurizer.feature(graph, e, emb)
        if f is not None:
            feats.append(f); keep.append(i)
    probs = np.zeros(len(edges), dtype=np.float32)
    if feats:
        scored = model.predict_proba(np.vstack(feats))
        for j, i in enumerate(keep):
            probs[i] = scored[j]
    return probs, emb


def refine_graph(featurizer, model, graph: Graph, cfg) -> Graph:
    """Prune + relabel an existing LLM graph."""
    mcfg = cfg.get(f"training.{METHOD}", {})
    prune = bool(mcfg.get("prune_below_threshold", True))
    relabel = bool(mcfg.get("relabel_to_best_relation", True))
    thr = model.threshold
    idx = graph.node_index()

    out_edges = []
    for e in graph.edges:
        best_rel, best_p = e.relation, None
        if relabel:
            s, t = idx.get(e.source), idx.get(e.target)
            cands = candidate_relations_for(s, t) if s and t else []
            cands = list(dict.fromkeys([e.relation, *cands]))  # current first, dedup
            variants = [Edge(e.source, e.target, r) for r in cands]
            probs, _ = _score_edges(featurizer, model, graph, variants)
            if len(probs):
                bi = int(np.argmax(probs))
                best_rel, best_p = cands[bi], float(probs[bi])
        else:
            best_p = float(_score_edges(featurizer, model, graph, [e])[0][0])

        if prune and best_p is not None and best_p < thr:
            continue  # drop implausible edge
        out_edges.append(Edge(e.source, e.target, best_rel, e.evidence,
                              round(best_p, 4) if best_p is not None else e.confidence))

    g = Graph(graph.patient_id, f"refined:{METHOD}", list(graph.nodes), out_edges)
    g.extra = {"refiner": METHOD, "n_edges_in": len(graph.edges), "n_edges_out": len(out_edges)}
    return g


def fully_connected_refine(featurizer, model, graph: Graph) -> Graph:
    """Keep only above-threshold edges from the type-plausible candidate set."""
    candidates = fully_connected_candidates(graph)
    probs, _ = _score_edges(featurizer, model, graph, candidates)
    thr = model.threshold
    out_edges = [
        Edge(c.source, c.target, c.relation, "", round(float(p), 4))
        for c, p in zip(candidates, probs) if p >= thr
    ]
    g = Graph(graph.patient_id, "refined:fully_connected_wm", list(graph.nodes), out_edges)
    g.extra = {"refiner": "fully_connected_wm", "n_candidates": len(candidates), "n_kept": len(out_edges)}
    return g


def main() -> None:
    ap = argparse.ArgumentParser(description="Refine LLM graphs with the edge classifier")
    ap.add_argument("--config", default="graph_jepa/config.yaml")
    ap.add_argument("--mode", choices=["refine", "fully_connected"], default="refine")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    setup_logging(args.log_level)
    cfg = load_config(args.config)

    model_path = cfg.path("paths.training_outputs") / METHOD / "model.pkl"
    if not model_path.exists():
        LOG.error("No trained model at %s. Run edge_plausibility_classifier.train first.", model_path)
        return
    model = TrainedModel.load(model_path)
    encoder = build_encoder(cfg)
    featurizer = EdgeFeaturizer(encoder)

    llm = load_graphs(cfg.path("paths.llm_graphs"))
    if not llm:
        LOG.error("No LLM graphs in %s. Run create_llm_graphs first.", cfg.path("paths.llm_graphs"))
        return
    split = _split_lookup(cfg)

    method_name = METHOD if args.mode == "refine" else "fully_connected_wm"
    out_dir = ensure_dir(cfg.path("paths.refined_graphs") / method_name)
    # We refine ALL patients so QA can use any; each graph records its split so
    # tier-1 / QA evaluation can restrict to held-out (test) patients.
    counts = {"train": 0, "val": 0, "test": 0, "unknown": 0}
    for pid, g in llm.items():
        if args.mode == "fully_connected":
            refined = fully_connected_refine(featurizer, model, g)
        else:
            refined = refine_graph(featurizer, model, g, cfg)
        sp = split.get(pid, "unknown")
        refined.extra["split"] = sp
        counts[sp] = counts.get(sp, 0) + 1
        save_graph(out_dir, refined)
    write_json(out_dir / "_manifest.json", {"method": method_name, "mode": args.mode,
                                            "n_graphs": len(llm), "by_split": counts})
    LOG.info("DONE refine (%s): %d graphs -> %s | by split: %s", args.mode, len(llm), out_dir, counts)


if __name__ == "__main__":
    main()
