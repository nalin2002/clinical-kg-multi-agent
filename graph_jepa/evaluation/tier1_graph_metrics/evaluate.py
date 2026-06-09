"""Tier-1 graph-level evaluation: refined graphs vs silver reference graphs.

Compares each method's graphs against the silver clinician-note-derived
references on:
  * Edge Precision / Recall / F1   (direction- and relation-sensitive)
  * Relation Accuracy              (correct relation among endpoint-matched edges)
  * Graph Edit Distance (approx)   (edge insert+delete to reach the reference)

Methods compared (config ``evaluation.methods``), e.g.:
  * ``llm_only``                    — raw EIR graph (no refinement)
  * ``edge_plausibility_classifier``— default world-model refiner
  * ``fully_connected_wm``          — FullyConnected+WM ablation

By default only HELD-OUT (test-split) patients are scored, because the world
model was trained on the silver graphs — scoring on training patients would be
circular. Use ``--split all`` to override (clearly a transductive number).

Run::

    python -m graph_jepa.evaluation.tier1_graph_metrics.evaluate \
        --config graph_jepa/config.yaml [--split test|val|train|all]

Outputs (under graph_jepa/evaluation/outputs/tier1/):
  * per_patient.csv, aggregate.csv, summary.json, summary.md
"""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path
from typing import Dict, List

from graph_jepa.common.graph_schema import Graph
from graph_jepa.common.graph_utils import (
    edge_prf,
    graph_edit_distance_approx,
    relation_accuracy,
)
from graph_jepa.common.io_utils import (
    LOG,
    ensure_dir,
    load_config,
    load_graphs,
    read_json,
    setup_logging,
    write_json,
)


def _method_dir(cfg, method: str) -> Path:
    if method == "llm_only":
        return cfg.path("paths.llm_graphs")
    return cfg.path("paths.refined_graphs") / method


def _split_lookup(cfg) -> Dict[str, str]:
    mp = cfg.path("paths.training_outputs") / "edge_plausibility_classifier" / "metrics.json"
    if not mp.exists():
        return {}
    split = read_json(mp).get("split", {})
    return {pid: name for name, ids in split.items() for pid in ids}


def evaluate_method(method: str, graphs: Dict[str, Graph], silver: Dict[str, Graph],
                    pids: List[str], thr: float) -> List[dict]:
    rows = []
    for pid in pids:
        g, ref = graphs.get(pid), silver.get(pid)
        if g is None or ref is None:
            continue
        prf = edge_prf(g, ref, thr)
        rel = relation_accuracy(g, ref, thr)
        ged = graph_edit_distance_approx(g, ref, thr)
        rows.append({
            "patient_id": pid, "method": method,
            "edge_precision": round(prf["precision"], 4),
            "edge_recall": round(prf["recall"], 4),
            "edge_f1": round(prf["f1"], 4),
            "relation_accuracy": round(rel["relation_accuracy"], 4),
            "ged": ged["ged"], "ged_normalized": round(ged["ged_normalized"], 4),
            "pred_edges": prf["pred"], "ref_edges": prf["ref"], "matched_edges": prf["matched"],
        })
    return rows


def _aggregate(rows: List[dict]) -> List[dict]:
    by_method: Dict[str, List[dict]] = {}
    for r in rows:
        by_method.setdefault(r["method"], []).append(r)
    agg = []
    metrics = ["edge_precision", "edge_recall", "edge_f1", "relation_accuracy", "ged_normalized"]
    for method, rs in by_method.items():
        row = {"method": method, "n_patients": len(rs)}
        for m in metrics:
            vals = [r[m] for r in rs]
            row[f"mean_{m}"] = round(statistics.mean(vals), 4) if vals else 0.0
            row[f"std_{m}"] = round(statistics.pstdev(vals), 4) if len(vals) > 1 else 0.0
        agg.append(row)
    return agg


def _write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    ensure_dir(path.parent)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _markdown(agg: List[dict], split: str) -> str:
    lines = [f"# Tier-1 Graph Metrics (split = `{split}`)", "",
             "Refined graphs vs **silver** clinician-note-derived references. "
             "Higher Edge F1 / Relation Accuracy is better; lower normalized GED is better.", "",
             "| Method | n | Edge P | Edge R | Edge F1 | Rel. Acc | norm GED |",
             "|---|--:|--:|--:|--:|--:|--:|"]
    for r in sorted(agg, key=lambda x: x.get("mean_edge_f1", 0), reverse=True):
        lines.append(
            f"| {r['method']} | {r['n_patients']} | {r['mean_edge_precision']:.3f} | "
            f"{r['mean_edge_recall']:.3f} | {r['mean_edge_f1']:.3f} | "
            f"{r['mean_relation_accuracy']:.3f} | {r['mean_ged_normalized']:.3f} |")
    lines += ["", "_Silver references are LLM-extracted from clinician notes, not gold "
              "human-curated graphs; treat absolute values as relative comparisons._"]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Tier-1 graph metrics vs silver references")
    ap.add_argument("--config", default="graph_jepa/config.yaml")
    ap.add_argument("--split", default="test", choices=["test", "val", "train", "all"])
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    setup_logging(args.log_level)
    cfg = load_config(args.config)
    thr = float(cfg.get("evaluation.node_match_threshold", 0.80))

    silver = load_graphs(cfg.path("paths.silver_graphs"))
    if not silver:
        LOG.error("No silver graphs. Run create_silver_graphs first.")
        return

    split_map = _split_lookup(cfg)
    if args.split == "all" or not split_map:
        pids = sorted(silver.keys())
        if args.split != "all":
            LOG.warning("no split metadata found; scoring ALL patients (transductive)")
    else:
        pids = sorted(pid for pid in silver if split_map.get(pid) == args.split)
    LOG.info("scoring %d patients (split=%s)", len(pids), args.split)

    methods = cfg.get("evaluation.methods", ["llm_only", "edge_plausibility_classifier"])
    all_rows: List[dict] = []
    for method in methods:
        graphs = load_graphs(_method_dir(cfg, method))
        if not graphs:
            LOG.warning("method %s: no graphs found, skipping", method)
            continue
        rows = evaluate_method(method, graphs, silver, pids, thr)
        LOG.info("method %s: scored %d patients", method, len(rows))
        all_rows.extend(rows)

    if not all_rows:
        LOG.error("No rows produced — are refined graphs present?")
        return

    out = ensure_dir(cfg.path("paths.eval_outputs") / "tier1")
    agg = _aggregate(all_rows)
    _write_csv(out / "per_patient.csv", all_rows)
    _write_csv(out / "aggregate.csv", agg)
    write_json(out / "summary.json", {"split": args.split, "node_match_threshold": thr, "aggregate": agg})
    (out / "summary.md").write_text(_markdown(agg, args.split), encoding="utf-8")
    LOG.info("DONE tier-1 -> %s", out)
    print("\n" + _markdown(agg, args.split) + "\n")


if __name__ == "__main__":
    main()
