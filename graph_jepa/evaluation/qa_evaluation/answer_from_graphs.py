"""Answer each QA question using ONLY a knowledge graph (no transcript access).

For every method (llm_only, edge_plausibility_classifier, fully_connected_wm,
...) and every patient QA set, the answerer LLM sees a serialized graph and the
question — never the transcript — so the score reflects how useful that graph is
as a standalone knowledge source.

Run::

    python -m graph_jepa.evaluation.qa_evaluation.answer_from_graphs \
        --config graph_jepa/config.yaml [--methods llm_only edge_plausibility_classifier]

Output: graph_jepa/evaluation/outputs/qa_answers/<method>/<patient_id>.json
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

from graph_jepa.common.graph_schema import Graph
from graph_jepa.common.io_utils import (
    LOG,
    ensure_dir,
    load_config,
    load_graphs,
    read_json,
    setup_logging,
    write_json,
)
from graph_jepa.common.llm_utils import LLMClient, load_prompt


def serialize_graph(graph: Graph) -> str:
    """Render a graph as readable typed triples + a node inventory for the LLM."""
    idx = graph.node_index()
    lines = ["NODES:"]
    for n in graph.nodes:
        lines.append(f"  - [{n.type}] {n.name}")
    lines.append("RELATIONS:")
    if not graph.edges:
        lines.append("  (none)")
    for e in graph.edges:
        s, t = idx.get(e.source), idx.get(e.target)
        if s and t:
            conf = f" (conf={e.confidence:.2f})" if e.confidence not in (None, 1.0) else ""
            lines.append(f"  - [{s.type}] {s.name} --{e.relation}--> [{t.type}] {t.name}{conf}")
    return "\n".join(lines)


def _method_dir(cfg, method: str) -> Path:
    if method == "llm_only":
        return cfg.path("paths.llm_graphs")
    return cfg.path("paths.refined_graphs") / method


def main() -> None:
    ap = argparse.ArgumentParser(description="Answer QA questions from graphs only")
    ap.add_argument("--config", default="graph_jepa/config.yaml")
    ap.add_argument("--methods", nargs="+", default=None, help="override config evaluation.methods")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    setup_logging(args.log_level)
    cfg = load_config(args.config)
    qa_dir = cfg.path("paths.qa_sets")
    qa_sets = {p.stem: read_json(p) for p in sorted(qa_dir.glob("*.json")) if not p.stem.startswith("_")}
    if not qa_sets:
        LOG.error("No QA sets in %s. Run generate_qa_sets first.", qa_dir)
        return

    provider = cfg.get("evaluation.qa.answerer_provider") or cfg.get("llm.provider")
    client = LLMClient(cfg, provider=provider)
    template = load_prompt("graph_qa_answer")
    methods = args.methods or cfg.get("evaluation.methods", ["llm_only", "edge_plausibility_classifier"])

    pids = sorted(qa_sets.keys())
    if args.limit:
        pids = pids[: args.limit]

    for method in methods:
        graphs = load_graphs(_method_dir(cfg, method))
        if not graphs:
            LOG.warning("method %s: no graphs, skipping", method)
            continue
        out_dir = ensure_dir(cfg.path("paths.eval_outputs") / "qa_answers" / method)
        n_q = 0
        for pid in pids:
            g = graphs.get(pid)
            if g is None:
                continue
            graph_text = serialize_graph(g)
            answers = []
            for q in qa_sets[pid].get("questions", []):
                prompt = template.replace("{graph}", graph_text).replace("{question}", q["question"])
                try:
                    ans = client.complete("You answer strictly from the provided graph.", prompt).strip()
                except Exception as exc:  # noqa: BLE001
                    LOG.warning("%s/%s %s: answer failed: %s", method, pid, q["question_id"], exc)
                    ans = ""
                answers.append({"question_id": q["question_id"], "question": q["question"],
                                "predicted_answer": ans})
                n_q += 1
            write_json(out_dir / f"{pid}.json", {"patient_id": pid, "method": method, "answers": answers})
        LOG.info("method %s: answered %d questions -> %s", method, n_q, out_dir)

    LOG.info("DONE answering from graphs.")


if __name__ == "__main__":
    main()
