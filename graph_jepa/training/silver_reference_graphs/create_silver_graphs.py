"""Create SILVER-STANDARD reference graphs from clinician notes.

For every ACI-Bench patient with a note, an LLM extracts a structured clinical
knowledge graph (the 7-type / 6-relation schema) from the NOTE ONLY. These are
the positive supervision signal for the world-model refiners and the reference
graphs for tier-1 evaluation.

IMPORTANT: these are *silver-standard, clinician-note-derived* references — NOT
gold human-curated graphs. The note is clinician-written, but the graph
structure is LLM-extracted and unverified.

Run::

    python -m graph_jepa.training.silver_reference_graphs.create_silver_graphs \
        --config graph_jepa/config.yaml [--limit N] [--overwrite]

Output: graph_jepa/training/outputs/silver_reference_graphs/<patient_id>.json
"""

from __future__ import annotations

import argparse

from graph_jepa.common.graph_schema import Edge, Graph, Node, validate_graph
from graph_jepa.common.io_utils import (
    LOG,
    discover_patients,
    ensure_dir,
    graph_path,
    load_config,
    save_graph,
    setup_logging,
    write_json,
)
from graph_jepa.common.llm_utils import LLMClient, load_prompt


def graph_from_llm_json(patient_id: str, payload: dict) -> Graph:
    """Build a canonical :class:`Graph` from the LLM's silver-extraction JSON."""
    nodes = []
    for i, n in enumerate(payload.get("nodes", []), 1):
        nodes.append(
            Node(
                id=str(n.get("id") or f"N_{i:03d}"),
                name=n.get("name") or n.get("text") or "",
                type=n.get("type", ""),
                normalized_name=n.get("normalized_name", ""),
            )
        )
    valid_ids = {n.id for n in nodes}
    edges = []
    for e in payload.get("edges", []):
        src, tgt = str(e.get("source") or e.get("source_id")), str(e.get("target") or e.get("target_id"))
        if src not in valid_ids or tgt not in valid_ids:
            continue  # drop dangling edges the LLM hallucinated
        edges.append(
            Edge(
                source=src,
                target=tgt,
                relation=e.get("relation") or e.get("type") or "",
                evidence=e.get("evidence", ""),
                confidence=float(e.get("confidence", 1.0)),
            )
        )
    return Graph(patient_id=patient_id, source="clinician_note", nodes=nodes, edges=edges)


def main() -> None:
    ap = argparse.ArgumentParser(description="Create silver reference graphs from clinician notes")
    ap.add_argument("--config", default="graph_jepa/config.yaml")
    ap.add_argument("--limit", type=int, default=None, help="cap number of patients (smoke test)")
    ap.add_argument("--overwrite", action="store_true", help="re-extract even if output exists")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    setup_logging(args.log_level)
    cfg = load_config(args.config)
    out_dir = ensure_dir(cfg.path("paths.silver_graphs"))
    client = LLMClient(cfg)  # uses llm.provider
    system = load_prompt("silver_graph_extraction")

    patients = discover_patients(cfg)
    if args.limit:
        patients = patients[: args.limit]
    with_notes = [p for p in patients if p.has_note]
    LOG.info("silver graphs: %d/%d patients have notes (provider=%s)",
             len(with_notes), len(patients), client.provider)

    n_ok = n_skip = n_fail = 0
    for p in with_notes:
        dest = graph_path(out_dir, p.patient_id)
        if dest.exists() and not args.overwrite:
            n_skip += 1
            continue
        note = p.read_note()
        if not note or not note.strip():
            LOG.warning("%s: empty note, skipping", p.patient_id)
            continue
        try:
            payload = client.complete_json(system, f"=== CLINICIAN NOTE ===\n{note}\n\nReturn ONLY the JSON object.")
            graph = graph_from_llm_json(p.patient_id, payload)
        except Exception as exc:  # noqa: BLE001
            LOG.error("%s: extraction failed: %s", p.patient_id, exc)
            n_fail += 1
            continue
        warns = validate_graph(graph)
        if warns:
            LOG.debug("%s: %d schema warnings", p.patient_id, len(warns))
        save_graph(out_dir, graph)
        n_ok += 1
        LOG.info("%s: %d nodes, %d edges", p.patient_id, len(graph.nodes), len(graph.edges))

    write_json(out_dir / "_manifest.json",
               {"created": n_ok, "skipped_existing": n_skip, "failed": n_fail,
                "provider": client.provider, "model": client.model})
    LOG.info("DONE silver graphs: %d created, %d skipped, %d failed -> %s",
             n_ok, n_skip, n_fail, out_dir)


if __name__ == "__main__":
    main()
