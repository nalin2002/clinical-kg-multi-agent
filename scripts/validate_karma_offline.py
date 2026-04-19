"""Offline validation for the KARMA pipeline's rule-based agents.

Loads the naive sub-KGs in ``data/naive_results/sub_kgs``, runs them through
the Schema Agent + Entity/Conflict Resolver (no LLM calls), and reports how
many node-type remaps / duplicate-edge drops happened.  Also writes the
cleaned sub-KGs to ``/tmp/karma_schema_only/`` so ``dump_graph`` can be run
against them for an end-to-end sanity check.

Usage:
    python scripts/validate_karma_offline.py
"""

import json
from pathlib import Path

from Clinical_KG_OS_LLM.karma_kg_extraction import (
    schema_agent,
    entity_conflict_agent,
)
from Clinical_KG_OS_LLM.paths import project_root


def main() -> None:
    root = project_root()
    src_dir = root / "data" / "naive_results" / "sub_kgs"
    out_dir = Path("/tmp/karma_schema_only")
    out_dir.mkdir(parents=True, exist_ok=True)

    totals = {
        "sub_kgs": 0,
        "pre_nodes": 0,
        "pre_edges": 0,
        "post_nodes": 0,
        "post_edges": 0,
        "node_type_remapped": 0,
        "node_dropped_unknown_type": 0,
        "edge_type_remapped": 0,
        "edge_dropped_unknown_type": 0,
        "duplicate_nodes_merged": 0,
        "duplicate_edges_dropped": 0,
        "dangling_edges_dropped": 0,
    }

    for path in sorted(src_dir.glob("RES*.json")):
        kg = json.loads(path.read_text())
        totals["sub_kgs"] += 1
        totals["pre_nodes"] += len(kg.get("nodes", []))
        totals["pre_edges"] += len(kg.get("edges", []))

        kg, schema_report = schema_agent(kg)
        kg, dedupe_report = entity_conflict_agent(kg)

        for k, v in schema_report.items():
            if k in totals:
                totals[k] += v
        for k, v in dedupe_report.items():
            if k in totals:
                totals[k] += v

        totals["post_nodes"] += len(kg["nodes"])
        totals["post_edges"] += len(kg["edges"])

        (out_dir / path.name).write_text(json.dumps(kg, indent=2, ensure_ascii=False))

    print("=" * 60)
    print("KARMA rule-based agents - offline validation on naive sub-KGs")
    print("=" * 60)
    for k, v in totals.items():
        print(f"  {k:30s}  {v}")
    print(f"\nCleaned sub-KGs written to {out_dir}")


if __name__ == "__main__":
    main()
