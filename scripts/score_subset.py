"""Score a student KG against a subset of the curated baseline.

The curated unified graph spans all 20 patients, so comparing a 3-patient
student KG against it makes recall look artificially low.  This script
filters the curated baseline down to just the ``--res-ids`` you ran, then
runs the normal similarity scorer.

Usage:
    python scripts/score_subset.py \
        --student ./my_kg_karma_unified/unified_graph_my_kg_karma.json \
        --baseline ./data/human_curated/unified_graph_curated.json \
        --res-ids RES0198 RES0199 RES0200
"""

import argparse
import json
from pathlib import Path

from Clinical_KG_OS_LLM.kg_similarity_scorer import compute_similarity, print_report


def filter_baseline(baseline: dict, res_ids: set[str]) -> dict:
    """Keep only nodes that appear in ``res_ids`` and edges whose res_id is in set."""
    kept_nodes = []
    for n in baseline.get("nodes", []):
        occs = [o for o in n.get("occurrences", []) if o.get("res_id") in res_ids]
        if occs:
            kept = dict(n)
            kept["occurrences"] = occs
            kept_nodes.append(kept)

    kept_ids = {n["id"] for n in kept_nodes}
    kept_edges = [
        e for e in baseline.get("edges", [])
        if e.get("res_id") in res_ids
        and e.get("source_id") in kept_ids
        and e.get("target_id") in kept_ids
    ]
    return {"nodes": kept_nodes, "edges": kept_edges}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--student", required=True)
    p.add_argument("--baseline", required=True)
    p.add_argument("--res-ids", nargs="+", required=True)
    args = p.parse_args()

    student = json.loads(Path(args.student).read_text())
    baseline = json.loads(Path(args.baseline).read_text())
    filtered = filter_baseline(baseline, set(args.res_ids))

    out_path = Path(args.baseline).with_name(
        f"curated_subset_{'_'.join(sorted(args.res_ids))}.json"
    )
    out_path.write_text(json.dumps(filtered, indent=2))
    print(f"Subset baseline: {len(filtered['nodes'])} nodes, "
          f"{len(filtered['edges'])} edges -> {out_path}")
    print()

    result = compute_similarity(student, filtered)
    print_report(result, args.student, str(out_path))


if __name__ == "__main__":
    main()
