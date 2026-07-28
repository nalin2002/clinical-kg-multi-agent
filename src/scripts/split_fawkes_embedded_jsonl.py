#!/usr/bin/env python3
"""Split the monolithic Fawkes embedded JSONL into per-admission sub-KG JSONs.

Reads each line from the JSONL, extracts nodes/edges/note/note_embedding,
and writes one ``<subject_id>_<hadm_id>.json`` per admission under
``train/`` or ``test/``.  The output format matches what
``MimicSubKGGraphBuilder`` expects for Graph-JEPA v5/v6 training.

Usage:
    python split_fawkes_embedded_jsonl.py \\
        data/fawkes-training-graph-embedded-260615/fawkes_training_graph_full_embedded_260615.jsonl

    python split_fawkes_embedded_jsonl.py \\
        data/fawkes-training-graph-embedded-260615/fawkes_training_graph_full_embedded_260615.jsonl \\
        --out outputs/fawkes_embedded_split/sub_kgs \\
        --test-patients 200 --split-seed 42
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path


DEFAULT_OUT = Path("outputs/fawkes_embedded_split/sub_kgs")
EDGE_LABEL_KEYS = (
    "model", "omop_src", "omop_src_cos", "omop_dst", "omop_dst_cos",
    "omop_lca_dist", "drug_link_cos", "rxcui", "dx_disease_cos",
    "matched_disease", "het_treats_ctd", "het_treats_cpd", "het_drug_cos",
    "het_dx_cos", "het_resembles_drd", "het_presents_dps", "prov_in_note",
    "prov_ratio",
)


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            yield lineno, json.loads(line)


def flatten_edge(edge: dict) -> dict:
    """Flatten nested ``labels`` dict onto the edge for v5/v6 compatibility."""
    out = {
        "source": edge["source"],
        "target": edge["target"],
        "relation": edge.get("relation", ""),
        "confidence": edge.get("confidence", 1.0),
        "evidence": edge.get("evidence", "structured"),
    }
    labels = edge.get("labels", {})
    if isinstance(labels, dict):
        for k in EDGE_LABEL_KEYS:
            out[k] = labels.get(k)
    else:
        for k in EDGE_LABEL_KEYS:
            out[k] = edge.get(k)
    return out


def split_assignments(
    subject_ids: list[str],
    test_patients: int,
    seed: int,
) -> dict[str, str]:
    """Assign subject_ids to train/test splits deterministically."""
    unique = sorted(set(subject_ids))
    rng = random.Random(seed)
    rng.shuffle(unique)
    test_set = set(unique[:test_patients])
    return {sid: ("test" if sid in test_set else "train") for sid in unique}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("jsonl", help="Fawkes embedded JSONL file")
    parser.add_argument("--out", default=str(DEFAULT_OUT),
                        help=f"Output directory (default: {DEFAULT_OUT})")
    parser.add_argument("--test-patients", type=int, default=200,
                        help="Number of patients to place in test/ (default: 200)")
    parser.add_argument("--split-seed", type=int, default=42,
                        help="Seed for train/test split (default: 42)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional limit on number of rows to process")
    parser.add_argument("--flatten-labels", action="store_true", default=True,
                        help="Flatten edge labels dict onto each edge (default: True)")
    parser.add_argument("--no-flatten-labels", action="store_false", dest="flatten_labels",
                        help="Keep nested labels dict on edges")
    args = parser.parse_args()

    source = Path(args.jsonl)
    out_dir = Path(args.out)

    print(f"[split] pass 1: collecting subject_ids from {source} ...")
    rows = []
    for lineno, row in iter_jsonl(source):
        rows.append(row)
        if args.limit is not None and len(rows) >= args.limit:
            break

    sid_list = [str(row.get("subject_id", "")) for row in rows]
    split_map = split_assignments(sid_list, args.test_patients, args.split_seed)
    unique_subjects = len(set(sid_list))

    train_dir = out_dir / "train"
    test_dir = out_dir / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    print(f"[split] pass 2: writing {len(rows)} graphs "
          f"({unique_subjects} subjects, test_patients={args.test_patients}) ...")

    written = Counter()
    total_nodes = Counter()
    total_edges = Counter()
    for row in rows:
        subject_id = str(row.get("subject_id", ""))
        hadm_id = str(row.get("hadm_id", ""))
        split = split_map.get(subject_id, "train")

        nodes = row.get("nodes", [])
        edges = row.get("edges", [])
        if args.flatten_labels:
            edges = [flatten_edge(e) for e in edges]

        graph = {
            "subject_id": subject_id,
            "hadm_id": hadm_id,
            "hadm_ids": [hadm_id],
            "split": split,
            "nodes": nodes,
            "edges": edges,
            "n_nodes": len(nodes),
            "n_edges": len(edges),
        }

        note = row.get("note")
        if note:
            graph["note"] = note
        note_emb = row.get("note_embedding")
        if note_emb:
            graph["note_embedding"] = note_emb
            graph["embed_model"] = row.get("embed_model", "")
            graph["embed_dim"] = row.get("embed_dim", len(note_emb))

        dest_dir = test_dir if split == "test" else train_dir
        path = dest_dir / f"{subject_id}_{hadm_id}.json"
        path.write_text(json.dumps(graph, indent=2, ensure_ascii=False),
                        encoding="utf-8")
        written[split] += 1
        total_nodes[split] += len(nodes)
        total_edges[split] += len(edges)

    manifest = {
        "method": "split_fawkes_embedded_jsonl",
        "source": str(source),
        "output_dir": str(out_dir),
        "split_seed": args.split_seed,
        "test_patients": args.test_patients,
        "flatten_labels": args.flatten_labels,
        "total_graphs": sum(written.values()),
        "train_graphs": written["train"],
        "test_graphs": written["test"],
        "unique_subjects": unique_subjects,
        "total_nodes": sum(total_nodes.values()),
        "total_edges": sum(total_edges.values()),
        "train_nodes": total_nodes["train"],
        "test_nodes": total_nodes["test"],
        "train_edges": total_edges["train"],
        "test_edges": total_edges["test"],
    }
    (out_dir / "_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"[split] done: {sum(written.values())} graphs -> {out_dir}")
    print(f"[split]   train={written['train']} test={written['test']}")
    print(f"[split]   nodes: train={total_nodes['train']} test={total_nodes['test']}")
    print(f"[split]   edges: train={total_edges['train']} test={total_edges['test']}")


if __name__ == "__main__":
    main()
