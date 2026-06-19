#!/usr/bin/env python3
"""
canonicalize_to_curator_260426.py
==================================

Post-process a unified-graph JSON (output of dump_graph.py) by rewriting each
student node's `text` field to the nearest curator KB label of the same
entity type, IF the BGE-M3 cosine similarity is above a threshold.

Edges follow nodes for free because edges reference `source_id`/`target_id`,
not text. So canonicalizing a node's text field automatically updates every
edge endpoint that points to it.

Why this exists
---------------
The scorer's fuzzy-match (`SequenceMatcher >= 0.80`) misses lay→clinical
bridges that are semantically obvious to a human and to BGE-M3 but
lexically distant. Examples:
    swelling in my legs   ↔ peripheral edema             (cosine ~0.72)
    low pumping function  ↔ decreased ejection fraction (cosine ~0.74)
    really tired          ↔ fatigue                     (cosine ~0.66)

This step rewrites those student node labels to their curator-canonical
form so the downstream fuzzy-match scorer recognizes them as hits.

The original text is preserved in `text_original` and the cosine score in
`canonicalized_via_bge_cosine` so the rewrite is auditable.

Usage
-----
    python canonicalize_to_curator_260426.py \
        --unified  path/to/unified_graph_*.json \
        --kb       path/to/curated_kb.json \
        --output   path/to/unified_graph_*_canonicalized.json \
        [--threshold 0.70]

Created: 260426
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

DEFAULT_THRESHOLD = 0.70


def load_curated_kb_by_type(kb_path: Path) -> dict[str, list[str]]:
    """Returns {entity_type: [label, ...]} from a curated KB JSON list."""
    entries = json.loads(kb_path.read_text())
    by_type: dict[str, list[str]] = defaultdict(list)
    for e in entries:
        et = (e.get("entity_type") or "").upper()
        label = (e.get("label") or "").strip()
        if et and label:
            by_type[et].append(label)
    # Dedupe within each type while preserving order
    return {t: list(dict.fromkeys(labels)) for t, labels in by_type.items()}


def load_bge_m3():
    """Load the BGE-M3 embedding model used by dump_graph (same model)."""
    try:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    except ImportError:
        sys.exit(
            "llama_index.embeddings.huggingface required: "
            "pip install llama-index-embeddings-huggingface"
        )
    print("[canonicalize] loading BGE-M3 …", flush=True)
    return HuggingFaceEmbedding(model_name="BAAI/bge-m3")


def cosine(a, b) -> float:
    import numpy as np
    a = np.asarray(a)
    b = np.asarray(b)
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / n) if n > 0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--unified", required=True,
                    help="Path to unified_graph_*.json (output of dump_graph.py)")
    ap.add_argument("--kb", required=True,
                    help="Path to curated KB JSON (same shape as curated_kb_260419.json)")
    ap.add_argument("--output", required=True,
                    help="Path to write the canonicalized unified graph")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                    help=f"BGE-M3 cosine threshold for rewriting (default {DEFAULT_THRESHOLD})")
    ap.add_argument("--report", default=None,
                    help="Optional path to write a JSON report of the rewrites")
    args = ap.parse_args()

    unified_path = Path(args.unified)
    kb_path = Path(args.kb)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[canonicalize] unified : {unified_path}")
    print(f"[canonicalize] kb      : {kb_path}")
    print(f"[canonicalize] output  : {out_path}")
    print(f"[canonicalize] threshold: {args.threshold}")

    # ── Load ──────────────────────────────────────────────────────────────────
    unified = json.loads(unified_path.read_text())
    kb_by_type = load_curated_kb_by_type(kb_path)
    print(f"[canonicalize] curator labels by type:")
    for t, labels in kb_by_type.items():
        print(f"  {t:18} {len(labels):4} labels")

    # ── Embed curator labels (once per type) ──────────────────────────────────
    embed_model = load_bge_m3()
    label_embs_by_type: dict[str, list] = {}
    for t, labels in kb_by_type.items():
        label_embs_by_type[t] = embed_model.get_text_embedding_batch(labels)

    # ── Walk student nodes ────────────────────────────────────────────────────
    rewrites = []   # report entries
    n_nodes = 0
    n_rewritten = 0
    n_no_curator_for_type = 0

    for node in unified.get("nodes", []):
        n_nodes += 1
        ntype = (node.get("type") or "").upper()
        ntext = (node.get("text") or "").strip()
        if not ntext:
            continue
        labels = kb_by_type.get(ntype)
        if not labels:
            n_no_curator_for_type += 1
            continue
        label_embs = label_embs_by_type[ntype]

        # Embed the student node text
        node_emb = embed_model.get_text_embedding(ntext)

        # Find best curator label of same type
        best_idx = -1
        best_sim = 0.0
        for i, lemb in enumerate(label_embs):
            s = cosine(node_emb, lemb)
            if s > best_sim:
                best_sim = s
                best_idx = i

        if best_idx >= 0 and best_sim >= args.threshold:
            best_label = labels[best_idx]
            if best_label.strip().lower() == ntext.strip().lower():
                # already canonical, no rewrite
                continue
            # Rewrite
            rewrites.append({
                "node_id": node["id"],
                "type": ntype,
                "text_original": ntext,
                "text_canonical": best_label,
                "cosine": round(best_sim, 4),
            })
            node["text_original"] = ntext
            node["text"] = best_label
            node["canonicalized_via_bge_cosine"] = round(best_sim, 4)
            n_rewritten += 1

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path.write_text(json.dumps(unified, indent=2, ensure_ascii=False))
    print(f"\n[canonicalize] DONE.")
    print(f"  nodes scanned:                      {n_nodes}")
    print(f"  nodes with no curator labels (skip): {n_no_curator_for_type}")
    print(f"  nodes rewritten to canonical:        {n_rewritten}")
    print(f"  saved: {out_path}")

    if args.report:
        Path(args.report).write_text(json.dumps({
            "unified_in": str(unified_path),
            "kb": str(kb_path),
            "threshold": args.threshold,
            "n_nodes_scanned": n_nodes,
            "n_no_curator_for_type": n_no_curator_for_type,
            "n_rewritten": n_rewritten,
            "rewrites": rewrites,
        }, indent=2, ensure_ascii=False))
        print(f"  report: {args.report}")


if __name__ == "__main__":
    main()
