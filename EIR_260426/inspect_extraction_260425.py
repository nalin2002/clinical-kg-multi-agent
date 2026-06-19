#!/usr/bin/env python3
"""
inspect_extraction_260425.py
============================

Dump v8/v9/v10 per-patient extraction output for a single transcript:
all 13 categories (7 entity + 6 edge), grouped by bucket, with turn IDs
and quote snippets, ready for manual review.

Usage:
    python eir/inspect_extraction_260425.py <run_dir> <res_id>

Example:
    python eir/inspect_extraction_260425.py \\
        eir/eir_results/smoke_test_glm47_specialized_parallel_v10_aci_bench/260425_230340 \\
        RES_D2N008

Created: 260425
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ENTITY_CATEGORIES = [
    "SYMPTOM", "DIAGNOSIS", "TREATMENT", "PROCEDURE",
    "LOCATION", "MEDICAL_HISTORY", "LAB_RESULT",
]
EDGE_CATEGORIES = [
    "INDICATES", "CAUSES", "CONFIRMS", "RULES_OUT", "TAKEN_FOR", "LOCATED_AT",
]


def show_category(res_dir: Path, res_id: str, cat: str, is_edge: bool) -> None:
    f = res_dir / f"{res_id}_{cat}.json"
    if not f.exists():
        print(f"\n=== {cat}  (MISSING file) ===")
        return
    try:
        j = json.load(open(f))
    except Exception as e:
        print(f"\n=== {cat}  (JSON ERROR: {e}) ===")
        return

    items: list[tuple[str, str, str, str]] = []
    for bucket in ("matched", "other", "fine_grained"):
        for x in j.get(bucket, []) or []:
            if is_edge:
                label = f"{x.get('src','?')} → {x.get('dst','?')}"
            else:
                label = x.get("label") or x.get("text") or "?"
            tid = x.get("turn_id") or "?"
            quote = (x.get("quote") or "")[:80]
            items.append((bucket, label, tid, quote))

    status = j.get("status", "?")
    n = len(items)
    print(f"\n=== {cat}  ({n} items, status={status}) ===")
    if not items:
        print("  (empty)")
        return
    for bucket, label, tid, quote in items:
        print(f"  [{bucket:13}] {label}")
        print(f"  {' ':15} ({tid}) {quote!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", help="Path to extraction run directory (contains RES*/ subdirs)")
    ap.add_argument("res_id", help="Patient ID (e.g. RES_D2N008 or RES0198)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    res_dir = run_dir / args.res_id
    if not res_dir.exists():
        sys.exit(f"ERROR: res dir not found: {res_dir}")

    print("=" * 78)
    print(f"PATIENT: {args.res_id}")
    print(f"SOURCE:  {run_dir}")
    print("=" * 78)

    print("\n" + "─" * 78)
    print("ENTITIES (Stage 1)")
    print("─" * 78)
    for cat in ENTITY_CATEGORIES:
        show_category(res_dir, args.res_id, cat, is_edge=False)

    print("\n" + "─" * 78)
    print("EDGES (Stage 2)")
    print("─" * 78)
    for cat in EDGE_CATEGORIES:
        show_category(res_dir, args.res_id, cat, is_edge=True)


if __name__ == "__main__":
    main()
