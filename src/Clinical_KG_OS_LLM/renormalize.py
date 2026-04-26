"""
Re-normalize existing sub-KGs without re-extracting.

Reads all RES*.json files in an input directory, applies the current
normalize_text() + validate_knowledge_graph() logic to each node, and
writes updated sub-KGs to an output directory.

Usage:
    python renormalize.py --input ./my_kg_karma_sc --output ./my_kg_karma_sc_renorm

Then run the usual merge + score steps on the output dir.
"""

import json
import argparse
from pathlib import Path

from Clinical_KG_OS_LLM.kg_extraction import (
    validate_knowledge_graph,
    normalize_text,
)


def main():
    parser = argparse.ArgumentParser(description="Re-normalize existing sub-KGs")
    parser.add_argument("--input", type=str, required=True,
                        help="Input directory containing sub-KG JSON files")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for re-normalized sub-KGs")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("RES*.json"))
    print(f"Found {len(files)} sub-KG files")
    print("=" * 60)

    for f in files:
        with open(f) as fp:
            kg = json.load(fp)

        before_n = len(kg.get('nodes', []))
        before_e = len(kg.get('edges', []))

        # Re-apply normalize_text to every node (validate_knowledge_graph
        # calls normalize_text internally, so we just re-run the full
        # validation pass — this is idempotent but re-normalizes with the
        # current code).
        kg_normalized = validate_knowledge_graph(kg)

        after_n = len(kg_normalized.get('nodes', []))
        after_e = len(kg_normalized.get('edges', []))

        # Save
        out_file = output_dir / f.name
        with open(out_file, 'w') as fp:
            json.dump(kg_normalized, fp, indent=2, ensure_ascii=False)

        print(f"  {f.stem}: {before_n}n/{before_e}e -> {after_n}n/{after_e}e")

    # Also copy _stats.json if present, for bookkeeping
    stats_src = input_dir / "_stats.json"
    if stats_src.exists():
        import shutil
        shutil.copy(stats_src, output_dir / "_stats.json")
        print(f"\nCopied _stats.json")

    print("=" * 60)
    print(f"Done. Re-normalized {len(files)} files to {output_dir}/")
    print("Next: run dump_graph.py + kg_similarity_scorer.py on the new dir")


if __name__ == "__main__":
    main()
