#!/usr/bin/env python3
"""Prepare Fawkes MIMIC notes JSONL rows for the cooperative KG pipeline.

The cooperative extractor expects transcript folders shaped like:

    <out>/RES.../RES....txt
    <out>/RES.../RES..._note.txt  # optional supporting context

Fawkes rows are not dialogues, so this adapter writes the structured
``input_block`` as the main text and the generated ``note`` as supporting
context. The extractor will then read both through its normal transcript +
optional note path.

Usage:
    python prepare_fawkes_for_cooperative.py path/to/notes.jsonl
    python prepare_fawkes_for_cooperative.py path/to/notes.jsonl --limit 10
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_OUT = Path("data/fawkes_mimic_notes/transcripts")


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def safe_id(value: Any, fallback: str) -> str:
    text = clean_text(value) or fallback
    text = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")
    return text or fallback


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_number, json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_number}: invalid JSON: {exc}") from exc


def write_row(row: dict[str, Any], source_path: Path, row_number: int, out_dir: Path) -> Path:
    subject_id = safe_id(row.get("subject_id"), f"row{row_number}")
    hadm_id = safe_id(row.get("hadm_id"), "no_hadm")
    # Keep the ID underscore-free before the output suffix. dump_graph infers
    # res_id by splitting KG filenames on "_"; IDs like RES_FAWKES_... would
    # collapse to RES_FAWKES during unification.
    res_id = f"RESFAWKES{subject_id}H{hadm_id}"
    res_dir = out_dir / res_id
    res_dir.mkdir(parents=True, exist_ok=True)

    main_text = clean_text(row.get("input_block")) or clean_text(row.get("note"))
    note_text = clean_text(row.get("note"))
    if not main_text:
        raise SystemExit(f"{source_path}:{row_number}: missing both input_block and note")

    (res_dir / f"{res_id}.txt").write_text(main_text + "\n", encoding="utf-8")
    if note_text:
        (res_dir / f"{res_id}_note.txt").write_text(note_text + "\n", encoding="utf-8")

    meta = {
        "source": "fawkes-mimic-notes",
        "source_path": str(source_path),
        "source_row": row_number,
        "subject_id": row.get("subject_id"),
        "hadm_id": row.get("hadm_id"),
        "gen_version": row.get("gen_version"),
        "shard": row.get("shard"),
        "input_block_chars": len(main_text),
        "note_chars": len(note_text),
        "counts": {
            key: row.get(key)
            for key in (
                "n_symptoms",
                "n_diagnoses",
                "n_medical_history",
                "n_procedures",
                "n_medications",
                "n_microbiology",
                "n_services",
            )
        },
    }
    (res_dir / f"{res_id}_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return res_dir / f"{res_id}.txt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", help="Fawkes MIMIC notes JSONL file")
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUT),
        help=f"Output transcript root (default: {DEFAULT_OUT})",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit")
    args = parser.parse_args()

    source_path = Path(args.jsonl)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for row_number, row in iter_jsonl(source_path):
        if args.limit is not None and written >= args.limit:
            break
        write_row(row, source_path, row_number, out_dir)
        written += 1

    manifest = {
        "source_path": str(source_path),
        "output_dir": str(out_dir),
        "rows_written": written,
    }
    (out_dir / "_fawkes_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[fawkes-prep] wrote {written} transcript folder(s) -> {out_dir}")


if __name__ == "__main__":
    main()
