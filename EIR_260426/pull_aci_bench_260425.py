#!/usr/bin/env python3
"""
pull_aci_bench_260425.py
========================

Pull one or more ACI-Bench encounters from Hugging Face
(`mkieffer/ACI-Bench`), convert the inline `[doctor]/[patient]` speaker
tags into our `D-N` / `P-N` turn-ID format, and write them under
`eir/eir_aci_bench/transcripts/RES_<ENCOUNTER_ID>/RES_<ENCOUNTER_ID>.txt`.
Also drops the paired clinical note alongside as
`RES_<ENCOUNTER_ID>_note.txt` and a small meta JSON.

Usage:
    python eir/pull_aci_bench_260425.py D2N008 D2N018
    python eir/pull_aci_bench_260425.py --all-train     # all 20 train
    python eir/pull_aci_bench_260425.py D2N003 --print  # echo transcript

The output folder shape mirrors `data/transcripts/RES0XXX/` exactly so
the v10 ACI-Bench wrapper can pick it up by overriding TRANSCRIPT_DIR.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

EIR_ROOT = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = EIR_ROOT / "eir_aci_bench" / "transcripts"
HF_DATASET = "mkieffer/ACI-Bench"


def aci_to_turn_format(dialogue: str) -> tuple[str, int, int]:
    """Convert '[doctor] ... [patient] ...' to 'D-1: ...\nP-1: ...' lines."""
    parts = re.split(r'\[(doctor|patient|dragon)\]\s*', dialogue)
    d_count = p_count = 0
    lines: list[str] = []
    for i in range(1, len(parts) - 1, 2):
        role = parts[i]
        text = parts[i + 1].strip()
        if not text:
            continue
        if role == "doctor":
            d_count += 1
            lines.append(f"D-{d_count}: {text}")
        elif role == "patient":
            p_count += 1
            lines.append(f"P-{p_count}: {text}")
        # ignore [dragon] turns (virtassist subset only)
    return "\n".join(lines), d_count, p_count


def write_one(row: dict, out_dir: Path) -> Path:
    eid = row["encounter_id"]
    res_id = f"RES_{eid}"
    transcript, d_n, p_n = aci_to_turn_format(row["dialogue"])
    res_dir = out_dir / res_id
    res_dir.mkdir(parents=True, exist_ok=True)

    (res_dir / f"{res_id}.txt").write_text(transcript + "\n", encoding="utf-8")
    (res_dir / f"{res_id}_note.txt").write_text(row["note"], encoding="utf-8")
    (res_dir / f"{res_id}_meta.json").write_text(
        json.dumps(
            {
                "source": "ACI-Bench",
                "hf_dataset": HF_DATASET,
                "encounter_id": eid,
                "doctor_turns": d_n,
                "patient_turns": p_n,
                "transcript_chars": len(transcript),
                "note_chars": len(row["note"]),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return res_dir / f"{res_id}.txt"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "encounter_ids",
        nargs="*",
        help="ACI-Bench encounter ids (e.g. D2N008 D2N018). Omit if --all-train.",
    )
    ap.add_argument(
        "--all-train",
        action="store_true",
        help="Pull all encounters from the train split (20 patients).",
    )
    ap.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUT_DIR),
        help=f"Output base dir. Default: {DEFAULT_OUT_DIR}",
    )
    ap.add_argument(
        "--print",
        action="store_true",
        help="Echo the converted transcript to stdout after writing.",
    )
    args = ap.parse_args()

    if not args.encounter_ids and not args.all_train:
        sys.exit("ERROR: pass at least one encounter_id, or --all-train.")

    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("datasets package required: pip install datasets")

    out_dir = Path(args.output_dir)

    # ACI-Bench has 5 splits (train/valid/test1/test2/test3); we look across all.
    splits = ["train", "valid", "test1", "test2", "test3"]
    by_id: dict[str, dict] = {}
    for sp in splits:
        try:
            ds = load_dataset(HF_DATASET, split=sp)
        except Exception:
            continue
        for r in ds:
            by_id[r["encounter_id"]] = r

    if args.all_train:
        ds_train = load_dataset(HF_DATASET, split="train")
        wanted = [r["encounter_id"] for r in ds_train]
    else:
        wanted = args.encounter_ids

    missing = [e for e in wanted if e not in by_id]
    if missing:
        sys.exit(f"ERROR: encounter(s) not found in HF dataset: {missing}")

    print(f"[aci-bench] pulling {len(wanted)} encounter(s) → {out_dir}")
    for eid in wanted:
        path = write_one(by_id[eid], out_dir)
        meta = json.loads(path.with_name(f"RES_{eid}_meta.json").read_text())
        print(
            f"  ✓ {eid}  →  {path.relative_to(EIR_ROOT.parent)}  "
            f"({meta['doctor_turns']}D/{meta['patient_turns']}P, "
            f"{meta['transcript_chars']} chars)"
        )
        if args.print:
            print()
            print("=" * 72)
            print(path.read_text())
            print("=" * 72)


if __name__ == "__main__":
    main()
