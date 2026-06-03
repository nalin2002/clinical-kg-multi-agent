"""ACI-Bench data loader for the multi-agent + Graph-JEPA pipeline.

Pulls doctor-patient encounters from the Hugging Face dataset
``mkieffer/ACI-Bench`` and writes them in the exact on-disk shape the
multi-agent extractor expects::

    <out>/RES_<ENCOUNTER>/RES_<ENCOUNTER>.txt        # transcript
    <out>/RES_<ENCOUNTER>/RES_<ENCOUNTER>_note.txt   # paired clinical note
    <out>/RES_<ENCOUNTER>/RES_<ENCOUNTER>_meta.json  # provenance

The transcript uses the same **bracketed** turn-tag format as the in-corpus
transcripts under ``data/transcripts`` (``[D-1] D: ...`` / ``[P-1] P: ...``),
which is what ``multi_agent_cooperative_kg`` keys on (regex ``\\[([PD]-\\d+)\\]``).

ACI-Bench ships as three Hugging Face *configs* (subsets) - ``aci``,
``virtassist`` and ``virtscribe`` - each with ``train/valid/test1/test2/test3``
splits. By default every subset is pulled and processed; restrict with
``--subsets``.

CLI::

    PYTHONPATH=src python -m aci_bench --split train --out data/aci_bench/transcripts
    PYTHONPATH=src python -m aci_bench --all --out data/aci_bench/transcripts
    PYTHONPATH=src python -m aci_bench --subsets aci virtscribe --split train
    PYTHONPATH=src python -m aci_bench --ids D2N008 D2N018 --out data/aci_bench/transcripts

Requires the ``datasets`` package (``pip install datasets``).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

from datasets import load_dataset


HF_DATASET = "mkieffer/ACI-Bench"
# ACI-Bench subsets are exposed as separate HF dataset configs.
CONFIGS = ("aci", "virtassist", "virtscribe")
SPLITS = ("train", "valid", "test1", "test2", "test3")
DEFAULT_OUT = Path("data/aci_bench/transcripts")


def dialogue_to_bracketed_turns(dialogue: str) -> tuple[str, int, int]:
    """Convert ``[doctor] ... [patient] ...`` into bracketed ``[D-N] D:`` lines.

    Returns ``(transcript_text, doctor_turns, patient_turns)``. ``[dragon]``
    turns (present only in the virtassist subset) are ignored.
    """
    parts = re.split(r"\[(doctor|patient|dragon)\]\s*", dialogue or "")
    d_count = p_count = 0
    lines: list[str] = []
    for i in range(1, len(parts) - 1, 2):
        role = parts[i]
        text = re.sub(r"\s+", " ", parts[i + 1].strip())
        if not text:
            continue
        if role == "doctor":
            d_count += 1
            lines.append(f"[D-{d_count}] D: {text}")
        elif role == "patient":
            p_count += 1
            lines.append(f"[P-{p_count}] P: {text}")
    return "\n\n".join(lines) + "\n", d_count, p_count


def write_encounter(row: dict, out_dir: Path) -> Path:
    eid = row["encounter_id"]
    res_id = f"RES_{eid}"
    transcript, d_n, p_n = dialogue_to_bracketed_turns(row["dialogue"])
    note = row.get("note", "") or ""

    res_dir = out_dir / res_id
    res_dir.mkdir(parents=True, exist_ok=True)
    txt_path = res_dir / f"{res_id}.txt"
    txt_path.write_text(transcript, encoding="utf-8")
    (res_dir / f"{res_id}_note.txt").write_text(note, encoding="utf-8")
    (res_dir / f"{res_id}_meta.json").write_text(
        json.dumps(
            {
                "source": "ACI-Bench",
                "hf_dataset": HF_DATASET,
                "encounter_id": eid,
                "subset": row.get("_subset"),
                "split": row.get("_split"),
                "doctor_turns": d_n,
                "patient_turns": p_n,
                "transcript_chars": len(transcript),
                "note_chars": len(note),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return txt_path


def _load_rows(splits: Iterable[str], subsets: Iterable[str]) -> dict[str, dict]:
    """Load encounters across the requested subsets x splits, keyed by id.

    Each ACI-Bench subset (``aci`` / ``virtassist`` / ``virtscribe``) is a
    distinct HF config, so we load every requested ``(config, split)`` pair.
    Rows are tagged with their originating ``_subset`` / ``_split``.
    """
    by_id: dict[str, dict] = {}
    for cfg in subsets:
        for sp in splits:
            try:
                ds = load_dataset(HF_DATASET, name=cfg, split=sp)
            except Exception as exc:  # a split may not exist for a config
                print(f"[aci-bench] skipping {cfg}/{sp}: {exc}", flush=True)
                continue
            n = 0
            for row in ds:
                eid = row["encounter_id"]
                prev = by_id.get(eid)
                if prev is not None and prev["_subset"] != cfg:
                    print(f"[aci-bench] WARNING: encounter id {eid} found in both "
                          f"{prev['_subset']} and {cfg}; keeping {cfg}.", flush=True)
                rec = dict(row)
                rec["_subset"] = cfg
                rec["_split"] = sp
                by_id[eid] = rec
                n += 1
            print(f"[aci-bench] loaded {cfg}/{sp}: {n} encounters", flush=True)
    return by_id


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(DEFAULT_OUT),
                    help=f"Output transcripts dir (default: {DEFAULT_OUT})")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--split", choices=SPLITS, default=None,
                       help="Pull every encounter from this split.")
    group.add_argument("--ids", nargs="+", default=None,
                       help="Pull specific encounter ids (e.g. D2N008 D2N018).")
    ap.add_argument("--all", action="store_true",
                    help="Pull every encounter across all splits.")
    ap.add_argument("--subsets", nargs="+", choices=CONFIGS, default=list(CONFIGS),
                    help="ACI-Bench subsets to pull (default: all three).")
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap the number of encounters written (after ordering).")
    args = ap.parse_args()

    out_dir = Path(args.out)

    # Decide which splits we need to scan.
    if args.ids or args.all:
        scan_splits = SPLITS
    else:
        scan_splits = (args.split,) if args.split else ("train",)

    by_id = _load_rows(scan_splits, args.subsets)
    if not by_id:
        raise SystemExit("[aci-bench] no encounters loaded from Hugging Face.")

    if args.ids:
        missing = [e for e in args.ids if e not in by_id]
        if missing:
            raise SystemExit(f"[aci-bench] encounter(s) not found: {missing}")
        wanted = list(args.ids)
    else:
        wanted = sorted(by_id)

    if args.limit is not None:
        wanted = wanted[: args.limit]

    print(f"[aci-bench] writing {len(wanted)} encounter(s) -> {out_dir}", flush=True)
    for eid in wanted:
        path = write_encounter(by_id[eid], out_dir)
        meta = json.loads(path.with_name(f"RES_{eid}_meta.json").read_text())
        print(f"  + {eid:>8}  [{meta['subset']}/{meta['split']}]  "
              f"{meta['doctor_turns']}D/{meta['patient_turns']}P  "
              f"{meta['transcript_chars']} chars  -> {path}", flush=True)


if __name__ == "__main__":
    main()
