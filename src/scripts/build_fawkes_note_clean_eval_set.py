#!/usr/bin/env python3
"""Select note-bearing eval graphs that no JEPA variant was trained on.

``data/fawkes_mimic_latest_admission_graphs/sub_kgs/test`` already carries the
768-dim Clinical-ModernBERT note embedding, but a handful of its patients also
appear in the corpora v6/v12/v16 trained on. This drops any graph whose
``subject_id`` shows up in a training corpus, so the survivors are clean at the
patient level for every model in the comparison.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

DEFAULT_SOURCE = Path("data/fawkes_mimic_latest_admission_graphs/sub_kgs/test")
DEFAULT_OUT = Path("outputs/fawkes_note_clean_eval/sub_kgs/test")
DEFAULT_TRAIN_CORPORA = [
    # v6 training split, and the same 4000 admissions v16 trained on.
    Path("data/fawkes_mimic_latest_admission_graphs/sub_kgs/train"),
    # the v8 scored corpus behind the v12 baseline.
    Path("outputs/fawkes_mimic_raw_global/sub_kgs/train"),
    Path("outputs/fawkes_mimic_raw_global/sub_kgs/test"),
]


def _graph_files(directory: Path) -> list[Path]:
    return [
        path
        for path in sorted(directory.glob("*.json"))
        if not path.name.startswith("_")
    ]


def _training_subject_ids(directories: list[Path]) -> set[str]:
    subjects: set[str] = set()
    for directory in directories:
        if not directory.is_dir():
            raise FileNotFoundError(f"training corpus not found: {directory}")
        for path in _graph_files(directory):
            subjects.add(str(json.loads(path.read_text())["subject_id"]))
    return subjects


def build(source: Path, out: Path, train_corpora: list[Path], note_dim: int) -> dict:
    if not source.is_dir():
        raise FileNotFoundError(f"source split not found: {source}")
    train_subjects = _training_subject_ids(train_corpora)

    kept, dropped_seen_patient, dropped_no_note = [], [], []
    for path in _graph_files(source):
        graph = json.loads(path.read_text())
        note = graph.get("note_embedding")
        if not note or len(note) != note_dim:
            dropped_no_note.append(path.name)
            continue
        if str(graph["subject_id"]) in train_subjects:
            dropped_seen_patient.append(path.name)
            continue
        kept.append(path)

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    for path in kept:
        shutil.copy2(path, out / path.name)

    manifest = {
        "source": str(source),
        "train_corpora": [str(p) for p in train_corpora],
        "note_embedding_dim": note_dim,
        "graphs_in_source": len(_graph_files(source)),
        "graphs_kept": len(kept),
        "dropped_patient_in_training_corpus": len(dropped_seen_patient),
        "dropped_missing_note_embedding": len(dropped_no_note),
        "kept_files": [path.name for path in kept],
    }
    (out.parent / "_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--note-dim", type=int, default=768)
    parser.add_argument(
        "--train-corpus",
        action="append",
        default=None,
        help="Directory of graphs a model trained on; repeatable.",
    )
    args = parser.parse_args()

    corpora = (
        [Path(p) for p in args.train_corpus]
        if args.train_corpus
        else DEFAULT_TRAIN_CORPORA
    )
    manifest = build(Path(args.source), Path(args.out), corpora, args.note_dim)
    print(
        f"[note-clean-eval] source={manifest['graphs_in_source']} "
        f"kept={manifest['graphs_kept']} "
        f"dropped_seen_patient={manifest['dropped_patient_in_training_corpus']} "
        f"dropped_no_note={manifest['dropped_missing_note_embedding']}"
    )
    print(f"[note-clean-eval] wrote -> {args.out}")


if __name__ == "__main__":
    main()
