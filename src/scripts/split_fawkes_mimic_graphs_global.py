#!/usr/bin/env python3
"""Globally split Fawkes MIMIC sub-KG JSON graphs into train/test folders.

This consumes all graph JSONs under directories like:

    outputs/fawkes_mimic_graphs_*/sub_kgs/{train,test}/*.json

The split is made after collecting the full graph pool, so each source chunk
does not keep its own independent train/test assignment.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path("outputs")
DEFAULT_SOURCE_GLOB = "fawkes_mimic_graphs_*/sub_kgs"
DEFAULT_OUT = Path("outputs/fawkes_mimic_global/sub_kgs")


@dataclass(frozen=True)
class GraphRecord:
    path: Path
    data: dict[str, Any]
    graph_key: tuple[str, str]
    patient_key: str

    @property
    def output_name(self) -> str:
        return self.path.name


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    if "nodes" not in data or "edges" not in data:
        return None
    return data


def _ids_from_path(path: Path) -> tuple[str, str]:
    parts = path.stem.split("_", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return path.stem, ""


def _graph_key(path: Path, data: dict[str, Any]) -> tuple[str, str]:
    path_subject_id, path_hadm_id = _ids_from_path(path)
    subject_id = str(data.get("subject_id") or path_subject_id)
    hadm_id = data.get("hadm_id")
    if hadm_id is None:
        hadm_ids = data.get("hadm_ids")
        if isinstance(hadm_ids, list) and hadm_ids:
            hadm_id = hadm_ids[0]
    return subject_id, str(hadm_id or path_hadm_id)


def _discover_source_roots(root: Path, source_glob: str, out_dir: Path) -> list[Path]:
    out_resolved = out_dir.resolve()
    roots: list[Path] = []
    for candidate in sorted(root.glob(source_glob)):
        if not candidate.is_dir():
            continue
        if candidate.resolve() == out_resolved:
            continue
        roots.append(candidate)
    return roots


def collect_graphs(source_roots: list[Path]) -> tuple[list[GraphRecord], dict[str, int]]:
    records: list[GraphRecord] = []
    seen_keys: set[tuple[str, str]] = set()
    stats = {
        "json_files_seen": 0,
        "non_graph_json_files": 0,
        "duplicate_graphs": 0,
    }

    for source_root in source_roots:
        for path in sorted(source_root.rglob("*.json")):
            if path.name.startswith("_"):
                continue
            stats["json_files_seen"] += 1
            data = _read_json(path)
            if data is None:
                stats["non_graph_json_files"] += 1
                continue

            graph_key = _graph_key(path, data)
            if graph_key in seen_keys:
                stats["duplicate_graphs"] += 1
                continue
            seen_keys.add(graph_key)
            records.append(
                GraphRecord(
                    path=path,
                    data=data,
                    graph_key=graph_key,
                    patient_key=graph_key[0],
                )
            )

    return records, stats


def split_records(
    records: list[GraphRecord],
    *,
    test_frac: float,
    test_count: int | None,
    seed: int,
    split_unit: str,
) -> dict[tuple[str, str], str]:
    if not records:
        raise SystemExit("No graph JSON files found in the matched source folders.")
    if not 0 <= test_frac <= 1:
        raise SystemExit("--test-frac must be between 0 and 1")
    if test_count is not None and test_count < 0:
        raise SystemExit("--test-count must be non-negative")

    if split_unit == "patient":
        units = sorted({record.patient_key for record in records})
    elif split_unit == "graph":
        units = sorted(f"{record.graph_key[0]}_{record.graph_key[1]}" for record in records)
    else:
        raise SystemExit(f"Unknown split unit: {split_unit}")

    desired_test_units = (
        test_count
        if test_count is not None
        else int(round(len(units) * test_frac))
    )
    desired_test_units = min(desired_test_units, len(units))

    rng = random.Random(seed)
    shuffled = units[:]
    rng.shuffle(shuffled)
    test_units = set(shuffled[:desired_test_units])

    split_by_key: dict[tuple[str, str], str] = {}
    for record in records:
        unit = (
            record.patient_key
            if split_unit == "patient"
            else f"{record.graph_key[0]}_{record.graph_key[1]}"
        )
        split_by_key[record.graph_key] = "test" if unit in test_units else "train"
    return split_by_key


def _ensure_writable_split_dirs(out_dir: Path, overwrite: bool) -> tuple[Path, Path]:
    train_dir = out_dir / "train"
    test_dir = out_dir / "test"
    existing = [
        path
        for split_dir in (train_dir, test_dir)
        if split_dir.exists()
        for path in split_dir.glob("*.json")
    ]
    if existing and not overwrite:
        raise SystemExit(
            f"{out_dir} already contains split JSON files. "
            "Pass --overwrite to replace them."
        )
    if overwrite:
        for path in existing:
            path.unlink()
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    return train_dir, test_dir


def write_split(
    records: list[GraphRecord],
    split_by_key: dict[tuple[str, str], str],
    *,
    out_dir: Path,
    source_roots: list[Path],
    source_stats: dict[str, int],
    split_unit: str,
    seed: int,
    test_frac: float,
    test_count: int | None,
    overwrite: bool,
) -> dict[str, Any]:
    train_dir, test_dir = _ensure_writable_split_dirs(out_dir, overwrite)
    written_names: set[str] = set()
    manifest_records: list[dict[str, Any]] = []

    for record in sorted(records, key=lambda item: item.output_name):
        if record.output_name in written_names:
            raise SystemExit(f"Output filename collision: {record.output_name}")
        written_names.add(record.output_name)

        split = split_by_key[record.graph_key]
        graph = dict(record.data)
        graph["split"] = split
        dest = (test_dir if split == "test" else train_dir) / record.output_name
        dest.write_text(
            json.dumps(graph, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        manifest_records.append(
            {
                "graph_key": list(record.graph_key),
                "split": split,
                "source_path": str(record.path),
                "output_path": str(dest),
            }
        )

    train_graphs = sum(1 for item in manifest_records if item["split"] == "train")
    test_graphs = sum(1 for item in manifest_records if item["split"] == "test")
    manifest = {
        "method": "fawkes_mimic_global_train_test_split",
        "output_dir": str(out_dir),
        "source_roots": [str(path) for path in source_roots],
        "split_unit": split_unit,
        "seed": seed,
        "test_frac": test_frac,
        "test_count": test_count,
        "graphs": len(records),
        "train_graphs": train_graphs,
        "test_graphs": test_graphs,
        "source_stats": source_stats,
        "records": manifest_records,
    }
    (out_dir / "_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default=str(DEFAULT_ROOT),
        help=f"Root containing fawkes_mimic_graphs_* folders (default: {DEFAULT_ROOT})",
    )
    parser.add_argument(
        "--source-glob",
        default=DEFAULT_SOURCE_GLOB,
        help=f"Glob under --root for source sub_kgs folders (default: {DEFAULT_SOURCE_GLOB})",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUT),
        help=f"Output sub_kgs directory with train/test folders (default: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--test-frac",
        type=float,
        default=0.2,
        help="Fraction of split units assigned to test (default: 0.2)",
    )
    parser.add_argument(
        "--test-count",
        type=int,
        default=None,
        help="Exact number of split units assigned to test; overrides --test-frac",
    )
    parser.add_argument(
        "--split-unit",
        choices=["patient", "graph"],
        default="patient",
        help="Split by patient to avoid patient leakage, or by graph (default: patient)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic split seed")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing JSON files under the output train/test folders",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned split counts without writing files",
    )
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out)
    source_roots = _discover_source_roots(root, args.source_glob, out_dir)
    if not source_roots:
        raise SystemExit(
            f"No source folders matched {args.source_glob!r} under {root}."
        )

    records, source_stats = collect_graphs(source_roots)
    split_by_key = split_records(
        records,
        test_frac=args.test_frac,
        test_count=args.test_count,
        seed=args.seed,
        split_unit=args.split_unit,
    )
    train_graphs = sum(
        1 for record in records if split_by_key[record.graph_key] == "train"
    )
    test_graphs = sum(
        1 for record in records if split_by_key[record.graph_key] == "test"
    )

    print(f"[fawkes-global-split] sources={len(source_roots)} graphs={len(records)}")
    print(
        "[fawkes-global-split] split "
        f"unit={args.split_unit} train={train_graphs} test={test_graphs}"
    )
    if source_stats["duplicate_graphs"]:
        print(
            "[fawkes-global-split] skipped duplicate graph(s): "
            f"{source_stats['duplicate_graphs']}"
        )
    if source_stats["non_graph_json_files"]:
        print(
            "[fawkes-global-split] skipped non-graph JSON file(s): "
            f"{source_stats['non_graph_json_files']}"
        )

    if args.dry_run:
        print("[fawkes-global-split] dry run; no files written")
        return

    manifest = write_split(
        records,
        split_by_key,
        out_dir=out_dir,
        source_roots=source_roots,
        source_stats=source_stats,
        split_unit=args.split_unit,
        seed=args.seed,
        test_frac=args.test_frac,
        test_count=args.test_count,
        overwrite=args.overwrite,
    )
    print(f"[fawkes-global-split] wrote split -> {out_dir}")
    print(f"[fawkes-global-split] manifest -> {out_dir / '_manifest.json'}")
    print(
        "[fawkes-global-split] final "
        f"train={manifest['train_graphs']} test={manifest['test_graphs']}"
    )


if __name__ == "__main__":
    main()
