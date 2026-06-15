#!/usr/bin/env python3
"""Convert Fawkes parquet tables into raw Graph-JEPA sub-KG JSONs.

Unlike ``prepare_fawkes_graphs_for_v4.py``, this converter does not retype
nodes, remap relations, or reverse edges. It preserves the parquet schema,
deduplicates edges by ``(source_id, target_id, relation)``, and only adds the
aliases required by Graph-JEPA:

* node: ``id`` and ``text``
* edge: ``source``, ``target``, and ``type``

Usage:
    python src/scripts/prepare_fawkes_graphs_raw.py
    python src/scripts/prepare_fawkes_graphs_raw.py --root data/... --out outputs/...
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_ROOT = Path("data/fawkes-mimic-graphs-complete-v8-rows2000-3000-260613")
DEFAULT_OUT = Path("outputs/fawkes_mimic_graphs_raw_2k_3k/sub_kgs")
METHOD = "fawkes_mimic_graphs_complete_v8_parquet_raw"


def clean_value(value: Any) -> Any:
    """Return JSON-serializable values, replacing pandas/numpy NA with None."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, dict):
        return {str(key): clean_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_value(item) for item in value]
    if hasattr(value, "tolist"):
        try:
            return clean_value(value.tolist())
        except (TypeError, ValueError):
            pass
    if hasattr(value, "item"):
        try:
            return clean_value(value.item())
        except (TypeError, ValueError):
            pass
    return value


def read_table(root: Path, name: str) -> pd.DataFrame:
    table_dir = root / name
    files = sorted(table_dir.glob("*.parquet"))
    if not files:
        raise SystemExit(f"No parquet files found under {table_dir}")
    return pd.concat((pd.read_parquet(path) for path in files), ignore_index=True)


def row_dict(row: pd.Series) -> dict[str, Any]:
    return {
        str(key): clean_value(value)
        for key, value in row.to_dict().items()
    }


def graph_key(subject_id: Any, hadm_id: Any) -> tuple[str, str]:
    return str(clean_value(subject_id)), str(clean_value(hadm_id))


def build_node(row: pd.Series) -> dict[str, Any]:
    data = row_dict(row)
    node_id = str(data.get("node_id") or "")
    node_type = str(data.get("type") or "")
    if not node_id or not node_type:
        raise ValueError("node row must contain node_id and type")

    name = str(data.get("name") or data.get("normalized_name") or node_id)
    return {
        **data,
        "id": node_id,
        "text": name,
        "mimic_type": node_type,
        "origin": METHOD,
    }


def build_edge(row: pd.Series) -> dict[str, Any]:
    data = row_dict(row)
    source_id = str(data.get("source_id") or "")
    target_id = str(data.get("target_id") or "")
    relation = str(data.get("relation") or "")
    if not source_id or not target_id or not relation:
        raise ValueError(
            "edge row must contain source_id, target_id, and relation"
        )

    return {
        **data,
        "source": source_id,
        "target": target_id,
        "type": relation,
        "mimic_relation": relation,
        "origin": METHOD,
    }


def deduplicate_edges(edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    deduplicated: list[dict[str, Any]] = []
    for edge in edges:
        key = (
            str(edge["source_id"]),
            str(edge["target_id"]),
            str(edge["type"]),
        )
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(edge)
    return deduplicated


def split_assignments(
    keys: list[tuple[Any, Any]],
    *,
    test_patients: int,
    seed: int,
) -> tuple[dict[tuple[Any, Any], str], dict[str, Any]]:
    if test_patients < 0:
        raise SystemExit("--test-patients must be non-negative")

    subject_counts = Counter(graph_key(key[0], key[1])[0] for key in keys)
    eligible_subjects = sorted(
        subject_id
        for subject_id, count in subject_counts.items()
        if count == 1
    )
    if test_patients > len(eligible_subjects):
        raise SystemExit(
            f"--test-patients={test_patients} requested, but only "
            f"{len(eligible_subjects)} single-admission patient(s) are eligible"
        )

    rng = random.Random(seed)
    test_subjects = set(rng.sample(eligible_subjects, test_patients))
    assignments = {
        key: (
            "test"
            if graph_key(key[0], key[1])[0] in test_subjects
            else "train"
        )
        for key in keys
    }
    return assignments, {
        "train_graphs": sum(
            split == "train" for split in assignments.values()
        ),
        "test_graphs": sum(
            split == "test" for split in assignments.values()
        ),
        "test_patients": test_patients,
        "test_seed": seed,
        "eligible_single_admission_patients": len(eligible_subjects),
        "subjects_with_multiple_admissions": sum(
            count > 1 for count in subject_counts.values()
        ),
        "test_subject_ids": sorted(test_subjects),
    }


def prepare(
    root: Path,
    out_dir: Path,
    *,
    limit: int | None = None,
    test_patients: int = 0,
    split_seed: int = 0,
) -> dict[str, Any]:
    train_dir = out_dir / "train"
    test_dir = out_dir / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    patients = read_table(root, "patients")
    nodes = read_table(root, "nodes")
    edges = read_table(root, "edges")

    patient_by_key = {
        graph_key(row["subject_id"], row["hadm_id"]): row_dict(row)
        for _, row in patients.iterrows()
    }
    node_groups = dict(tuple(nodes.groupby(["subject_id", "hadm_id"], sort=True)))
    edge_groups = dict(tuple(edges.groupby(["subject_id", "hadm_id"], sort=True)))

    keys = sorted(set(node_groups) | set(edge_groups))
    if limit is not None:
        keys = keys[:limit]
    split_by_key, split_manifest = split_assignments(
        keys,
        test_patients=test_patients,
        seed=split_seed,
    )

    written_by_split: Counter[str] = Counter()
    nodes_by_split: Counter[str] = Counter()
    edges_by_split: Counter[str] = Counter()
    node_types: Counter[str] = Counter()
    edge_types: Counter[str] = Counter()
    typed_relations: Counter[tuple[str, str, str]] = Counter()

    for raw_key in keys:
        subject_id, hadm_id = graph_key(raw_key[0], raw_key[1])
        graph_nodes = [
            build_node(row)
            for _, row in node_groups.get(raw_key, pd.DataFrame()).iterrows()
        ]
        graph_edges = [
            build_edge(row)
            for _, row in edge_groups.get(raw_key, pd.DataFrame()).iterrows()
        ]
        graph_edges = deduplicate_edges(graph_edges)
        if not graph_nodes:
            continue

        split = split_by_key[raw_key]
        graph = {
            "subject_id": subject_id,
            "hadm_id": hadm_id,
            "hadm_ids": [hadm_id],
            "split": split,
            "nodes": graph_nodes,
            "edges": graph_edges,
            "_method": METHOD,
            "_source_root": str(root),
            "_patient": patient_by_key.get((subject_id, hadm_id), {}),
        }
        output_dir = test_dir if split == "test" else train_dir
        output_path = output_dir / f"{subject_id}_{hadm_id}.json"
        output_path.write_text(
            json.dumps(graph, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        written_by_split[split] += 1
        nodes_by_split[split] += len(graph_nodes)
        edges_by_split[split] += len(graph_edges)
        node_types.update(str(node["type"]) for node in graph_nodes)
        edge_types.update(str(edge["type"]) for edge in graph_edges)
        typed_relations.update(
            (
                str(edge.get("source_type") or ""),
                str(edge["type"]),
                str(edge.get("target_type") or ""),
            )
            for edge in graph_edges
        )

    manifest = {
        "method": METHOD,
        "source_root": str(root),
        "output_dir": str(out_dir),
        "graphs": sum(written_by_split.values()),
        "total_nodes": sum(nodes_by_split.values()),
        "total_edges": sum(edges_by_split.values()),
        "splits": {
            **split_manifest,
            "written_train_graphs": written_by_split["train"],
            "written_test_graphs": written_by_split["test"],
            "train_nodes": nodes_by_split["train"],
            "test_nodes": nodes_by_split["test"],
            "train_edges": edges_by_split["train"],
            "test_edges": edges_by_split["test"],
        },
        "schema": {
            "node_types": dict(sorted(node_types.items())),
            "edge_types": dict(sorted(edge_types.items())),
            "typed_relations": [
                {
                    "source_type": source_type,
                    "relation": relation,
                    "target_type": target_type,
                    "count": count,
                }
                for (
                    source_type,
                    relation,
                    target_type,
                ), count in sorted(typed_relations.items())
            ],
        },
    }
    (out_dir / "_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default=str(DEFAULT_ROOT),
        help=f"Downloaded Fawkes graph dataset root (default: {DEFAULT_ROOT})",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUT),
        help=f"Output train/test graph directory (default: {DEFAULT_OUT})",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional graph limit")
    parser.add_argument(
        "--test-patients",
        type=int,
        default=0,
        help="Number of single-admission patients to place in test/",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=0,
        help="Deterministic seed for selecting test patients",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    manifest = prepare(
        Path(args.root),
        Path(args.out),
        limit=args.limit,
        test_patients=args.test_patients,
        split_seed=args.split_seed,
    )
    print(
        f"[fawkes-graphs-raw] wrote {manifest['graphs']} graph(s) "
        f"-> {manifest['output_dir']}"
    )
    print(
        "[fawkes-graphs-raw] "
        f"nodes={manifest['total_nodes']} edges={manifest['total_edges']}"
    )


if __name__ == "__main__":
    main()
