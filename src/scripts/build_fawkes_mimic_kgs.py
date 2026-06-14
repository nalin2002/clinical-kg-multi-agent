#!/usr/bin/env python3
"""Build MIMIC-style KG JSONs from a Fawkes MIMIC notes JSONL shard.

The Fawkes rows already contain structured clinical fields alongside the
generated note. This script turns those fields into per-admission sub-KGs that
match the existing MIMIC sub-KG shape used by Graph-JEPA scoring.

Usage:
    python build_fawkes_mimic_kgs.py data/fawkes-mimic-notes-.../notes_v3_shard_00_260612.jsonl
    python build_fawkes_mimic_kgs.py path/to/shard.jsonl --limit 5
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_OUT = Path("outputs/fawkes_mimic_notes/sub_kgs")
METHOD = "fawkes_mimic_notes_structured_jsonl"


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def normalized_name(value: str) -> str:
    return clean_text(value).lower()


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def value_name(value: Any, *keys: str) -> str:
    if isinstance(value, dict):
        for key in keys:
            text = clean_text(value.get(key))
            if text:
                return text
        for key in ("name", "description", "label", "organism", "specimen"):
            text = clean_text(value.get(key))
            if text:
                return text
        return compact_json(value)
    return clean_text(value)


def is_missing_entity(text: str) -> bool:
    return normalized_name(text) in {"", "unknown", "none", "nan", "null"}


def list_values(row: dict[str, Any], key: str) -> list[Any]:
    value = row.get(key) or []
    return value if isinstance(value, list) else [value]


def patient_label(row: dict[str, Any]) -> str:
    age = row.get("anchor_age")
    gender = clean_text(row.get("gender")).lower()
    if age is not None and gender:
        return f"{age}-year-old {gender}"
    if age is not None:
        return f"{age}-year-old patient"
    return f"patient {row.get('subject_id', '')}".strip()


def build_graph(row: dict[str, Any], source_path: Path, row_number: int) -> dict[str, Any]:
    subject_id = str(row.get("subject_id", ""))
    hadm_id = str(row.get("hadm_id", ""))
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    node_ids: dict[tuple[str, str], str] = {}

    def add_node(
        node_type: str,
        text: str,
        *,
        source_tables: list[str],
        evidence: str,
        metadata: dict[str, Any] | None = None,
    ) -> str | None:
        text = clean_text(text)
        if is_missing_entity(text):
            return None
        key = (node_type, normalized_name(text))
        existing = node_ids.get(key)
        if existing:
            return existing

        node_id = f"N{len(nodes) + 1}"
        node_ids[key] = node_id
        nodes.append(
            {
                "id": node_id,
                "type": node_type,
                "name": text,
                "normalized_name": normalized_name(text),
                "source_tables": source_tables,
                "evidence": evidence,
                "hadm_ids": [hadm_id] if hadm_id else [],
                "metadata": metadata or {},
                "origin": METHOD,
                "text": text,
                "mimic_type": node_type,
            }
        )
        return node_id

    def add_edge(
        source_id: str | None,
        target_id: str | None,
        relation: str,
        *,
        evidence: str,
        source_tables: list[str],
        confidence: float = 0.95,
    ) -> None:
        if not source_id or not target_id or source_id == target_id:
            return
        source = next((n for n in nodes if n["id"] == source_id), None)
        target = next((n for n in nodes if n["id"] == target_id), None)
        if source is None or target is None:
            return
        key = (source_id, target_id, relation, hadm_id)
        if any(
            (
                e["source_id"],
                e["target_id"],
                e["type"],
                str(e.get("hadm_id", "")),
            )
            == key
            for e in edges
        ):
            return
        edges.append(
            {
                "source": source_id,
                "target": target_id,
                "relation": relation,
                "source_type": source["type"],
                "target_type": target["type"],
                "hadm_id": hadm_id,
                "evidence": evidence,
                "source_tables": source_tables,
                "confidence": confidence,
                "origin": METHOD,
                "source_id": source_id,
                "target_id": target_id,
                "type": relation,
                "mimic_relation": relation,
            }
        )

    patient_id = add_node(
        "PATIENT",
        patient_label(row),
        source_tables=["patients", "admissions"],
        evidence=compact_json(
            {
                "gender": row.get("gender"),
                "anchor_age": row.get("anchor_age"),
                "race": row.get("race"),
            }
        ),
        metadata={
            "subject_id": subject_id,
            "hadm_id": hadm_id,
            "gender": row.get("gender"),
            "anchor_age": row.get("anchor_age"),
            "race": row.get("race"),
            "admission_type": row.get("admission_type"),
            "admission_location": row.get("admission_location"),
            "discharge_location": row.get("discharge_location"),
            "died_in_hospital": row.get("died_in_hospital"),
            "locations": row.get("locations") or [],
            "drg": row.get("drg") or [],
        },
    )

    primary_dx_text = clean_text(row.get("diagnosis_primary"))
    diagnosis_ids: list[str] = []
    for diagnosis in [primary_dx_text, *list_values(row, "diagnoses")]:
        text = value_name(diagnosis, "name", "description")
        node_id = add_node(
            "DIAGNOSIS",
            text,
            source_tables=["diagnoses"],
            evidence=f'"{text}" listed in diagnoses',
            metadata={"primary": normalized_name(text) == normalized_name(primary_dx_text)},
        )
        if node_id:
            diagnosis_ids.append(node_id)
            add_edge(
                patient_id,
                node_id,
                "HAS_DIAGNOSIS",
                evidence=f'"{text}" listed in diagnoses',
                source_tables=["diagnoses"],
            )

    primary_dx_id = diagnosis_ids[0] if diagnosis_ids else None

    for symptom in list_values(row, "symptoms"):
        text = value_name(symptom, "name", "description")
        node_id = add_node(
            "SYMPTOM",
            text,
            source_tables=["diagnoses"],
            evidence=f'"{text}" listed in symptoms',
        )
        add_edge(
            node_id,
            primary_dx_id,
            "INDICATES",
            evidence=f'"{text}" listed in symptoms',
            source_tables=["diagnoses"],
            confidence=0.80,
        )

    for history in list_values(row, "medical_history"):
        text = value_name(history, "name", "description")
        node_id = add_node(
            "MEDICAL_HISTORY",
            text,
            source_tables=["diagnoses"],
            evidence=f'"{text}" listed in medical_history',
        )
        add_edge(
            patient_id,
            node_id,
            "ASSOCIATED_WITH",
            evidence=f'"{text}" listed in medical_history',
            source_tables=["diagnoses"],
            confidence=0.80,
        )

    for medication in list_values(row, "medications"):
        text = value_name(medication, "name")
        node_id = add_node(
            "MEDICATION",
            text,
            source_tables=["medications"],
            evidence=f'"{text}" listed in medications',
            metadata=medication if isinstance(medication, dict) else {},
        )
        add_edge(
            patient_id,
            node_id,
            "TAKES_MEDICATION",
            evidence=f'"{text}" listed in medications',
            source_tables=["medications"],
        )

    for procedure in list_values(row, "procedures"):
        text = value_name(procedure, "name", "description")
        node_id = add_node(
            "PROCEDURE",
            text,
            source_tables=["procedures"],
            evidence=f'"{text}" listed in procedures',
        )
        add_edge(
            patient_id,
            node_id,
            "UNDERWENT_PROCEDURE",
            evidence=f'"{text}" listed in procedures',
            source_tables=["procedures"],
        )
        add_edge(
            node_id,
            primary_dx_id,
            "PERFORMED_FOR",
            evidence=f'"{text}" listed in procedures',
            source_tables=["procedures", "diagnoses"],
            confidence=0.75,
        )

    for lab_test in list_values(row, "lab_tests"):
        text = value_name(lab_test, "name", "label", "description")
        node_id = add_node(
            "LAB_TEST",
            text,
            source_tables=["lab_tests"],
            evidence=f'"{text}" listed in lab_tests',
            metadata=lab_test if isinstance(lab_test, dict) else {},
        )
        add_edge(
            patient_id,
            node_id,
            "HAD_LAB_TEST",
            evidence=f'"{text}" listed in lab_tests',
            source_tables=["lab_tests"],
        )
        add_edge(
            primary_dx_id,
            node_id,
            "DIAGNOSED_BY",
            evidence=f'"{text}" listed in lab_tests',
            source_tables=["lab_tests", "diagnoses"],
            confidence=0.75,
        )

    for microbiology in list_values(row, "microbiology"):
        text = value_name(microbiology, "organism", "specimen")
        node_id = add_node(
            "MICROBIOLOGY",
            text,
            source_tables=["microbiology"],
            evidence=f'"{text}" listed in microbiology',
            metadata=microbiology if isinstance(microbiology, dict) else {},
        )
        add_edge(
            patient_id,
            node_id,
            "HAD_MICROBIOLOGY",
            evidence=f'"{text}" listed in microbiology',
            source_tables=["microbiology"],
        )
        add_edge(
            node_id,
            primary_dx_id,
            "CONFIRMS",
            evidence=f'"{text}" listed in microbiology',
            source_tables=["microbiology", "diagnoses"],
            confidence=0.75,
        )

    for service in list_values(row, "services"):
        text = value_name(service, "name", "description")
        node_id = add_node(
            "SERVICE",
            text,
            source_tables=["services"],
            evidence=f'"{text}" listed in services',
        )
        for diagnosis_id in diagnosis_ids:
            add_edge(
                node_id,
                diagnosis_id,
                "MANAGED_FOR",
                evidence=f'"{text}" listed in services',
                source_tables=["services", "diagnoses"],
                confidence=0.85,
            )

    return {
        "subject_id": subject_id,
        "hadm_id": hadm_id,
        "hadm_ids": [hadm_id] if hadm_id else [],
        "nodes": nodes,
        "edges": edges,
        "_method": METHOD,
        "_source_path": str(source_path),
        "_source_row": row_number,
        "_meta": {
            "gen_version": row.get("gen_version"),
            "shard": row.get("shard"),
            "note_chars": len(clean_text(row.get("note"))),
            "input_block_chars": len(clean_text(row.get("input_block"))),
            "n_sentences": row.get("n_sentences"),
            "entity_coverage": row.get("entity_coverage"),
            "disposition_faithful": row.get("disposition_faithful"),
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
        },
    }


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", help="Fawkes MIMIC notes JSONL shard")
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUT),
        help=f"Output directory for per-admission KG JSONs (default: {DEFAULT_OUT})",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit")
    args = parser.parse_args()

    source_path = Path(args.jsonl)
    out_dir = Path(args.out)
    rows = iter_jsonl(source_path)
    if args.limit is not None:
        rows = rows[: args.limit]

    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    total_nodes = 0
    total_edges = 0
    for row_number, row in enumerate(rows, start=1):
        graph = build_graph(row, source_path, row_number)
        subject_id = graph["subject_id"] or f"row{row_number}"
        hadm_id = graph["hadm_id"] or "no_hadm"
        output_path = out_dir / f"{subject_id}_{hadm_id}.json"
        output_path.write_text(json.dumps(graph, indent=2, ensure_ascii=False), encoding="utf-8")
        written += 1
        total_nodes += len(graph["nodes"])
        total_edges += len(graph["edges"])

    manifest = {
        "method": METHOD,
        "source_path": str(source_path),
        "output_dir": str(out_dir),
        "graphs": written,
        "total_nodes": total_nodes,
        "total_edges": total_edges,
    }
    (out_dir / "_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"[fawkes-mimic-kgs] wrote {written} graph(s) -> {out_dir}")
    print(f"[fawkes-mimic-kgs] total nodes={total_nodes} edges={total_edges}")


if __name__ == "__main__":
    main()
