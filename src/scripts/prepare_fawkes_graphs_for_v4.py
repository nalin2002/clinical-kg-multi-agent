#!/usr/bin/env python3
"""Convert Fawkes graph parquet tables into v4 training-ready sub-KG JSONs.

Graph-JEPA v4 already knows how to train from MIMIC-style sub-KG JSON files via:

    --data mimic-subkgs --mimic-subkg-path <json-dir>

The Fawkes graph dataset is stored as three parquet tables:

    patients/train-*.parquet
    nodes/train-*.parquet
    edges/train-*.parquet

This script joins those tables by ``(subject_id, hadm_id)`` and writes one
``<subject_id>_<hadm_id>.json`` graph per admission under ``train/`` or
``test/``. Dataset-specific schema repairs happen here, during conversion, so
v4 training does not learn known conversion mismatches as negative edges.
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
DEFAULT_OUT = Path("outputs/fawkes_mimic_graphs_2k_3k/sub_kgs")
METHOD = "fawkes_mimic_graphs_complete_v8_parquet"
REPAIRED_METHOD = f"{METHOD}_v4_schema_repaired"
HIGH_CONFIDENCE_THRESHOLD = 0.80
INFECTION_LIKE_TERMS = (
    "abscess",
    "bacteremia",
    "bacteriuria",
    "cellulitis",
    "cholangitis",
    "cystitis",
    "empyema",
    "endocarditis",
    "infect",
    "meningitis",
    "osteomyelitis",
    "peritonitis",
    "pneumonia",
    "pyelonephritis",
    "sepsis",
    "septic",
    "urinary tract infection",
)


def normalize_text(value: Any) -> str:
    return str(clean_value(value) or "").strip().lower()


def truthy(value: Any) -> bool:
    value = clean_value(value)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "t", "yes", "y"}
    return bool(value)


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
        return {str(k): clean_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_value(v) for v in value]
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
    return {str(key): clean_value(value) for key, value in row.to_dict().items()}


def graph_key(subject_id: Any, hadm_id: Any) -> tuple[str, str]:
    return str(clean_value(subject_id)), str(clean_value(hadm_id))


def value_name(value: Any) -> str:
    value = clean_value(value)
    if isinstance(value, dict):
        for key in ("name", "description", "label", "organism", "specimen"):
            text = str(value.get(key) or "").strip()
            if text:
                return text
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value or "").strip()


def patient_symptom_names(patient: dict[str, Any]) -> set[str]:
    symptoms = patient.get("symptoms") or []
    if not isinstance(symptoms, list):
        symptoms = [symptoms]
    return {
        normalize_text(value_name(symptom))
        for symptom in symptoms
        if normalize_text(value_name(symptom))
    }


def build_node(
    row: pd.Series,
    *,
    symptom_names: set[str],
    repair_counts: Counter[str],
) -> dict[str, Any]:
    data = row_dict(row)
    node_id = str(data.pop("node_id"))
    raw_type = str(data.pop("type", "") or "").upper()
    name = str(data.get("name") or data.get("normalized_name") or node_id)
    normalized_name = str(data.get("normalized_name") or name).lower()
    node_type = raw_type
    schema_repair = None

    if (
        raw_type == "DIAGNOSIS"
        and (
            truthy(data.get("presenting"))
            or normalize_text(normalized_name) in symptom_names
        )
    ):
        node_type = "SYMPTOM"
        repair_counts["presenting_diagnosis_to_symptom_nodes"] += 1
        schema_repair = {
            "action": "retype_presenting_diagnosis_to_symptom",
            "from_type": raw_type,
            "to_type": node_type,
        }

    node = {
        **data,
        "id": node_id,
        "type": node_type,
        "mimic_type": raw_type,
        "name": name,
        "normalized_name": normalized_name,
        "text": name,
        "evidence": "fawkes graph parquet",
        "origin": REPAIRED_METHOD,
    }
    if schema_repair is not None:
        node["schema_repair"] = schema_repair
    return node


def node_name(node: dict[str, Any] | None, fallback: str) -> str:
    if not node:
        return fallback
    return str(
        node.get("name")
        or node.get("normalized_name")
        or node.get("text")
        or fallback
    )


def is_infection_like_problem(node: dict[str, Any] | None) -> bool:
    if not node:
        return False
    text = normalize_text(
        " ".join(
            str(node.get(key) or "")
            for key in ("name", "normalized_name", "text")
        )
    )
    return any(term in text for term in INFECTION_LIKE_TERMS)


def confidence_at_least(value: Any, threshold: float = HIGH_CONFIDENCE_THRESHOLD) -> bool:
    value = clean_value(value)
    try:
        return float(value) >= threshold
    except (TypeError, ValueError):
        return False


def confidence_value(edge: dict[str, Any]) -> float:
    try:
        return float(clean_value(edge.get("confidence")))
    except (TypeError, ValueError):
        return float("-inf")


def deduplicate_edges(
    edges: list[dict[str, Any]],
    *,
    repair_counts: Counter[str],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for edge in edges:
        key = (
            str(edge.get("source_id") or ""),
            str(edge.get("target_id") or ""),
            str(edge.get("type") or ""),
        )
        grouped.setdefault(key, []).append(edge)

    deduped: list[dict[str, Any]] = []
    for group in grouped.values():
        if len(group) == 1:
            deduped.append(group[0])
            continue
        kept = max(group, key=confidence_value)
        kept = dict(kept)
        kept["deduplicated_count"] = len(group)
        repair_counts["duplicate_edge_groups_collapsed"] += 1
        repair_counts["duplicate_edges_removed"] += len(group) - 1
        deduped.append(kept)
    return deduped


def build_edge(
    row: pd.Series,
    *,
    node_by_id: dict[str, dict[str, Any]],
    repair_counts: Counter[str],
) -> dict[str, Any]:
    data = row_dict(row)
    raw_source_id = str(data.get("source_id") or data.get("source") or "")
    raw_target_id = str(data.get("target_id") or data.get("target") or "")
    raw_relation = str(data.get("relation") or data.get("type") or "").upper()
    source_id = raw_source_id
    target_id = raw_target_id
    relation = raw_relation
    source_node = node_by_id.get(source_id)
    target_node = node_by_id.get(target_id)
    source_type = str(
        (source_node or {}).get("type")
        or data.get("source_type")
        or ""
    ).upper()
    target_type = str(
        (target_node or {}).get("type")
        or data.get("target_type")
        or ""
    ).upper()
    schema_repair = None

    if (
        source_type == "MEDICATION"
        and raw_relation == "MANAGED_FOR"
        and target_type == "DIAGNOSIS"
    ):
        source_id, target_id = target_id, source_id
        source_node, target_node = target_node, source_node
        source_type, target_type = target_type, source_type
        relation = "TREATED_BY"
        repair_counts["medication_managed_for_to_diagnosis_treated_by_edges"] += 1
        schema_repair = {
            "action": "remap_medication_managed_for_to_treated_by",
            "from": {
                "source_id": raw_source_id,
                "target_id": raw_target_id,
                "type": raw_relation,
            },
            "to": {
                "source_id": source_id,
                "target_id": target_id,
                "type": relation,
            },
        }
    elif (
        source_type == "MEDICATION"
        and raw_relation in {"CONFIRMS", "INDICATES"}
        and target_type == "DIAGNOSIS"
    ):
        source_id, target_id = target_id, source_id
        source_node, target_node = target_node, source_node
        source_type, target_type = target_type, source_type
        relation = "TREATED_BY"
        repair_counts["medication_evidence_to_diagnosis_treated_by_edges"] += 1
        schema_repair = {
            "action": "remap_medication_evidence_to_treated_by",
            "from": {
                "source_id": raw_source_id,
                "target_id": raw_target_id,
                "type": raw_relation,
            },
            "to": {
                "source_id": source_id,
                "target_id": target_id,
                "type": relation,
            },
        }
    elif (
        source_type == "MEDICATION"
        and raw_relation == "MANAGED_FOR"
        and target_type == "MICROBIOLOGY"
    ):
        relation = "TARGETS_ORGANISM"
        repair_counts["medication_managed_for_to_microbiology_targets_edges"] += 1
        schema_repair = {
            "action": "remap_medication_microbiology_edge_to_targets_organism",
            "from": {
                "source_id": raw_source_id,
                "target_id": raw_target_id,
                "type": raw_relation,
            },
            "to": {
                "source_id": source_id,
                "target_id": target_id,
                "type": relation,
            },
        }
    elif (
        source_type == "PATIENT"
        and raw_relation == "HAS_DIAGNOSIS"
        and target_type == "SYMPTOM"
    ):
        relation = "ASSOCIATED_WITH"
        repair_counts["patient_has_diagnosis_to_symptom_associated_edges"] += 1
        schema_repair = {
            "action": "remap_patient_symptom_diagnosis_edge_to_associated_with",
            "from": {
                "source_id": raw_source_id,
                "target_id": raw_target_id,
                "type": raw_relation,
            },
            "to": {
                "source_id": source_id,
                "target_id": target_id,
                "type": relation,
            },
        }
    elif (
        source_type == "MEDICATION"
        and raw_relation == "MANAGED_FOR"
        and target_type == "SYMPTOM"
    ):
        relation = "ASSOCIATED_WITH"
        repair_counts["medication_managed_for_to_symptom_associated_edges"] += 1
        schema_repair = {
            "action": "remap_medication_symptom_edge_to_associated_with",
            "from": {
                "source_id": raw_source_id,
                "target_id": raw_target_id,
                "type": raw_relation,
            },
            "to": {
                "source_id": source_id,
                "target_id": target_id,
                "type": relation,
            },
        }
    elif (
        source_type == "MEDICATION"
        and raw_relation == "MANAGED_FOR"
        and target_type == "PROCEDURE"
    ):
        relation = "USED_DURING"
        repair_counts["medication_managed_for_to_procedure_used_during_edges"] += 1
        schema_repair = {
            "action": "remap_medication_procedure_edge_to_used_during",
            "from": {
                "source_id": raw_source_id,
                "target_id": raw_target_id,
                "type": raw_relation,
            },
            "to": {
                "source_id": source_id,
                "target_id": target_id,
                "type": relation,
            },
        }
    elif (
        source_type == "MEDICATION"
        and raw_relation == "MANAGED_FOR"
        and target_type == "MEDICATION"
    ):
        relation = "ASSOCIATED_WITH"
        repair_counts["medication_managed_for_to_medication_associated_edges"] += 1
        schema_repair = {
            "action": "remap_medication_to_medication_edge_to_associated_with",
            "from": {
                "source_id": raw_source_id,
                "target_id": raw_target_id,
                "type": raw_relation,
            },
            "to": {
                "source_id": source_id,
                "target_id": target_id,
                "type": relation,
            },
        }
    elif (
        source_type == "MEDICATION"
        and raw_relation == "COMPLICATED_BY"
        and target_type == "DIAGNOSIS"
    ):
        relation = "CAUSES"
        repair_counts["medication_complicated_by_to_diagnosis_causes_edges"] += 1
        schema_repair = {
            "action": "remap_medication_complication_to_causes",
            "from": {
                "source_id": raw_source_id,
                "target_id": raw_target_id,
                "type": raw_relation,
            },
            "to": {
                "source_id": source_id,
                "target_id": target_id,
                "type": relation,
            },
        }
    elif (
        source_type == "PATIENT"
        and raw_relation == "MANAGED_FOR"
        and target_type == "MEDICATION"
    ):
        relation = "TAKES_MEDICATION"
        repair_counts["patient_managed_for_medication_takes_edges"] += 1
        schema_repair = {
            "action": "remap_patient_managed_for_medication_to_takes_medication",
            "from": {
                "source_id": raw_source_id,
                "target_id": raw_target_id,
                "type": raw_relation,
            },
            "to": {
                "source_id": source_id,
                "target_id": target_id,
                "type": relation,
            },
        }
    elif (
        source_type == "DIAGNOSIS"
        and raw_relation == "INDICATES"
        and target_type == "DIAGNOSIS"
    ):
        relation = "ASSOCIATED_WITH"
        repair_counts["diagnosis_indicates_diagnosis_associated_edges"] += 1
        schema_repair = {
            "action": "remap_diagnosis_indicates_diagnosis_to_associated_with",
            "from": {
                "source_id": raw_source_id,
                "target_id": raw_target_id,
                "type": raw_relation,
            },
            "to": {
                "source_id": source_id,
                "target_id": target_id,
                "type": relation,
            },
        }
    elif (
        source_type == "DIAGNOSIS"
        and raw_relation == "COMPLICATED_BY"
        and target_type == "MICROBIOLOGY"
    ):
        source_id, target_id = target_id, source_id
        source_node, target_node = target_node, source_node
        source_type, target_type = target_type, source_type
        relation = "CONFIRMS"
        repair_counts["diagnosis_microbiology_complicated_by_to_confirms_edges"] += 1
        schema_repair = {
            "action": "remap_diagnosis_microbiology_complication_to_confirms",
            "from": {
                "source_id": raw_source_id,
                "target_id": raw_target_id,
                "type": raw_relation,
            },
            "to": {
                "source_id": source_id,
                "target_id": target_id,
                "type": relation,
            },
        }
    elif (
        source_type == "DIAGNOSIS"
        and raw_relation == "COMPLICATED_BY"
        and target_type == "MEDICATION"
    ):
        relation = "ASSOCIATED_WITH"
        repair_counts["diagnosis_complicated_by_medication_associated_edges"] += 1
        schema_repair = {
            "action": "remap_diagnosis_medication_complication_to_associated_with",
            "from": {
                "source_id": raw_source_id,
                "target_id": raw_target_id,
                "type": raw_relation,
            },
            "to": {
                "source_id": source_id,
                "target_id": target_id,
                "type": relation,
            },
        }
    elif (
        source_type == "MICROBIOLOGY"
        and raw_relation in {"CONFIRMS", "INDICATES"}
        and target_type == "SYMPTOM"
    ):
        if is_infection_like_problem(target_node):
            relation = "CONFIRMS"
            repair_counts["microbiology_evidence_to_symptom_confirms_edges"] += 1
            action = "remap_microbiology_symptom_evidence_to_confirms"
        else:
            relation = "ASSOCIATED_WITH"
            repair_counts["microbiology_evidence_to_symptom_associated_edges"] += 1
            action = "remap_microbiology_symptom_evidence_to_associated_with"
        schema_repair = {
            "action": action,
            "from": {
                "source_id": raw_source_id,
                "target_id": raw_target_id,
                "type": raw_relation,
            },
            "to": {
                "source_id": source_id,
                "target_id": target_id,
                "type": relation,
            },
        }
    elif (
        source_type == "MICROBIOLOGY"
        and raw_relation == "INDICATES"
        and target_type == "DIAGNOSIS"
    ):
        relation = "CONFIRMS"
        repair_counts["microbiology_indicates_diagnosis_confirms_edges"] += 1
        schema_repair = {
            "action": "remap_microbiology_indicates_diagnosis_to_confirms",
            "from": {
                "source_id": raw_source_id,
                "target_id": raw_target_id,
                "type": raw_relation,
            },
            "to": {
                "source_id": source_id,
                "target_id": target_id,
                "type": relation,
            },
        }
    elif (
        source_type == "PROCEDURE"
        and target_type == "DIAGNOSIS"
        and raw_relation in {"COMPLICATED_BY", "INDICATES"}
    ):
        relation = "PERFORMED_FOR"
        repair_counts["procedure_relation_to_performed_for_edges"] += 1
        schema_repair = {
            "action": "remap_procedure_relation_to_performed_for",
            "from": {
                "source_id": raw_source_id,
                "target_id": raw_target_id,
                "type": raw_relation,
            },
            "to": {
                "source_id": source_id,
                "target_id": target_id,
                "type": relation,
            },
        }
    elif (
        source_type == "PROCEDURE"
        and raw_relation == "CONFIRMS"
        and target_type == "SYMPTOM"
    ):
        if confidence_at_least(data.get("confidence")):
            relation = "CONFIRMS"
            repair_counts["procedure_confirms_symptom_high_confidence_edges"] += 1
            action = "retain_high_confidence_procedure_confirms_symptom"
        else:
            relation = "ASSOCIATED_WITH"
            repair_counts["procedure_confirms_symptom_low_confidence_associated_edges"] += 1
            action = "remap_low_confidence_procedure_confirms_symptom_to_associated_with"
        schema_repair = {
            "action": action,
            "confidence_threshold": HIGH_CONFIDENCE_THRESHOLD,
            "from": {
                "source_id": raw_source_id,
                "target_id": raw_target_id,
                "type": raw_relation,
            },
            "to": {
                "source_id": source_id,
                "target_id": target_id,
                "type": relation,
            },
        }
    elif (
        source_type == "PROCEDURE"
        and raw_relation == "CONFIRMS"
        and target_type == "MICROBIOLOGY"
    ):
        relation = "DETECTS"
        repair_counts["procedure_confirms_microbiology_detects_edges"] += 1
        schema_repair = {
            "action": "remap_procedure_confirms_microbiology_to_detects",
            "from": {
                "source_id": raw_source_id,
                "target_id": raw_target_id,
                "type": raw_relation,
            },
            "to": {
                "source_id": source_id,
                "target_id": target_id,
                "type": relation,
            },
        }
    elif (
        source_type == "DIAGNOSIS"
        and target_type == "PROCEDURE"
        and raw_relation == "COMPLICATED_BY"
    ):
        source_id, target_id = target_id, source_id
        source_node, target_node = target_node, source_node
        source_type, target_type = target_type, source_type
        relation = "PERFORMED_FOR"
        repair_counts["diagnosis_procedure_complicated_by_to_performed_for_edges"] += 1
        schema_repair = {
            "action": "remap_diagnosis_procedure_complication_to_performed_for",
            "from": {
                "source_id": raw_source_id,
                "target_id": raw_target_id,
                "type": raw_relation,
            },
            "to": {
                "source_id": source_id,
                "target_id": target_id,
                "type": relation,
            },
        }
    elif (
        source_type == "MICROBIOLOGY"
        and raw_relation == "COMPLICATED_BY"
        and target_type == "DIAGNOSIS"
    ):
        relation = "CONFIRMS"
        repair_counts["microbiology_complicated_by_to_diagnosis_confirms_edges"] += 1
        schema_repair = {
            "action": "remap_microbiology_complicated_by_to_confirms",
            "from": {
                "source_id": raw_source_id,
                "target_id": raw_target_id,
                "type": raw_relation,
            },
            "to": {
                "source_id": source_id,
                "target_id": target_id,
                "type": relation,
            },
        }
    elif (
        source_type == "PROCEDURE"
        and target_type == "SYMPTOM"
        and raw_relation == "COMPLICATED_BY"
    ):
        relation = "COMPLICATED_BY"
        repair_counts["procedure_complicated_by_symptom_edges"] += 1
        schema_repair = {
            "action": "retain_procedure_complicated_by_symptom",
            "from": {
                "source_id": raw_source_id,
                "target_id": raw_target_id,
                "type": raw_relation,
            },
            "to": {
                "source_id": source_id,
                "target_id": target_id,
                "type": relation,
            },
        }
    elif (
        source_type == "PROCEDURE"
        and target_type == "SYMPTOM"
        and raw_relation == "INDICATES"
    ):
        relation = "PERFORMED_FOR"
        repair_counts["procedure_indicates_symptom_performed_for_edges"] += 1
        schema_repair = {
            "action": "remap_procedure_indicates_symptom_to_performed_for",
            "from": {
                "source_id": raw_source_id,
                "target_id": raw_target_id,
                "type": raw_relation,
            },
            "to": {
                "source_id": source_id,
                "target_id": target_id,
                "type": relation,
            },
        }
    elif (
        source_type == "SYMPTOM"
        and raw_relation == "COMPLICATED_BY"
        and target_type == "MICROBIOLOGY"
    ):
        if is_infection_like_problem(source_node):
            source_id, target_id = target_id, source_id
            source_node, target_node = target_node, source_node
            source_type, target_type = target_type, source_type
            relation = "CONFIRMS"
            repair_counts["symptom_microbiology_complication_to_confirms_edges"] += 1
            action = "remap_symptom_microbiology_complication_to_confirms"
        else:
            relation = "ASSOCIATED_WITH"
            repair_counts["symptom_microbiology_complication_associated_edges"] += 1
            action = "remap_symptom_microbiology_complication_to_associated_with"
        schema_repair = {
            "action": action,
            "from": {
                "source_id": raw_source_id,
                "target_id": raw_target_id,
                "type": raw_relation,
            },
            "to": {
                "source_id": source_id,
                "target_id": target_id,
                "type": relation,
            },
        }
    elif (
        source_type == "SYMPTOM"
        and raw_relation == "COMPLICATED_BY"
        and target_type == "PROCEDURE"
    ):
        source_id, target_id = target_id, source_id
        source_node, target_node = target_node, source_node
        source_type, target_type = target_type, source_type
        relation = "PERFORMED_FOR"
        repair_counts["symptom_procedure_complication_to_performed_for_edges"] += 1
        schema_repair = {
            "action": "remap_symptom_procedure_complication_to_performed_for",
            "from": {
                "source_id": raw_source_id,
                "target_id": raw_target_id,
                "type": raw_relation,
            },
            "to": {
                "source_id": source_id,
                "target_id": target_id,
                "type": relation,
            },
        }
    elif (
        source_type == "SYMPTOM"
        and raw_relation == "COMPLICATED_BY"
        and target_type == "DIAGNOSIS"
    ):
        relation = "INDICATES"
        repair_counts["symptom_complicated_by_to_indicates_edges"] += 1
        schema_repair = {
            "action": "remap_symptom_complicated_by_to_indicates",
            "from": {
                "source_id": raw_source_id,
                "target_id": raw_target_id,
                "type": raw_relation,
            },
            "to": {
                "source_id": source_id,
                "target_id": target_id,
                "type": relation,
            },
        }
    elif (
        source_type == "DIAGNOSIS"
        and raw_relation == "COMPLICATED_BY"
        and target_type == "SYMPTOM"
    ):
        relation = "CAUSES"
        repair_counts["diagnosis_complicated_by_to_symptom_causes_edges"] += 1
        schema_repair = {
            "action": "remap_diagnosis_symptom_complication_to_causes",
            "from": {
                "source_id": raw_source_id,
                "target_id": raw_target_id,
                "type": raw_relation,
            },
            "to": {
                "source_id": source_id,
                "target_id": target_id,
                "type": relation,
            },
        }
    elif (
        source_type == "SYMPTOM"
        and target_type == "SYMPTOM"
        and raw_relation in {"COMPLICATED_BY", "INDICATES"}
    ):
        relation = "CO_OCCURS_WITH"
        repair_counts["symptom_to_symptom_unconstrained_edges"] += 1
        schema_repair = {
            "action": "remap_symptom_to_symptom_edge_to_co_occurs_with",
            "from": {
                "source_id": raw_source_id,
                "target_id": raw_target_id,
                "type": raw_relation,
            },
            "to": {
                "source_id": source_id,
                "target_id": target_id,
                "type": relation,
            },
        }

    edge = {
        **data,
        "source": source_id,
        "target": target_id,
        "source_id": source_id,
        "target_id": target_id,
        "relation": relation,
        "type": relation,
        "source_name": node_name(source_node, source_id),
        "source_type": source_type,
        "target_name": node_name(target_node, target_id),
        "target_type": target_type,
        "evidence": data.get("evidence") or "fawkes graph parquet",
        "origin": REPAIRED_METHOD,
    }
    if schema_repair is not None:
        edge["schema_repair"] = schema_repair
    return edge


def split_assignments(
    keys: list[tuple[Any, Any]],
    *,
    test_patients: int,
    seed: int,
) -> tuple[dict[tuple[Any, Any], str], dict[str, Any]]:
    if test_patients < 0:
        raise SystemExit("--test-patients must be non-negative")

    subject_counts = Counter(graph_key(raw_key[0], raw_key[1])[0] for raw_key in keys)
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
        raw_key: (
            "test"
            if graph_key(raw_key[0], raw_key[1])[0] in test_subjects
            else "train"
        )
        for raw_key in keys
    }
    return assignments, {
        "train_graphs": sum(1 for split in assignments.values() if split == "train"),
        "test_graphs": sum(1 for split in assignments.values() if split == "test"),
        "test_patients": test_patients,
        "test_seed": seed,
        "eligible_single_admission_patients": len(eligible_subjects),
        "subjects_with_multiple_admissions": sum(
            1 for count in subject_counts.values() if count > 1
        ),
        "test_subject_ids": sorted(test_subjects),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default=str(DEFAULT_ROOT),
        help=f"Downloaded Fawkes graph dataset root (default: {DEFAULT_ROOT})",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUT),
        help=(
            "Output directory for train/test graph folders "
            f"(default: {DEFAULT_OUT})"
        ),
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional graph limit")
    parser.add_argument(
        "--test-patients",
        type=int,
        default=0,
        help=(
            "Number of single-admission patients to place in test/. "
            "All remaining admissions are written to train/."
        ),
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=0,
        help="Deterministic seed for selecting test patients",
    )
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out)
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
    if args.limit is not None:
        keys = keys[: args.limit]
    split_by_key, split_manifest = split_assignments(
        keys,
        test_patients=args.test_patients,
        seed=args.split_seed,
    )

    written = 0
    total_nodes = 0
    total_edges = 0
    written_by_split: Counter[str] = Counter()
    total_nodes_by_split: Counter[str] = Counter()
    total_edges_by_split: Counter[str] = Counter()
    repair_counts: Counter[str] = Counter()
    for raw_key in keys:
        subject_id, hadm_id = graph_key(raw_key[0], raw_key[1])
        patient = patient_by_key.get((subject_id, hadm_id), {})
        symptom_names = patient_symptom_names(patient)
        graph_nodes = [
            build_node(row, symptom_names=symptom_names, repair_counts=repair_counts)
            for _, row in node_groups.get(raw_key, pd.DataFrame()).iterrows()
        ]
        node_by_id = {str(node["id"]): node for node in graph_nodes}
        graph_edges = [
            build_edge(row, node_by_id=node_by_id, repair_counts=repair_counts)
            for _, row in edge_groups.get(raw_key, pd.DataFrame()).iterrows()
        ]
        graph_edges = deduplicate_edges(graph_edges, repair_counts=repair_counts)
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
            "_method": REPAIRED_METHOD,
            "_source_root": str(root),
            "_patient": patient,
        }
        output_dir = test_dir if split == "test" else train_dir
        output_path = output_dir / f"{subject_id}_{hadm_id}.json"
        output_path.write_text(
            json.dumps(graph, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        written += 1
        total_nodes += len(graph_nodes)
        total_edges += len(graph_edges)
        written_by_split[split] += 1
        total_nodes_by_split[split] += len(graph_nodes)
        total_edges_by_split[split] += len(graph_edges)

    manifest = {
        "method": REPAIRED_METHOD,
        "source_root": str(root),
        "output_dir": str(out_dir),
        "graphs": written,
        "total_nodes": total_nodes,
        "total_edges": total_edges,
        "splits": {
            **split_manifest,
            "written_train_graphs": written_by_split["train"],
            "written_test_graphs": written_by_split["test"],
            "train_nodes": total_nodes_by_split["train"],
            "test_nodes": total_nodes_by_split["test"],
            "train_edges": total_edges_by_split["train"],
            "test_edges": total_edges_by_split["test"],
        },
        "repairs": dict(sorted(repair_counts.items())),
    }
    (out_dir / "_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"[fawkes-graphs-v4] wrote {written} graph(s) -> {out_dir}")
    print(
        "[fawkes-graphs-v4] splits "
        f"train={written_by_split['train']} test={written_by_split['test']}"
    )
    print(f"[fawkes-graphs-v4] total nodes={total_nodes} edges={total_edges}")
    print(f"[fawkes-graphs-v4] repairs={dict(sorted(repair_counts.items()))}")


if __name__ == "__main__":
    main()
