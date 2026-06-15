"""PyG graph tensor conversion for Graph-JEPA v5."""

from __future__ import annotations

import math
import re
from typing import Sequence

import torch

from graph_jepa.schema import EDGE_TYPE_TO_IDX, PatientGraph
from graph_jepa_v4.data import to_graph_data as _to_graph_data


_ADMIN_ARTIFACT_EXACT = {
    "bag",
    "syringe",
    "vial",
}
_ADMIN_ARTIFACT_PATTERN = re.compile(
    r"(?:\bflush\b|sterile water|mini bag plus|iso-osmotic dextrose|\bsyringe\b)",
    re.IGNORECASE,
)


def _confidence_value(edge: dict) -> float:
    try:
        value = float(edge.get("confidence"))
    except (TypeError, ValueError):
        return float("nan")
    return value if math.isfinite(value) else float("nan")


def _node_name(node: dict | None) -> str:
    if not node:
        return ""
    return str(
        node.get("name")
        or node.get("normalized_name")
        or node.get("text")
        or ""
    ).strip()


def _is_clinical_artifact_node(node: dict | None) -> bool:
    if not node:
        return False
    node_type = str(node.get("type") or "").upper()
    name = _node_name(node)
    normalized = name.lower()
    if node_type == "MEDICATION":
        return (
            normalized in _ADMIN_ARTIFACT_EXACT
            or bool(_ADMIN_ARTIFACT_PATTERN.search(name))
        )
    if node_type == "MICROBIOLOGY":
        return normalized == "cancelled"
    return False


def to_graph_data(graph: PatientGraph, encoder):
    """Convert a graph and preserve aligned fine-tuning edge metadata."""

    data = _to_graph_data(graph, encoder)
    id_to_idx = graph.id_to_index()
    node_by_id = {
        str(node.get("id")): node
        for node in graph.nodes
        if node.get("id")
    }
    confidence = []
    is_llm = []
    artifact = []
    for edge in graph.edges:
        source = id_to_idx.get(edge["source_id"])
        target = id_to_idx.get(edge["target_id"])
        relation = EDGE_TYPE_TO_IDX.get(edge["type"])
        if source is None or target is None or relation is None:
            continue
        confidence.append(_confidence_value(edge))
        is_llm.append(str(edge.get("evidence") or "").strip().lower() == "llm")
        artifact.append(
            _is_clinical_artifact_node(node_by_id.get(str(edge["source_id"])))
            or _is_clinical_artifact_node(node_by_id.get(str(edge["target_id"])))
        )
    data.edge_llm_confidence = torch.tensor(confidence, dtype=torch.float)
    data.edge_is_llm = torch.tensor(is_llm, dtype=torch.bool)
    data.edge_clinical_artifact = torch.tensor(artifact, dtype=torch.bool)
    return data


class PatientGraphDataset:
    """Lazy in-memory PyG dataset carrying v5 edge-confidence metadata."""

    def __init__(self, graphs: Sequence[PatientGraph], encoder):
        self.graphs = list(graphs)
        self.encoder = encoder
        self._cache: list[object | None] = [None] * len(self.graphs)

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int):
        if self._cache[idx] is None:
            self._cache[idx] = to_graph_data(self.graphs[idx], self.encoder)
        return self._cache[idx]

__all__ = ["PatientGraphDataset", "to_graph_data"]
