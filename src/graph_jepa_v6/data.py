"""PyG graph tensor conversion for Graph-JEPA v6."""

from __future__ import annotations

import math
import re
from typing import Sequence

import torch

from graph_jepa.schema import EDGE_TYPE_TO_IDX, NODE_TYPE_TO_IDX, PatientGraph
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


def _node_text(node: dict | None) -> str:
    if not node:
        return ""
    return str(
        node.get("text")
        or node.get("name")
        or node.get("normalized_name")
        or node.get("id")
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


def _truthy(value) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() not in {"", "0", "0.0", "false", "no", "none"}


def _edge_endpoint(edge: dict, primary: str, fallback: str) -> str:
    return str(edge.get(primary) or edge.get(fallback) or "")


def _edge_relation(edge: dict) -> str:
    return str(edge.get("type") or edge.get("relation") or "")


def _graph_with_aliases(graph: PatientGraph) -> PatientGraph:
    nodes = []
    changed = False
    for node in graph.nodes:
        if node.get("text"):
            nodes.append(node)
            continue
        normalized = dict(node)
        normalized["text"] = _node_text(node)
        nodes.append(normalized)
        changed = True

    edges = []
    for edge in graph.edges:
        source_id = _edge_endpoint(edge, "source_id", "source")
        target_id = _edge_endpoint(edge, "target_id", "target")
        relation = _edge_relation(edge)
        if (
            edge.get("source_id") == source_id
            and edge.get("target_id") == target_id
            and edge.get("type") == relation
        ):
            edges.append(edge)
            continue
        normalized = dict(edge)
        normalized["source_id"] = source_id
        normalized["target_id"] = target_id
        normalized["type"] = relation
        edges.append(normalized)
        changed = True
    if not changed:
        return graph
    return PatientGraph(nodes=nodes, edges=edges, extra=graph.extra)


def _note_norm(text) -> str:
    return " ".join(str(text or "").lower().split())


def _note_text(graph: PatientGraph) -> str:
    return str(graph.extra.get("note") or graph.extra.get("input_block") or "")


def _note_embedding(graph: PatientGraph, note_embedding_dim: int) -> list[float] | None:
    raw = graph.extra.get("note_embedding")
    if raw is None:
        raw = graph.extra.get("note_embeddings")
    if raw is None:
        return None
    values = [float(value) for value in raw]
    if len(values) != note_embedding_dim:
        subject = graph.extra.get("subject_id", graph.extra.get("_source_path", "unknown"))
        raise ValueError(
            "note_embedding dim "
            f"{len(values)} != {note_embedding_dim} for graph {subject}"
        )
    return values


def _edge_prov_in_note(edge: dict) -> bool:
    labels = edge.get("labels")
    if isinstance(labels, dict) and _truthy(labels.get("prov_in_note")):
        return True
    return _truthy(edge.get("prov_in_note"))


def _grounded_note_nodes(graph: PatientGraph, note_ground_by: str) -> set[int]:
    ground_by = str(note_ground_by or "prov").lower()
    id_to_idx = graph.id_to_index()
    grounded: set[int] = set()

    if ground_by == "prov":
        for edge in graph.edges:
            if not _edge_prov_in_note(edge):
                continue
            source = id_to_idx.get(_edge_endpoint(edge, "source_id", "source"))
            target = id_to_idx.get(_edge_endpoint(edge, "target_id", "target"))
            if source is not None:
                grounded.add(source)
            if target is not None:
                grounded.add(target)
        return grounded

    if ground_by == "name":
        note = _note_norm(_note_text(graph))
        for idx, node in enumerate(graph.nodes):
            name = _note_norm(_node_name(node))
            if len(name) >= 3 and name in note:
                grounded.add(idx)
        return grounded

    if ground_by == "all":
        patient_type = NODE_TYPE_TO_IDX["PATIENT"]
        for idx, node_type in enumerate(graph.node_type_indices()):
            if node_type != patient_type:
                grounded.add(idx)
        return grounded

    raise ValueError(
        "note_ground_by must be one of 'prov', 'name', or 'all', "
        f"got {note_ground_by!r}"
    )


def _append_note_features(
    data,
    graph: PatientGraph,
    *,
    note_embedding_dim: int,
    note_ground_by: str,
):
    grounded = _grounded_note_nodes(graph, note_ground_by)
    note = _note_embedding(graph, note_embedding_dim)
    if note is None:
        note_vector = data.x.new_zeros((note_embedding_dim,))
        note_present = False
    else:
        note_vector = torch.tensor(note, dtype=data.x.dtype, device=data.x.device)
        note_present = True

    note_features = data.x.new_zeros((data.num_nodes, note_embedding_dim))
    if grounded:
        grounded_idx = torch.tensor(
            sorted(grounded),
            dtype=torch.long,
            device=data.x.device,
        )
        note_features[grounded_idx] = note_vector
    data.x = torch.cat([data.x, note_features], dim=-1)
    data.note_grounded_mask = torch.zeros(
        data.num_nodes,
        dtype=torch.bool,
        device=data.x.device,
    )
    if grounded:
        data.note_grounded_mask[grounded_idx] = True
    data.n_note_grounded = torch.tensor([len(grounded)], dtype=torch.long)
    data.note_embedding_present = torch.tensor([note_present], dtype=torch.bool)
    return data


def to_graph_data(
    graph: PatientGraph,
    encoder,
    *,
    use_note_embeddings: bool = True,
    note_embedding_dim: int = 768,
    note_ground_by: str = "prov",
):
    """Convert a graph, adding localized note embeddings to node features."""

    graph = _graph_with_aliases(graph)
    data = _to_graph_data(graph, encoder)
    if use_note_embeddings:
        data = _append_note_features(
            data,
            graph,
            note_embedding_dim=note_embedding_dim,
            note_ground_by=note_ground_by,
        )
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
        source_id = _edge_endpoint(edge, "source_id", "source")
        target_id = _edge_endpoint(edge, "target_id", "target")
        source = id_to_idx.get(source_id)
        target = id_to_idx.get(target_id)
        relation = EDGE_TYPE_TO_IDX.get(_edge_relation(edge))
        if source is None or target is None or relation is None:
            continue
        confidence.append(_confidence_value(edge))
        is_llm.append(str(edge.get("evidence") or "").strip().lower() == "llm")
        artifact.append(
            _is_clinical_artifact_node(node_by_id.get(source_id))
            or _is_clinical_artifact_node(node_by_id.get(target_id))
        )
    data.edge_llm_confidence = torch.tensor(confidence, dtype=torch.float)
    data.edge_is_llm = torch.tensor(is_llm, dtype=torch.bool)
    data.edge_clinical_artifact = torch.tensor(artifact, dtype=torch.bool)
    return data


class PatientGraphDataset:
    """Lazy in-memory PyG dataset carrying v6 note and confidence metadata."""

    def __init__(
        self,
        graphs: Sequence[PatientGraph],
        encoder,
        *,
        use_note_embeddings: bool = True,
        note_embedding_dim: int = 768,
        note_ground_by: str = "prov",
    ):
        self.graphs = list(graphs)
        self.encoder = encoder
        self.use_note_embeddings = use_note_embeddings
        self.note_embedding_dim = note_embedding_dim
        self.note_ground_by = note_ground_by
        self._cache: list[object | None] = [None] * len(self.graphs)

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int):
        if self._cache[idx] is None:
            self._cache[idx] = to_graph_data(
                self.graphs[idx],
                self.encoder,
                use_note_embeddings=self.use_note_embeddings,
                note_embedding_dim=self.note_embedding_dim,
                note_ground_by=self.note_ground_by,
            )
        return self._cache[idx]

__all__ = ["PatientGraphDataset", "to_graph_data"]
