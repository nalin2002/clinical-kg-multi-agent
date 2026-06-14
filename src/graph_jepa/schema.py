"""Graph schema + pipeline-JSON round-tripping.

This module is intentionally torch-free so the data contract can be exercised
without the heavy ML dependencies.

Data contract (verified against the pipeline outputs):
  * nodes : {id, text, type, evidence, turn_id}  (unified adds occurrences/res_id)
  * edges : {source_id, target_id, type, evidence, turn_id}  (+ occurrences/res_id)

The refinement layer round-trips this exact shape and, by default, only adds
fields (``jepa_score``, ``jepa_flag``) to edges. Graph-JEPA v2 can optionally
append unverified ``jepa_suggested`` edges between existing nodes.
"""

from __future__ import annotations

import enum
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple


class NodeType(enum.Enum):
    SYMPTOM = "SYMPTOM"
    DIAGNOSIS = "DIAGNOSIS"
    TREATMENT = "TREATMENT"
    PROCEDURE = "PROCEDURE"
    LOCATION = "LOCATION"
    MEDICAL_HISTORY = "MEDICAL_HISTORY"
    LAB_RESULT = "LAB_RESULT"
    PATIENT = "PATIENT"
    MEDICATION = "MEDICATION"
    LAB_TEST = "LAB_TEST"
    MICROBIOLOGY = "MICROBIOLOGY"
    SERVICE = "SERVICE"
    PROCUREMENT = "PROCUREMENT"


class EdgeType(enum.Enum):
    CAUSES = "CAUSES"
    INDICATES = "INDICATES"
    LOCATED_AT = "LOCATED_AT"
    RULES_OUT = "RULES_OUT"
    TAKEN_FOR = "TAKEN_FOR"
    CONFIRMS = "CONFIRMS"
    ADMINISTERED_DURING = "ADMINISTERED_DURING"
    ASSOCIATED_WITH = "ASSOCIATED_WITH"
    COMPLICATED_BY = "COMPLICATED_BY"
    CO_OCCURS_WITH = "CO_OCCURS_WITH"
    DIAGNORED_BY = "DIAGNORED_BY"
    DIAGNOSED_BY = "DIAGNOSED_BY"
    HAD_LAB_TEST = "HAD_LAB_TEST"
    HAD_MICROBIOLOGY = "HAD_MICROBIOLOGY"
    HAD_PROCEDURE = "HAD_PROCEDURE"
    HAS_DIAGNOSIS = "HAS_DIAGNOSIS"
    HAS_MEDICATION = "HAS_MEDICATION"
    HAS_MICROBIOLOGY = "HAS_MICROBIOLOGY"
    INVESTIGATED_BY = "INVESTIGATED_BY"
    MANAGED_BY_SERVICE = "MANAGED_BY_SERVICE"
    MANAGED_FOR = "MANAGED_FOR"
    MANAGES_FOR = "MANAGES_FOR"
    MONITORED_BY = "MONITORED_BY"
    PART_OF_REGIMEN = "PART_OF_REGIMEN"
    PERFORMED_FOR = "PERFORMED_FOR"
    TAKES_MEDICATION = "TAKES_MEDICATION"
    TARGETS_ORGANISM = "TARGETS_ORGANISM"
    TARGET_ORGANISM = "TARGET_ORGANISM"
    TREATED_BY = "TREATED_BY"
    UNDERWENT_PROCEDURE = "UNDERWENT_PROCEDURE"
    USED_DURING = "USED_DURING"
    DETECTS = "DETECTS"


# Stable index maps (used to build embeddings / one-hots consistently).
NODE_TYPE_TO_IDX: Dict[str, int] = {t.value: i for i, t in enumerate(NodeType)}
EDGE_TYPE_TO_IDX: Dict[str, int] = {t.value: i for i, t in enumerate(EdgeType)}
IDX_TO_NODE_TYPE: Dict[int, str] = {i: t for t, i in NODE_TYPE_TO_IDX.items()}
IDX_TO_EDGE_TYPE: Dict[int, str] = {i: t for t, i in EDGE_TYPE_TO_IDX.items()}

NUM_NODE_TYPES = len(NODE_TYPE_TO_IDX)
NUM_EDGE_TYPES = len(EDGE_TYPE_TO_IDX)


@dataclass
class PatientGraph:
    """A single patient/transcript knowledge graph.

    ``nodes`` and ``edges`` hold the *raw* pipeline dicts so unknown/extra
    fields survive a round-trip untouched. ``extra`` captures any top-level keys
    besides ``nodes``/``edges`` (e.g. ``_method``, ``_source``).
    """

    nodes: List[Dict[str, Any]] = field(default_factory=list)
    edges: List[Dict[str, Any]] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    # ----- convenience accessors -------------------------------------------------
    @property
    def node_ids(self) -> List[str]:
        return [n["id"] for n in self.nodes]

    def id_to_index(self) -> Dict[str, int]:
        return {n["id"]: i for i, n in enumerate(self.nodes)}

    def node_type_indices(self) -> List[int]:
        return [NODE_TYPE_TO_IDX[n["type"]] for n in self.nodes]

    def node_encoder_keys(self) -> List[Tuple[str, str]]:
        """``(type, text)`` pairs used as the shared-node-space encoder key."""
        return [(n["type"], n.get("text", "")) for n in self.nodes]

    def edge_index_pairs(self) -> List[Tuple[int, int]]:
        """Edges as ``(src_idx, tgt_idx)`` using node-list ordering.

        Edges whose endpoints are not present in ``nodes`` are skipped (defensive
        against malformed graphs); use :meth:`valid_edge_mask` to learn which.
        """
        idx = self.id_to_index()
        pairs: List[Tuple[int, int]] = []
        for e in self.edges:
            s, t = idx.get(e["source_id"]), idx.get(e["target_id"])
            if s is not None and t is not None:
                pairs.append((s, t))
        return pairs

    def valid_edge_mask(self) -> List[bool]:
        idx = self.id_to_index()
        return [e["source_id"] in idx and e["target_id"] in idx for e in self.edges]

    def edge_type_indices(self) -> List[int]:
        return [EDGE_TYPE_TO_IDX[e["type"]] for e in self.edges if e["type"] in EDGE_TYPE_TO_IDX]

    # ----- (de)serialisation -----------------------------------------------------
    @classmethod
    def from_pipeline_json(cls, data: Dict[str, Any]) -> "PatientGraph":
        if "nodes" not in data or "edges" not in data:
            raise ValueError("pipeline JSON must contain 'nodes' and 'edges'")
        extra = {k: v for k, v in data.items() if k not in ("nodes", "edges")}
        return cls(
            nodes=[dict(n) for n in data["nodes"]],
            edges=[dict(e) for e in data["edges"]],
            extra=extra,
        )

    @classmethod
    def load(cls, path: str | Path) -> "PatientGraph":
        with open(path, "r") as f:
            return cls.from_pipeline_json(json.load(f))

    def to_pipeline_json(self) -> Dict[str, Any]:
        """Reconstruct the pipeline dict, preserving original key order.

        ``nodes`` first, then ``edges``, then any ``extra`` top-level keys -
        matching the layout produced by the pipeline.
        """
        out: Dict[str, Any] = {"nodes": self.nodes, "edges": self.edges}
        out.update(self.extra)
        return out

    def save(self, path: str | Path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_pipeline_json(), f, indent=2)

    def annotate_edges(self, scores: List[float], flags: List[str]) -> None:
        """Attach ``jepa_score`` / ``jepa_flag`` to each edge in order.

        ``scores`` / ``flags`` must align 1:1 with ``self.edges``.
        """
        if len(scores) != len(self.edges) or len(flags) != len(self.edges):
            raise ValueError("scores/flags length must match number of edges")
        for e, s, fl in zip(self.edges, scores, flags):
            e["jepa_score"] = round(float(s), 6)
            e["jepa_flag"] = fl
