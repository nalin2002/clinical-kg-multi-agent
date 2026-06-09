"""Canonical clinical knowledge-graph schema for graph_jepa.

A single in-memory representation (:class:`Graph`) is used everywhere — silver
reference graphs, LLM-extracted graphs, and refined graphs all round-trip
through the same shape so every downstream tool reads one format.

On-disk schema (the format written/read by :mod:`io_utils`)::

    {
      "patient_id": "RES_D2N001",
      "source": "clinician_note" | "llm_transcript" | "refined:<method>",
      "nodes": [
        {"id": "...", "name": "...", "type": "...", "normalized_name": "...",
         "evidence": "...", "turn_id": "..."}
      ],
      "edges": [
        {"source": "<node id>", "target": "<node id>", "relation": "...",
         "evidence": "...", "confidence": 0.0-1.0}
      ]
    }

The EIR 13-agent extractor emits a sibling schema
(``nodes[].text`` / ``edges[].source_id`` / ``edges[].type``); converters here
translate it into the canonical form so the rest of the pipeline is unaware of
EIR's field names.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

# --------------------------------------------------------------------------- #
# Controlled vocabulary (mirrors the 7 node types / 6 relations the EIR
# extractor and the human-curated ACI-Bench KGs use).
# --------------------------------------------------------------------------- #
NODE_TYPES: List[str] = [
    "SYMPTOM",
    "DIAGNOSIS",
    "TREATMENT",
    "PROCEDURE",
    "LOCATION",
    "MEDICAL_HISTORY",
    "LAB_RESULT",
]

RELATION_TYPES: List[str] = [
    "INDICATES",
    "CAUSES",
    "LOCATED_AT",
    "RULES_OUT",
    "TAKEN_FOR",
    "CONFIRMS",
]

# Typed clinical common-sense: which (source_type, relation) -> {target_types}
# are clinically plausible. Used to (a) sample "invalid clinical relation"
# negatives at training time and (b) flag type-violating edges at refine time.
RELATION_SCHEMA: Dict[str, Dict[str, List[str]]] = {
    "SYMPTOM": {"INDICATES": ["DIAGNOSIS"], "LOCATED_AT": ["LOCATION"]},
    "DIAGNOSIS": {"CAUSES": ["SYMPTOM"]},
    "TREATMENT": {"TAKEN_FOR": ["SYMPTOM", "DIAGNOSIS"]},
    "PROCEDURE": {"RULES_OUT": ["DIAGNOSIS"], "CONFIRMS": ["DIAGNOSIS"]},
    "MEDICAL_HISTORY": {"CAUSES": ["DIAGNOSIS"]},
    "LAB_RESULT": {"INDICATES": ["DIAGNOSIS"], "CONFIRMS": ["DIAGNOSIS"]},
}


def is_clinically_plausible(src_type: str, relation: str, tgt_type: str) -> bool:
    """True if ``(src_type) -[relation]-> (tgt_type)`` obeys RELATION_SCHEMA."""
    return tgt_type in RELATION_SCHEMA.get(src_type, {}).get(relation, [])


def normalize_text(text: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation edges — for matching."""
    text = (text or "").lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .,:;-")


# --------------------------------------------------------------------------- #
# Dataclasses
# --------------------------------------------------------------------------- #
@dataclass
class Node:
    id: str
    name: str
    type: str
    normalized_name: str = ""
    evidence: str = ""
    turn_id: str = ""

    def __post_init__(self) -> None:
        if not self.normalized_name:
            self.normalized_name = normalize_text(self.name)
        self.type = (self.type or "").upper()

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "normalized_name": self.normalized_name,
            "evidence": self.evidence,
            "turn_id": self.turn_id,
        }


@dataclass
class Edge:
    source: str
    target: str
    relation: str
    evidence: str = ""
    confidence: float = 1.0

    def __post_init__(self) -> None:
        self.relation = (self.relation or "").upper()

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "target": self.target,
            "relation": self.relation,
            "evidence": self.evidence,
            "confidence": self.confidence,
        }


@dataclass
class Graph:
    patient_id: str
    source: str
    nodes: List[Node] = field(default_factory=list)
    edges: List[Edge] = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    # ----- indexing helpers ------------------------------------------------ #
    def node_index(self) -> Dict[str, Node]:
        return {n.id: n for n in self.nodes}

    def get_node(self, node_id: str) -> Optional[Node]:
        return self.node_index().get(node_id)

    def has_valid_endpoints(self, edge: Edge) -> bool:
        idx = self.node_index()
        return edge.source in idx and edge.target in idx

    def triples(self) -> List[tuple]:
        """``(normalized_src_name, relation, normalized_tgt_name)`` per edge."""
        idx = self.node_index()
        out = []
        for e in self.edges:
            s, t = idx.get(e.source), idx.get(e.target)
            if s and t:
                out.append((s.normalized_name, e.relation, t.normalized_name))
        return out

    # ----- serialization --------------------------------------------------- #
    def to_dict(self) -> dict:
        d = {
            "patient_id": self.patient_id,
            "source": self.source,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
        }
        if self.extra:
            d["extra"] = self.extra
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Graph":
        """Load the canonical schema. Tolerates the EIR schema transparently."""
        if _looks_like_eir(d):
            return cls.from_eir_dict(d)
        nodes = [
            Node(
                id=str(n["id"]),
                name=n.get("name") or n.get("text") or "",
                type=n.get("type", ""),
                normalized_name=n.get("normalized_name", ""),
                evidence=n.get("evidence", ""),
                turn_id=str(n.get("turn_id", "")),
            )
            for n in d.get("nodes", [])
        ]
        edges = [
            Edge(
                source=str(e.get("source") or e.get("source_id")),
                target=str(e.get("target") or e.get("target_id")),
                relation=e.get("relation") or e.get("type") or "",
                evidence=e.get("evidence", ""),
                confidence=float(e.get("confidence", 1.0)),
            )
            for e in d.get("edges", [])
        ]
        return cls(
            patient_id=str(d.get("patient_id") or d.get("res_id") or "unknown"),
            source=d.get("source", "unknown"),
            nodes=nodes,
            edges=edges,
            extra=d.get("extra", {}),
        )

    # ----- EIR interop ----------------------------------------------------- #
    @classmethod
    def from_eir_dict(cls, d: dict, patient_id: str = "", source: str = "llm_transcript") -> "Graph":
        """Convert an EIR per-patient KG ({nodes:[{id,text,type}], edges:[{source_id,target_id,type}]})."""
        nodes = [
            Node(
                id=str(n["id"]),
                name=n.get("text") or n.get("name") or "",
                type=n.get("type", ""),
                evidence=n.get("evidence", ""),
                turn_id=str(n.get("turn_id", "")),
            )
            for n in d.get("nodes", [])
        ]
        edges = [
            Edge(
                source=str(e.get("source_id") or e.get("source")),
                target=str(e.get("target_id") or e.get("target")),
                relation=e.get("type") or e.get("relation") or "",
                evidence=e.get("evidence", ""),
                confidence=float(e.get("confidence", 1.0)),
            )
            for e in d.get("edges", [])
        ]
        return cls(
            patient_id=str(patient_id or d.get("patient_id") or d.get("res_id") or "unknown"),
            source=source,
            nodes=nodes,
            edges=edges,
        )


def _looks_like_eir(d: dict) -> bool:
    """Heuristic: EIR nodes use 'text' and edges use 'source_id'."""
    nodes = d.get("nodes") or []
    edges = d.get("edges") or []
    node_eir = bool(nodes) and "text" in nodes[0] and "name" not in nodes[0]
    edge_eir = bool(edges) and "source_id" in edges[0] and "source" not in edges[0]
    return node_eir or edge_eir


def new_node_ids(prefix: str, count: int, start: int = 1) -> List[str]:
    """Deterministic node-id generator, e.g. ``N_001`` ... ``N_NNN``."""
    return [f"{prefix}_{i:03d}" for i in range(start, start + count)]


def validate_graph(graph: Graph) -> List[str]:
    """Return a list of human-readable schema warnings (empty == clean)."""
    warnings: List[str] = []
    idx = graph.node_index()
    if len(idx) != len(graph.nodes):
        warnings.append("duplicate node ids")
    for n in graph.nodes:
        if n.type not in NODE_TYPES:
            warnings.append(f"node {n.id}: unknown type {n.type!r}")
    for e in graph.edges:
        if e.relation not in RELATION_TYPES:
            warnings.append(f"edge {e.source}->{e.target}: unknown relation {e.relation!r}")
        if e.source not in idx or e.target not in idx:
            warnings.append(f"edge {e.source}->{e.target}: dangling endpoint")
    return warnings
