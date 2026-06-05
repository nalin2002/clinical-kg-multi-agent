"""Datasets and graph sources for Graph-JEPA.

Three things live here:

* :data:`RELATION_SCHEMA` - the typed relation rules ``(src_type, relation) ->
  {allowed tgt_types}``. Used both to generate *plausible* synthetic edges and
  to sample *type-violating* negatives at train time.
* :class:`SyntheticGraphGenerator` - builds :class:`PatientGraph` objects from a
  small clinical vocabulary (torch-free; safe to run anywhere).
* :class:`MimicGraphBuilder` - documented stub with a frozen interface mapping
  MIMIC-IV tables to the shared node/edge schema. Raises a clear
  ``NotImplementedError`` until the real tables are wired in.
* :func:`to_pyg_data` / :class:`PatientGraphDataset` - convert graphs to
  PyTorch-Geometric ``Data`` (torch imported lazily).
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

from .schema import (
    EDGE_TYPE_TO_IDX,
    NODE_TYPE_TO_IDX,
    EdgeType,
    NodeType,
    PatientGraph,
)

# --------------------------------------------------------------------------- #
# Typed relation schema: which (src_type, relation) -> tgt_types are plausible.
# This encodes clinical common-sense and is the supervision signal for the
# edge-plausibility head (positives obey it; type-violating negatives break it).
# --------------------------------------------------------------------------- #
RELATION_SCHEMA: Dict[Tuple[str, str], Set[str]] = {
    (NodeType.SYMPTOM.value, EdgeType.INDICATES.value): {
        NodeType.DIAGNOSIS.value,
    },
    (NodeType.SYMPTOM.value, EdgeType.LOCATED_AT.value): {
        NodeType.LOCATION.value,
    },
    (NodeType.TREATMENT.value, EdgeType.TAKEN_FOR.value): {
        NodeType.SYMPTOM.value,
        NodeType.DIAGNOSIS.value,
    },
    (NodeType.PROCEDURE.value, EdgeType.RULES_OUT.value): {
        NodeType.DIAGNOSIS.value,
    },
    (NodeType.PROCEDURE.value, EdgeType.CONFIRMS.value): {
        NodeType.DIAGNOSIS.value,
    },
    (NodeType.MEDICAL_HISTORY.value, EdgeType.CAUSES.value): {
        NodeType.DIAGNOSIS.value,
    },
    (NodeType.DIAGNOSIS.value, EdgeType.CAUSES.value): {
        NodeType.SYMPTOM.value,
    },
    (NodeType.LAB_RESULT.value, EdgeType.INDICATES.value): {
        NodeType.DIAGNOSIS.value,
    },
    (NodeType.LAB_RESULT.value, EdgeType.CONFIRMS.value): {
        NodeType.DIAGNOSIS.value,
    },
}


def is_plausible_typed(src_type: str, relation: str, tgt_type: str) -> bool:
    """True if ``(src_type) -[relation]-> (tgt_type)`` obeys the typed schema."""
    return tgt_type in RELATION_SCHEMA.get((src_type, relation), set())


# --------------------------------------------------------------------------- #
# Small clinical vocabulary per node type (illustrative, not exhaustive).
# --------------------------------------------------------------------------- #
VOCAB: Dict[str, List[str]] = {
    NodeType.SYMPTOM.value: [
        "fever", "cough", "fatigue", "shortness of breath", "chest pain",
        "nasal congestion", "sore throat", "headache", "nausea", "dizziness",
        "loss of taste", "abdominal pain", "rash", "joint pain", "wheezing",
    ],
    NodeType.DIAGNOSIS.value: [
        "covid-19", "influenza", "common cold", "pneumonia", "asthma",
        "bronchitis", "type 2 diabetes", "hypertension", "migraine", "gastritis",
    ],
    NodeType.TREATMENT.value: [
        "acetaminophen", "ibuprofen", "insulin", "albuterol inhaler",
        "amoxicillin", "decongestant", "rest and fluids", "isolation",
        "antihistamine", "metformin",
    ],
    NodeType.PROCEDURE.value: [
        "covid swab", "chest x-ray", "blood draw", "chest auscultation",
        "spirometry", "ecg", "ct scan",
    ],
    NodeType.LOCATION.value: [
        "chest", "throat", "lungs", "head", "abdomen", "sinuses", "right arm",
    ],
    NodeType.MEDICAL_HISTORY.value: [
        "family history of diabetes", "prior pneumonia", "smoking history",
        "daycare exposure", "hospital worker", "seasonal allergies",
    ],
    NodeType.LAB_RESULT.value: [
        "elevated wbc", "positive pcr", "high glucose", "low oxygen saturation",
        "elevated crp", "abnormal chest imaging",
    ],
}


class SyntheticGraphGenerator:
    """Generates plausible patient-state :class:`PatientGraph` objects.

    Each graph centers on one or two diagnoses and wires up symptoms, treatments,
    procedures, locations, history and labs using :data:`RELATION_SCHEMA`, so the
    generated edges are clinically *plausible* by construction.
    """

    def __init__(self, seed: int = 0, min_nodes: int = 8, max_nodes: int = 24):
        self.rng = random.Random(seed)
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes

    def _sample(self, node_type: str, k: int) -> List[str]:
        pool = VOCAB[node_type]
        k = min(k, len(pool))
        return self.rng.sample(pool, k)

    def generate(self) -> PatientGraph:
        nodes: List[dict] = []
        edges: List[dict] = []
        # type -> list of node ids of that type
        by_type: Dict[str, List[str]] = {t: [] for t in NODE_TYPE_TO_IDX}
        counter = [0]

        def add_node(node_type: str, text: str) -> str:
            counter[0] += 1
            nid = f"S_{counter[0]:03d}"
            nodes.append({
                "id": nid, "text": text, "type": node_type,
                "evidence": text, "turn_id": "",
            })
            by_type[node_type].append(nid)
            return nid

        def add_edge(src: str, rel: str, tgt: str) -> None:
            edges.append({
                "source_id": src, "target_id": tgt, "type": rel,
                "evidence": "", "turn_id": "",
            })

        # Core: 1-2 diagnoses
        for dx in self._sample(NodeType.DIAGNOSIS.value, self.rng.randint(1, 2)):
            add_node(NodeType.DIAGNOSIS.value, dx)

        # Symptoms -> INDICATES -> diagnosis (and sometimes LOCATED_AT location)
        for sym in self._sample(NodeType.SYMPTOM.value, self.rng.randint(2, 5)):
            sid = add_node(NodeType.SYMPTOM.value, sym)
            dx = self.rng.choice(by_type[NodeType.DIAGNOSIS.value])
            add_edge(sid, EdgeType.INDICATES.value, dx)
            if self.rng.random() < 0.4:
                loc = add_node(NodeType.LOCATION.value,
                               self.rng.choice(VOCAB[NodeType.LOCATION.value]))
                add_edge(sid, EdgeType.LOCATED_AT.value, loc)

        # Treatments -> TAKEN_FOR -> symptom or diagnosis
        sym_or_dx = by_type[NodeType.SYMPTOM.value] + by_type[NodeType.DIAGNOSIS.value]
        for tx in self._sample(NodeType.TREATMENT.value, self.rng.randint(1, 3)):
            tid = add_node(NodeType.TREATMENT.value, tx)
            add_edge(tid, EdgeType.TAKEN_FOR.value, self.rng.choice(sym_or_dx))

        # Procedures -> RULES_OUT / CONFIRMS -> diagnosis
        for proc in self._sample(NodeType.PROCEDURE.value, self.rng.randint(0, 2)):
            pid = add_node(NodeType.PROCEDURE.value, proc)
            rel = self.rng.choice([EdgeType.RULES_OUT.value, EdgeType.CONFIRMS.value])
            add_edge(pid, rel, self.rng.choice(by_type[NodeType.DIAGNOSIS.value]))

        # Medical history -> CAUSES -> diagnosis
        for hx in self._sample(NodeType.MEDICAL_HISTORY.value, self.rng.randint(0, 2)):
            hid = add_node(NodeType.MEDICAL_HISTORY.value, hx)
            add_edge(hid, EdgeType.CAUSES.value,
                     self.rng.choice(by_type[NodeType.DIAGNOSIS.value]))

        # Lab results -> INDICATES / CONFIRMS -> diagnosis
        for lab in self._sample(NodeType.LAB_RESULT.value, self.rng.randint(0, 2)):
            lid = add_node(NodeType.LAB_RESULT.value, lab)
            rel = self.rng.choice([EdgeType.INDICATES.value, EdgeType.CONFIRMS.value])
            add_edge(lid, rel, self.rng.choice(by_type[NodeType.DIAGNOSIS.value]))

        return PatientGraph(nodes=nodes, edges=edges,
                            extra={"_method": "synthetic"})

    def generate_many(self, n: int) -> List[PatientGraph]:
        return [self.generate() for _ in range(n)]


class MimicGraphBuilder:
    """Frozen interface for building :class:`PatientGraph`s from MIMIC-IV.

    This is a documented stub: the interface is fixed now so real MIMIC-IV data
    can be plugged in later *without code changes elsewhere*. :meth:`build`
    raises :class:`NotImplementedError` until the tables are provided.

    MIMIC-IV table -> node-type mapping (frozen contract)::

        diagnoses_icd  (+ d_icd_diagnoses)   -> DIAGNOSIS
        prescriptions                         -> TREATMENT
        labevents      (+ d_labitems)        -> LAB_RESULT
        procedures_icd (+ d_icd_procedures)  -> PROCEDURE
        (clinical notes, optional)           -> SYMPTOM / MEDICAL_HISTORY

    Edges are derived per hospital admission (``hadm_id``):

        * co-occurrence within an admission, and
        * temporal ordering from ``charttime`` / ``starttime`` (history/labs ->
          diagnosis -> treatment), mapped onto the 6 relation types via
          :data:`RELATION_SCHEMA`.

    One :class:`PatientGraph` is produced per admission (``hadm_id``).
    """

    #: table file name -> destination node type
    TABLE_MAPPING: Dict[str, str] = {
        "diagnoses_icd": NodeType.DIAGNOSIS.value,
        "prescriptions": NodeType.TREATMENT.value,
        "labevents": NodeType.LAB_RESULT.value,
        "procedures_icd": NodeType.PROCEDURE.value,
    }

    def __init__(self, mimic_root: str | Path, include_notes: bool = False):
        self.mimic_root = Path(mimic_root)
        self.include_notes = include_notes

    def build(self) -> List[PatientGraph]:
        raise NotImplementedError(
            "MimicGraphBuilder is a stub. Provide MIMIC-IV tables under "
            f"'{self.mimic_root}' and implement table loading per TABLE_MAPPING "
            f"({self.TABLE_MAPPING}). The PatientGraph schema and the rest of the "
            "Graph-JEPA pipeline are unchanged; only this builder needs filling in."
        )


class AciBenchGraphBuilder:
    """Load ACI-Bench KG JSONs for Graph-JEPA training.

    This consumes already-built KG JSONs, either from the multi-agent extractor
    (for example ``outputs/aci_bench/sub_kgs``) or curated reference KGs (for
    example ``EIR_260426/eir_aci_bench/transcripts``).  It deliberately does not
    run extraction from transcripts; training starts from graph JSONs.
    """

    DEFAULT_CANDIDATES = (
        Path("outputs/aci_bench/sub_kgs"),
        Path("EIR_260426/eir_aci_bench/transcripts"),
        Path("outputs/aci_bench_smoke/sub_kgs"),
    )

    def __init__(
        self,
        kg_path: str | Path | None = None,
        *,
        limit: int | None = None,
        pattern: str = "*.json",
    ):
        self.kg_path = Path(kg_path) if kg_path else None
        self.limit = limit
        self.pattern = pattern

    def _roots(self) -> List[Path]:
        if self.kg_path is not None:
            return [self.kg_path]
        return [p for p in self.DEFAULT_CANDIDATES if p.exists()]

    def _paths(self, root: Path) -> List[Path]:
        if root.is_file():
            return [root]
        if root.is_dir():
            return sorted(root.rglob(self.pattern))
        return []

    @staticmethod
    def _load_graph(path: Path) -> PatientGraph | None:
        try:
            graph = PatientGraph.load(path)
        except (ValueError, KeyError, json.JSONDecodeError):
            return None
        if not graph.nodes:
            return None
        graph.extra.setdefault("_source_path", str(path))
        return graph

    def build(self) -> List[PatientGraph]:
        graphs: List[PatientGraph] = []
        for root in self._roots():
            for path in self._paths(root):
                graph = self._load_graph(path)
                if graph is None:
                    continue
                graphs.append(graph)
                if self.limit is not None and len(graphs) >= self.limit:
                    return graphs
        if not graphs:
            roots = self._roots()
            root_msg = ", ".join(str(p) for p in roots) if roots else "no existing default paths"
            raise ValueError(
                "No ACI-Bench KG graphs found. Pass --aci-kg-path to a KG JSON "
                "file/directory, or run ACI-Bench KG extraction first. Checked: "
                f"{root_msg}"
            )
        return graphs


# --------------------------------------------------------------------------- #
# PyG conversion (torch imported lazily so the above stays torch-free).
# --------------------------------------------------------------------------- #
def to_pyg_data(graph: PatientGraph, encoder):
    """Convert a :class:`PatientGraph` to a PyTorch-Geometric ``Data`` object.

    Node features ``x`` are encoder embeddings; ``node_type`` and ``edge_type``
    are long tensors of schema indices; ``edge_index`` is ``[2, E]``. Only edges
    with valid endpoints and known relation types are included.
    """
    import torch
    from torch_geometric.data import Data

    keys = graph.node_encoder_keys()
    x = torch.from_numpy(encoder.encode(keys)).float()
    node_type = torch.tensor(graph.node_type_indices(), dtype=torch.long)

    id_to_idx = graph.id_to_index()
    src, dst, etype = [], [], []
    for e in graph.edges:
        s, t = id_to_idx.get(e["source_id"]), id_to_idx.get(e["target_id"])
        if s is None or t is None or e["type"] not in EDGE_TYPE_TO_IDX:
            continue
        src.append(s)
        dst.append(t)
        etype.append(EDGE_TYPE_TO_IDX[e["type"]])

    if src:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_type = torch.tensor(etype, dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_type = torch.zeros((0,), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index)
    data.node_type = node_type
    data.edge_type = edge_type
    data.num_nodes = x.size(0)
    return data


class PatientGraphDataset:
    """A PyG-compatible sequence of ``Data`` objects.

    Implemented as a plain in-memory sequence (PyG ``DataLoader`` accepts any
    sequence of ``Data``), avoiding the on-disk ``InMemoryDataset`` ceremony for
    these small graphs. ``Data`` objects are built lazily on first access.
    """

    def __init__(self, graphs: Sequence[PatientGraph], encoder):
        self.graphs = list(graphs)
        self.encoder = encoder
        self._cache: List[object] = [None] * len(self.graphs)

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int):
        if self._cache[idx] is None:
            self._cache[idx] = to_pyg_data(self.graphs[idx], self.encoder)
        return self._cache[idx]
