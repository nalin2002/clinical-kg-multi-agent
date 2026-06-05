"""Torch-only graph conversion for Graph-JEPA v2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from graph_jepa.schema import EDGE_TYPE_TO_IDX, PatientGraph


@dataclass(frozen=True)
class GraphData:
    """Minimal graph tensor container used by v2.

    It intentionally mirrors the tiny subset of PyG ``Data`` used by this
    package, without requiring torch-geometric.
    """

    x: torch.Tensor
    edge_index: torch.Tensor
    edge_type: torch.Tensor
    num_nodes: int

    def to(self, device: torch.device | str) -> "GraphData":
        return GraphData(
            x=self.x.to(device),
            edge_index=self.edge_index.to(device),
            edge_type=self.edge_type.to(device),
            num_nodes=self.num_nodes,
        )


def to_graph_data(graph: PatientGraph, encoder) -> GraphData:
    keys = graph.node_encoder_keys()
    x = torch.from_numpy(encoder.encode(keys)).float()
    id_to_idx = graph.id_to_index()
    src, dst, etype = [], [], []
    for e in graph.edges:
        s = id_to_idx.get(e["source_id"])
        t = id_to_idx.get(e["target_id"])
        r = EDGE_TYPE_TO_IDX.get(e["type"])
        if s is None or t is None or r is None:
            continue
        src.append(s)
        dst.append(t)
        etype.append(r)

    if src:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_type = torch.tensor(etype, dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_type = torch.zeros((0,), dtype=torch.long)
    return GraphData(x=x, edge_index=edge_index, edge_type=edge_type, num_nodes=x.size(0))


class PatientGraphDataset:
    """Lazy in-memory tensor dataset for v2."""

    def __init__(self, graphs: Sequence[PatientGraph], encoder):
        self.graphs = list(graphs)
        self.encoder = encoder
        self._cache: list[GraphData | None] = [None] * len(self.graphs)

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> GraphData:
        if self._cache[idx] is None:
            self._cache[idx] = to_graph_data(self.graphs[idx], self.encoder)
        return self._cache[idx]
