"""PyG graph tensor conversion for Graph-JEPA v3.

Training uses ``torch_geometric.loader.DataLoader`` so multiple patient graphs
can be collated into one mini-batch while PyG handles ``edge_index`` offsets.
"""

from __future__ import annotations

from typing import Sequence

import torch

from graph_jepa.schema import EDGE_TYPE_TO_IDX, PatientGraph
from torch_geometric.data import Data



def to_graph_data(graph: PatientGraph, encoder):
    """Convert a :class:`PatientGraph` to a PyG ``Data`` object."""

    keys = graph.node_encoder_keys()
    x = torch.from_numpy(encoder.encode(keys)).float()
    node_type = torch.tensor(graph.node_type_indices(), dtype=torch.long)
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

    data = Data(x=x, edge_index=edge_index)
    data.node_type = node_type
    data.edge_type = edge_type
    data.num_nodes = x.size(0)
    return data


class PatientGraphDataset:
    """Lazy in-memory PyG dataset for v3."""

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
