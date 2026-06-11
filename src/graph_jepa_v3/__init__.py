"""Graph-JEPA v3 for clinical KG graph revision.

This package is intentionally parallel to :mod:`graph_jepa_v2`. It keeps the
patch/subgraph JEPA backbone, then trains the edge head to revise noisy KGs by
recovering hidden true edges and rejecting injected false edges.
"""

from graph_jepa.schema import EdgeType, NodeType, PatientGraph

from .config import Config

__all__ = ["Config", "EdgeType", "GraphJEPAv3", "NodeType", "PatientGraph"]


def __getattr__(name: str):
    if name == "GraphJEPAv3":
        from .model import GraphJEPAv3

        return GraphJEPAv3
    raise AttributeError(name)
