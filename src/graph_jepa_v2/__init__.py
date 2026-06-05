"""Subgraph-patch Graph-JEPA v2 for clinical KG refinement.

This package is intentionally parallel to :mod:`graph_jepa`.  It reuses the
clinical KG schema, encoders, and data adapters from v1, but replaces the
node-mask objective with a patch/subgraph JEPA objective inspired by the
Graph-JEPA research line.
"""

from graph_jepa.schema import EdgeType, NodeType, PatientGraph

from .config import Config

__all__ = ["Config", "EdgeType", "GraphJEPAv2", "NodeType", "PatientGraph"]


def __getattr__(name: str):
    if name == "GraphJEPAv2":
        from .model import GraphJEPAv2

        return GraphJEPAv2
    raise AttributeError(name)
