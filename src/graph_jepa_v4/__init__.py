"""Graph-JEPA v4 staged training for clinical KG graph revision.

v4 keeps the v3 graph-revision model, but trains it as experiment C:
masked/JEPA pretraining first, then joint JEPA + edge-revision fine-tuning.
"""

from graph_jepa.schema import EdgeType, NodeType, PatientGraph

from .config import Config

__all__ = ["Config", "EdgeType", "GraphJEPAv4", "NodeType", "PatientGraph"]


def __getattr__(name: str):
    if name == "GraphJEPAv4":
        from .model import GraphJEPAv4

        return GraphJEPAv4
    raise AttributeError(name)
