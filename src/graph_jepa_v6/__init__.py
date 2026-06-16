"""Graph-JEPA v6 staged training for clinical KG graph revision.

v6 keeps v5 masked pretraining and candidate-ranking fine-tuning, then adds
localized note embeddings to node features.
"""

from graph_jepa.schema import EdgeType, NodeType, PatientGraph

from .config import Config

__all__ = ["Config", "EdgeType", "GraphJEPAv6", "NodeType", "PatientGraph"]


def __getattr__(name: str):
    if name == "GraphJEPAv6":
        from .model import GraphJEPAv6

        return GraphJEPAv6
    raise AttributeError(name)
