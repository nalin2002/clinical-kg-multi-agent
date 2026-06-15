"""Graph-JEPA v5 staged training for clinical KG graph revision.

v5 keeps v4 masked pretraining and schema-aware revision, then adds a
candidate-ranking fine-tuning objective: hidden true edges must score above
hard, schema-valid distractor candidates from the same patient graph.
"""

from graph_jepa.schema import EdgeType, NodeType, PatientGraph

from .config import Config

__all__ = ["Config", "EdgeType", "GraphJEPAv5", "NodeType", "PatientGraph"]


def __getattr__(name: str):
    if name == "GraphJEPAv5":
        from .model import GraphJEPAv5

        return GraphJEPAv5
    raise AttributeError(name)
