"""Graph-JEPA refinement layer for the clinical multi-agent KG pipeline.

A consistency-scoring layer that runs *after* the multi-agent pipeline. It reads
the existing KG JSON, embeds nodes in a shared BGE-M3 space, runs a hybrid
Graph-JEPA to produce context-aware node latents, and annotates each candidate
edge with a plausibility score (``jepa_score``) and flag (``jepa_flag``).

By default it never adds or removes facts; it only adds fields to edges.

The light-weight, torch-free pieces (``schema``, ``encoders.MockEncoder``,
``data.SyntheticGraphGenerator``) import without ``torch`` / ``torch_geometric``
so the data round-trip can be exercised in environments without the heavy deps.
The model / train / score modules require ``torch`` + ``torch_geometric``.
"""

from .schema import EdgeType, NodeType, PatientGraph

__all__ = ["EdgeType", "NodeType", "PatientGraph"]
