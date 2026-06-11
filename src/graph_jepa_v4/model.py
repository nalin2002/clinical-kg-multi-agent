"""Graph-JEPA v4 model aliases.

The v4 experiment changes the training schedule, not the v3 graph-revision
architecture, so this module keeps the model behavior identical and gives the
checkpoint/module a v4 class name.
"""

from __future__ import annotations

from graph_jepa_v3.model import (
    EdgePlausibilityHead,
    GraphJEPAv3,
    GraphNodeEncoder,
    PatchTransformer,
    PygMessageLayer,
    TypedMessageLayer,
    _sample_revision_negatives,
    update_ema,
    vicreg_terms,
)


class GraphJEPAv4(GraphJEPAv3):
    """v3 graph-revision model trained with the v4 staged schedule."""


__all__ = [
    "EdgePlausibilityHead",
    "GraphJEPAv4",
    "GraphNodeEncoder",
    "PatchTransformer",
    "PygMessageLayer",
    "TypedMessageLayer",
    "_sample_revision_negatives",
    "update_ema",
    "vicreg_terms",
]
