"""Hyperparameter dataclasses for the Graph-JEPA refinement layer.

Kept deliberately small. These are plain dataclasses (no torch import) so they
can be constructed/serialised anywhere, including the torch-free code paths.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict


@dataclass
class ModelConfig:
    """Architecture hyperparameters.

    ``in_dim`` is the node-encoder output dimension. It must match whatever
    encoder is used at train *and* score time (BGE-M3 is 1024; MockEncoder is
    configurable and defaults to 256). It is recorded in the checkpoint so the
    scorer can validate compatibility.
    """

    in_dim: int = 256
    hidden_dim: int = 128
    latent_dim: int = 128
    num_layers: int = 2
    conv: str = "gine"  # "gine" | "gat"
    dropout: float = 0.1
    num_node_types: int = 7
    num_relations: int = 6
    ema_decay: float = 0.99
    predictor_hidden: int = 128


@dataclass
class TrainConfig:
    """Training-loop hyperparameters."""

    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-5
    mask_ratio: float = 0.25  # fraction of nodes in the masked target region
    vicreg_weight: float = 1.0
    edge_head_weight: float = 1.0
    jepa_weight: float = 1.0
    neg_per_pos: int = 1  # negative edges sampled per positive edge
    seed: int = 0
    # Synthetic-data generation (only used with --data synthetic)
    synthetic_graphs: int = 256
    synthetic_min_nodes: int = 8
    synthetic_max_nodes: int = 24


@dataclass
class ScoreConfig:
    """Inference / scoring hyperparameters.

    ``jepa_score`` is a convex combination of the edge-head probability and a
    structural consistency term derived from the JEPA energy::

        jepa_score = alpha * p_head + (1 - alpha) * structural_score

    Edges are flagged ``inconsistent`` below ``inconsistent_threshold`` and
    ``weak`` below ``weak_threshold`` (otherwise ``ok``).
    """

    alpha: float = 0.5
    weak_threshold: float = 0.5
    inconsistent_threshold: float = 0.25
    energy_temperature: float = 1.0  # structural_score = exp(-energy / T)
    prune_threshold: float | None = None  # None => never prune (annotate only)


@dataclass
class Config:
    """Top-level bundle persisted alongside checkpoints."""

    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    score: ScoreConfig = field(default_factory=ScoreConfig)
    encoder: str = "mock"  # "mock" | "bge" | "sapbert"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        return cls(
            model=ModelConfig(**d.get("model", {})),
            train=TrainConfig(**d.get("train", {})),
            score=ScoreConfig(**d.get("score", {})),
            encoder=d.get("encoder", "mock"),
        )
