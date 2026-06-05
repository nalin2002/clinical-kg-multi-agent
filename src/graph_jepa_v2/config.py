"""Configuration dataclasses for Graph-JEPA v2.

The v2 package keeps the same clinical KG data contract as :mod:`graph_jepa`,
but changes the self-supervised objective from masked node prediction to
subgraph-patch prediction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict


@dataclass
class ModelConfig:
    """Architecture hyperparameters for the patch-based model."""

    in_dim: int = 256
    hidden_dim: int = 160
    latent_dim: int = 160
    num_gnn_layers: int = 2
    conv: str = "gine"  # "gine" | "gat"
    gnn_backend: str = "pyg"  # "pyg" | "torch"
    dropout: float = 0.1
    num_relations: int = 6
    ema_decay: float = 0.996

    # Patch/subgraph representation.
    num_patches: int = 8
    patch_pe_dim: int = 8
    patch_layers: int = 2
    patch_heads: int = 4
    patch_mlp_ratio: float = 2.0

    # Prediction heads.
    predictor_hidden: int = 256


@dataclass
class TrainConfig:
    """Training-loop hyperparameters."""

    epochs: int = 40
    lr: float = 8e-4
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    seed: int = 0
    batch_size: int = 16
    num_workers: int = 0

    # JEPA patch task.
    context_patches: int = 1
    target_patches: int = 4
    ema_start: float = 0.996
    ema_end: float = 1.0
    jepa_weight: float = 1.0
    edge_head_weight: float = 0.7
    vicreg_var_weight: float = 0.5
    vicreg_cov_weight: float = 0.04

    # Synthetic-data generation (only used with --data synthetic).
    synthetic_graphs: int = 256
    synthetic_min_nodes: int = 8
    synthetic_max_nodes: int = 28


@dataclass
class ScoreConfig:
    """Inference hyperparameters."""

    alpha: float = 0.5
    weak_threshold: float = 0.5
    inconsistent_threshold: float = 0.25
    energy_temperature: float = 1.0
    prune_threshold: float | None = None


@dataclass
class Config:
    """Top-level checkpoint configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    score: ScoreConfig = field(default_factory=ScoreConfig)
    encoder: str = "mock"  # "mock" | "bge" | "sapbert"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        model_dict = dict(d.get("model", {}))
        if model_dict and "gnn_backend" not in model_dict:
            model_dict["gnn_backend"] = "torch"

        return cls(
            model=ModelConfig(**model_dict),
            train=TrainConfig(**d.get("train", {})),
            score=ScoreConfig(**d.get("score", {})),
            encoder=d.get("encoder", "mock"),
        )
