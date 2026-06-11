"""Configuration dataclasses for Graph-JEPA v4.

v4 runs experiment C: first train the masked/JEPA objective by itself, then
fine-tune jointly with graph-revision edge plausibility.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict

from graph_jepa.schema import NUM_EDGE_TYPES


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
    num_relations: int = NUM_EDGE_TYPES
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

    # Total epochs is derived as pretrain_epochs + finetune_epochs when training.
    epochs: int = 80
    pretrain_epochs: int = 40
    finetune_epochs: int = 40
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
    revision_weight: float = 1.0
    revision_mask_ratio: float = 0.25
    revision_neg_per_pos: int = 3
    vicreg_var_weight: float = 0.5
    vicreg_cov_weight: float = 0.04

    # Synthetic-data generation (only used with --data synthetic).
    synthetic_graphs: int = 256
    synthetic_min_nodes: int = 8
    synthetic_max_nodes: int = 28


@dataclass
class ScoreConfig:
    """Inference hyperparameters."""

    alpha: float = 0.9
    weak_threshold: float = 0.5
    inconsistent_threshold: float = 0.25
    weak_threshold_by_relation: Dict[str, float] = field(default_factory=lambda: {
        "CONFIRMS": 0.30,
        "LOCATED_AT": 0.15,
    })
    inconsistent_threshold_by_relation: Dict[str, float] = field(default_factory=lambda: {
        "CONFIRMS": 0.15,
        "LOCATED_AT": 0.08,
    })
    energy_temperature: float = 1.0
    prune_threshold: float | None = None
    candidate_threshold: float = 0.85
    candidate_threshold_by_relation: Dict[str, float] = field(default_factory=lambda: {
        "INDICATES": 0.95,
        "CONFIRMS": 0.96,
        "TAKEN_FOR": 0.95,
        "CAUSES": 0.98,
        "LOCATED_AT": 0.95,
        "RULES_OUT": 0.99,
    })
    disabled_candidate_relations: list[str] = field(default_factory=lambda: [
    ])
    max_candidate_edges: int = 50
    require_shared_res_id: bool = True
    skip_negated_candidates: bool = True


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

        train_dict = dict(d.get("train", {}))
        train_dict.pop("edge_head_weight", None)
        if "pretrain_epochs" not in train_dict:
            train_dict["pretrain_epochs"] = 0
        if "finetune_epochs" not in train_dict:
            train_dict["finetune_epochs"] = train_dict.get("epochs", 40)
        train_dict["epochs"] = (
            int(train_dict["pretrain_epochs"]) + int(train_dict["finetune_epochs"])
        )

        return cls(
            model=ModelConfig(**model_dict),
            train=TrainConfig(**train_dict),
            score=ScoreConfig(**d.get("score", {})),
            encoder=d.get("encoder", "mock"),
        )
