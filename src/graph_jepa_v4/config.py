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
        "ADMINISTERED_DURING": 0.85,
        "DIAGNOSED_BY": 0.88,
        "HAD_LAB_TEST": 0.90,
        "HAD_MICROBIOLOGY": 0.90,
        "HAS_DIAGNOSIS": 0.90,
        "INVESTIGATED_BY": 0.88,
        "MANAGED_BY_SERVICE": 0.90,
        "MANAGED_FOR": 0.85,
        "MONITORED_BY": 0.88,
        "PERFORMED_FOR": 0.88,
        "TAKES_MEDICATION": 0.90,
        "TARGETS_ORGANISM": 0.90,
        "TREATED_BY": 0.88,
        "UNDERWENT_PROCEDURE": 0.90,
    })
    inconsistent_threshold_by_relation: Dict[str, float] = field(default_factory=lambda: {
        "CONFIRMS": 0.15,
        "LOCATED_AT": 0.08,
        "ADMINISTERED_DURING": 0.65,
        "DIAGNOSED_BY": 0.65,
        "HAD_LAB_TEST": 0.75,
        "HAD_MICROBIOLOGY": 0.75,
        "HAS_DIAGNOSIS": 0.75,
        "INVESTIGATED_BY": 0.65,
        "MANAGED_BY_SERVICE": 0.75,
        "MANAGED_FOR": 0.65,
        "MONITORED_BY": 0.65,
        "PERFORMED_FOR": 0.65,
        "TAKES_MEDICATION": 0.75,
        "TARGETS_ORGANISM": 0.70,
        "TREATED_BY": 0.65,
        "UNDERWENT_PROCEDURE": 0.75,
    })
    energy_temperature: float = 1.0
    prune_threshold: float | None = None
    schema_invalid_score: float = 0.0
    schema_unconstrained_relations: list[str] = field(default_factory=lambda: [
        "ASSOCIATED_WITH",
        "CO_OCCURS_WITH",
    ])
    candidate_threshold: float = 0.85
    candidate_threshold_by_relation: Dict[str, float] = field(default_factory=lambda: {
        "INDICATES": 0.95,
        "CONFIRMS": 0.98,
        "TAKEN_FOR": 0.95,
        "CAUSES": 0.98,
        "LOCATED_AT": 0.95,
        "RULES_OUT": 0.99,
        "ADMINISTERED_DURING": 0.98,
        "COMPLICATED_BY": 0.995,
        "DIAGNOSED_BY": 0.985,
        "HAD_LAB_TEST": 0.97,
        "HAD_MICROBIOLOGY": 0.97,
        "HAS_DIAGNOSIS": 0.96,
        "INVESTIGATED_BY": 0.99,
        "MANAGED_BY_SERVICE": 0.97,
        "MANAGED_FOR": 0.99,
        "MONITORED_BY": 0.98,
        "PART_OF_REGIMEN": 0.995,
        "PERFORMED_FOR": 0.995,
        "TAKES_MEDICATION": 0.96,
        "TARGETS_ORGANISM": 0.985,
        "TREATED_BY": 0.975,
        "UNDERWENT_PROCEDURE": 0.96,
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

        score_dict = dict(d.get("score", {}))
        default_score = ScoreConfig()
        for key in (
            "weak_threshold_by_relation",
            "inconsistent_threshold_by_relation",
        ):
            if key in score_dict:
                merged = dict(getattr(default_score, key))
                merged.update(score_dict[key])
                score_dict[key] = merged
        if "candidate_threshold_by_relation" in score_dict:
            merged = dict(default_score.candidate_threshold_by_relation)
            for relation, threshold in score_dict["candidate_threshold_by_relation"].items():
                merged[relation] = max(
                    float(threshold),
                    merged.get(relation, default_score.candidate_threshold),
                )
            score_dict["candidate_threshold_by_relation"] = merged

        return cls(
            model=ModelConfig(**model_dict),
            train=TrainConfig(**train_dict),
            score=ScoreConfig(**score_dict),
            encoder=d.get("encoder", "mock"),
        )
