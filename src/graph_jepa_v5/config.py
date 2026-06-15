"""Configuration dataclasses for Graph-JEPA v5."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict

from graph_jepa_v4.config import ModelConfig, ScoreConfig, TrainConfig as V4TrainConfig


@dataclass
class TrainConfig(V4TrainConfig):
    """Training-loop hyperparameters.

    v5 adds a candidate-ranking objective for joint fine-tuning.  A subset of
    real schema-valid edges is hidden from message passing, then the edge head
    must rank each hidden edge above schema-valid distractors from the same
    patient graph.
    """

    ranking_weight: float = 1.0
    ranking_mask_ratio: float = 0.25
    ranking_neg_per_pos: int = 8
    ranking_max_pos: int = 256
    ranking_temperature: float = 0.2
    llm_confidence_negatives: bool = False
    llm_negative_threshold: float = 0.3
    llm_positive_threshold: float = 0.8
    llm_negative_threshold_by_relation: Dict[str, float] = field(
        default_factory=lambda: {
            "ASSOCIATED_WITH": 0.5,
            "CAUSES": 0.5,
            "CO_OCCURS_WITH": 0.5,
            "COMPLICATED_BY": 0.65,
            "CONFIRMS": 0.65,
            "INDICATES": 0.60,
            "MANAGED_FOR": 0.70,
            "PERFORMED_FOR": 0.5,
            "TARGETS_ORGANISM": 0.5,
            "TREATED_BY": 0.5,
            "USED_DURING": 0.5,
        }
    )
    llm_positive_threshold_by_relation: Dict[str, float] = field(
        default_factory=lambda: {
            "ASSOCIATED_WITH": 0.85,
            "CAUSES": 0.9,
            "CO_OCCURS_WITH": 0.85,
            "COMPLICATED_BY": 0.9,
            "CONFIRMS": 0.95,
            "INDICATES": 0.85,
            "MANAGED_FOR": 0.8,
            "PERFORMED_FOR": 0.9,
            "TARGETS_ORGANISM": 0.9,
            "TREATED_BY": 0.85,
            "USED_DURING": 0.9,
        }
    )
    llm_negative_weight: float = 0.2
    clinical_artifact_filters: bool = False


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
        defaults = TrainConfig()
        for key in (
            "ranking_weight",
            "ranking_mask_ratio",
            "ranking_neg_per_pos",
            "ranking_max_pos",
            "ranking_temperature",
            "llm_confidence_negatives",
            "llm_negative_threshold",
            "llm_positive_threshold",
            "clinical_artifact_filters",
        ):
            train_dict.setdefault(key, getattr(defaults, key))
        for key in (
            "llm_negative_threshold_by_relation",
            "llm_positive_threshold_by_relation",
        ):
            merged = dict(getattr(defaults, key))
            merged.update(train_dict.get(key, {}))
            train_dict[key] = merged
        train_dict.setdefault(
            "llm_negative_weight",
            defaults.llm_negative_weight,
        )
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
