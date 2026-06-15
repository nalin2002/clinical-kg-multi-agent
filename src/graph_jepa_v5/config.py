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
        ):
            train_dict.setdefault(key, getattr(defaults, key))
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
