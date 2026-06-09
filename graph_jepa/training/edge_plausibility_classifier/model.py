"""The edge-plausibility model: a light sklearn classifier (CPU, no GPU).

Wrapping sklearn keeps the DEFAULT approach trivial to train within a 5-day
deadline. The model scores a triple's plausibility in [0,1]; threshold tuning
and graph refinement live in train.py / refine_graphs.py.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class TrainedModel:
    """Bundle persisted to disk: classifier + tuned threshold + provenance."""

    clf: object
    threshold: float
    feature_dim: int
    encoder_name: str
    model_type: str

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if X.shape[0] == 0:
            return np.zeros((0,), dtype=np.float32)
        return self.clf.predict_proba(X)[:, 1].astype(np.float32)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        return path

    @staticmethod
    def load(path: str | Path) -> "TrainedModel":
        with open(path, "rb") as f:
            return pickle.load(f)


def build_classifier(model_type: str, hidden_layer_sizes, max_iter: int, seed: int):
    """Construct an untrained sklearn estimator."""
    if model_type == "logreg":
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(max_iter=max_iter, class_weight="balanced", random_state=seed)
    if model_type == "mlp":
        from sklearn.neural_network import MLPClassifier

        return MLPClassifier(
            hidden_layer_sizes=tuple(hidden_layer_sizes),
            max_iter=max_iter,
            early_stopping=True,
            random_state=seed,
        )
    raise ValueError(f"unknown model_type: {model_type!r}")


def tune_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Pick the probability threshold that maximises F1 on validation data."""
    if len(y_true) == 0:
        return 0.5
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 19):
        pred = (y_prob >= t).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t
