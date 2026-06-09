"""Train the edge-plausibility classifier (DEFAULT world-model refiner).

Pipeline:
1. Load silver reference graphs (positives).
2. Patient-level train/val/test split (no leakage).
3. For each training graph, sample corruption-based negatives
   (relation replacement, source-target swap, random target, invalid clinical
   relation).
4. Featurize positives + negatives, fit an sklearn classifier.
5. Tune the decision threshold for F1 on the validation split.
6. Save the model + threshold + metrics + the split.

Run::

    python -m graph_jepa.training.edge_plausibility_classifier.train \
        --config graph_jepa/config.yaml
"""

from __future__ import annotations

import argparse
import random

import numpy as np

from graph_jepa.common.encoders import build_encoder
from graph_jepa.common.graph_utils import sample_negatives
from graph_jepa.common.io_utils import (
    LOG,
    ensure_dir,
    load_config,
    load_graphs,
    setup_logging,
    split_patients,
    write_json,
)
from graph_jepa.training.edge_plausibility_classifier.featurize import (
    EdgeFeaturizer,
    build_dataset,
)
from graph_jepa.training.edge_plausibility_classifier.model import (
    TrainedModel,
    build_classifier,
    tune_threshold,
)

METHOD = "edge_plausibility_classifier"


def _prf(y_true: np.ndarray, y_prob: np.ndarray, thr: float) -> dict:
    if len(y_true) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "n": 0}
    pred = (y_prob >= thr).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4), "n": int(len(y_true))}


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the edge-plausibility classifier")
    ap.add_argument("--config", default="graph_jepa/config.yaml")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    setup_logging(args.log_level)
    cfg = load_config(args.config)
    seed = int(cfg.get("training.seed", 13))
    rng = random.Random(seed)
    np.random.seed(seed)

    mcfg = cfg.get(f"training.{METHOD}", {})
    n_neg = int(cfg.get("training.negatives_per_positive", 4))
    corruption_mix = cfg.get("training.corruption_mix", {})

    silver = load_graphs(cfg.path("paths.silver_graphs"))
    if not silver:
        LOG.error("No silver graphs found in %s. Run create_silver_graphs first.",
                  cfg.path("paths.silver_graphs"))
        return
    split = split_patients(cfg, list(silver.keys()))
    LOG.info("silver graphs: %d | split train/val/test = %d/%d/%d",
             len(silver), len(split["train"]), len(split["val"]), len(split["test"]))

    encoder = build_encoder(cfg)
    featurizer = EdgeFeaturizer(encoder)

    def subset(ids):
        return {pid: silver[pid] for pid in ids if pid in silver}

    def negs(graphs):
        return {pid: sample_negatives(g, n_neg, corruption_mix, rng) for pid, g in graphs.items()}

    train_g, val_g = subset(split["train"]), subset(split["val"])
    Xtr, ytr, _ = build_dataset(featurizer, train_g, negs(train_g))
    Xva, yva, _ = build_dataset(featurizer, val_g, negs(val_g))
    LOG.info("train examples: %d (%d pos), val examples: %d", len(ytr), int(ytr.sum()), len(yva))
    if len(ytr) == 0:
        LOG.error("No training examples produced (empty silver graphs?). Aborting.")
        return

    clf = build_classifier(
        mcfg.get("model_type", "mlp"),
        mcfg.get("hidden_layer_sizes", [256, 64]),
        int(mcfg.get("max_iter", 400)),
        seed,
    )
    LOG.info("fitting %s ...", mcfg.get("model_type", "mlp"))
    clf.fit(Xtr, ytr)

    # Threshold: tuned on val unless a fixed value is configured.
    val_prob = clf.predict_proba(Xva)[:, 1] if len(yva) else np.zeros(0)
    fixed = mcfg.get("threshold")
    threshold = float(fixed) if fixed is not None else tune_threshold(yva, val_prob)

    model = TrainedModel(
        clf=clf, threshold=threshold, feature_dim=Xtr.shape[1],
        encoder_name=getattr(encoder, "name", "unknown"),
        model_type=mcfg.get("model_type", "mlp"),
    )
    out_dir = ensure_dir(cfg.path("paths.training_outputs") / METHOD)
    model.save(out_dir / "model.pkl")

    metrics = {
        "threshold": round(threshold, 4),
        "encoder": model.encoder_name,
        "train": _prf(ytr, clf.predict_proba(Xtr)[:, 1], threshold),
        "val": _prf(yva, val_prob, threshold) if len(yva) else {},
        "split": split,
        "feature_dim": int(Xtr.shape[1]),
        "negatives_per_positive": n_neg,
    }
    write_json(out_dir / "metrics.json", metrics)
    LOG.info("DONE. threshold=%.3f  val=%s  -> %s", threshold, metrics.get("val"), out_dir)


if __name__ == "__main__":
    main()
