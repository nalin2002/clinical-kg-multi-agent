"""Train the graph-level contrastive plausibility scorer (Setup B world model).

Self-supervised: trains on the EIR 13-agent TRANSCRIPT graphs (``llm_graphs``),
not silver. For each graph, several corrupted copies are produced via
:func:`graph_utils.corrupt_graph`; a margin-ranking loss pushes
``score(real) - score(corrupt) > margin``, so the model learns the corpus's
dominant clinical structure. Silver graphs are reserved for evaluation (no
leakage); a small graph-wise holdout is used only for rank-accuracy monitoring.

Run::

    python -m graph_jepa.training.graph_contrastive_scorer.train --config graph_jepa/config.yaml
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List

import numpy as np

from ...common.encoders import build_encoder
from ...common.graph_schema import Graph
from ...common.graph_utils import corrupt_graph
from ...common.io_utils import (
    Config,
    LOG,
    ensure_dir,
    load_config,
    load_training_graphs,
    setup_logging,
    write_json,
)
from .model import build_model, graph_vector, input_dim

_STRATEGIES = [
    "edge_deletion",
    "invalid_edge_addition",
    "relation_label_replacement",
    "direction_flip",
]


def _require_torch():
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "pip install torch — needed for this ablation (graph_contrastive_scorer)"
        ) from exc
    import torch

    return torch


def _encoder_dim(encoder) -> int:
    """Embedding dimensionality of the configured encoder."""
    return int(getattr(encoder, "dim"))


def _build_pairs(
    graphs: List[Graph],
    encoder,
    emb_dim: int,
    edits_per_graph: int,
    rng: random.Random,
) -> List[tuple]:
    """For each non-empty graph build ``(real_vec, corrupt_vec)`` pairs."""
    pairs: List[tuple] = []
    for g in graphs:
        if not g.edges:
            continue
        real_vec = graph_vector(encoder, g, emb_dim)
        n_corrupt = max(1, min(edits_per_graph, 8))
        for _ in range(n_corrupt):
            strat = rng.choice(_STRATEGIES)
            corrupt = corrupt_graph(g, strat, rng)
            pairs.append((real_vec, graph_vector(encoder, corrupt, emb_dim)))
    return pairs


def _evaluate(model, pairs: List[tuple]) -> float:
    """Fraction of pairs where score(real) > score(corrupt)."""
    if not pairs:
        return 0.0
    torch = _require_torch()
    model.eval()
    real = torch.from_numpy(np.vstack([p[0] for p in pairs])).float()
    corrupt = torch.from_numpy(np.vstack([p[1] for p in pairs])).float()
    with torch.no_grad():
        s_real = model(real).squeeze(-1)
        s_corrupt = model(corrupt).squeeze(-1)
        return float((s_real > s_corrupt).float().mean().item())


def train(cfg: Config) -> Path:
    """Train the scorer and write ``model.pt`` + ``config_used.json``."""
    torch = _require_torch()

    seed = int(cfg.get("training.seed", 13))
    rng = random.Random(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    tcfg = cfg.get("training.graph_contrastive_scorer", {}) or {}
    epochs = int(tcfg.get("epochs", 30))
    lr = float(tcfg.get("lr", 1e-3))
    hidden_dim = int(tcfg.get("hidden_dim", 128))
    margin = float(tcfg.get("margin", 0.3))
    edits_per_graph = int(tcfg.get("edits_per_graph", 50))

    # Setup B: self-supervised on the EIR transcript graphs (silver = eval only).
    train_graphs, val_graphs, source = load_training_graphs(cfg)
    if not train_graphs:
        raise FileNotFoundError(
            f"no training graphs under paths.{source} — run create_llm_graphs first"
        )
    LOG.info(
        "graph_contrastive_scorer: training on %d '%s' graphs (+%d monitor) — silver reserved for eval",
        len(train_graphs), source, len(val_graphs),
    )

    encoder = build_encoder(cfg)
    emb_dim = _encoder_dim(encoder)
    in_dim = input_dim(emb_dim)

    train_pairs = _build_pairs(train_graphs, encoder, emb_dim, edits_per_graph, rng)
    val_pairs = _build_pairs(val_graphs, encoder, emb_dim, edits_per_graph, rng)
    if not train_pairs:
        raise ValueError("no training pairs produced (all silver graphs empty?)")
    LOG.info("built %d train / %d val ranking pairs", len(train_pairs), len(val_pairs))

    model = build_model(in_dim, hidden_dim)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MarginRankingLoss(margin=margin)

    real_t = torch.from_numpy(np.vstack([p[0] for p in train_pairs])).float()
    corrupt_t = torch.from_numpy(np.vstack([p[1] for p in train_pairs])).float()
    target = torch.ones(real_t.shape[0])  # want score(real) > score(corrupt)

    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        s_real = model(real_t).squeeze(-1)
        s_corrupt = model(corrupt_t).squeeze(-1)
        loss = loss_fn(s_real, s_corrupt, target)
        loss.backward()
        opt.step()
        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            val_acc = _evaluate(model, val_pairs)
            LOG.info(
                "epoch %d/%d  loss=%.4f  val_rank_acc=%.3f",
                epoch + 1,
                epochs,
                float(loss.item()),
                val_acc,
            )

    out_dir = ensure_dir(cfg.path("paths.training_outputs") / "graph_contrastive_scorer")
    model_path = out_dir / "model.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "in_dim": in_dim,
            "hidden_dim": hidden_dim,
            "emb_dim": emb_dim,
            "encoder_name": getattr(encoder, "name", "unknown"),
        },
        model_path,
    )
    write_json(
        out_dir / "config_used.json",
        {
            "epochs": epochs,
            "lr": lr,
            "hidden_dim": hidden_dim,
            "margin": margin,
            "edits_per_graph": edits_per_graph,
            "seed": seed,
            "in_dim": in_dim,
            "emb_dim": emb_dim,
            "encoder_name": getattr(encoder, "name", "unknown"),
            "n_train_graphs": len(train_graphs),
            "n_val_graphs": len(val_graphs),
            "n_train_pairs": len(train_pairs),
            "final_val_rank_acc": _evaluate(model, val_pairs),
        },
    )
    LOG.info("saved scorer checkpoint -> %s", model_path)
    return model_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="graph_jepa/config.yaml")
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
