"""Train the JEPA-style masked-edge relation predictor (Setup B world model).

Self-supervised: trains on the EIR 13-agent TRANSCRIPT graphs (``llm_graphs``),
not silver. For each graph a fraction (``mask_ratio``) of edges is masked; the
model encodes the *visible* graph into a context vector and learns to recover
each masked edge's RELATION label from ``[emb(src), emb(tgt), context]`` via
cross-entropy. It thus learns the dominant clinical structure of the corpus and
can later correct minority inconsistencies in a graph. Silver graphs are NOT
used in training — they are reserved for evaluation (no leakage). A small
graph-wise holdout is used only for accuracy monitoring.

Run::

    python -m graph_jepa.training.masked_edge_prediction.train --config graph_jepa/config.yaml
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np

from ...common.encoders import build_encoder
from ...common.graph_schema import RELATION_TYPES, Graph
from ...common.io_utils import (
    Config,
    LOG,
    ensure_dir,
    load_config,
    load_training_graphs,
    setup_logging,
    write_json,
)
from .model import (
    build_model,
    context_vector,
    edge_input,
    input_dim,
    relation_to_index,
    visible_node_embeddings,
)


def _require_torch():
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "pip install torch — needed for this ablation (masked_edge_prediction)"
        ) from exc
    import torch

    return torch


def _encoder_dim(encoder) -> int:
    return int(getattr(encoder, "dim"))


def _build_examples(
    graphs: List[Graph],
    encoder,
    emb_dim: int,
    mask_ratio: float,
    rng: random.Random,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build ``(X, y)`` masked-edge training examples.

    For each graph, mask ``ceil(mask_ratio * n_edges)`` edges (at least one when
    the graph has edges), encode the visible graph, and emit one example per
    masked edge whose relation is in-vocabulary and whose endpoints survive in
    the visible node set.
    """
    xs: List[np.ndarray] = []
    ys: List[int] = []
    for g in graphs:
        if not g.edges:
            continue
        n = len(g.edges)
        k = max(1, int(np.ceil(mask_ratio * n)))
        masked_idx = set(rng.sample(range(n), min(k, n)))
        visible_edges = [e for i, e in enumerate(g.edges) if i not in masked_idx]
        visible_graph = Graph(g.patient_id, g.source, list(g.nodes), visible_edges)
        emb = visible_node_embeddings(encoder, visible_graph)
        ctx = context_vector(emb, emb_dim)
        for i in masked_idx:
            e = g.edges[i]
            label = relation_to_index(e.relation)
            if label is None:
                continue
            x = edge_input(emb, e.source, e.target, ctx, emb_dim)
            if x is None:
                continue
            xs.append(x)
            ys.append(label)
    if not xs:
        return np.zeros((0, input_dim(emb_dim)), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.vstack(xs).astype(np.float32), np.asarray(ys, dtype=np.int64)


def _accuracy(model, X: np.ndarray, y: np.ndarray) -> float:
    if X.shape[0] == 0:
        return 0.0
    torch = _require_torch()
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X).float())
        pred = logits.argmax(dim=-1).numpy()
    return float((pred == y).mean())


def train(cfg: Config) -> Path:
    """Train the masked-edge predictor and write ``model.pt`` + run record."""
    torch = _require_torch()

    seed = int(cfg.get("training.seed", 13))
    rng = random.Random(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    tcfg = cfg.get("training.masked_edge_prediction", {}) or {}
    epochs = int(tcfg.get("epochs", 40))
    lr = float(tcfg.get("lr", 1e-3))
    hidden_dim = int(tcfg.get("hidden_dim", 128))
    mask_ratio = float(tcfg.get("mask_ratio", 0.25))

    # Setup B: self-supervised on the EIR transcript graphs (silver = eval only).
    train_graphs, val_graphs, source = load_training_graphs(cfg)
    if not train_graphs:
        raise FileNotFoundError(
            f"no training graphs under paths.{source} — run create_llm_graphs first"
        )
    LOG.info(
        "masked_edge_prediction: training on %d '%s' graphs (+%d monitor) — silver reserved for eval",
        len(train_graphs), source, len(val_graphs),
    )

    encoder = build_encoder(cfg)
    emb_dim = _encoder_dim(encoder)
    in_dim = input_dim(emb_dim)
    n_rel = len(RELATION_TYPES)

    X_tr, y_tr = _build_examples(train_graphs, encoder, emb_dim, mask_ratio, rng)
    X_va, y_va = _build_examples(val_graphs, encoder, emb_dim, mask_ratio, rng)
    if X_tr.shape[0] == 0:
        raise ValueError("no masked-edge training examples (empty / OOV-only graphs?)")
    LOG.info("built %d train / %d val masked-edge examples", X_tr.shape[0], X_va.shape[0])

    model = build_model(in_dim, hidden_dim, n_rel)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    Xt = torch.from_numpy(X_tr).float()
    yt = torch.from_numpy(y_tr).long()

    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model(Xt)
        loss = loss_fn(logits, yt)
        loss.backward()
        opt.step()
        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            LOG.info(
                "epoch %d/%d  loss=%.4f  train_acc=%.3f  val_acc=%.3f",
                epoch + 1,
                epochs,
                float(loss.item()),
                _accuracy(model, X_tr, y_tr),
                _accuracy(model, X_va, y_va),
            )

    out_dir = ensure_dir(cfg.path("paths.training_outputs") / "masked_edge_prediction")
    model_path = out_dir / "model.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "in_dim": in_dim,
            "hidden_dim": hidden_dim,
            "emb_dim": emb_dim,
            "n_relations": n_rel,
            "relation_types": RELATION_TYPES,
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
            "mask_ratio": mask_ratio,
            "seed": seed,
            "in_dim": in_dim,
            "emb_dim": emb_dim,
            "encoder_name": getattr(encoder, "name", "unknown"),
            "n_train_graphs": len(train_graphs),
            "n_val_graphs": len(val_graphs),
            "n_train_examples": int(X_tr.shape[0]),
            "n_val_examples": int(X_va.shape[0]),
            "final_val_acc": _accuracy(model, X_va, y_va),
        },
    )
    LOG.info("saved masked-edge predictor -> %s", model_path)
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
