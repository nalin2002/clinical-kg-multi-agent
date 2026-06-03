"""Inference: annotate pipeline KG edges with ``jepa_score`` / ``jepa_flag``.

For each edge ``(src, rel, tgt)`` the score combines two signals:

* **edge-head probability** ``p_head = sigmoid(head(z_src, z_tgt, rel))`` from the
  supervised typed-plausibility head, and
* **structural consistency** ``exp(-energy / T)`` where ``energy`` is the distance
  between the JEPA predictor's reconstruction of the masked target endpoint and
  the EMA target encoder's latent for that node.

::

    jepa_score = alpha * p_head + (1 - alpha) * structural_score

By default this is **annotate-only** - no edges are added or removed. Passing
``--prune-threshold T`` drops edges whose ``jepa_score < T`` (opt-in).

CLI::

    python -m graph_jepa.score --input <dir-or-json> --checkpoint checkpoints/graph_jepa.pt --output <dir-or-json>
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import List, Tuple

import torch

from .config import Config
from .data import to_pyg_data
from .encoders import build_encoder
from .model import GraphJEPA
from .schema import EDGE_TYPE_TO_IDX, PatientGraph


def _load(checkpoint: str, device: torch.device, bge_cache: str):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = Config.from_dict(ckpt["config"])
    model = GraphJEPA(cfg.model).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    if cfg.encoder == "bge":
        encoder = build_encoder("bge", cache_dir=bge_cache)
    else:
        encoder = build_encoder("mock", mock_dim=cfg.model.in_dim)
    return model, encoder, cfg


def _flag(score: float, cfg: Config) -> str:
    if score < cfg.score.inconsistent_threshold:
        return "inconsistent"
    if score < cfg.score.weak_threshold:
        return "weak"
    return "ok"


@torch.no_grad()
def score_graph(graph: PatientGraph, model: GraphJEPA, encoder, cfg: Config,
                device: torch.device) -> Tuple[List[float], List[str]]:
    """Return per-edge ``(scores, flags)`` aligned 1:1 with ``graph.edges``."""
    scores = [0.0] * len(graph.edges)
    flags = ["inconsistent"] * len(graph.edges)

    if not graph.nodes:
        return scores, flags

    data = to_pyg_data(graph, encoder).to(device)
    z_full = model.encode_context(data, None)      # online, full graph
    z_tgt = model.encode_target(data)              # EMA target, full graph
    latent_scale = math.sqrt(cfg.model.latent_dim)
    id_to_idx = graph.id_to_index()

    for ei, e in enumerate(graph.edges):
        s = id_to_idx.get(e["source_id"])
        t = id_to_idx.get(e["target_id"])
        rel = EDGE_TYPE_TO_IDX.get(e["type"])
        if s is None or t is None or rel is None:
            # Cannot validate (dangling endpoint / unknown relation): keep but flag.
            continue

        rel_t = torch.tensor([rel], dtype=torch.long, device=device)
        logit = model.edge_head(z_full[s:s + 1], z_full[t:t + 1], rel_t)
        p_head = torch.sigmoid(logit).item()

        # Structural consistency: mask the target endpoint, predict, compare.
        node_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
        node_mask[t] = True
        z_masked = model.encode_context(data, node_mask)
        pred = model.predictor(z_masked[node_mask])
        energy = torch.norm(pred - z_tgt[t:t + 1], dim=-1).item() / latent_scale
        structural = math.exp(-energy / cfg.score.energy_temperature)

        score = cfg.score.alpha * p_head + (1.0 - cfg.score.alpha) * structural
        scores[ei] = score
        flags[ei] = _flag(score, cfg)

    return scores, flags


def _iter_inputs(input_path: Path) -> List[Path]:
    if input_path.is_dir():
        return sorted(input_path.glob("*.json"))
    return [input_path]


def _prune(graph: PatientGraph, threshold: float) -> int:
    before = len(graph.edges)
    graph.edges = [e for e in graph.edges if e.get("jepa_score", 1.0) >= threshold]
    return before - len(graph.edges)


def run(args) -> None:
    device = torch.device(args.device)
    model, encoder, cfg = _load(args.checkpoint, device, args.bge_cache)
    if args.prune_threshold is not None:
        cfg.score.prune_threshold = args.prune_threshold

    input_path = Path(args.input)
    output_path = Path(args.output)
    inputs = _iter_inputs(input_path)
    output_is_dir = input_path.is_dir()
    if output_is_dir:
        output_path.mkdir(parents=True, exist_ok=True)

    for jf in inputs:
        try:
            graph = PatientGraph.load(jf)
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            # Skip sidecar JSON that isn't a patient graph (e.g. _stats.json)
            # so a whole output directory can be scored in one pass.
            if output_is_dir:
                print(f"{jf.name}: skipped (not a KG: {exc})")
                continue
            raise
        scores, flags = score_graph(graph, model, encoder, cfg, device)
        graph.annotate_edges(scores, flags)

        pruned = 0
        if cfg.score.prune_threshold is not None:
            pruned = _prune(graph, cfg.score.prune_threshold)

        dest = output_path / jf.name if output_is_dir else output_path
        graph.save(dest)
        flagged = sum(1 for f in flags if f != "ok")
        print(f"{jf.name}: {len(graph.nodes)} nodes, {len(scores)} edges scored, "
              f"{flagged} flagged" + (f", {pruned} pruned" if pruned else ""))


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Annotate KG edges with Graph-JEPA plausibility scores")
    p.add_argument("--input", required=True, help="KG JSON file or directory")
    p.add_argument("--checkpoint", required=True, help="trained checkpoint .pt")
    p.add_argument("--output", required=True, help="output file or directory")
    p.add_argument("--device", default="cpu")
    p.add_argument("--bge-cache", default=".cache/graph_jepa/bge")
    p.add_argument("--prune-threshold", type=float, default=None,
                   help="opt-in: drop edges with jepa_score below this value")
    return p


def main(argv=None) -> None:
    run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()
