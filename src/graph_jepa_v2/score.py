"""Score clinical KG edges with Graph-JEPA v2."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import List, Tuple

import torch

from graph_jepa.encoders import build_encoder
from graph_jepa.schema import EDGE_TYPE_TO_IDX, PatientGraph

from .config import Config
from .data import to_graph_data
from .model import GraphJEPAv2
from .patches import build_patch_data


def _load(checkpoint: str, device: torch.device, encoder_cache: str):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = Config.from_dict(ckpt["config"])
    model = GraphJEPAv2(cfg.model).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    if cfg.encoder in ("bge", "sapbert"):
        encoder = build_encoder(cfg.encoder, cache_dir=encoder_cache)
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
def score_graph(
    graph: PatientGraph,
    model: GraphJEPAv2,
    encoder,
    cfg: Config,
    device: torch.device,
) -> Tuple[List[float], List[str]]:
    scores = [0.0] * len(graph.edges)
    flags = ["inconsistent"] * len(graph.edges)

    if not graph.nodes:
        return scores, flags

    data = to_graph_data(graph, encoder).to(device)
    patch_data = build_patch_data(
        data,
        num_patches=cfg.model.num_patches,
        patch_pe_dim=cfg.model.patch_pe_dim,
        generator=None,
    ).to(device)
    if patch_data.num_patches == 0:
        return scores, flags

    z_nodes = model.encode_nodes(data)
    id_to_idx = graph.id_to_index()

    # Cache patch energies because many edges share the same target patch.
    patch_energy: dict[int, float] = {}

    for ei, e in enumerate(graph.edges):
        s = id_to_idx.get(e["source_id"])
        t = id_to_idx.get(e["target_id"])
        rel = EDGE_TYPE_TO_IDX.get(e["type"])
        if s is None or t is None or rel is None:
            continue

        rel_t = torch.tensor([rel], dtype=torch.long, device=device)
        logit = model.edge_head(z_nodes[s:s + 1], z_nodes[t:t + 1], rel_t)
        p_head = torch.sigmoid(logit).item()

        target_patch = int(patch_data.assignment[t].item())
        if target_patch not in patch_energy:
            idx = torch.tensor([target_patch], dtype=torch.long, device=device)
            patch_energy[target_patch] = float(
                model.patch_prediction_energy(data, patch_data, idx)[0].item()
            )
        structural = math.exp(-patch_energy[target_patch] / cfg.score.energy_temperature)
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
    model, encoder, cfg = _load(args.checkpoint, device, args.encoder_cache)
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
        print(
            f"{jf.name}: {len(graph.nodes)} nodes, {len(scores)} edges scored, "
            f"{flagged} flagged" + (f", {pruned} pruned" if pruned else "")
        )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Annotate KG edges with Graph-JEPA v2 scores")
    p.add_argument("--input", required=True, help="KG JSON file or directory")
    p.add_argument("--checkpoint", required=True, help="trained v2 checkpoint .pt")
    p.add_argument("--output", required=True, help="output file or directory")
    p.add_argument("--device", default="cpu")
    p.add_argument("--encoder-cache", default=".cache/graph_jepa_v2/encoder")
    p.add_argument("--prune-threshold", type=float, default=None)
    return p


def main(argv=None) -> None:
    run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()
