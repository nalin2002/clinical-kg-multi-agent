"""Score clinical KG edges with Graph-JEPA v4."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import List, Tuple

import torch

import graph_jepa_v3.score as _v3
from graph_jepa.encoders import build_encoder
from graph_jepa.schema import EDGE_TYPE_TO_IDX, PatientGraph

from .config import Config
from .data import to_graph_data
from .model import GraphJEPAv4
from .patches import build_patch_data

RELATIONS = _v3.RELATIONS
NEGATED_OR_ABSENT_MARKERS = _v3.NEGATED_OR_ABSENT_MARKERS

_annotate_revision_actions = _v3._annotate_revision_actions
_candidate_threshold = _v3._candidate_threshold
_flag = _v3._flag
_iter_candidate_specs = _v3._iter_candidate_specs
_iter_inputs = _v3._iter_inputs
_load_graph_for_scoring = _v3._load_graph_for_scoring
_normalise_relation = _v3._normalise_relation
_prune = _v3._prune
_revision_action = _v3._revision_action
_schema_compatibility_errors = _v3._schema_compatibility_errors
_update_disabled_relations = _v3._update_disabled_relations
_update_relation_thresholds = _v3._update_relation_thresholds
_validate_checkpoint_relation_capacity = _v3._validate_checkpoint_relation_capacity
score_graph = _v3.score_graph


def _load(checkpoint: str, device: torch.device, encoder_cache: str):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = Config.from_dict(ckpt["config"])
    model = GraphJEPAv4(cfg.model).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    if cfg.encoder in ("bge", "sapbert"):
        encoder = build_encoder(cfg.encoder, cache_dir=encoder_cache)
    else:
        encoder = build_encoder("mock", mock_dim=cfg.model.in_dim)
    return model, encoder, cfg


def _append_candidate_edges(
    graph: PatientGraph,
    scored: List[Tuple[float, int, int, str]],
    cfg: Config,
) -> int:
    for rank, (score, s_idx, t_idx, relation) in enumerate(scored, start=1):
        graph.edges.append(
            {
                "source_id": graph.nodes[s_idx]["id"],
                "target_id": graph.nodes[t_idx]["id"],
                "type": relation,
                "evidence": "",
                "turn_id": "",
                "jepa_score": round(float(score), 6),
                "jepa_flag": _flag(score, cfg, relation),
                "jepa_revision_action": _revision_action(
                    score,
                    cfg,
                    relation=relation,
                    candidate=True,
                ),
                "jepa_suggested": True,
                "jepa_unverified": True,
                "jepa_candidate_rank": rank,
                "jepa_source": "graph_jepa_v4_candidate_generation",
            }
        )
    return len(scored)


@torch.no_grad()
def add_candidate_edges(
    graph: PatientGraph,
    model: GraphJEPAv4,
    encoder,
    cfg: Config,
    device: torch.device,
) -> int:
    """Add high-scoring missing edges between existing nodes."""
    if not graph.nodes or cfg.score.max_candidate_edges <= 0:
        return 0

    candidates = _iter_candidate_specs(graph, cfg)
    if not candidates:
        return 0

    data = to_graph_data(graph, encoder).to(device)
    patch_data = build_patch_data(
        data,
        num_patches=cfg.model.num_patches,
        patch_pe_dim=cfg.model.patch_pe_dim,
        generator=None,
    ).to(device)
    if patch_data.num_patches == 0:
        return 0

    z_nodes = model.encode_nodes(data)
    patch_energy: dict[int, float] = {}
    scored: List[Tuple[float, int, int, str]] = []

    for s_idx, t_idx, relation in candidates:
        rel = EDGE_TYPE_TO_IDX.get(relation)
        if rel is None:
            continue
        threshold = _candidate_threshold(relation, cfg)
        if threshold is None:
            continue

        rel_t = torch.tensor([rel], dtype=torch.long, device=device)
        logit = model.edge_head(z_nodes[s_idx:s_idx + 1], z_nodes[t_idx:t_idx + 1], rel_t)
        p_head = torch.sigmoid(logit).item()

        target_patch = int(patch_data.assignment[t_idx].item())
        if target_patch not in patch_energy:
            idx = torch.tensor([target_patch], dtype=torch.long, device=device)
            patch_energy[target_patch] = float(
                model.patch_prediction_energy(data, patch_data, idx)[0].item()
            )
        structural = math.exp(-patch_energy[target_patch] / cfg.score.energy_temperature)
        score = cfg.score.alpha * p_head + (1.0 - cfg.score.alpha) * structural
        if score >= threshold:
            scored.append((score, s_idx, t_idx, relation))

    scored.sort(key=lambda item: item[0], reverse=True)
    scored = scored[:cfg.score.max_candidate_edges]
    return _append_candidate_edges(graph, scored, cfg)


def run(args) -> None:
    device = torch.device(args.device)
    model, encoder, cfg = _load(args.checkpoint, device, args.encoder_cache)
    if args.prune_threshold is not None:
        cfg.score.prune_threshold = args.prune_threshold
    if args.candidate_threshold is not None:
        cfg.score.candidate_threshold = args.candidate_threshold
        cfg.score.candidate_threshold_by_relation = {
            relation: args.candidate_threshold
            for relation in RELATIONS
        }
    _update_relation_thresholds(
        cfg.score.weak_threshold_by_relation,
        args.review_weak_threshold,
    )
    _update_relation_thresholds(
        cfg.score.inconsistent_threshold_by_relation,
        args.review_inconsistent_threshold,
    )
    _update_relation_thresholds(
        cfg.score.candidate_threshold_by_relation,
        args.candidate_threshold_relation,
    )
    cfg.score.disabled_candidate_relations = _update_disabled_relations(
        cfg.score.disabled_candidate_relations,
        enable=args.enable_candidate_relation,
        disable=args.disable_candidate_relation,
    )
    if args.max_candidates is not None:
        cfg.score.max_candidate_edges = args.max_candidates
    if args.allow_cross_res_candidates:
        cfg.score.require_shared_res_id = False
    if args.allow_negated_candidates:
        cfg.score.skip_negated_candidates = False

    input_path = Path(args.input)
    output_path = Path(args.output)
    inputs = _iter_inputs(input_path)
    output_is_dir = input_path.is_dir()
    if output_is_dir:
        output_path.mkdir(parents=True, exist_ok=True)

    for jf in inputs:
        try:
            graph = _load_graph_for_scoring(jf)
            _validate_checkpoint_relation_capacity(graph, cfg)
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            if output_is_dir:
                print(f"{jf.name}: skipped (not a KG: {exc})")
                continue
            raise

        scores, flags = score_graph(graph, model, encoder, cfg, device)
        graph.annotate_edges(scores, flags)
        _annotate_revision_actions(graph, cfg)

        added = 0
        if args.add_candidates:
            added = add_candidate_edges(graph, model, encoder, cfg, device)
            _annotate_revision_actions(graph, cfg)

        pruned = 0
        if cfg.score.prune_threshold is not None:
            pruned = _prune(graph, cfg.score.prune_threshold)

        dest = output_path / jf.name if output_is_dir else output_path
        graph.save(dest)
        flagged = sum(1 for f in flags if f != "ok")
        print(
            f"{jf.name}: {len(graph.nodes)} nodes, {len(scores)} edges scored, "
            f"{flagged} flagged"
            + (f", {added} candidates added" if added else "")
            + (f", {pruned} pruned" if pruned else "")
        )


def build_arg_parser():
    p = _v3.build_arg_parser()
    p.description = "Revise KG edges with Graph-JEPA v4 scores"
    for action in p._actions:
        if action.dest == "checkpoint":
            action.help = "trained v4 checkpoint .pt"
        elif action.dest == "encoder_cache":
            action.default = ".cache/graph_jepa_v4/encoder"
    return p


def main(argv=None) -> None:
    run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()
