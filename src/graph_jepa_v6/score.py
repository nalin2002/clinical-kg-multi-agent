"""Score clinical KG edges with Graph-JEPA v6."""

from __future__ import annotations

import json
from pathlib import Path

import torch

import graph_jepa_v3.score as _v3
import graph_jepa_v4.score as _v4

from .config import Config
from .data import to_graph_data
from .model import GraphJEPAv6
from .training import build_checkpoint_encoder


def _install_v6_data_conversion() -> None:
    _v3.to_graph_data = to_graph_data
    _v4.to_graph_data = to_graph_data


def _load(checkpoint: str, device: torch.device, encoder_cache: str):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = Config.from_dict(ckpt["config"])
    model = GraphJEPAv6(cfg.model).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    encoder = build_checkpoint_encoder(cfg, encoder_cache)
    return model, encoder, cfg


def run(args) -> None:
    device = torch.device(args.device)
    model, encoder, cfg = _load(args.checkpoint, device, args.encoder_cache)
    _install_v6_data_conversion()
    if args.prune_threshold is not None:
        cfg.score.prune_threshold = args.prune_threshold
    if args.candidate_threshold is not None:
        cfg.score.candidate_threshold = args.candidate_threshold
        cfg.score.candidate_threshold_by_relation = {
            relation: args.candidate_threshold
            for relation in _v4.RELATIONS
        }
    _v4._update_relation_thresholds(
        cfg.score.weak_threshold_by_relation,
        args.review_weak_threshold,
    )
    _v4._update_relation_thresholds(
        cfg.score.inconsistent_threshold_by_relation,
        args.review_inconsistent_threshold,
    )
    _v4._update_relation_thresholds(
        cfg.score.candidate_threshold_by_relation,
        args.candidate_threshold_relation,
    )
    cfg.score.disabled_candidate_relations = _v4._update_disabled_relations(
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
    inputs = _v4._iter_inputs(input_path)
    output_is_dir = input_path.is_dir()
    if output_is_dir:
        output_path.mkdir(parents=True, exist_ok=True)

    for jf in inputs:
        try:
            graph = _v4._load_graph_for_scoring(jf)
            _v4._validate_checkpoint_relation_capacity(graph, cfg)
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            if output_is_dir:
                print(f"{jf.name}: skipped (not a KG: {exc})")
                continue
            raise

        scores, flags = _v4.score_graph(graph, model, encoder, cfg, device)
        graph.annotate_edges(scores, flags)
        _v4._annotate_revision_actions(graph, cfg)

        added = 0
        if args.add_candidates:
            added = _v4.add_candidate_edges(graph, model, encoder, cfg, device)
            _v4._annotate_revision_actions(graph, cfg)

        pruned = 0
        if cfg.score.prune_threshold is not None:
            pruned = _v4._prune(graph, cfg.score.prune_threshold)

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
    p = _v4.build_arg_parser()
    p.description = "Revise KG edges with Graph-JEPA v6 scores"
    for action in p._actions:
        if action.dest == "checkpoint":
            action.help = "trained v6 checkpoint .pt"
        elif action.dest == "encoder_cache":
            action.default = ".cache/graph_jepa_v6/encoder"
    return p


def main(argv=None) -> None:
    run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()
