"""Score clinical KG edges with Graph-JEPA v3.

By default this annotates existing edges only. ``--add-candidates`` appends
high-scoring missing edges, while ``--prune-threshold`` removes low-scoring
existing edges. Both actions use the same graph-revision edge score.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import List, Tuple

import torch

from graph_jepa.data import RELATION_SCHEMA, adapt_mimic_subkg
from graph_jepa.encoders import build_encoder
from graph_jepa.schema import EDGE_TYPE_TO_IDX, NODE_TYPE_TO_IDX, PatientGraph

from .config import Config
from .data import to_graph_data
from .model import GraphJEPAv3
from .patches import build_patch_data


RELATIONS = set(EDGE_TYPE_TO_IDX)

NEGATED_OR_ABSENT_MARKERS = (
    "absent ",
    "no ",
    "denies ",
    "denied ",
    "negative ",
    "without ",
    "unremarkable",
)


def _normalise_relation(relation: str) -> str:
    return relation.strip().upper()


def _update_relation_thresholds(target: dict[str, float], values: list[str] | None) -> None:
    if not values:
        return
    for value in values:
        if "=" not in value:
            raise ValueError(f"relation threshold must be RELATION=VALUE, got {value!r}")
        relation, raw_threshold = value.split("=", 1)
        relation = _normalise_relation(relation)
        if relation not in RELATIONS:
            raise ValueError(f"unknown relation for threshold override: {relation!r}")
        target[relation] = float(raw_threshold)


def _update_disabled_relations(
    disabled: list[str],
    *,
    enable: list[str] | None,
    disable: list[str] | None,
) -> list[str]:
    disabled_set = {_normalise_relation(relation) for relation in disabled}
    for relation in enable or []:
        relation = _normalise_relation(relation)
        if relation not in RELATIONS:
            raise ValueError(f"unknown relation to enable for candidates: {relation!r}")
        disabled_set.discard(relation)
    for relation in disable or []:
        relation = _normalise_relation(relation)
        if relation not in RELATIONS:
            raise ValueError(f"unknown relation to disable for candidates: {relation!r}")
        disabled_set.add(relation)
    return sorted(disabled_set)


def _load(checkpoint: str, device: torch.device, encoder_cache: str):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = Config.from_dict(ckpt["config"])
    model = GraphJEPAv3(cfg.model).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    if cfg.encoder in ("bge", "sapbert"):
        encoder = build_encoder(cfg.encoder, cache_dir=encoder_cache)
    else:
        encoder = build_encoder("mock", mock_dim=cfg.model.in_dim)
    return model, encoder, cfg


def _review_thresholds(relation: str | None, cfg: Config) -> tuple[float, float]:
    if relation is None:
        return cfg.score.inconsistent_threshold, cfg.score.weak_threshold
    relation = _normalise_relation(relation)
    inconsistent = cfg.score.inconsistent_threshold_by_relation.get(
        relation,
        cfg.score.inconsistent_threshold,
    )
    weak = cfg.score.weak_threshold_by_relation.get(
        relation,
        cfg.score.weak_threshold,
    )
    return inconsistent, weak


def _flag(score: float, cfg: Config, relation: str | None = None) -> str:
    inconsistent_threshold, weak_threshold = _review_thresholds(relation, cfg)
    if score < inconsistent_threshold:
        return "inconsistent"
    if score < weak_threshold:
        return "weak"
    return "ok"


def _candidate_threshold(relation: str, cfg: Config) -> float | None:
    relation = _normalise_relation(relation)
    if relation in {_normalise_relation(r) for r in cfg.score.disabled_candidate_relations}:
        return None
    return cfg.score.candidate_threshold_by_relation.get(
        relation,
        cfg.score.candidate_threshold,
    )


def _node_res_ids(node: dict) -> set[str]:
    occurrences = node.get("occurrences") or []
    return {
        str(occ.get("res_id"))
        for occ in occurrences
        if occ.get("res_id")
    }


def _has_res_id_context(src: dict, tgt: dict) -> bool:
    return bool(_node_res_ids(src) or _node_res_ids(tgt))


def _shares_res_id(src: dict, tgt: dict) -> bool:
    src_res = _node_res_ids(src)
    tgt_res = _node_res_ids(tgt)
    if not src_res or not tgt_res:
        return False
    return bool(src_res & tgt_res)


def _is_negated_or_absent(node: dict) -> bool:
    text = " ".join(
        str(node.get(key, "")).lower()
        for key in ("text", "evidence")
    ).strip()
    if not text:
        return False
    return any(marker in text for marker in NEGATED_OR_ABSENT_MARKERS)


def _candidate_context_ok(src: dict, tgt: dict, cfg: Config) -> bool:
    if cfg.score.skip_negated_candidates and _is_negated_or_absent(src):
        return False
    if (
        cfg.score.require_shared_res_id
        and _has_res_id_context(src, tgt)
        and not _shares_res_id(src, tgt)
    ):
        return False
    return True


def _revision_action(
    score: float,
    cfg: Config,
    *,
    relation: str | None = None,
    candidate: bool = False,
) -> str:
    if candidate:
        return "add_candidate"
    if cfg.score.prune_threshold is not None and score < cfg.score.prune_threshold:
        return "prune_candidate"
    _inconsistent_threshold, weak_threshold = _review_thresholds(relation, cfg)
    if score < weak_threshold:
        return "review"
    return "keep"


def _iter_candidate_specs(graph: PatientGraph, cfg: Config) -> List[Tuple[int, int, str]]:
    """Return missing schema-valid ``(src_idx, tgt_idx, relation)`` candidates."""
    existing = {
        (e.get("source_id"), e.get("target_id"), e.get("type"))
        for e in graph.edges
    }
    by_source_type: dict[str, List[Tuple[str, set[str]]]] = {}
    for (src_type, relation), target_types in RELATION_SCHEMA.items():
        by_source_type.setdefault(src_type, []).append((relation, target_types))

    candidates: List[Tuple[int, int, str]] = []
    for s_idx, src in enumerate(graph.nodes):
        src_id = src.get("id")
        if not src_id:
            continue
        for relation, target_types in by_source_type.get(src.get("type"), []):
            for t_idx, tgt in enumerate(graph.nodes):
                tgt_id = tgt.get("id")
                if s_idx == t_idx or not tgt_id or tgt.get("type") not in target_types:
                    continue
                if (src_id, tgt_id, relation) in existing:
                    continue
                if not _candidate_context_ok(src, tgt, cfg):
                    continue
                candidates.append((s_idx, t_idx, relation))
    return candidates


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
                "jepa_source": "graph_jepa_v3_candidate_generation",
            }
        )
    return len(scored)


@torch.no_grad()
def score_graph(
    graph: PatientGraph,
    model: GraphJEPAv3,
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
        flags[ei] = _flag(score, cfg, e["type"])

    return scores, flags


@torch.no_grad()
def add_candidate_edges(
    graph: PatientGraph,
    model: GraphJEPAv3,
    encoder,
    cfg: Config,
    device: torch.device,
) -> int:
    """Add high-scoring missing edges between existing nodes.

    Candidates are schema-valid triples that are absent from the input graph.
    Added edges are marked ``jepa_suggested`` / ``jepa_unverified`` because this
    scorer does not produce transcript evidence.
    """
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


def _iter_inputs(input_path: Path) -> List[Path]:
    if input_path.is_dir():
        return sorted(input_path.glob("*.json"))
    return [input_path]


def _schema_compatibility_errors(graph: PatientGraph) -> list[str]:
    unknown_node_types = sorted(
        {
            str(node.get("type"))
            for node in graph.nodes
            if node.get("type") not in NODE_TYPE_TO_IDX
        }
    )
    errors: list[str] = []
    if unknown_node_types:
        errors.append(f"unknown node types: {', '.join(unknown_node_types)}")

    unknown_edge_types = sorted(
        {
            str(edge.get("type") or edge.get("relation"))
            for edge in graph.edges
            if (edge.get("type") or edge.get("relation")) not in EDGE_TYPE_TO_IDX
        }
    )
    if unknown_edge_types:
        errors.append(f"unknown edge types: {', '.join(unknown_edge_types)}")

    malformed_edges = sum(
        1
        for edge in graph.edges
        if not {"source_id", "target_id", "type"} <= edge.keys()
    )
    if malformed_edges:
        errors.append(f"{malformed_edges} edges missing source_id/target_id/type")
    return errors


def _looks_like_mimic_subkg(data: dict) -> bool:
    if "subject_id" in data or "hadm_ids" in data:
        return True
    mimic_only_node_types = {
        "PATIENT",
        "MEDICATION",
        "LAB_TEST",
        "MICROBIOLOGY",
        "SERVICE",
    }
    if any(
        str(node.get("type", "")).upper() in mimic_only_node_types
        for node in data.get("nodes", [])
    ):
        return True
    return any(
        "source" in edge or "target" in edge or "relation" in edge
        for edge in data.get("edges", [])
    )


def _load_graph_for_scoring(path: Path) -> PatientGraph:
    with open(path, "r") as f:
        data = json.load(f)

    graph = PatientGraph.from_pipeline_json(data)
    errors = _schema_compatibility_errors(graph)
    if not errors:
        return graph

    if _looks_like_mimic_subkg(data):
        graph = adapt_mimic_subkg(data, source_path=path)
        errors = _schema_compatibility_errors(graph)
        if not errors:
            return graph

    raise ValueError(
        "KG is not compatible with the JEPA scoring schema "
        f"({'; '.join(errors)})"
    )


def _validate_checkpoint_relation_capacity(graph: PatientGraph, cfg: Config) -> None:
    required = 0
    for edge in graph.edges:
        rel_idx = EDGE_TYPE_TO_IDX.get(edge.get("type"))
        if rel_idx is not None:
            required = max(required, rel_idx + 1)
    if required > cfg.model.num_relations:
        raise ValueError(
            "checkpoint relation embedding is too small for this graph "
            f"(checkpoint num_relations={cfg.model.num_relations}, "
            f"required>={required}). Retrain Graph-JEPA with the expanded "
            "MIMIC schema before scoring raw MIMIC relations."
        )


def _prune(graph: PatientGraph, threshold: float) -> int:
    before = len(graph.edges)
    graph.edges = [e for e in graph.edges if e.get("jepa_score", 1.0) >= threshold]
    return before - len(graph.edges)


def _annotate_revision_actions(graph: PatientGraph, cfg: Config) -> None:
    for edge in graph.edges:
        if edge.get("jepa_suggested"):
            edge["jepa_revision_action"] = _revision_action(
                float(edge.get("jepa_score", 0.0)),
                cfg,
                relation=edge.get("type"),
                candidate=True,
            )
        else:
            edge["jepa_revision_action"] = _revision_action(
                float(edge.get("jepa_score", 0.0)),
                cfg,
                relation=edge.get("type"),
            )


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


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Revise KG edges with Graph-JEPA v3 scores")
    p.add_argument("--input", required=True, help="KG JSON file or directory")
    p.add_argument("--checkpoint", required=True, help="trained v3 checkpoint .pt")
    p.add_argument("--output", required=True, help="output file or directory")
    p.add_argument("--device", default="cpu")
    p.add_argument("--encoder-cache", default=".cache/graph_jepa_v3/encoder")
    p.add_argument("--prune-threshold", type=float, default=None)
    p.add_argument(
        "--add-candidates",
        action="store_true",
        help="add high-scoring missing schema-valid edges between existing nodes",
    )
    p.add_argument(
        "--candidate-threshold",
        type=float,
        default=None,
        help="global minimum jepa_score for enabled candidate relations",
    )
    p.add_argument(
        "--candidate-threshold-relation",
        action="append",
        default=None,
        metavar="RELATION=VALUE",
        help="override candidate-add threshold for one relation; repeatable",
    )
    p.add_argument(
        "--review-weak-threshold",
        action="append",
        default=None,
        metavar="RELATION=VALUE",
        help="override existing-edge weak/review threshold for one relation; repeatable",
    )
    p.add_argument(
        "--review-inconsistent-threshold",
        action="append",
        default=None,
        metavar="RELATION=VALUE",
        help="override existing-edge inconsistent threshold for one relation; repeatable",
    )
    p.add_argument(
        "--enable-candidate-relation",
        action="append",
        default=None,
        metavar="RELATION",
        help="enable candidate additions for a relation disabled by the score config",
    )
    p.add_argument(
        "--disable-candidate-relation",
        action="append",
        default=None,
        metavar="RELATION",
        help="disable candidate additions for a relation; repeatable",
    )
    p.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help="maximum number of candidate edges to add per graph",
    )
    p.add_argument(
        "--allow-cross-res-candidates",
        action="store_true",
        help="allow unified-graph candidates whose endpoints never share a res_id",
    )
    p.add_argument(
        "--allow-negated-candidates",
        action="store_true",
        help="allow absent/negated findings as sources for added candidates",
    )
    return p


def main(argv=None) -> None:
    run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()
