"""Leave-one-out edge-recovery evaluation for Graph-JEPA v6."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import torch

from graph_jepa.schema import IDX_TO_EDGE_TYPE
from graph_jepa_v3.model import _allowed_target_type_indices
from graph_jepa_v4.model import _schema_edge_masks

from .data import PatientGraphDataset
from .model import _confidence_supervision_masks
from .training import (
    add_data_args,
    build_checkpoint_encoder,
    build_graphs,
    load_model_checkpoint,
)


def _chance_mrr(candidate_count: float) -> float:
    if candidate_count <= 1:
        return 1.0
    return (math.log(candidate_count) + 0.5772) / candidate_count


def _trusted_edge_masks(data, cfg) -> tuple[torch.Tensor, torch.Tensor]:
    schema_valid, _schema_invalid, unconstrained = _schema_edge_masks(
        data,
        allow_unconstrained=False,
    )
    trusted_positive, _weak_negative, _ignored = _confidence_supervision_masks(
        data,
        enabled=cfg.train.llm_confidence_negatives,
        negative_threshold=cfg.train.llm_negative_threshold,
        positive_threshold=cfg.train.llm_positive_threshold,
        negative_threshold_by_relation=cfg.train.llm_negative_threshold_by_relation,
        positive_threshold_by_relation=cfg.train.llm_positive_threshold_by_relation,
        clinical_artifact_filters=cfg.train.clinical_artifact_filters,
    )
    positive_mask = schema_valid & trusted_positive
    message_mask = (schema_valid | unconstrained) & trusted_positive
    return positive_mask, message_mask


def _positive_triples(data, positive_mask: torch.Tensor) -> set[tuple[int, int, int]]:
    return {
        (int(source), int(target), int(relation))
        for source, target, relation in zip(
            data.edge_index[0, positive_mask].detach().cpu().tolist(),
            data.edge_index[1, positive_mask].detach().cpu().tolist(),
            data.edge_type[positive_mask].detach().cpu().tolist(),
        )
    }


def _target_candidates(
    data,
    *,
    source: int,
    target: int,
    relation: int,
    true_triples: set[tuple[int, int, int]],
    candidate_mode: str,
) -> list[int]:
    node_type = getattr(data, "node_type", None)
    if node_type is None:
        return []
    node_type_cpu = node_type.detach().cpu().tolist()
    target_type = int(node_type_cpu[target])

    if candidate_mode == "same-type":
        allowed_types = {target_type}
    elif candidate_mode == "schema":
        allowed_types = _allowed_target_type_indices(node_type_cpu[source], relation)
    else:
        raise ValueError(f"unknown candidate_mode: {candidate_mode!r}")
    if not allowed_types:
        return []

    other_true_tails = {
        t
        for s, t, r in true_triples
        if s == source and r == relation and t != target
    }
    return [
        idx
        for idx, candidate_type in enumerate(node_type_cpu)
        if idx != source
        and int(candidate_type) in allowed_types
        and idx not in other_true_tails
    ]


def _empty_metrics() -> dict:
    return {
        "mrr": float("nan"),
        "hits1": 0.0,
        "hits3": 0.0,
        "hits10": 0.0,
        "n": 0,
        "per_rel": [],
    }


@torch.no_grad()
def leave_one_out_recovery(
    model,
    graphs: Iterable,
    cfg,
    device: torch.device,
    *,
    cap: int = 40000,
    candidate_mode: str = "schema",
) -> dict:
    """Mask one trusted edge at a time and rank its target against candidates."""

    if cap <= 0:
        return _empty_metrics()

    model.eval()
    rel_rr: dict[int, list[float]] = defaultdict(list)
    rel_hits: dict[int, dict[int, int]] = defaultdict(lambda: {1: 0, 3: 0, 10: 0})
    rel_candidates: dict[int, list[int]] = defaultdict(list)
    total_hits = {1: 0, 3: 0, 10: 0}
    total_rr: list[float] = []
    total_n = 0

    for data in graphs:
        if total_n >= cap:
            break
        data = data.to(device)
        edge_count = int(data.edge_index.size(1))
        if edge_count < 1:
            continue
        if edge_count and int(data.edge_type.max().item()) >= cfg.model.num_relations:
            raise ValueError(
                "graph contains relation id outside checkpoint relation capacity "
                f"({int(data.edge_type.max().item())} >= {cfg.model.num_relations})"
            )

        positive_mask, base_message_mask = _trusted_edge_masks(data, cfg)
        positive_indices = positive_mask.nonzero(as_tuple=False).flatten()
        if positive_indices.numel() == 0:
            continue
        true_triples = _positive_triples(data, positive_mask)

        for edge_idx in positive_indices.detach().cpu().tolist():
            if total_n >= cap:
                break
            edge_idx = int(edge_idx)
            source = int(data.edge_index[0, edge_idx].item())
            target = int(data.edge_index[1, edge_idx].item())
            relation = int(data.edge_type[edge_idx].item())
            if source == target:
                continue

            candidates = _target_candidates(
                data,
                source=source,
                target=target,
                relation=relation,
                true_triples=true_triples,
                candidate_mode=candidate_mode,
            )
            if target not in candidates or len(candidates) < 2:
                continue

            message_mask = base_message_mask.clone()
            message_mask[edge_idx] = False
            z_nodes = model.context_node_encoder(
                data.x,
                data.edge_index[:, message_mask],
                data.edge_type[message_mask],
            )
            cand_t = torch.tensor(candidates, dtype=torch.long, device=device)
            src_t = torch.full(
                (len(candidates),),
                source,
                dtype=torch.long,
                device=device,
            )
            rel_t = torch.full(
                (len(candidates),),
                relation,
                dtype=torch.long,
                device=device,
            )
            logits = model.edge_head(z_nodes[src_t], z_nodes[cand_t], rel_t)
            target_idx = int((cand_t == target).nonzero(as_tuple=False)[0, 0].item())
            rank = int((logits > logits[target_idx]).sum().item()) + 1
            reciprocal_rank = 1.0 / rank

            rel_rr[relation].append(reciprocal_rank)
            rel_candidates[relation].append(len(candidates))
            total_rr.append(reciprocal_rank)
            for k in total_hits:
                if rank <= k:
                    rel_hits[relation][k] += 1
                    total_hits[k] += 1
            total_n += 1

    per_rel = []
    for relation in sorted(rel_rr, key=lambda rel: -len(rel_rr[rel])):
        n = len(rel_rr[relation])
        mean_candidates = float(sum(rel_candidates[relation]) / max(n, 1))
        per_rel.append(
            {
                "rel": IDX_TO_EDGE_TYPE.get(relation, f"rel{relation}"),
                "n": n,
                "mrr": float(sum(rel_rr[relation]) / n),
                "h1": rel_hits[relation][1] / n,
                "h3": rel_hits[relation][3] / n,
                "h10": rel_hits[relation][10] / n,
                "C": mean_candidates,
                "chance_mrr": _chance_mrr(mean_candidates),
                "chance_h1": 1.0 / mean_candidates if mean_candidates >= 1 else 1.0,
            }
        )

    if not total_rr:
        return _empty_metrics()
    return {
        "mrr": float(sum(total_rr) / len(total_rr)),
        "hits1": total_hits[1] / total_n,
        "hits3": total_hits[3] / total_n,
        "hits10": total_hits[10] / total_n,
        "n": total_n,
        "per_rel": per_rel,
    }


def _filter_graphs(
    graphs: list,
    *,
    split: str | None,
    start: int,
    limit: int | None,
) -> list:
    if split:
        graphs = [
            graph
            for graph in graphs
            if str(graph.extra.get("split", "")).lower() == split.lower()
        ]
    if start < 0:
        raise ValueError("--start-graph must be non-negative")
    graphs = graphs[start:]
    if limit is not None:
        graphs = graphs[:limit]
    return graphs


def run(args) -> dict:
    device = torch.device(args.device)
    model, cfg = load_model_checkpoint(args.checkpoint, device)
    encoder = build_checkpoint_encoder(cfg, args.encoder_cache)
    graphs = build_graphs(args, cfg)
    graphs = _filter_graphs(
        graphs,
        split=args.split,
        start=args.start_graph,
        limit=args.max_graphs,
    )
    if not graphs:
        raise ValueError("no graphs selected for evaluation")

    dataset = PatientGraphDataset(
        graphs,
        encoder,
        use_note_embeddings=cfg.model.use_note_embeddings,
        note_embedding_dim=cfg.model.note_embedding_dim,
        note_ground_by=cfg.model.note_ground_by,
    )
    metrics = leave_one_out_recovery(
        model,
        (dataset[idx] for idx in range(len(dataset))),
        cfg,
        device,
        cap=args.cap,
        candidate_mode=args.candidate_mode,
    )

    print(
        "[LOO] "
        f"graphs={len(dataset)} candidate_mode={args.candidate_mode} "
        f"MRR={metrics['mrr']:.3f} H@1={metrics['hits1']:.3f} "
        f"H@3={metrics['hits3']:.3f} H@10={metrics['hits10']:.3f} "
        f"over {metrics['n']} edges"
    )
    for row in metrics["per_rel"]:
        print(
            "[LOO] "
            f"{row['rel']:<22} n={row['n']:<5} C={row['C']:.0f} "
            f"MRR={row['mrr']:.3f} (chance {row['chance_mrr']:.3f}) "
            f"H@1={row['h1']:.3f} H@3={row['h3']:.3f} H@10={row['h10']:.3f}"
        )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "checkpoint": args.checkpoint,
            "candidate_mode": args.candidate_mode,
            "cap": args.cap,
            "graphs": len(dataset),
            "metrics": metrics,
        }
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return metrics


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a Graph-JEPA v6 checkpoint with leave-one-out edge recovery"
        )
    )
    add_data_args(parser)
    parser.add_argument("--checkpoint", default="checkpoints/graph_jepa_v6.pt")
    parser.add_argument("--encoder-cache", default=".cache/graph_jepa_v6/encoder")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cap", type=int, default=40000)
    parser.add_argument(
        "--candidate-mode",
        choices=["schema", "same-type"],
        default="schema",
        help="schema matches v6 ranking training; same-type mirrors the v12 script",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Optional graph.extra['split'] value to keep, e.g. train or test",
    )
    parser.add_argument("--start-graph", type=int, default=0)
    parser.add_argument("--max-graphs", type=int, default=None)
    parser.add_argument("--output", default=None, help="Optional JSON metrics path")
    return parser


def main(argv=None) -> None:
    run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()
