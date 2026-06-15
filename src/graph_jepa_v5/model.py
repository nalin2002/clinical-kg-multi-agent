"""Graph-JEPA v5 model.

v5 inherits v4 schema-aware message passing and revision loss, then adds a
candidate-ranking objective for fine-tuning edge additions.  The ranking task
hides real schema-valid edges and asks the edge head to score each hidden true
edge above hard, schema-valid candidate distractors from the same patient graph.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F
from graph_jepa_v3.model import (
    EdgePlausibilityHead,
    GraphNodeEncoder,
    PatchTransformer,
    PygMessageLayer,
    TypedMessageLayer,
    _allowed_relation_indices,
    _allowed_target_type_indices,
    _graph_bounds,
    _sample_revision_negatives,
    update_ema,
    vicreg_terms,
)
from graph_jepa_v4.model import (
    GraphJEPAv4,
    _schema_edge_masks,
    sanitized_graph_data,
)


def _batch_bounds(data):
    batch = getattr(data, "batch", None)
    ptr = getattr(data, "ptr", None)
    if batch is None or ptr is None:
        return None, None
    return batch.detach().cpu().tolist(), ptr.detach().cpu().tolist()


def _sample_without_replacement(
    candidates: list[tuple[int, int, int]],
    *,
    limit: int,
) -> list[tuple[int, int, int]]:
    if limit <= 0 or not candidates:
        return []
    if len(candidates) <= limit:
        order = torch.randperm(len(candidates)).tolist()
    else:
        order = torch.randperm(len(candidates))[:limit].tolist()
    return [candidates[int(idx)] for idx in order]


def _candidate_distractors_for_positive(
    data,
    *,
    source: int,
    target: int,
    relation: int,
    node_type_cpu: list[int],
    existing: set[tuple[int, int, int]],
    batch_cpu,
    ptr_cpu,
    limit: int,
) -> list[tuple[int, int, int]]:
    """Build hard schema-valid distractors for one hidden positive edge."""

    relation_candidates = [
        (source, target, alt_relation)
        for alt_relation in _allowed_relation_indices(
            node_type_cpu[source],
            node_type_cpu[target],
        )
        if alt_relation != relation
    ]

    allowed_targets = _allowed_target_type_indices(node_type_cpu[source], relation)
    target_candidates: list[tuple[int, int, int]] = []
    source_candidates: list[tuple[int, int, int]] = []
    lo, hi = _graph_bounds(data, source, batch_cpu, ptr_cpu)
    for candidate in range(lo, hi):
        if candidate in (source, target):
            continue
        if node_type_cpu[candidate] in allowed_targets:
            target_candidates.append((source, candidate, relation))

        candidate_target_types = _allowed_target_type_indices(
            node_type_cpu[candidate],
            relation,
        )
        if node_type_cpu[target] in candidate_target_types:
            source_candidates.append((candidate, target, relation))

    buckets = [target_candidates, source_candidates, relation_candidates]
    out: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int, int]] = set()
    per_bucket = max(1, limit // max(1, len(buckets)))
    for bucket in buckets:
        available = [
            candidate
            for candidate in bucket
            if candidate not in existing and candidate not in seen
        ]
        take = min(per_bucket, limit - len(out))
        for candidate in _sample_without_replacement(available, limit=take):
            seen.add(candidate)
            out.append(candidate)
        if len(out) >= limit:
            return out

    if len(out) < limit:
        remaining = [
            candidate
            for bucket in buckets
            for candidate in bucket
            if candidate not in existing and candidate not in seen
        ]
        for candidate in _sample_without_replacement(
            remaining,
            limit=limit - len(out),
        ):
            seen.add(candidate)
            out.append(candidate)
    return out


def _select_hidden_positive_indices(
    positive_indices: torch.Tensor,
    *,
    mask_ratio: float,
    max_pos: int,
) -> torch.Tensor:
    if positive_indices.numel() == 0 or max_pos <= 0:
        return positive_indices.new_zeros((0,))

    target = int(round(float(mask_ratio) * int(positive_indices.numel())))
    target = max(1, target)
    target = min(target, int(positive_indices.numel()), int(max_pos))
    perm = torch.randperm(int(positive_indices.numel()), device=positive_indices.device)
    return positive_indices[perm[:target]]


class GraphJEPAv5(GraphJEPAv4):
    """v4 architecture plus hidden-edge candidate ranking."""

    def candidate_ranking_loss(
        self,
        data,
        *,
        mask_ratio: float,
        neg_per_pos: int,
        max_pos: int,
        temperature: float,
    ) -> Tuple[torch.Tensor, dict]:
        """Rank hidden true edges above hard schema-valid candidate distractors."""

        if data.edge_index.size(1) == 0 or neg_per_pos <= 0:
            zero = self.edge_head.net[0].weight.sum() * 0.0
            return zero, {
                "ranking_ce": 0.0,
                "ranking_pos": 0,
                "ranking_neg": 0,
                "ranking_hidden": 0,
            }

        schema_valid, _schema_invalid, unconstrained = _schema_edge_masks(
            data,
            allow_unconstrained=False,
        )
        positive_indices = schema_valid.nonzero(as_tuple=False).flatten()
        hidden_indices = _select_hidden_positive_indices(
            positive_indices,
            mask_ratio=mask_ratio,
            max_pos=max_pos,
        )
        if hidden_indices.numel() == 0:
            zero = self.edge_head.net[0].weight.sum() * 0.0
            return zero, {
                "ranking_ce": 0.0,
                "ranking_pos": 0,
                "ranking_neg": 0,
                "ranking_hidden": 0,
            }

        node_type = getattr(data, "node_type", None)
        if node_type is None:
            zero = self.edge_head.net[0].weight.sum() * 0.0
            return zero, {
                "ranking_ce": 0.0,
                "ranking_pos": 0,
                "ranking_neg": 0,
                "ranking_hidden": int(hidden_indices.numel()),
            }
        node_type_cpu = node_type.detach().cpu().tolist()
        src_cpu = data.edge_index[0].detach().cpu().tolist()
        dst_cpu = data.edge_index[1].detach().cpu().tolist()
        rel_cpu = data.edge_type.detach().cpu().tolist()
        existing = {
            (int(s), int(t), int(r))
            for s, t, r in zip(src_cpu, dst_cpu, rel_cpu)
        }
        batch_cpu, ptr_cpu = _batch_bounds(data)

        groups: list[tuple[int, int, int, list[tuple[int, int, int]]]] = []
        for edge_idx in hidden_indices.detach().cpu().tolist():
            edge_idx = int(edge_idx)
            source = int(src_cpu[edge_idx])
            target = int(dst_cpu[edge_idx])
            relation = int(rel_cpu[edge_idx])
            negatives = _candidate_distractors_for_positive(
                data,
                source=source,
                target=target,
                relation=relation,
                node_type_cpu=node_type_cpu,
                existing=existing,
                batch_cpu=batch_cpu,
                ptr_cpu=ptr_cpu,
                limit=neg_per_pos,
            )
            if negatives:
                groups.append((source, target, relation, negatives))

        if not groups:
            zero = self.edge_head.net[0].weight.sum() * 0.0
            return zero, {
                "ranking_ce": 0.0,
                "ranking_pos": 0,
                "ranking_neg": 0,
                "ranking_hidden": int(hidden_indices.numel()),
            }

        hidden_mask = torch.zeros(
            data.edge_index.size(1),
            dtype=torch.bool,
            device=data.edge_index.device,
        )
        hidden_mask[hidden_indices] = True
        message_mask = (schema_valid | unconstrained) & ~hidden_mask
        z = self.context_node_encoder(
            data.x,
            data.edge_index[:, message_mask],
            data.edge_type[message_mask],
        )

        temperature = max(float(temperature), 1e-6)
        losses = []
        total_neg = 0
        device = data.edge_index.device
        for source, target, relation, negatives in groups:
            pos_rel = torch.tensor([relation], dtype=torch.long, device=device)
            pos_logit = self.edge_head(
                z[source:source + 1],
                z[target:target + 1],
                pos_rel,
            )
            neg_src, neg_dst, neg_rel = zip(*negatives)
            neg_src_t = torch.tensor(neg_src, dtype=torch.long, device=device)
            neg_dst_t = torch.tensor(neg_dst, dtype=torch.long, device=device)
            neg_rel_t = torch.tensor(neg_rel, dtype=torch.long, device=device)
            neg_logit = self.edge_head(z[neg_src_t], z[neg_dst_t], neg_rel_t)
            logits = torch.cat([pos_logit, neg_logit], dim=0).unsqueeze(0)
            target_class = torch.zeros((1,), dtype=torch.long, device=device)
            losses.append(F.cross_entropy(logits / temperature, target_class))
            total_neg += len(negatives)

        loss = torch.stack(losses).mean()
        return loss, {
            "ranking_ce": float(loss.detach()),
            "ranking_pos": len(groups),
            "ranking_neg": total_neg,
            "ranking_hidden": int(hidden_indices.numel()),
        }


__all__ = [
    "EdgePlausibilityHead",
    "GraphJEPAv5",
    "GraphNodeEncoder",
    "PatchTransformer",
    "PygMessageLayer",
    "TypedMessageLayer",
    "_sample_revision_negatives",
    "sanitized_graph_data",
    "update_ema",
    "vicreg_terms",
]
