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
from graph_jepa.schema import IDX_TO_EDGE_TYPE
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
    _observed_invalid_negatives,
    _relation_balanced_bce,
    _reversed_schema_invalid_negatives,
    _schema_edge_masks,
    _unique_negative_triples,
    _with_edge_mask,
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


def _confidence_supervision_masks(
    data,
    *,
    enabled: bool,
    negative_threshold: float,
    positive_threshold: float,
    negative_threshold_by_relation: dict[str, float] | None = None,
    positive_threshold_by_relation: dict[str, float] | None = None,
    clinical_artifact_filters: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return trusted-positive, weak-negative, and ignored edge masks."""

    num_edges = int(data.edge_index.size(1))
    device = data.edge_index.device
    all_edges = torch.ones((num_edges,), dtype=torch.bool, device=device)
    no_edges = torch.zeros((num_edges,), dtype=torch.bool, device=device)
    artifact = getattr(data, "edge_clinical_artifact", None)
    if (
        not clinical_artifact_filters
        or artifact is None
        or int(artifact.numel()) != num_edges
    ):
        artifact = no_edges
    else:
        artifact = artifact.to(device=device, dtype=torch.bool)

    confidence = getattr(data, "edge_llm_confidence", None)
    if (
        not enabled
        or confidence is None
        or int(confidence.numel()) != num_edges
    ):
        return all_edges & ~artifact, artifact, no_edges

    confidence = confidence.to(device)
    scored = torch.isfinite(confidence)
    is_llm = getattr(data, "edge_is_llm", None)
    if is_llm is None or int(is_llm.numel()) != num_edges:
        is_llm = scored
    else:
        is_llm = is_llm.to(device=device, dtype=torch.bool)

    negative_by_relation = {
        str(relation).upper(): float(threshold)
        for relation, threshold in (negative_threshold_by_relation or {}).items()
    }
    positive_by_relation = {
        str(relation).upper(): float(threshold)
        for relation, threshold in (positive_threshold_by_relation or {}).items()
    }
    relation_indices = data.edge_type.detach().cpu().tolist()
    negative_values = [
        negative_by_relation.get(
            str(IDX_TO_EDGE_TYPE.get(int(relation), "")).upper(),
            float(negative_threshold),
        )
        for relation in relation_indices
    ]
    positive_values = [
        positive_by_relation.get(
            str(IDX_TO_EDGE_TYPE.get(int(relation), "")).upper(),
            float(positive_threshold),
        )
        for relation in relation_indices
    ]
    negative_cutoff = torch.tensor(
        negative_values,
        dtype=confidence.dtype,
        device=device,
    )
    positive_cutoff = torch.tensor(
        positive_values,
        dtype=confidence.dtype,
        device=device,
    )

    scored_llm = is_llm & scored
    low = scored_llm & (confidence <= negative_cutoff)
    high = scored_llm & (confidence >= positive_cutoff) & ~low
    ignored = is_llm & ~low & ~high
    trusted_positive = (~is_llm | high) & ~artifact
    weak_negative = (low | artifact) & ~trusted_positive
    ignored &= ~artifact
    return trusted_positive, weak_negative, ignored


def _clinical_artifact_mask(data, *, enabled: bool) -> torch.Tensor:
    num_edges = int(data.edge_index.size(1))
    artifact = getattr(data, "edge_clinical_artifact", None)
    if not enabled or artifact is None or int(artifact.numel()) != num_edges:
        return torch.zeros(
            (num_edges,),
            dtype=torch.bool,
            device=data.edge_index.device,
        )
    return artifact.to(device=data.edge_index.device, dtype=torch.bool)


def _edge_triples(data, mask: torch.Tensor) -> set[tuple[int, int, int]]:
    return {
        (int(source), int(target), int(relation))
        for source, target, relation in zip(
            data.edge_index[0, mask].tolist(),
            data.edge_index[1, mask].tolist(),
            data.edge_type[mask].tolist(),
        )
    }


def confidence_sanitized_graph_data(
    data,
    *,
    enabled: bool,
    negative_threshold: float,
    positive_threshold: float,
    negative_threshold_by_relation: dict[str, float] | None = None,
    positive_threshold_by_relation: dict[str, float] | None = None,
    clinical_artifact_filters: bool = False,
):
    """Keep only schema-valid, trusted edges for fine-tuning message passing."""

    valid, _invalid, unconstrained = _schema_edge_masks(
        data,
        allow_unconstrained=True,
    )
    trusted_positive, _weak_negative, _ignored = _confidence_supervision_masks(
        data,
        enabled=enabled,
        negative_threshold=negative_threshold,
        positive_threshold=positive_threshold,
        negative_threshold_by_relation=negative_threshold_by_relation,
        positive_threshold_by_relation=positive_threshold_by_relation,
        clinical_artifact_filters=clinical_artifact_filters,
    )
    keep = (valid | unconstrained) & trusted_positive
    masked = _with_edge_mask(data, keep)
    for field in (
        "edge_llm_confidence",
        "edge_is_llm",
        "edge_clinical_artifact",
    ):
        values = getattr(data, field, None)
        if values is not None and int(values.numel()) == int(keep.numel()):
            setattr(masked, field, values[keep])
    return masked


def _weighted_relation_balanced_bce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    relation: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    if bool(torch.all(weights == 1.0)):
        return _relation_balanced_bce(logits, labels, relation)

    per_example = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    relation_losses = []
    for rel in relation.unique():
        rel_mask = relation == rel
        label_losses = []
        for label in (0.0, 1.0):
            cell_mask = rel_mask & (labels == label)
            if not bool(cell_mask.any()):
                continue
            cell_weights = weights[cell_mask]
            label_losses.append((per_example[cell_mask] * cell_weights).mean())
        if label_losses:
            relation_losses.append(torch.stack(label_losses).mean())
    if not relation_losses:
        return (per_example * weights).mean()
    return torch.stack(relation_losses).mean()


class GraphJEPAv5(GraphJEPAv4):
    """v4 architecture plus hidden-edge candidate ranking."""

    def revision_loss(
        self,
        data,
        *,
        mask_ratio: float,
        neg_per_pos: int,
        llm_confidence_negatives: bool = False,
        llm_negative_threshold: float = 0.3,
        llm_positive_threshold: float = 0.8,
        llm_negative_threshold_by_relation: dict[str, float] | None = None,
        llm_positive_threshold_by_relation: dict[str, float] | None = None,
        llm_negative_weight: float = 0.2,
        clinical_artifact_filters: bool = False,
    ) -> Tuple[torch.Tensor, dict]:
        """Use trusted positives, ignored uncertain edges, and weak negatives."""

        if data.edge_index.size(1) == 0:
            zero = self.edge_head.net[0].weight.sum() * 0.0
            return zero, {
                "revision_bce": 0.0,
                "revision_pos": 0,
                "revision_neg": 0,
                "revision_schema_neg": 0,
                "revision_invalid_neg": 0,
                "revision_reverse_neg": 0,
                "revision_llm_neg": 0,
                "revision_artifact_neg": 0,
                "revision_llm_ignored": 0,
                "revision_hidden": 0,
            }

        schema_valid, schema_invalid, unconstrained = _schema_edge_masks(
            data,
            allow_unconstrained=False,
        )
        trusted_positive, weak_negative, llm_ignored = _confidence_supervision_masks(
            data,
            enabled=llm_confidence_negatives,
            negative_threshold=llm_negative_threshold,
            positive_threshold=llm_positive_threshold,
            negative_threshold_by_relation=llm_negative_threshold_by_relation,
            positive_threshold_by_relation=llm_positive_threshold_by_relation,
            clinical_artifact_filters=clinical_artifact_filters,
        )
        artifact = _clinical_artifact_mask(
            data,
            enabled=clinical_artifact_filters,
        )
        pos_mask = schema_valid & trusted_positive
        invalid_mask = schema_invalid
        weak_negative_mask = weak_negative & ~schema_invalid
        llm_weak_mask = weak_negative_mask & ~artifact
        artifact_negative_mask = weak_negative_mask & artifact
        message_mask = (schema_valid | unconstrained) & trusted_positive
        pos_idx = pos_mask.nonzero(as_tuple=False).flatten()
        device = data.edge_index.device

        hidden = torch.zeros(data.edge_index.size(1), dtype=torch.bool, device=device)
        if pos_idx.numel() > 0 and mask_ratio > 0.0:
            hidden_pos = torch.rand(pos_idx.numel(), device=device) < mask_ratio
            if not bool(hidden_pos.any()):
                hidden_pos[
                    int(torch.randint(pos_idx.numel(), (1,), device=device).item())
                ] = True
            hidden[pos_idx[hidden_pos]] = True

        keep = message_mask & ~hidden
        z = self.context_node_encoder(
            data.x,
            data.edge_index[:, keep],
            data.edge_type[keep],
        )

        if pos_idx.numel() > 0:
            pos_src = data.edge_index[0, pos_mask]
            pos_dst = data.edge_index[1, pos_mask]
            pos_rel = data.edge_type[pos_mask]
            pos_logit = self.edge_head(z[pos_src], z[pos_dst], pos_rel)
        else:
            pos_src = pos_dst = pos_rel = torch.zeros(
                (0,),
                dtype=torch.long,
                device=device,
            )
            pos_logit = z.new_zeros((0,))

        positives = {
            (int(s), int(t), int(r))
            for s, t, r in zip(pos_src.tolist(), pos_dst.tolist(), pos_rel.tolist())
        }
        pos_data = _with_edge_mask(data, pos_mask)
        schema_neg = _sample_revision_negatives(pos_data, neg_per_pos=neg_per_pos)
        invalid_neg = _observed_invalid_negatives(data, invalid_mask)
        reverse_neg = _reversed_schema_invalid_negatives(
            data,
            pos_src,
            pos_dst,
            pos_rel,
        )
        weak_neg = _observed_invalid_negatives(data, weak_negative_mask)
        weak_negative_triples = _edge_triples(data, weak_negative_mask)
        llm_negative_triples = _edge_triples(data, llm_weak_mask)
        artifact_negative_triples = _edge_triples(data, artifact_negative_mask)
        negative_exclusions = _edge_triples(
            data,
            ~(weak_negative_mask | schema_invalid),
        )
        neg_src, neg_dst, neg_rel = _unique_negative_triples(
            [schema_neg, invalid_neg, reverse_neg, weak_neg],
            positives | negative_exclusions,
        )

        if neg_src.numel() > 0:
            neg_logit = self.edge_head(z[neg_src], z[neg_dst], neg_rel)
            neg_weights = torch.tensor(
                [
                    (
                        float(llm_negative_weight)
                        if (int(s), int(t), int(r)) in weak_negative_triples
                        else 1.0
                    )
                    for s, t, r in zip(
                        neg_src.tolist(),
                        neg_dst.tolist(),
                        neg_rel.tolist(),
                    )
                ],
                dtype=neg_logit.dtype,
                device=device,
            )
        else:
            neg_logit = z.new_zeros((0,))
            neg_weights = z.new_zeros((0,))

        if pos_logit.numel() == 0 and neg_logit.numel() == 0:
            zero = self.edge_head.net[0].weight.sum() * 0.0
            return zero, {
                "revision_bce": 0.0,
                "revision_pos": 0,
                "revision_neg": 0,
                "revision_schema_neg": int(schema_neg[0].numel()),
                "revision_invalid_neg": int(invalid_neg[0].numel()),
                "revision_reverse_neg": int(reverse_neg[0].numel()),
                "revision_llm_neg": 0,
                "revision_artifact_neg": 0,
                "revision_llm_ignored": int(llm_ignored.sum().item()),
                "revision_hidden": int(hidden.sum().item()),
            }

        logits = torch.cat([pos_logit, neg_logit])
        labels = torch.cat([torch.ones_like(pos_logit), torch.zeros_like(neg_logit)])
        relations = torch.cat([pos_rel, neg_rel])
        weights = torch.cat([torch.ones_like(pos_logit), neg_weights])
        loss = _weighted_relation_balanced_bce(logits, labels, relations, weights)
        included_llm_neg = sum(
            1
            for s, t, r in zip(
                neg_src.tolist(),
                neg_dst.tolist(),
                neg_rel.tolist(),
            )
            if (int(s), int(t), int(r)) in llm_negative_triples
        )
        included_artifact_neg = sum(
            1
            for s, t, r in zip(
                neg_src.tolist(),
                neg_dst.tolist(),
                neg_rel.tolist(),
            )
            if (int(s), int(t), int(r)) in artifact_negative_triples
        )
        return loss, {
            "revision_bce": float(loss.detach()),
            "revision_pos": int(pos_logit.numel()),
            "revision_neg": int(neg_logit.numel()),
            "revision_schema_neg": int(schema_neg[0].numel()),
            "revision_invalid_neg": int(invalid_neg[0].numel()),
            "revision_reverse_neg": int(reverse_neg[0].numel()),
            "revision_llm_neg": included_llm_neg,
            "revision_artifact_neg": included_artifact_neg,
            "revision_llm_ignored": int(llm_ignored.sum().item()),
            "revision_hidden": int(hidden.sum().item()),
        }

    def candidate_ranking_loss(
        self,
        data,
        *,
        mask_ratio: float,
        neg_per_pos: int,
        max_pos: int,
        temperature: float,
        llm_confidence_negatives: bool = False,
        llm_negative_threshold: float = 0.3,
        llm_positive_threshold: float = 0.8,
        llm_negative_threshold_by_relation: dict[str, float] | None = None,
        llm_positive_threshold_by_relation: dict[str, float] | None = None,
        clinical_artifact_filters: bool = False,
    ) -> Tuple[torch.Tensor, dict]:
        """Rank hidden true edges above hard schema-valid candidate distractors."""

        if data.edge_index.size(1) == 0 or neg_per_pos <= 0:
            zero = self.edge_head.net[0].weight.sum() * 0.0
            return zero, {
                "ranking_ce": 0.0,
                "ranking_pos": 0,
                "ranking_neg": 0,
                "ranking_hidden": 0,
                "ranking_llm_excluded": 0,
                "ranking_artifact_excluded": 0,
            }

        schema_valid, _schema_invalid, unconstrained = _schema_edge_masks(
            data,
            allow_unconstrained=False,
        )
        trusted_positive, weak_negative, llm_ignored = _confidence_supervision_masks(
            data,
            enabled=llm_confidence_negatives,
            negative_threshold=llm_negative_threshold,
            positive_threshold=llm_positive_threshold,
            negative_threshold_by_relation=llm_negative_threshold_by_relation,
            positive_threshold_by_relation=llm_positive_threshold_by_relation,
            clinical_artifact_filters=clinical_artifact_filters,
        )
        artifact = _clinical_artifact_mask(
            data,
            enabled=clinical_artifact_filters,
        )
        positive_indices = (schema_valid & trusted_positive).nonzero(
            as_tuple=False,
        ).flatten()
        llm_excluded = int(((weak_negative & ~artifact) | llm_ignored).sum().item())
        artifact_excluded = int(artifact.sum().item())
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
                "ranking_llm_excluded": llm_excluded,
                "ranking_artifact_excluded": artifact_excluded,
            }

        node_type = getattr(data, "node_type", None)
        if node_type is None:
            zero = self.edge_head.net[0].weight.sum() * 0.0
            return zero, {
                "ranking_ce": 0.0,
                "ranking_pos": 0,
                "ranking_neg": 0,
                "ranking_hidden": int(hidden_indices.numel()),
                "ranking_llm_excluded": llm_excluded,
                "ranking_artifact_excluded": artifact_excluded,
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
                "ranking_llm_excluded": llm_excluded,
                "ranking_artifact_excluded": artifact_excluded,
            }

        hidden_mask = torch.zeros(
            data.edge_index.size(1),
            dtype=torch.bool,
            device=data.edge_index.device,
        )
        hidden_mask[hidden_indices] = True
        message_mask = (
            (schema_valid | unconstrained)
            & trusted_positive
            & ~hidden_mask
        )
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
            "ranking_llm_excluded": llm_excluded,
            "ranking_artifact_excluded": artifact_excluded,
        }


__all__ = [
    "EdgePlausibilityHead",
    "GraphJEPAv5",
    "GraphNodeEncoder",
    "PatchTransformer",
    "PygMessageLayer",
    "TypedMessageLayer",
    "_sample_revision_negatives",
    "confidence_sanitized_graph_data",
    "sanitized_graph_data",
    "update_ema",
    "vicreg_terms",
]
