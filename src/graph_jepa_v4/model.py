"""Graph-JEPA v4 model.

The architecture is inherited from v3, while the training objective is made
schema-aware: typed-invalid observed edges are removed from message passing and
trained as explicit negatives instead of positives.
"""

from __future__ import annotations

import copy
from typing import Tuple

import torch
import torch.nn.functional as F

from graph_jepa.data import RELATION_SCHEMA, canonical_relation
from graph_jepa.schema import IDX_TO_EDGE_TYPE, IDX_TO_NODE_TYPE
from graph_jepa_v3.model import (
    EdgePlausibilityHead,
    GraphJEPAv3,
    GraphNodeEncoder,
    PatchTransformer,
    PygMessageLayer,
    TypedMessageLayer,
    _sample_revision_negatives,
    update_ema,
    vicreg_terms,
)

SCHEMA_UNCONSTRAINED_RELATIONS = {"ASSOCIATED_WITH", "CO_OCCURS_WITH"}


def _is_unconstrained_relation(relation: str) -> bool:
    return canonical_relation(relation) in SCHEMA_UNCONSTRAINED_RELATIONS


def _is_schema_valid_type_triple(
    src_type: str | None,
    relation: str | None,
    tgt_type: str | None,
    *,
    allow_unconstrained: bool,
) -> bool:
    relation = canonical_relation(relation or "")
    if allow_unconstrained and _is_unconstrained_relation(relation):
        return True
    return tgt_type in RELATION_SCHEMA.get((src_type, relation), set())


def _schema_edge_masks(
    data,
    *,
    allow_unconstrained: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(valid, invalid, unconstrained)`` masks over ``data.edge_index``."""
    device = data.edge_index.device
    num_edges = int(data.edge_index.size(1))
    if num_edges == 0:
        empty = torch.zeros((0,), dtype=torch.bool, device=device)
        return empty, empty, empty

    node_type = getattr(data, "node_type", None)
    if node_type is None:
        valid = torch.ones((num_edges,), dtype=torch.bool, device=device)
        empty = torch.zeros((num_edges,), dtype=torch.bool, device=device)
        return valid, empty, empty

    src = data.edge_index[0].detach().cpu().tolist()
    dst = data.edge_index[1].detach().cpu().tolist()
    rel = data.edge_type.detach().cpu().tolist()
    node_type_cpu = node_type.detach().cpu().tolist()

    valid_values: list[bool] = []
    unconstrained_values: list[bool] = []
    for s, t, r in zip(src, dst, rel):
        src_type = IDX_TO_NODE_TYPE.get(int(node_type_cpu[int(s)]))
        tgt_type = IDX_TO_NODE_TYPE.get(int(node_type_cpu[int(t)]))
        relation = IDX_TO_EDGE_TYPE.get(int(r))
        unconstrained = bool(relation and _is_unconstrained_relation(relation))
        valid_values.append(
            _is_schema_valid_type_triple(
                src_type,
                relation,
                tgt_type,
                allow_unconstrained=allow_unconstrained,
            )
        )
        unconstrained_values.append(unconstrained)

    valid = torch.tensor(valid_values, dtype=torch.bool, device=device)
    unconstrained = torch.tensor(unconstrained_values, dtype=torch.bool, device=device)
    invalid = ~valid & ~unconstrained
    return valid, invalid, unconstrained


def _with_edge_mask(data, mask: torch.Tensor):
    masked = copy.copy(data)
    masked.edge_index = data.edge_index[:, mask]
    masked.edge_type = data.edge_type[mask]
    return masked


def sanitized_graph_data(data):
    """Return a view with typed-invalid edges removed from message passing."""
    valid, _invalid, unconstrained = _schema_edge_masks(
        data,
        allow_unconstrained=True,
    )
    return _with_edge_mask(data, valid | unconstrained)


def _triples_to_tensors(
    triples: list[tuple[int, int, int]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not triples:
        empty = torch.zeros((0,), dtype=torch.long, device=device)
        return empty, empty, empty
    src, dst, rel = zip(*triples)
    return (
        torch.tensor(src, dtype=torch.long, device=device),
        torch.tensor(dst, dtype=torch.long, device=device),
        torch.tensor(rel, dtype=torch.long, device=device),
    )


def _unique_negative_triples(
    parts: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    positives: set[tuple[int, int, int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    triples: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int, int]] = set()
    device = parts[0][0].device if parts else torch.device("cpu")
    for src_t, dst_t, rel_t in parts:
        device = src_t.device
        for s, t, r in zip(src_t.tolist(), dst_t.tolist(), rel_t.tolist()):
            triple = (int(s), int(t), int(r))
            if triple in positives or triple in seen:
                continue
            seen.add(triple)
            triples.append(triple)
    return _triples_to_tensors(triples, device)


def _observed_invalid_negatives(
    data,
    invalid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if invalid_mask.numel() == 0 or not bool(invalid_mask.any()):
        empty = torch.zeros((0,), dtype=torch.long, device=data.edge_index.device)
        return empty, empty, empty
    return (
        data.edge_index[0, invalid_mask],
        data.edge_index[1, invalid_mask],
        data.edge_type[invalid_mask],
    )


def _is_schema_valid_index_triple(data, s: int, t: int, r: int) -> bool:
    node_type = getattr(data, "node_type", None)
    if node_type is None:
        return True
    src_type = IDX_TO_NODE_TYPE.get(int(node_type[int(s)].item()))
    tgt_type = IDX_TO_NODE_TYPE.get(int(node_type[int(t)].item()))
    relation = IDX_TO_EDGE_TYPE.get(int(r))
    return _is_schema_valid_type_triple(
        src_type,
        relation,
        tgt_type,
        allow_unconstrained=False,
    )


def _reversed_schema_invalid_negatives(
    data,
    pos_src: torch.Tensor,
    pos_dst: torch.Tensor,
    pos_rel: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    triples: list[tuple[int, int, int]] = []
    for s, t, r in zip(pos_src.tolist(), pos_dst.tolist(), pos_rel.tolist()):
        s = int(s)
        t = int(t)
        r = int(r)
        if s == t:
            continue
        if not _is_schema_valid_index_triple(data, t, s, r):
            triples.append((t, s, r))
    return _triples_to_tensors(triples, data.edge_index.device)


def _relation_balanced_bce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    relation: torch.Tensor,
) -> torch.Tensor:
    per_example = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    relation_losses = []
    for rel in relation.unique():
        rel_mask = relation == rel
        label_losses = []
        for label in (0.0, 1.0):
            label_mask = rel_mask & (labels == label)
            if bool(label_mask.any()):
                label_losses.append(per_example[label_mask].mean())
        if label_losses:
            relation_losses.append(torch.stack(label_losses).mean())
    if not relation_losses:
        return per_example.mean()
    return torch.stack(relation_losses).mean()


class GraphJEPAv4(GraphJEPAv3):
    """v3 architecture with schema-aware v4 training losses."""

    def revision_loss(
        self,
        data,
        *,
        mask_ratio: float,
        neg_per_pos: int,
    ) -> Tuple[torch.Tensor, dict]:
        """Schema-aware joint graph-revision loss.

        Positives are only schema-valid observed typed edges. Invalid observed
        typed edges are explicit negatives. Message passing uses sanitized graph
        structure so the encoder does not propagate through known-bad typed
        directions.
        """
        if data.edge_index.size(1) == 0:
            zero = self.edge_head.net[0].weight.sum() * 0.0
            return zero, {
                "revision_bce": 0.0,
                "revision_pos": 0,
                "revision_neg": 0,
                "revision_schema_neg": 0,
                "revision_invalid_neg": 0,
                "revision_reverse_neg": 0,
                "revision_hidden": 0,
            }

        schema_valid, schema_invalid, unconstrained = _schema_edge_masks(
            data,
            allow_unconstrained=False,
        )
        message_mask = schema_valid | unconstrained
        pos_mask = schema_valid
        pos_idx = pos_mask.nonzero(as_tuple=False).flatten()
        device = data.edge_index.device

        hidden = torch.zeros(data.edge_index.size(1), dtype=torch.bool, device=device)
        if pos_idx.numel() > 0 and mask_ratio > 0.0:
            hidden_pos = torch.rand(pos_idx.numel(), device=device) < mask_ratio
            if not bool(hidden_pos.any()):
                hidden_pos[int(torch.randint(pos_idx.numel(), (1,), device=device).item())] = True
            hidden[pos_idx[hidden_pos]] = True

        keep = message_mask & ~hidden
        z = self.context_node_encoder(data.x, data.edge_index[:, keep], data.edge_type[keep])

        if pos_idx.numel() > 0:
            pos_src = data.edge_index[0, pos_mask]
            pos_dst = data.edge_index[1, pos_mask]
            pos_rel = data.edge_type[pos_mask]
            pos_logit = self.edge_head(z[pos_src], z[pos_dst], pos_rel)
        else:
            pos_src = pos_dst = pos_rel = torch.zeros((0,), dtype=torch.long, device=device)
            pos_logit = z.new_zeros((0,))

        positives = {
            (int(s), int(t), int(r))
            for s, t, r in zip(pos_src.tolist(), pos_dst.tolist(), pos_rel.tolist())
        }
        pos_data = _with_edge_mask(data, pos_mask)
        schema_neg = _sample_revision_negatives(pos_data, neg_per_pos=neg_per_pos)
        invalid_neg = _observed_invalid_negatives(data, schema_invalid)
        reverse_neg = _reversed_schema_invalid_negatives(data, pos_src, pos_dst, pos_rel)
        neg_src, neg_dst, neg_rel = _unique_negative_triples(
            [schema_neg, invalid_neg, reverse_neg],
            positives,
        )

        if neg_src.numel() > 0:
            neg_logit = self.edge_head(z[neg_src], z[neg_dst], neg_rel)
        else:
            neg_logit = z.new_zeros((0,))

        if pos_logit.numel() == 0 and neg_logit.numel() == 0:
            zero = self.edge_head.net[0].weight.sum() * 0.0
            return zero, {
                "revision_bce": 0.0,
                "revision_pos": 0,
                "revision_neg": 0,
                "revision_schema_neg": 0,
                "revision_invalid_neg": int(invalid_neg[0].numel()),
                "revision_reverse_neg": int(reverse_neg[0].numel()),
                "revision_hidden": int(hidden.sum().item()),
            }

        logits = torch.cat([pos_logit, neg_logit])
        labels = torch.cat([torch.ones_like(pos_logit), torch.zeros_like(neg_logit)])
        relations = torch.cat([pos_rel, neg_rel])
        loss = _relation_balanced_bce(logits, labels, relations)
        return loss, {
            "revision_bce": float(loss.detach()),
            "revision_pos": int(pos_logit.numel()),
            "revision_neg": int(neg_logit.numel()),
            "revision_schema_neg": int(schema_neg[0].numel()),
            "revision_invalid_neg": int(invalid_neg[0].numel()),
            "revision_reverse_neg": int(reverse_neg[0].numel()),
            "revision_hidden": int(hidden.sum().item()),
        }


__all__ = [
    "EdgePlausibilityHead",
    "GraphJEPAv4",
    "GraphNodeEncoder",
    "PatchTransformer",
    "PygMessageLayer",
    "TypedMessageLayer",
    "_sample_revision_negatives",
    "_schema_edge_masks",
    "sanitized_graph_data",
    "update_ema",
    "vicreg_terms",
]
