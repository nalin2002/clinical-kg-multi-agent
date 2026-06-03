"""Graph-JEPA model: context/target encoders, predictor, masking, losses, head.

Hybrid objective:

1. **JEPA (relation-agnostic, self-supervised)**: mask a connected node region,
   encode the *visible* graph with the online context encoder, and predict the
   masked nodes' latents as produced by an EMA **target** encoder over the full
   graph. Trained with a latent regression loss + a VICReg anti-collapse term.

2. **Typed edge plausibility (lightweight, supervised)**: an
   :class:`EdgePlausibilityHead` scores ``(z_src, z_tgt, relation)`` triples,
   trained with co-occurrence positives vs. corrupted negatives.

Requires ``torch`` and ``torch_geometric``.
"""

from __future__ import annotations

from collections import deque
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GINEConv

from .config import ModelConfig


def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )


class GraphEncoder(nn.Module):
    """GNN node encoder (GINE or GAT) with a learnable mask token.

    Relation embeddings are used as edge features so the encoder is message-
    passing over typed edges, but the JEPA objective itself stays relation-
    agnostic (it predicts latents of masked nodes regardless of relation).
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.input_proj = nn.Linear(cfg.in_dim, cfg.hidden_dim)
        self.mask_token = nn.Parameter(torch.zeros(cfg.in_dim))
        nn.init.normal_(self.mask_token, std=0.02)
        self.relation_emb = nn.Embedding(cfg.num_relations, cfg.hidden_dim)

        self.convs = nn.ModuleList()
        for _ in range(cfg.num_layers):
            if cfg.conv == "gine":
                self.convs.append(
                    GINEConv(_mlp(cfg.hidden_dim, cfg.hidden_dim, cfg.hidden_dim),
                             edge_dim=cfg.hidden_dim)
                )
            elif cfg.conv == "gat":
                self.convs.append(
                    GATConv(cfg.hidden_dim, cfg.hidden_dim, heads=1,
                            edge_dim=cfg.hidden_dim)
                )
            else:
                raise ValueError(f"unknown conv: {cfg.conv!r}")
        self.norms = nn.ModuleList(nn.LayerNorm(cfg.hidden_dim)
                                   for _ in range(cfg.num_layers))
        self.out_proj = nn.Linear(cfg.hidden_dim, cfg.latent_dim)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if node_mask is not None and node_mask.any():
            x = x.clone()
            x[node_mask] = self.mask_token.to(x.dtype)

        h = self.input_proj(x)
        edge_attr = self.relation_emb(edge_type) if edge_type.numel() else \
            torch.zeros((0, self.cfg.hidden_dim), device=x.device)

        for conv, norm in zip(self.convs, self.norms):
            h_in = h
            if edge_index.numel():
                h = conv(h, edge_index, edge_attr)
            else:  # isolated nodes: convs are no-ops on empty edges
                h = h
            h = norm(h)
            h = F.relu(h)
            h = self.dropout(h)
            h = h + h_in  # residual
        return self.out_proj(h)


class Predictor(nn.Module):
    """Maps context latents at masked positions to predicted target latents."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.net = _mlp(cfg.latent_dim, cfg.predictor_hidden, cfg.latent_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class EdgePlausibilityHead(nn.Module):
    """Scores a typed edge ``(z_src, z_tgt, relation)`` -> logit."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.relation_emb = nn.Embedding(cfg.num_relations, cfg.latent_dim)
        self.net = _mlp(3 * cfg.latent_dim, cfg.latent_dim, 1)

    def forward(
        self,
        z_src: torch.Tensor,
        z_tgt: torch.Tensor,
        relation: torch.Tensor,
    ) -> torch.Tensor:
        r = self.relation_emb(relation)
        return self.net(torch.cat([z_src, z_tgt, r], dim=-1)).squeeze(-1)


@torch.no_grad()
def update_ema(online: nn.Module, target: nn.Module, decay: float) -> None:
    """In-place EMA update: ``target = decay * target + (1 - decay) * online``."""
    for p_o, p_t in zip(online.parameters(), target.parameters()):
        p_t.mul_(decay).add_(p_o.detach(), alpha=1.0 - decay)
    for b_o, b_t in zip(online.buffers(), target.buffers()):
        b_t.copy_(b_o)


def subgraph_mask(
    edge_index: torch.Tensor,
    num_nodes: int,
    mask_ratio: float,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Boolean node mask over a random *connected* region (relation-agnostic).

    Grows a region by BFS from a random seed node over the undirected adjacency
    until ~``mask_ratio * num_nodes`` nodes are covered. Falls back to a single
    random node for tiny / edgeless graphs. (Temporal-aware masking can replace
    this later without changing the training loop.)
    """
    target = max(1, int(round(mask_ratio * num_nodes)))
    mask = torch.zeros(num_nodes, dtype=torch.bool)

    # Undirected adjacency.
    adj: List[List[int]] = [[] for _ in range(num_nodes)]
    if edge_index.numel():
        src = edge_index[0].tolist()
        dst = edge_index[1].tolist()
        for s, t in zip(src, dst):
            adj[s].append(t)
            adj[t].append(s)

    def _rand_int(n: int) -> int:
        return int(torch.randint(0, n, (1,), generator=generator).item())

    seed = _rand_int(num_nodes)
    selected = {seed}
    queue: deque[int] = deque([seed])
    while queue and len(selected) < target:
        cur = queue.popleft()
        neighbors = adj[cur][:]
        # shuffle neighbours for stochastic growth
        for i in range(len(neighbors) - 1, 0, -1):
            j = _rand_int(i + 1)
            neighbors[i], neighbors[j] = neighbors[j], neighbors[i]
        for nb in neighbors:
            if nb not in selected:
                selected.add(nb)
                queue.append(nb)
                if len(selected) >= target:
                    break

    # If the connected component is smaller than target, top up randomly.
    while len(selected) < target:
        selected.add(_rand_int(num_nodes))

    mask[list(selected)] = True
    # Never mask every node (context encoder needs visible nodes).
    if mask.all():
        mask[_rand_int(num_nodes)] = False
    return mask


def vicreg_terms(z: torch.Tensor, gamma: float = 1.0, eps: float = 1e-4
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
    """VICReg variance + covariance regularisation (anti-collapse).

    Returns ``(var_loss, cov_loss)``. ``var_loss`` is a hinge keeping per-dim
    std >= ``gamma``; ``cov_loss`` pushes off-diagonal covariances toward zero.
    """
    if z.size(0) < 2:
        zero = z.sum() * 0.0
        return zero, zero
    z = z - z.mean(dim=0, keepdim=True)
    std = torch.sqrt(z.var(dim=0) + eps)
    var_loss = F.relu(gamma - std).mean()
    n, d = z.shape
    cov = (z.T @ z) / (n - 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    cov_loss = off_diag.pow(2).sum() / d
    return var_loss, cov_loss


class GraphJEPA(nn.Module):
    """Bundles the online context encoder, EMA target encoder, predictor, head."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.context_encoder = GraphEncoder(cfg)
        self.target_encoder = GraphEncoder(cfg)
        # Target starts as a copy of context and is never trained by grad.
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)
        self.predictor = Predictor(cfg)
        self.edge_head = EdgePlausibilityHead(cfg)

    # ----- encoding helpers ------------------------------------------------------
    def encode_context(self, data, node_mask=None) -> torch.Tensor:
        return self.context_encoder(data.x, data.edge_index, data.edge_type, node_mask)

    @torch.no_grad()
    def encode_target(self, data) -> torch.Tensor:
        self.target_encoder.eval()
        return self.target_encoder(data.x, data.edge_index, data.edge_type, None)

    def update_target(self, decay: float) -> None:
        update_ema(self.context_encoder, self.target_encoder, decay)

    # ----- losses ----------------------------------------------------------------
    def jepa_loss(self, data, node_mask: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        z_ctx = self.encode_context(data, node_mask)         # online, masked
        z_tgt = self.encode_target(data)                     # EMA, full graph
        pred = self.predictor(z_ctx[node_mask])
        target = z_tgt[node_mask].detach()

        inv = F.smooth_l1_loss(pred, target)
        var_loss, cov_loss = vicreg_terms(z_ctx)
        # latent std for monitoring collapse
        with torch.no_grad():
            latent_std = z_ctx.std(dim=0).mean()
        loss = inv + var_loss + 0.04 * cov_loss
        return loss, {
            "jepa_inv": float(inv.detach()),
            "jepa_var": float(var_loss.detach()),
            "jepa_cov": float(cov_loss.detach()),
            "latent_std": float(latent_std),
        }

    def edge_loss(self, data) -> Tuple[torch.Tensor, dict]:
        """Binary plausibility loss: positives are real edges, negatives are
        relation/endpoint corruptions of those edges."""
        if data.edge_index.size(1) == 0:
            zero = self.context_encoder.out_proj.bias.sum() * 0.0
            return zero, {"edge_bce": 0.0}

        z = self.encode_context(data, None)
        src = data.edge_index[0]
        dst = data.edge_index[1]
        rel = data.edge_type
        pos_logit = self.edge_head(z[src], z[dst], rel)

        # Negatives: corrupt the target endpoint to a random node.
        num_nodes = z.size(0)
        neg_dst = torch.randint(0, num_nodes, dst.shape, device=z.device)
        neg_logit = self.edge_head(z[src], z[neg_dst], rel)

        logits = torch.cat([pos_logit, neg_logit])
        labels = torch.cat([torch.ones_like(pos_logit),
                            torch.zeros_like(neg_logit)])
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        return loss, {"edge_bce": float(loss.detach())}
