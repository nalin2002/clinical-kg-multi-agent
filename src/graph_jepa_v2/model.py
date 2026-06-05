"""Patch/subgraph Graph-JEPA model for clinical KG refinement."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .patches import PatchData, PatchTask, pool_nodes_to_patches, visible_mask


def _mlp(
    in_dim: int,
    hidden: int,
    out_dim: int,
    *,
    activation: type[nn.Module] = nn.GELU,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        activation(),
        nn.Linear(hidden, out_dim),
    )


class TypedMessageLayer(nn.Module):
    """Pure-torch typed message passing layer.

    Each directed KG edge sends a typed message from source to target and a
    separate reverse message from target to source.  That keeps relation
    direction visible while still letting evidence flow both ways in small KGs.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.relation_emb = nn.Embedding(cfg.num_relations, cfg.hidden_dim)
        self.msg = _mlp(2 * cfg.hidden_dim, cfg.hidden_dim, cfg.hidden_dim)
        self.rev_msg = _mlp(2 * cfg.hidden_dim, cfg.hidden_dim, cfg.hidden_dim)
        self.self_lin = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        if edge_index.numel() == 0:
            return self.self_lin(h)

        src = edge_index[0]
        dst = edge_index[1]
        r = self.relation_emb(edge_type)
        out = h.new_zeros(h.shape)
        deg = h.new_zeros((h.size(0), 1))

        fwd = self.msg(torch.cat([h[src], r], dim=-1))
        out.index_add_(0, dst, fwd)
        deg.index_add_(0, dst, h.new_ones((dst.numel(), 1)))

        rev = self.rev_msg(torch.cat([h[dst], r], dim=-1))
        out.index_add_(0, src, rev)
        deg.index_add_(0, src, h.new_ones((src.numel(), 1)))

        return self.self_lin(h) + out / deg.clamp_min(1.0)


class GraphNodeEncoder(nn.Module):
    """Typed-edge GNN used before patch pooling."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.input_proj = nn.Linear(cfg.in_dim, cfg.hidden_dim)
        if cfg.conv not in {"gine", "gat"}:
            raise ValueError(f"unknown conv: {cfg.conv!r}")
        self.layers = nn.ModuleList(
            TypedMessageLayer(cfg) for _ in range(cfg.num_gnn_layers)
        )
        self.norms = nn.ModuleList(
            nn.LayerNorm(cfg.hidden_dim) for _ in range(cfg.num_gnn_layers)
        )
        self.out_proj = nn.Linear(cfg.hidden_dim, cfg.latent_dim)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        h = self.input_proj(x)
        for layer, norm in zip(self.layers, self.norms):
            residual = h
            h = layer(h, edge_index, edge_type)
            h = norm(h)
            h = F.gelu(h)
            h = self.dropout(h)
            h = h + residual
        return self.out_proj(h)


class PatchTransformer(nn.Module):
    """Small transformer over patch tokens.

    Masked patches keep their positional signal but use a learned content token.
    Visible patches act as keys/values for context prediction.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(cfg.latent_dim))
        nn.init.normal_(self.mask_token, std=0.02)
        ff_dim = int(cfg.latent_dim * cfg.patch_mlp_ratio)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.latent_dim,
            nhead=cfg.patch_heads,
            dim_feedforward=ff_dim,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=cfg.patch_layers,
            enable_nested_tensor=False,
        )
        self.out_norm = nn.LayerNorm(cfg.latent_dim)

    def forward(
        self,
        content: torch.Tensor,
        pos: torch.Tensor,
        visible: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        tokens = content + pos
        key_padding_mask = None
        if visible is not None:
            if not bool(visible.any()):
                visible = torch.ones_like(visible)
            masked_content = self.mask_token.to(content.dtype).expand_as(content)
            tokens = torch.where(visible[:, None], tokens, masked_content + pos)
            key_padding_mask = (~visible).unsqueeze(0)
        out = self.encoder(tokens.unsqueeze(0), src_key_padding_mask=key_padding_mask)
        return self.out_norm(out.squeeze(0))


class EdgePlausibilityHead(nn.Module):
    """Scores a typed edge ``(z_src, z_tgt, relation)``."""

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
    for p_o, p_t in zip(online.parameters(), target.parameters()):
        p_t.mul_(decay).add_(p_o.detach(), alpha=1.0 - decay)
    for b_o, b_t in zip(online.buffers(), target.buffers()):
        b_t.copy_(b_o)


def vicreg_terms(z: torch.Tensor, gamma: float = 1.0, eps: float = 1e-4
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
    if z.size(0) < 2:
        zero = z.sum() * 0.0
        return zero, zero
    z = z - z.mean(dim=0, keepdim=True)
    std = torch.sqrt(z.var(dim=0) + eps)
    var_loss = F.relu(gamma - std).mean()
    cov = (z.T @ z) / max(1, z.size(0) - 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    cov_loss = off_diag.pow(2).sum() / z.size(1)
    return var_loss, cov_loss


class GraphJEPAv2(nn.Module):
    """Patch-based JEPA with a downstream typed edge head."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.context_node_encoder = GraphNodeEncoder(cfg)
        self.target_node_encoder = GraphNodeEncoder(cfg)
        self.context_patch_pos = _mlp(cfg.patch_pe_dim, cfg.latent_dim, cfg.latent_dim)
        self.target_patch_pos = _mlp(cfg.patch_pe_dim, cfg.latent_dim, cfg.latent_dim)
        self.context_patch_encoder = PatchTransformer(cfg)
        self.target_patch_encoder = PatchTransformer(cfg)

        self.target_node_encoder.load_state_dict(self.context_node_encoder.state_dict())
        self.target_patch_pos.load_state_dict(self.context_patch_pos.state_dict())
        self.target_patch_encoder.load_state_dict(self.context_patch_encoder.state_dict())
        for module in (
            self.target_node_encoder,
            self.target_patch_pos,
            self.target_patch_encoder,
        ):
            for p in module.parameters():
                p.requires_grad_(False)

        self.predictor = _mlp(2 * cfg.latent_dim, cfg.predictor_hidden, cfg.latent_dim)
        self.edge_head = EdgePlausibilityHead(cfg)

    def update_target(self, decay: float) -> None:
        update_ema(self.context_node_encoder, self.target_node_encoder, decay)
        update_ema(self.context_patch_pos, self.target_patch_pos, decay)
        update_ema(self.context_patch_encoder, self.target_patch_encoder, decay)

    def encode_nodes(self, data) -> torch.Tensor:
        return self.context_node_encoder(data.x, data.edge_index, data.edge_type)

    @torch.no_grad()
    def encode_target_nodes(self, data) -> torch.Tensor:
        self.target_node_encoder.eval()
        return self.target_node_encoder(data.x, data.edge_index, data.edge_type)

    def _context_patches(
        self,
        data,
        patch_data: PatchData,
        visible: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        node_z = self.encode_nodes(data)
        content = pool_nodes_to_patches(node_z, patch_data)
        pos = self.context_patch_pos(patch_data.patch_pos.to(node_z.device))
        patches = self.context_patch_encoder(content, pos, visible.to(node_z.device))
        return patches, pos

    @torch.no_grad()
    def _target_patches(self, data, patch_data: PatchData) -> torch.Tensor:
        self.target_node_encoder.eval()
        self.target_patch_pos.eval()
        self.target_patch_encoder.eval()
        node_z = self.encode_target_nodes(data)
        content = pool_nodes_to_patches(node_z, patch_data)
        pos = self.target_patch_pos(patch_data.patch_pos.to(node_z.device))
        return self.target_patch_encoder(content, pos, None)

    def jepa_loss(
        self,
        data,
        patch_data: PatchData,
        task: PatchTask,
        *,
        var_weight: float,
        cov_weight: float,
    ) -> Tuple[torch.Tensor, dict]:
        if patch_data.num_patches < 2 or task.target_idx.numel() == 0:
            zero = self.predictor[0].weight.sum() * 0.0
            return zero, {
                "jepa_inv": 0.0,
                "jepa_var": 0.0,
                "jepa_cov": 0.0,
                "patch_std": 0.0,
            }

        visible = visible_mask(patch_data.num_patches, task.context_idx).to(data.x.device)
        ctx, pos = self._context_patches(data, patch_data, visible)
        tgt = self._target_patches(data, patch_data)
        target_idx = task.target_idx.to(data.x.device)

        pred_in = torch.cat([ctx[target_idx], pos[target_idx]], dim=-1)
        pred = self.predictor(pred_in)
        target = tgt[target_idx].detach()
        inv = F.smooth_l1_loss(pred, target)
        var_loss, cov_loss = vicreg_terms(ctx[visible])
        loss = inv + var_weight * var_loss + cov_weight * cov_loss
        with torch.no_grad():
            patch_std = ctx.std(dim=0).mean() if ctx.size(0) > 1 else ctx.std()
        return loss, {
            "jepa_inv": float(inv.detach()),
            "jepa_var": float(var_loss.detach()),
            "jepa_cov": float(cov_loss.detach()),
            "patch_std": float(patch_std.detach()),
        }

    def edge_loss(self, data) -> Tuple[torch.Tensor, dict]:
        if data.edge_index.size(1) == 0:
            zero = self.edge_head.net[0].weight.sum() * 0.0
            return zero, {"edge_bce": 0.0}

        z = self.encode_nodes(data)
        src = data.edge_index[0]
        dst = data.edge_index[1]
        rel = data.edge_type
        pos_logit = self.edge_head(z[src], z[dst], rel)

        num_nodes = z.size(0)
        neg_dst = torch.randint(0, num_nodes, dst.shape, device=z.device)
        neg_logit = self.edge_head(z[src], z[neg_dst], rel)

        logits = torch.cat([pos_logit, neg_logit])
        labels = torch.cat([torch.ones_like(pos_logit), torch.zeros_like(neg_logit)])
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        return loss, {"edge_bce": float(loss.detach())}

    @torch.no_grad()
    def patch_prediction_energy(
        self,
        data,
        patch_data: PatchData,
        target_patch_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Return normalized prediction energy for target patches."""
        device = data.x.device
        target_patch_idx = target_patch_idx.to(device)
        visible = torch.ones(patch_data.num_patches, dtype=torch.bool, device=device)
        visible[target_patch_idx] = False
        if not bool(visible.any()):
            visible[target_patch_idx] = True

        ctx, pos = self._context_patches(data, patch_data, visible)
        tgt = self._target_patches(data, patch_data)
        pred = self.predictor(torch.cat([ctx[target_patch_idx], pos[target_patch_idx]], dim=-1))
        energy = torch.norm(pred - tgt[target_patch_idx], dim=-1)
        return energy / math.sqrt(self.cfg.latent_dim)

    @torch.no_grad()
    def encode_graph(self, data, patch_data: PatchData) -> torch.Tensor:
        patches = self._target_patches(data, patch_data)
        if patches.numel() == 0:
            return data.x.new_zeros((self.cfg.latent_dim,))
        return patches.mean(dim=0)
