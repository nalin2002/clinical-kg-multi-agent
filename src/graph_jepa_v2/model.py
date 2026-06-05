"""Patch/subgraph Graph-JEPA model for clinical KG refinement."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .patches import PatchData, PatchTask, pool_nodes_to_patches, visible_mask

try:
    from torch_geometric.nn import GATConv, GINEConv
except ImportError:  # pragma: no cover - exercised only without PyG installed.
    GATConv = None
    GINEConv = None


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


class PygMessageLayer(nn.Module):
    """PyTorch Geometric typed message passing layer."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if GINEConv is None or GATConv is None:
            raise ImportError(
                "Graph-JEPA v2 was configured with gnn_backend='pyg', but "
                "torch_geometric is not importable. Install torch-geometric or "
                "run with --gnn-backend torch."
            )
        self.cfg = cfg
        self.relation_emb = nn.Embedding(cfg.num_relations, cfg.hidden_dim)
        if cfg.conv == "gine":
            self.conv = GINEConv(
                _mlp(cfg.hidden_dim, cfg.hidden_dim, cfg.hidden_dim),
                edge_dim=cfg.hidden_dim,
            )
        elif cfg.conv == "gat":
            self.conv = GATConv(
                cfg.hidden_dim,
                cfg.hidden_dim,
                heads=1,
                edge_dim=cfg.hidden_dim,
                add_self_loops=False,
            )
        else:
            raise ValueError(f"unknown conv: {cfg.conv!r}")

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        if edge_index.numel() == 0:
            return h
        edge_attr = self.relation_emb(edge_type)
        return self.conv(h, edge_index, edge_attr)


class GraphNodeEncoder(nn.Module):
    """Typed-edge GNN used before patch pooling."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.input_proj = nn.Linear(cfg.in_dim, cfg.hidden_dim)
        if cfg.conv not in {"gine", "gat"}:
            raise ValueError(f"unknown conv: {cfg.conv!r}")
        if cfg.gnn_backend not in {"pyg", "torch"}:
            raise ValueError(f"unknown gnn_backend: {cfg.gnn_backend!r}")
        layer_cls = PygMessageLayer if cfg.gnn_backend == "pyg" else TypedMessageLayer
        self.layers = nn.ModuleList(
            layer_cls(cfg) for _ in range(cfg.num_gnn_layers)
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
        valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        squeeze = content.dim() == 2
        if squeeze:
            content = content.unsqueeze(0)
            pos = pos.unsqueeze(0)
            if visible is not None:
                visible = visible.unsqueeze(0)
            if valid is not None:
                valid = valid.unsqueeze(0)

        tokens = content + pos
        key_padding_mask = None
        if visible is not None:
            if valid is None:
                valid = torch.ones_like(visible)
            visible = visible & valid
            if visible.size(1) > 0:
                no_visible = ~visible.any(dim=1)
                if bool(no_visible.any()):
                    visible = visible.clone()
                    visible[no_visible, 0] = True
            masked_content = self.mask_token.to(content.dtype).expand_as(content)
            tokens = torch.where(visible.unsqueeze(-1), tokens, masked_content + pos)
            key_padding_mask = ~visible
        elif valid is not None:
            key_padding_mask = ~valid
            if key_padding_mask.size(1) > 0:
                no_valid = key_padding_mask.all(dim=1)
                if bool(no_valid.any()):
                    key_padding_mask = key_padding_mask.clone()
                    key_padding_mask[no_valid, 0] = False

        out = self.encoder(tokens, src_key_padding_mask=key_padding_mask)
        out = self.out_norm(out)
        return out.squeeze(0) if squeeze else out


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


def _pack_patches(
    values: torch.Tensor,
    patch_data: PatchData,
    fill_value: float | bool = 0.0,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if patch_data.patch_ptr is None:
        return values, None

    ptr = patch_data.patch_ptr.to(values.device)
    sizes = ptr[1:] - ptr[:-1]
    batch_size = max(0, int(ptr.numel()) - 1)
    max_patches = int(sizes.max().item()) if sizes.numel() else 0
    out_shape = (batch_size, max_patches) + tuple(values.shape[1:])
    if values.dtype == torch.bool:
        out = torch.full(
            out_shape,
            bool(fill_value),
            dtype=values.dtype,
            device=values.device,
        )
    else:
        out = values.new_full(out_shape, fill_value)
    valid = torch.zeros((batch_size, max_patches), dtype=torch.bool, device=values.device)

    for graph_idx in range(batch_size):
        lo = int(ptr[graph_idx].item())
        hi = int(ptr[graph_idx + 1].item())
        length = hi - lo
        if length <= 0:
            continue
        out[graph_idx, :length] = values[lo:hi]
        valid[graph_idx, :length] = True
    return out, valid


def _unpack_patches(values: torch.Tensor, patch_data: PatchData) -> torch.Tensor:
    if patch_data.patch_ptr is None:
        return values

    ptr = patch_data.patch_ptr.to(values.device)
    chunks = []
    for graph_idx in range(max(0, int(ptr.numel()) - 1)):
        lo = int(ptr[graph_idx].item())
        hi = int(ptr[graph_idx + 1].item())
        length = hi - lo
        if length > 0:
            chunks.append(values[graph_idx, :length])
    if not chunks:
        return values.new_zeros((0,) + tuple(values.shape[2:]))
    return torch.cat(chunks, dim=0)


def _sample_clean_negative_dst(data) -> tuple[torch.Tensor, torch.Tensor]:
    edge_index = data.edge_index
    device = edge_index.device
    src_cpu = edge_index[0].detach().cpu().tolist()
    dst_cpu = edge_index[1].detach().cpu().tolist()
    existing: dict[int, set[int]] = {}
    for s, t in zip(src_cpu, dst_cpu):
        existing.setdefault(int(s), set()).add(int(t))

    batch = getattr(data, "batch", None)
    ptr = getattr(data, "ptr", None)
    if batch is not None and ptr is not None:
        batch_cpu = batch.detach().cpu().tolist()
        ptr_cpu = ptr.detach().cpu().tolist()
    else:
        batch_cpu = None
        ptr_cpu = None

    neg_dst = []
    valid = []
    for s in src_cpu:
        s = int(s)
        if batch_cpu is not None and ptr_cpu is not None:
            graph_id = int(batch_cpu[s])
            lo = int(ptr_cpu[graph_id])
            hi = int(ptr_cpu[graph_id + 1])
        else:
            lo = 0
            hi = int(data.num_nodes)

        blocked = existing.get(s, set())
        candidates = [node for node in range(lo, hi) if node not in blocked]
        if not candidates:
            neg_dst.append(lo)
            valid.append(False)
            continue
        idx = int(torch.randint(len(candidates), (1,)).item())
        neg_dst.append(candidates[idx])
        valid.append(True)

    return (
        torch.tensor(neg_dst, dtype=torch.long, device=device),
        torch.tensor(valid, dtype=torch.bool, device=device),
    )


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
        visible = visible.to(node_z.device)
        if patch_data.patch_ptr is not None:
            content, valid = _pack_patches(content, patch_data)
            pos_dense, _ = _pack_patches(pos, patch_data)
            visible, _ = _pack_patches(visible, patch_data, fill_value=False)
            patches = self.context_patch_encoder(content, pos_dense, visible, valid)
            patches = _unpack_patches(patches, patch_data)
        else:
            patches = self.context_patch_encoder(content, pos, visible)
        return patches, pos

    @torch.no_grad()
    def _target_patches(self, data, patch_data: PatchData) -> torch.Tensor:
        self.target_node_encoder.eval()
        self.target_patch_pos.eval()
        self.target_patch_encoder.eval()
        node_z = self.encode_target_nodes(data)
        content = pool_nodes_to_patches(node_z, patch_data)
        pos = self.target_patch_pos(patch_data.patch_pos.to(node_z.device))
        if patch_data.patch_ptr is not None:
            content, valid = _pack_patches(content, patch_data)
            pos, _ = _pack_patches(pos, patch_data)
            patches = self.target_patch_encoder(content, pos, None, valid)
            return _unpack_patches(patches, patch_data)
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
        neg_dst, valid_neg = _sample_clean_negative_dst(data)
        if not bool(valid_neg.any()):
            zero = self.edge_head.net[0].weight.sum() * 0.0
            return zero, {"edge_bce": 0.0}

        src = src[valid_neg]
        dst = dst[valid_neg]
        rel = rel[valid_neg]
        neg_dst = neg_dst[valid_neg]
        pos_logit = self.edge_head(z[src], z[dst], rel)
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
