"""Shared training helpers for Graph-JEPA v5 scripts."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm

from graph_jepa.encoders import build_encoder
from graph_jepa_v4.training import (
    add_data_args,
    add_runtime_args,
    apply_common_train_args,
    build_graphs,
    build_optimizer,
    ema_decay,
    init_wandb,
)

from .config import Config
from .data import PatientGraphDataset
from .model import GraphJEPAv5, confidence_sanitized_graph_data, sanitized_graph_data
from .patches import build_patch_data, sample_patch_task

PRETRAIN_CHECKPOINT_NAME = "graph_jepa_v5_pretrain.pt"
FINAL_CHECKPOINT_NAME = "graph_jepa_v5.pt"
PRETRAIN_STAGE = "masked_pretrain"
FINETUNE_STAGE = "candidate_rank_finetune"


def build_train_loader(args, cfg: Config, encoder):
    graphs = build_graphs(args, cfg)
    dataset = PatientGraphDataset(graphs, encoder)
    loader_gen = torch.Generator().manual_seed(cfg.train.seed)
    loader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        generator=loader_gen,
    )
    return dataset, loader


def build_checkpoint_encoder(cfg: Config, encoder_cache: str):
    if cfg.encoder in ("bge", "sapbert"):
        return build_encoder(cfg.encoder, cache_dir=encoder_cache)
    return build_encoder("mock", mock_dim=cfg.model.in_dim)


def load_model_checkpoint(checkpoint: str, device: torch.device):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = Config.from_dict(ckpt["config"])
    model = GraphJEPAv5(cfg.model).to(device)
    model.load_state_dict(ckpt["state_dict"])
    return model, cfg


def save_checkpoint(
    model: GraphJEPAv5,
    cfg: Config,
    out: str,
    *,
    checkpoint_name: str,
    config_name: str,
) -> Path:
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / checkpoint_name
    torch.save({"state_dict": model.state_dict(), "config": cfg.to_dict()}, ckpt_path)
    with open(out_dir / config_name, "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)
    print(f"Saved checkpoint to {ckpt_path}")
    return ckpt_path


def train_epochs(
    model: GraphJEPAv5,
    optimizer: torch.optim.Optimizer,
    train_loader,
    cfg: Config,
    *,
    stage_name: str,
    epochs: int,
    use_revision: bool,
    device: torch.device,
    generator: torch.Generator,
    wandb_run=None,
) -> int:
    total_steps = max(1, epochs * len(train_loader))
    global_step = 0
    for epoch in range(epochs):
        model.train()
        agg = {
            "loss": 0.0,
            "jepa_inv": 0.0,
            "jepa_var": 0.0,
            "revision_bce": 0.0,
            "ranking_ce": 0.0,
            "ranking_pos": 0.0,
            "ranking_neg": 0.0,
            "ranking_llm_excluded": 0.0,
            "ranking_artifact_excluded": 0.0,
            "patch_std": 0.0,
            "schema_dropped": 0.0,
            "llm_dropped": 0.0,
            "revision_invalid_neg": 0.0,
            "revision_llm_neg": 0.0,
            "revision_artifact_neg": 0.0,
            "revision_llm_ignored": 0.0,
        }
        n = 0
        progress = tqdm(
            train_loader,
            desc=f"{stage_name} {epoch:03d}",
            total=len(train_loader),
            unit="batch",
            dynamic_ncols=True,
            leave=False,
        )
        for data in progress:
            data = data.to(device)
            if data.num_nodes < 2:
                continue
            message_data = sanitized_graph_data(data)
            schema_dropped = int(
                data.edge_index.size(1) - message_data.edge_index.size(1)
            )
            llm_dropped = 0
            if use_revision and (
                cfg.train.llm_confidence_negatives
                or cfg.train.clinical_artifact_filters
            ):
                schema_edges = int(message_data.edge_index.size(1))
                message_data = confidence_sanitized_graph_data(
                    data,
                    enabled=cfg.train.llm_confidence_negatives,
                    negative_threshold=cfg.train.llm_negative_threshold,
                    positive_threshold=cfg.train.llm_positive_threshold,
                    negative_threshold_by_relation=(
                        cfg.train.llm_negative_threshold_by_relation
                    ),
                    positive_threshold_by_relation=(
                        cfg.train.llm_positive_threshold_by_relation
                    ),
                    clinical_artifact_filters=(
                        cfg.train.clinical_artifact_filters
                    ),
                )
                llm_dropped = schema_edges - int(message_data.edge_index.size(1))

            patch_data = build_patch_data(
                message_data,
                num_patches=cfg.model.num_patches,
                patch_pe_dim=cfg.model.patch_pe_dim,
                generator=generator,
            ).to(device)
            task = sample_patch_task(
                patch_data,
                context_patches=cfg.train.context_patches,
                target_patches=cfg.train.target_patches,
                generator=generator,
            ).to(device)

            jepa, jlog = model.jepa_loss(
                message_data,
                patch_data,
                task,
                var_weight=cfg.train.vicreg_var_weight,
                cov_weight=cfg.train.vicreg_cov_weight,
            )
            if use_revision:
                revision, rlog = model.revision_loss(
                    data,
                    mask_ratio=cfg.train.revision_mask_ratio,
                    neg_per_pos=cfg.train.revision_neg_per_pos,
                    llm_confidence_negatives=cfg.train.llm_confidence_negatives,
                    llm_negative_threshold=cfg.train.llm_negative_threshold,
                    llm_positive_threshold=cfg.train.llm_positive_threshold,
                    llm_negative_threshold_by_relation=(
                        cfg.train.llm_negative_threshold_by_relation
                    ),
                    llm_positive_threshold_by_relation=(
                        cfg.train.llm_positive_threshold_by_relation
                    ),
                    llm_negative_weight=cfg.train.llm_negative_weight,
                    clinical_artifact_filters=(
                        cfg.train.clinical_artifact_filters
                    ),
                )
                ranking, klog = model.candidate_ranking_loss(
                    data,
                    mask_ratio=cfg.train.ranking_mask_ratio,
                    neg_per_pos=cfg.train.ranking_neg_per_pos,
                    max_pos=cfg.train.ranking_max_pos,
                    temperature=cfg.train.ranking_temperature,
                    llm_confidence_negatives=cfg.train.llm_confidence_negatives,
                    llm_negative_threshold=cfg.train.llm_negative_threshold,
                    llm_positive_threshold=cfg.train.llm_positive_threshold,
                    llm_negative_threshold_by_relation=(
                        cfg.train.llm_negative_threshold_by_relation
                    ),
                    llm_positive_threshold_by_relation=(
                        cfg.train.llm_positive_threshold_by_relation
                    ),
                    clinical_artifact_filters=(
                        cfg.train.clinical_artifact_filters
                    ),
                )
                loss = (
                    cfg.train.jepa_weight * jepa
                    + cfg.train.revision_weight * revision
                    + cfg.train.ranking_weight * ranking
                )
            else:
                rlog = {"revision_bce": 0.0, "revision_invalid_neg": 0.0}
                klog = {"ranking_ce": 0.0, "ranking_pos": 0, "ranking_neg": 0}
                loss = cfg.train.jepa_weight * jepa

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            optimizer.step()
            model.update_target(ema_decay(global_step, total_steps, cfg))
            global_step += 1

            agg["loss"] += float(loss.detach())
            agg["jepa_inv"] += jlog["jepa_inv"]
            agg["jepa_var"] += jlog["jepa_var"]
            agg["patch_std"] += jlog["patch_std"]
            agg["revision_bce"] += rlog["revision_bce"]
            agg["ranking_ce"] += klog["ranking_ce"]
            agg["ranking_pos"] += klog["ranking_pos"]
            agg["ranking_neg"] += klog["ranking_neg"]
            agg["ranking_llm_excluded"] += klog.get("ranking_llm_excluded", 0.0)
            agg["ranking_artifact_excluded"] += klog.get(
                "ranking_artifact_excluded",
                0.0,
            )
            agg["schema_dropped"] += schema_dropped
            agg["llm_dropped"] += llm_dropped
            agg["revision_invalid_neg"] += rlog.get("revision_invalid_neg", 0.0)
            agg["revision_llm_neg"] += rlog.get("revision_llm_neg", 0.0)
            agg["revision_artifact_neg"] += rlog.get(
                "revision_artifact_neg",
                0.0,
            )
            agg["revision_llm_ignored"] += rlog.get("revision_llm_ignored", 0.0)
            n += 1
            progress.set_postfix(
                loss=f"{agg['loss']/n:.4f}",
                jepa=f"{agg['jepa_inv']/n:.4f}",
                revision=f"{agg['revision_bce']/n:.4f}",
                ranking=f"{agg['ranking_ce']/n:.4f}",
            )

        denom = max(n, 1)
        metrics = {
            "epoch": epoch,
            "train/stage_is_joint": int(use_revision),
            "train/loss": agg["loss"] / denom,
            "train/jepa_inv": agg["jepa_inv"] / denom,
            "train/jepa_var": agg["jepa_var"] / denom,
            "train/revision_bce": agg["revision_bce"] / denom,
            "train/ranking_ce": agg["ranking_ce"] / denom,
            "train/ranking_pos": agg["ranking_pos"] / denom,
            "train/ranking_neg": agg["ranking_neg"] / denom,
            "train/ranking_llm_excluded": agg["ranking_llm_excluded"] / denom,
            "train/ranking_artifact_excluded": (
                agg["ranking_artifact_excluded"] / denom
            ),
            "train/revision_invalid_neg": agg["revision_invalid_neg"] / denom,
            "train/revision_llm_neg": agg["revision_llm_neg"] / denom,
            "train/revision_artifact_neg": (
                agg["revision_artifact_neg"] / denom
            ),
            "train/revision_llm_ignored": agg["revision_llm_ignored"] / denom,
            "train/schema_dropped_edges": agg["schema_dropped"] / denom,
            "train/llm_dropped_edges": agg["llm_dropped"] / denom,
            "train/patch_std": agg["patch_std"] / denom,
            "train/lr": cfg.train.lr,
            "train/global_step": global_step,
        }
        print(
            f"{stage_name} epoch {epoch:03d} | loss {metrics['train/loss']:.4f} "
            f"| jepa_inv {metrics['train/jepa_inv']:.4f} "
            f"| revision_bce {metrics['train/revision_bce']:.4f} "
            f"| ranking_ce {metrics['train/ranking_ce']:.4f} "
            f"| patch_std {metrics['train/patch_std']:.4f}"
        )
        if wandb_run:
            wandb_run.log(metrics, step=epoch)
    return global_step
