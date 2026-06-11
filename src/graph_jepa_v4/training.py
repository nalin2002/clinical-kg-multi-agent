"""Shared training helpers for Graph-JEPA v4 scripts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import List

import torch
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm

from graph_jepa.data import (
    AciBenchGraphBuilder,
    MimicGraphBuilder,
    MimicSubKGGraphBuilder,
    SyntheticGraphGenerator,
)
from graph_jepa.encoders import build_encoder
from graph_jepa.schema import PatientGraph

from .config import Config
from .data import PatientGraphDataset
from .model import GraphJEPAv4
from .patches import build_patch_data, sample_patch_task

PRETRAIN_CHECKPOINT_NAME = "graph_jepa_v4_pretrain.pt"
FINAL_CHECKPOINT_NAME = "graph_jepa_v4.pt"
PRETRAIN_STAGE = "masked_pretrain"
FINETUNE_STAGE = "joint_finetune"


def build_graphs(args, cfg: Config) -> List[PatientGraph]:
    if args.data == "synthetic":
        gen = SyntheticGraphGenerator(
            seed=cfg.train.seed,
            min_nodes=cfg.train.synthetic_min_nodes,
            max_nodes=cfg.train.synthetic_max_nodes,
        )
        return gen.generate_many(cfg.train.synthetic_graphs)
    if args.data == "mimic":
        return MimicGraphBuilder(args.mimic_root, include_notes=args.mimic_notes).build()
    if args.data == "mimic-subkgs":
        return MimicSubKGGraphBuilder(
            args.mimic_subkg_path,
            limit=args.mimic_subkg_limit,
        ).build()
    if args.data == "aci-bench":
        return AciBenchGraphBuilder(
            args.aci_kg_path,
            limit=args.aci_limit,
        ).build()
    raise ValueError(f"unknown --data: {args.data!r}")


def ema_decay(step: int, total_steps: int, cfg: Config) -> float:
    if total_steps <= 1:
        return cfg.train.ema_end
    progress = step / float(total_steps - 1)
    cosine = 0.5 * (1.0 - math.cos(math.pi * progress))
    return cfg.train.ema_start + cosine * (cfg.train.ema_end - cfg.train.ema_start)


def apply_common_train_args(args, cfg: Config) -> None:
    cfg.train.lr = args.lr
    cfg.train.batch_size = args.batch_size
    cfg.train.num_workers = args.num_workers
    cfg.train.context_patches = args.context_patches
    cfg.train.target_patches = args.target_patches
    cfg.train.synthetic_graphs = args.synthetic_graphs
    cfg.train.synthetic_min_nodes = args.synthetic_min_nodes
    cfg.train.synthetic_max_nodes = args.synthetic_max_nodes


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
    model = GraphJEPAv4(cfg.model).to(device)
    model.load_state_dict(ckpt["state_dict"])
    return model, cfg


def save_checkpoint(
    model: GraphJEPAv4,
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


def init_wandb(
    args,
    cfg: Config,
    dataset_size: int,
    *,
    script: str,
    checkpoint_name: str,
):
    if not args.wandb:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise SystemExit("wandb logging requested; install with `pip install wandb`.") from exc

    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name=args.wandb_run_name or None,
        tags=args.wandb_tags or None,
        mode=args.wandb_mode,
        config={
            "script": script,
            "checkpoint_name": checkpoint_name,
            "dataset_size": dataset_size,
            "cli": vars(args),
            "graph_jepa": cfg.to_dict(),
        },
    )


def train_epochs(
    model: GraphJEPAv4,
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
            "patch_std": 0.0,
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

            patch_data = build_patch_data(
                data,
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
                data,
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
                )
                loss = cfg.train.jepa_weight * jepa + cfg.train.revision_weight * revision
            else:
                rlog = {"revision_bce": 0.0}
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
            n += 1
            progress.set_postfix(
                loss=f"{agg['loss']/n:.4f}",
                jepa=f"{agg['jepa_inv']/n:.4f}",
                revision=f"{agg['revision_bce']/n:.4f}",
            )

        denom = max(n, 1)
        metrics = {
            "epoch": epoch,
            "train/stage_is_joint": int(use_revision),
            "train/loss": agg["loss"] / denom,
            "train/jepa_inv": agg["jepa_inv"] / denom,
            "train/jepa_var": agg["jepa_var"] / denom,
            "train/revision_bce": agg["revision_bce"] / denom,
            "train/patch_std": agg["patch_std"] / denom,
            "train/lr": cfg.train.lr,
            "train/global_step": global_step,
        }
        print(
            f"{stage_name} epoch {epoch:03d} | loss {metrics['train/loss']:.4f} "
            f"| jepa_inv {metrics['train/jepa_inv']:.4f} "
            f"| jepa_var {metrics['train/jepa_var']:.4f} "
            f"| revision_bce {metrics['train/revision_bce']:.4f} "
            f"| patch_std {metrics['train/patch_std']:.4f}"
        )
        if wandb_run:
            wandb_run.log(metrics, step=epoch)
    return global_step


def build_optimizer(model: GraphJEPAv4, cfg: Config) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )


def add_data_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--data",
        choices=["synthetic", "mimic", "mimic-subkgs", "aci-bench"],
        default="synthetic",
    )
    p.add_argument("--synthetic-graphs", type=int, default=256)
    p.add_argument("--synthetic-min-nodes", type=int, default=8)
    p.add_argument("--synthetic-max-nodes", type=int, default=28)
    p.add_argument("--mimic-root", default=None)
    p.add_argument("--mimic-notes", action="store_true")
    p.add_argument("--mimic-subkg-path", default=None,
                   help="MIMIC sub-KG JSON file or directory. Defaults to "
                        "outputs/mimic_4/sub_kgs.")
    p.add_argument("--mimic-subkg-limit", type=int, default=None,
                   help="Limit number of adapted MIMIC sub-KGs loaded for training/smoke tests.")
    p.add_argument("--aci-kg-path", default=None,
                   help="ACI-Bench KG JSON file or directory. Defaults to "
                        "outputs/aci_bench/sub_kgs, then curated EIR KGs, then smoke KGs.")
    p.add_argument("--aci-limit", type=int, default=None,
                   help="Limit number of ACI-Bench graphs loaded for training/smoke tests.")


def add_runtime_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--out", default="checkpoints/")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--context-patches", type=int, default=1)
    p.add_argument("--target-patches", type=int, default=4)
    p.add_argument("--wandb", action="store_true", help="Log training metrics to Weights & Biases")
    p.add_argument("--wandb-project", default="clinical-kg-graph-jepa")
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument("--wandb-tags", nargs="*", default=None)
    p.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
