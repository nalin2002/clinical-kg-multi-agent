"""Fine-tune Graph-JEPA v4 from a masked-pretrained checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .training import (
    FINAL_CHECKPOINT_NAME,
    FINETUNE_STAGE,
    PRETRAIN_CHECKPOINT_NAME,
    add_data_args,
    build_checkpoint_encoder,
    build_optimizer,
    build_train_loader,
    init_wandb,
    load_model_checkpoint,
    save_checkpoint,
    train_epochs,
)


def _apply_finetune_args(args, cfg) -> None:
    cfg.train.finetune_epochs = args.epochs
    cfg.train.epochs = cfg.train.pretrain_epochs + cfg.train.finetune_epochs
    cfg.train.lr = args.lr
    cfg.train.batch_size = args.batch_size
    cfg.train.num_workers = args.num_workers
    cfg.train.revision_weight = args.revision_weight
    cfg.train.revision_mask_ratio = args.revision_mask_ratio
    cfg.train.revision_neg_per_pos = args.revision_neg_per_pos
    cfg.train.synthetic_graphs = args.synthetic_graphs
    cfg.train.synthetic_min_nodes = args.synthetic_min_nodes
    cfg.train.synthetic_max_nodes = args.synthetic_max_nodes
    if args.context_patches is not None:
        cfg.train.context_patches = args.context_patches
    if args.target_patches is not None:
        cfg.train.target_patches = args.target_patches


def finetune(args) -> Path:
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive for fine-tuning")

    device = torch.device(args.device)
    model, cfg = load_model_checkpoint(args.checkpoint, device)
    _apply_finetune_args(args, cfg)

    torch.manual_seed(cfg.train.seed)
    generator = torch.Generator().manual_seed(cfg.train.seed)

    encoder = build_checkpoint_encoder(cfg, args.encoder_cache)
    dataset, train_loader = build_train_loader(args, cfg, encoder)
    print(
        f"Loaded {len(dataset)} graphs for joint fine-tuning "
        f"(checkpoint={args.checkpoint}, encoder={cfg.encoder}, "
        f"in_dim={cfg.model.in_dim}, patches={cfg.model.num_patches}, "
        f"batch_size={cfg.train.batch_size}, "
        f"pretrain_epochs={cfg.train.pretrain_epochs}, "
        f"finetune_epochs={cfg.train.finetune_epochs})"
    )
    wandb_run = init_wandb(
        args,
        cfg,
        len(dataset),
        script="graph_jepa_v4.finetune",
        checkpoint_name=FINAL_CHECKPOINT_NAME,
    )

    optimizer = build_optimizer(model, cfg)
    train_epochs(
        model,
        optimizer,
        train_loader,
        cfg,
        stage_name=FINETUNE_STAGE,
        epochs=cfg.train.finetune_epochs,
        use_revision=True,
        device=device,
        generator=generator,
        wandb_run=wandb_run,
    )
    ckpt_path = save_checkpoint(
        model,
        cfg,
        args.out,
        checkpoint_name=FINAL_CHECKPOINT_NAME,
        config_name="config_v4.json",
    )
    if wandb_run:
        wandb_run.summary["checkpoint_path"] = str(ckpt_path)
        wandb_run.finish()
    return ckpt_path


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fine-tune Graph-JEPA v4 edge plausibility")
    add_data_args(p)
    p.add_argument("--checkpoint", default=f"checkpoints/{PRETRAIN_CHECKPOINT_NAME}",
                   help="masked-pretrained Graph-JEPA v4 checkpoint")
    p.add_argument("--out", default="checkpoints/")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--encoder-cache", default=".cache/graph_jepa_v4/encoder")
    p.add_argument("--context-patches", type=int, default=None,
                   help="override checkpoint context patch count for fine-tuning")
    p.add_argument("--target-patches", type=int, default=None,
                   help="override checkpoint target patch count for fine-tuning")
    p.add_argument("--revision-weight", type=float, default=1.0,
                   help="Weight for the joint keep/prune/add edge-revision loss")
    p.add_argument("--revision-mask-ratio", type=float, default=0.25,
                   help="Fraction of true edges hidden from message passing before scoring")
    p.add_argument("--revision-neg-per-pos", type=int, default=3,
                   help="Schema-valid false edges sampled per true edge")
    p.add_argument("--wandb", action="store_true", help="Log training metrics to Weights & Biases")
    p.add_argument("--wandb-project", default="clinical-kg-graph-jepa")
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument("--wandb-tags", nargs="*", default=None)
    p.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    return p


def main(argv=None) -> None:
    finetune(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()
