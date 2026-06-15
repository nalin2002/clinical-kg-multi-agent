"""Fine-tune Graph-JEPA v5 from a masked-pretrained checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from graph_jepa.data import canonical_relation
from graph_jepa.schema import EDGE_TYPE_TO_IDX

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


def _relation_threshold_overrides(
    values,
    current: dict[str, float],
    *,
    option: str,
) -> dict[str, float]:
    thresholds = dict(current)
    for value in values or []:
        relation, separator, raw_threshold = str(value).partition("=")
        relation = canonical_relation(relation)
        if not separator or relation not in EDGE_TYPE_TO_IDX:
            raise ValueError(
                f"{option} must use a known RELATION=VALUE pair, got {value!r}"
            )
        try:
            threshold = float(raw_threshold)
        except ValueError as exc:
            raise ValueError(
                f"{option} threshold must be numeric, got {value!r}"
            ) from exc
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                f"{option} threshold must be between 0 and 1, got {value!r}"
            )
        thresholds[relation] = threshold
    return thresholds


def _apply_finetune_args(args, cfg) -> None:
    cfg.train.finetune_epochs = args.epochs
    cfg.train.epochs = cfg.train.pretrain_epochs + cfg.train.finetune_epochs
    cfg.train.lr = args.lr
    cfg.train.batch_size = args.batch_size
    cfg.train.num_workers = args.num_workers
    cfg.train.revision_weight = args.revision_weight
    cfg.train.revision_mask_ratio = args.revision_mask_ratio
    cfg.train.revision_neg_per_pos = args.revision_neg_per_pos
    cfg.train.ranking_weight = args.ranking_weight
    cfg.train.ranking_mask_ratio = args.ranking_mask_ratio
    cfg.train.ranking_neg_per_pos = args.ranking_neg_per_pos
    cfg.train.ranking_max_pos = args.ranking_max_pos
    cfg.train.ranking_temperature = args.ranking_temperature
    cfg.train.llm_confidence_negatives = args.llm_confidence_negatives
    cfg.train.llm_negative_threshold = args.llm_negative_threshold
    cfg.train.llm_positive_threshold = args.llm_positive_threshold
    cfg.train.llm_negative_weight = args.llm_negative_weight
    cfg.train.llm_negative_threshold_by_relation = _relation_threshold_overrides(
        args.llm_negative_threshold_relation,
        cfg.train.llm_negative_threshold_by_relation,
        option="--llm-negative-threshold-relation",
    )
    cfg.train.llm_positive_threshold_by_relation = _relation_threshold_overrides(
        args.llm_positive_threshold_relation,
        cfg.train.llm_positive_threshold_by_relation,
        option="--llm-positive-threshold-relation",
    )
    cfg.train.clinical_artifact_filters = args.clinical_artifact_filters
    cfg.train.synthetic_graphs = args.synthetic_graphs
    cfg.train.synthetic_min_nodes = args.synthetic_min_nodes
    cfg.train.synthetic_max_nodes = args.synthetic_max_nodes
    if args.context_patches is not None:
        cfg.train.context_patches = args.context_patches
    if args.target_patches is not None:
        cfg.train.target_patches = args.target_patches
    if not 0.0 <= cfg.train.llm_negative_threshold <= 1.0:
        raise ValueError("--llm-negative-threshold must be between 0 and 1")
    if not 0.0 <= cfg.train.llm_positive_threshold <= 1.0:
        raise ValueError("--llm-positive-threshold must be between 0 and 1")
    if cfg.train.llm_negative_threshold >= cfg.train.llm_positive_threshold:
        raise ValueError(
            "--llm-negative-threshold must be less than --llm-positive-threshold"
        )
    relation_names = (
        set(cfg.train.llm_negative_threshold_by_relation)
        | set(cfg.train.llm_positive_threshold_by_relation)
    )
    for relation in sorted(relation_names):
        negative = cfg.train.llm_negative_threshold_by_relation.get(
            relation,
            cfg.train.llm_negative_threshold,
        )
        positive = cfg.train.llm_positive_threshold_by_relation.get(
            relation,
            cfg.train.llm_positive_threshold,
        )
        if negative >= positive:
            raise ValueError(
                f"LLM negative threshold must be less than positive threshold "
                f"for {relation}: {negative} >= {positive}"
            )
    if cfg.train.llm_negative_weight <= 0.0:
        raise ValueError("--llm-negative-weight must be positive")


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
        f"Loaded {len(dataset)} graphs for v5 candidate-ranking fine-tuning "
        f"(checkpoint={args.checkpoint}, encoder={cfg.encoder}, "
        f"in_dim={cfg.model.in_dim}, patches={cfg.model.num_patches}, "
        f"batch_size={cfg.train.batch_size}, "
        f"pretrain_epochs={cfg.train.pretrain_epochs}, "
        f"finetune_epochs={cfg.train.finetune_epochs}, "
        f"ranking_neg_per_pos={cfg.train.ranking_neg_per_pos}, "
        f"llm_confidence_negatives={cfg.train.llm_confidence_negatives}, "
        f"clinical_artifact_filters={cfg.train.clinical_artifact_filters})"
    )
    wandb_run = init_wandb(
        args,
        cfg,
        len(dataset),
        script="graph_jepa_v5.finetune",
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
        config_name="config_v5.json",
    )
    if wandb_run:
        wandb_run.summary["checkpoint_path"] = str(ckpt_path)
        wandb_run.finish()
    return ckpt_path


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fine-tune Graph-JEPA v5 candidate ranking")
    add_data_args(p)
    p.add_argument("--checkpoint", default=f"checkpoints/{PRETRAIN_CHECKPOINT_NAME}",
                   help="masked-pretrained Graph-JEPA v5 checkpoint")
    p.add_argument("--out", default="checkpoints/")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--encoder-cache", default=".cache/graph_jepa_v5/encoder")
    p.add_argument("--context-patches", type=int, default=None,
                   help="override checkpoint context patch count for fine-tuning")
    p.add_argument("--target-patches", type=int, default=None,
                   help="override checkpoint target patch count for fine-tuning")
    p.add_argument("--revision-weight", type=float, default=1.0,
                   help="Weight for the schema-aware edge-revision BCE loss")
    p.add_argument("--revision-mask-ratio", type=float, default=0.25,
                   help="Fraction of true edges hidden before BCE scoring")
    p.add_argument("--revision-neg-per-pos", type=int, default=3,
                   help="Schema-valid false edges sampled per true edge for BCE")
    p.add_argument("--ranking-weight", type=float, default=1.0,
                   help="Weight for the hidden-edge candidate-ranking loss")
    p.add_argument("--ranking-mask-ratio", type=float, default=0.25,
                   help="Fraction of schema-valid true edges hidden for ranking")
    p.add_argument("--ranking-neg-per-pos", type=int, default=8,
                   help="Hard candidate distractors ranked against each hidden edge")
    p.add_argument("--ranking-max-pos", type=int, default=256,
                   help="Maximum hidden positive edges ranked per batch")
    p.add_argument("--ranking-temperature", type=float, default=0.2,
                   help="Softmax temperature for candidate ranking")
    p.add_argument(
        "--llm-confidence-negatives",
        "--llm-tri-state-supervision",
        dest="llm_confidence_negatives",
        action="store_true",
        help=(
            "Use high-confidence LLM edges as positives, ignore uncertain edges, "
            "and use low-confidence edges as weak negatives"
        ),
    )
    p.add_argument(
        "--llm-negative-threshold",
        type=float,
        default=0.3,
        help="LLM edges at or below this fallback confidence are weak negatives",
    )
    p.add_argument(
        "--llm-positive-threshold",
        type=float,
        default=0.8,
        help="LLM edges at or above this fallback confidence are trusted positives",
    )
    p.add_argument(
        "--llm-negative-weight",
        type=float,
        default=0.2,
        help="BCE weight for weak LLM/artifact negatives",
    )
    p.add_argument(
        "--llm-negative-threshold-relation",
        action="append",
        default=[],
        metavar="RELATION=VALUE",
        help="Override the weak-negative confidence cutoff for one relation",
    )
    p.add_argument(
        "--llm-positive-threshold-relation",
        action="append",
        default=[],
        metavar="RELATION=VALUE",
        help="Override the trusted-positive confidence cutoff for one relation",
    )
    p.add_argument(
        "--clinical-artifact-filters",
        action="store_true",
        help=(
            "Treat administration supplies/carriers and cancelled microbiology "
            "edges as weak negatives"
        ),
    )
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
