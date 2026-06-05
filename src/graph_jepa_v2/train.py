"""Train Graph-JEPA v2 with subgraph-patch prediction."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import List

import torch

from graph_jepa.data import MimicGraphBuilder, SyntheticGraphGenerator
from graph_jepa.encoders import build_encoder
from graph_jepa.schema import PatientGraph

from .config import Config
from .data import PatientGraphDataset
from .model import GraphJEPAv2
from .patches import build_patch_data, sample_patch_task

CHECKPOINT_NAME = "graph_jepa_v2.pt"


def _build_graphs(args, cfg: Config) -> List[PatientGraph]:
    if args.data == "synthetic":
        gen = SyntheticGraphGenerator(
            seed=cfg.train.seed,
            min_nodes=cfg.train.synthetic_min_nodes,
            max_nodes=cfg.train.synthetic_max_nodes,
        )
        return gen.generate_many(cfg.train.synthetic_graphs)
    if args.data == "mimic":
        return MimicGraphBuilder(args.mimic_root, include_notes=args.mimic_notes).build()
    raise ValueError(f"unknown --data: {args.data!r}")


def _ema_decay(step: int, total_steps: int, cfg: Config) -> float:
    if total_steps <= 1:
        return cfg.train.ema_end
    progress = step / float(total_steps - 1)
    cosine = 0.5 * (1.0 - math.cos(math.pi * progress))
    return cfg.train.ema_start + cosine * (cfg.train.ema_end - cfg.train.ema_start)


def train(args) -> Path:
    cfg = Config()
    cfg.encoder = args.encoder
    cfg.train.epochs = args.epochs
    cfg.train.lr = args.lr
    cfg.model.num_patches = args.num_patches
    cfg.model.patch_pe_dim = args.patch_pe_dim
    cfg.train.context_patches = args.context_patches
    cfg.train.target_patches = args.target_patches
    cfg.train.synthetic_graphs = args.synthetic_graphs
    cfg.train.synthetic_min_nodes = args.synthetic_min_nodes
    cfg.train.synthetic_max_nodes = args.synthetic_max_nodes

    encoder = build_encoder(args.encoder, mock_dim=args.mock_dim, cache_dir=args.encoder_cache)
    cfg.model.in_dim = encoder.dim

    torch.manual_seed(cfg.train.seed)
    gen = torch.Generator().manual_seed(cfg.train.seed)
    device = torch.device(args.device)

    graphs = _build_graphs(args, cfg)
    dataset = PatientGraphDataset(graphs, encoder)
    print(
        f"Loaded {len(dataset)} graphs (encoder={args.encoder}, "
        f"in_dim={cfg.model.in_dim}, patches={cfg.model.num_patches})"
    )

    model = GraphJEPAv2(cfg.model).to(device)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )

    total_steps = max(1, cfg.train.epochs * len(dataset))
    global_step = 0
    order = list(range(len(dataset)))
    for epoch in range(cfg.train.epochs):
        model.train()
        agg = {
            "loss": 0.0,
            "jepa_inv": 0.0,
            "jepa_var": 0.0,
            "edge_bce": 0.0,
            "patch_std": 0.0,
        }
        n = 0
        for i in order:
            data = dataset[i].to(device)
            if data.num_nodes < 2:
                continue

            patch_data = build_patch_data(
                data,
                num_patches=cfg.model.num_patches,
                patch_pe_dim=cfg.model.patch_pe_dim,
                generator=gen,
            ).to(device)
            task = sample_patch_task(
                patch_data,
                context_patches=cfg.train.context_patches,
                target_patches=cfg.train.target_patches,
                generator=gen,
            ).to(device)

            jepa, jlog = model.jepa_loss(
                data,
                patch_data,
                task,
                var_weight=cfg.train.vicreg_var_weight,
                cov_weight=cfg.train.vicreg_cov_weight,
            )
            edge, elog = model.edge_loss(data)
            loss = cfg.train.jepa_weight * jepa + cfg.train.edge_head_weight * edge

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            opt.step()
            model.update_target(_ema_decay(global_step, total_steps, cfg))
            global_step += 1

            agg["loss"] += float(loss.detach())
            agg["jepa_inv"] += jlog["jepa_inv"]
            agg["jepa_var"] += jlog["jepa_var"]
            agg["patch_std"] += jlog["patch_std"]
            agg["edge_bce"] += elog["edge_bce"]
            n += 1

        denom = max(n, 1)
        print(
            f"epoch {epoch:03d} | loss {agg['loss']/denom:.4f} "
            f"| jepa_inv {agg['jepa_inv']/denom:.4f} "
            f"| jepa_var {agg['jepa_var']/denom:.4f} "
            f"| edge_bce {agg['edge_bce']/denom:.4f} "
            f"| patch_std {agg['patch_std']/denom:.4f}"
        )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / CHECKPOINT_NAME
    torch.save({"state_dict": model.state_dict(), "config": cfg.to_dict()}, ckpt_path)
    with open(out_dir / "config_v2.json", "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)
    print(f"Saved checkpoint to {ckpt_path}")
    return ckpt_path


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train Graph-JEPA v2")
    p.add_argument("--data", choices=["synthetic", "mimic"], default="synthetic")
    p.add_argument("--out", default="checkpoints/")
    p.add_argument("--encoder", choices=["mock", "bge", "sapbert"], default="mock")
    p.add_argument("--mock-dim", type=int, default=256)
    p.add_argument("--encoder-cache", default=".cache/graph_jepa_v2/encoder")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--device", default="cpu")
    p.add_argument("--num-patches", type=int, default=8)
    p.add_argument("--patch-pe-dim", type=int, default=8)
    p.add_argument("--context-patches", type=int, default=1)
    p.add_argument("--target-patches", type=int, default=4)
    p.add_argument("--synthetic-graphs", type=int, default=256)
    p.add_argument("--synthetic-min-nodes", type=int, default=8)
    p.add_argument("--synthetic-max-nodes", type=int, default=28)
    p.add_argument("--mimic-root", default=None)
    p.add_argument("--mimic-notes", action="store_true")
    return p


def main(argv=None) -> None:
    train(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()
