"""Training loop for Graph-JEPA.

EMA target + AdamW. The hybrid loss combines the JEPA latent objective (+VICReg)
with the typed edge-plausibility head. Trains per-graph (graphs are small), so
the connected-region masking stays trivial.

CLI::

    python -m graph_jepa.train --data synthetic --out checkpoints/
    python -m graph_jepa.train --data mimic --mimic-root /path/to/mimic --out checkpoints/
    python -m graph_jepa.train --data synthetic --encoder bge --out checkpoints/

Requires torch + torch_geometric (and FlagEmbedding only for --encoder bge).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import torch

from .config import Config
from .data import MimicGraphBuilder, PatientGraphDataset, SyntheticGraphGenerator
from .encoders import build_encoder
from .model import GraphJEPA, subgraph_mask
from .schema import PatientGraph

CHECKPOINT_NAME = "graph_jepa.pt"


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


def train(args) -> Path:
    cfg = Config()
    cfg.encoder = args.encoder
    cfg.train.epochs = args.epochs
    cfg.train.lr = args.lr
    encoder = build_encoder(
        args.encoder, mock_dim=args.mock_dim, cache_dir=args.bge_cache
    )
    cfg.model.in_dim = encoder.dim

    torch.manual_seed(cfg.train.seed)
    gen = torch.Generator().manual_seed(cfg.train.seed)
    device = torch.device(args.device)

    graphs = _build_graphs(args, cfg)
    dataset = PatientGraphDataset(graphs, encoder)
    print(f"Loaded {len(dataset)} graphs (encoder={args.encoder}, "
          f"in_dim={cfg.model.in_dim})")

    model = GraphJEPA(cfg.model).to(device)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.train.lr, weight_decay=cfg.train.weight_decay,
    )

    model.train()
    order = list(range(len(dataset)))
    for epoch in range(cfg.train.epochs):
        torch.manual_seed(cfg.train.seed + epoch)
        agg = {"loss": 0.0, "jepa_inv": 0.0, "jepa_var": 0.0,
               "edge_bce": 0.0, "latent_std": 0.0}
        n = 0
        for i in order:
            data = dataset[i].to(device)
            if data.num_nodes < 2:
                continue
            mask = subgraph_mask(
                data.edge_index, data.num_nodes, cfg.train.mask_ratio, gen
            ).to(device)

            jepa, jlog = model.jepa_loss(data, mask)
            edge, elog = model.edge_loss(data)
            loss = (cfg.train.jepa_weight * jepa
                    + cfg.train.edge_head_weight * edge)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            model.update_target(cfg.model.ema_decay)

            agg["loss"] += float(loss.detach())
            agg["jepa_inv"] += jlog["jepa_inv"]
            agg["jepa_var"] += jlog["jepa_var"]
            agg["latent_std"] += jlog["latent_std"]
            agg["edge_bce"] += elog["edge_bce"]
            n += 1

        n = max(n, 1)
        print(f"epoch {epoch:03d} | loss {agg['loss']/n:.4f} "
              f"| jepa_inv {agg['jepa_inv']/n:.4f} "
              f"| jepa_var {agg['jepa_var']/n:.4f} "
              f"| edge_bce {agg['edge_bce']/n:.4f} "
              f"| latent_std {agg['latent_std']/n:.4f}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / CHECKPOINT_NAME
    torch.save({"state_dict": model.state_dict(), "config": cfg.to_dict()},
               ckpt_path)
    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)
    print(f"Saved checkpoint to {ckpt_path}")
    return ckpt_path


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train the Graph-JEPA refinement model")
    p.add_argument("--data", choices=["synthetic", "mimic"], default="synthetic")
    p.add_argument("--out", default="checkpoints/", help="output directory")
    p.add_argument("--encoder", choices=["mock", "bge", "sapbert"], default="mock")
    p.add_argument("--mock-dim", type=int, default=256,
                   help="MockEncoder dimension (ignored for bge)")
    p.add_argument("--bge-cache", default=".cache/graph_jepa/bge")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cpu")
    p.add_argument("--mimic-root", default=None, help="MIMIC-IV data root")
    p.add_argument("--mimic-notes", action="store_true",
                   help="include notes-derived SYMPTOM/MEDICAL_HISTORY nodes")
    return p


def main(argv=None) -> None:
    args = build_arg_parser().parse_args(argv)
    train(args)


if __name__ == "__main__":
    main()
