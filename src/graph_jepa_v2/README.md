# Graph-JEPA v2

This folder is a parallel implementation of the Graph-JEPA refinement layer. It
does not replace `graph_jepa`; it adds a more upstream-style patch/subgraph JEPA
core for clinical KG refinement.

## What changed

- Patient KGs are partitioned into connected patches with balanced BFS.
- A coarsened patch graph is built for each patient graph.
- Patch positional features include patch size, patch degree, and random-walk
  return probabilities on the coarsened graph.
- The node encoder defaults to PyTorch Geometric `GINEConv` / `GATConv`.
- The self-supervised task predicts target patch latents from context patches.
- The clinical typed edge plausibility head is kept as the downstream scorer.

## Train

```bash
PYTHONPATH=src python -m graph_jepa_v2.train \
  --data synthetic \
  --out checkpoints/
```

Train from ACI-Bench KG JSONs:

```bash
PYTHONPATH=src python -m graph_jepa_v2.train   --data aci-bench   --aci-kg-path outputs/aci_bench/sub_kgs/   --gnn-backend pyg   --conv gine   --out ckpts/  --encoder sapbert --wandb   --wandb-project audio_mental_health  --wandb-run-name exp1  --wandb-entity mangoesai
```

For the curated ACI-Bench reference KGs already present in this repo:

```bash
PYTHONPATH=src python -m graph_jepa_v2.train \
  --data aci-bench \
  --aci-kg-path EIR_260426/eir_aci_bench/transcripts/ \
  --gnn-backend pyg \
  --conv gine \
  --out checkpoints/
```

Useful knobs:

```bash
PYTHONPATH=src python -m graph_jepa_v2.train \
  --data synthetic \
  --num-patches 8 \
  --context-patches 1 \
  --target-patches 4 \
  --patch-pe-dim 8 \
  --batch-size 16 \
  --gnn-backend pyg \
  --conv gine \
  --out checkpoints/
```

Use `--gnn-backend torch` only as a dependency-light fallback when
`torch-geometric` is unavailable.

The checkpoint is written to `checkpoints/graph_jepa_v2.pt`.

## Score

```bash
PYTHONPATH=src python -m graph_jepa_v2.score \
  --input outputs/cooperative_20_enriched_v2/sub_kgs/RES0198_cooperative_multi_agent_enriched_v2.json \
  --checkpoint checkpoints/graph_jepa_v2.pt \
  --output RES0198_jepa_v2.json
```

The scorer writes the same edge fields as v1: `jepa_score` and `jepa_flag`.

Optionally add high-scoring missing edges between existing nodes:

```bash
PYTHONPATH=src python -m graph_jepa_v2.score \
  --input outputs/cooperative_20_enriched_v2/sub_kgs/RES0198_cooperative_multi_agent_enriched_v2.json \
  --checkpoint checkpoints/graph_jepa_v2.pt \
  --output RES0198_jepa_v2.json \
  --add-candidates \
  --candidate-threshold 0.7 \
  --max-candidates 50
```

Candidate edges are restricted to the typed relation schema and are marked
`jepa_suggested: true` / `jepa_unverified: true` because the model does not
produce transcript evidence.

## Design note

The implementation is conceptually inspired by the Graph-JEPA paper/repo shape:
patch/subgraph encoding, context-to-target prediction, and EMA target encoders.
It is written fresh for this clinical KG pipeline and keeps the current schema,
encoders, synthetic graph adapter, MIMIC-IV stub, and default annotate-only
scoring contract. Candidate edge generation is opt-in.
