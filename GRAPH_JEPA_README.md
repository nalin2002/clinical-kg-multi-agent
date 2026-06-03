# Graph-JEPA Refinement Layer

A consistency-scoring layer that runs **after** the multi-agent KG pipeline. It
reads the existing KG JSON, embeds nodes in a shared BGE-M3 space, runs a hybrid
Graph-JEPA to produce context-aware node latents, and annotates each candidate
edge with a plausibility score (`jepa_score ∈ [0,1]`) and a flag (`jepa_flag ∈
{ok, weak, inconsistent}`).

By default it is **annotate-only**: it never adds and (unless you opt in) never
removes facts. It only adds two fields to each edge.

It is built now against a **synthetic data adapter** and a **frozen MIMIC-IV
builder interface**, so real data can be plugged in later without code changes.

---

## Theory → code map

| Concept | Where |
| --- | --- |
| Shared node space: frozen BGE-M3 of `"{TYPE}: {text}"` | `encoders.BgeNodeEncoder` (`MockEncoder` for dev/CI) |
| Graph schema + pipeline JSON round-trip (annotate-only) | `schema.PatientGraph`, `schema.NodeType` / `EdgeType` |
| Typed relation rules (plausible vs. type-violating) | `data.RELATION_SCHEMA`, `data.is_plausible_typed` |
| Synthetic patient-state graphs | `data.SyntheticGraphGenerator` |
| MIMIC-IV builder (frozen stub) | `data.MimicGraphBuilder` |
| PyG conversion / dataset | `data.to_pyg_data`, `data.PatientGraphDataset` |
| Context encoder (GINE/GAT GNN) | `model.GraphEncoder` |
| EMA target encoder (no grad) | `model.GraphJEPA.target_encoder`, `model.update_ema` |
| Predictor (masked-latent regression) | `model.Predictor` |
| Relation-agnostic connected-region masking | `model.subgraph_mask` |
| JEPA latent loss + VICReg anti-collapse | `model.GraphJEPA.jepa_loss`, `model.vicreg_terms` |
| Typed edge-plausibility head | `model.EdgePlausibilityHead`, `model.GraphJEPA.edge_loss` |
| Training loop (EMA + AdamW, checkpointing) | `train.py` |
| Per-edge scoring (head + JEPA energy), optional prune | `score.py` |
| Hyperparameters | `config.py` |

### The hybrid objective

1. **JEPA (relation-agnostic, self-supervised).** Mask a random *connected*
   region of nodes (`subgraph_mask`), encode the visible graph with the online
   **context** encoder, and predict the masked nodes' latents as produced by an
   **EMA target** encoder over the full graph. Loss = smooth-L1 latent
   regression + a VICReg variance/covariance term that prevents representational
   collapse.
2. **Typed edge plausibility (lightweight, supervised).** `EdgePlausibilityHead`
   scores `(z_src, z_tgt, relation)`; positives are observed (co-occurrence)
   edges, negatives are endpoint corruptions.

### How an edge is scored at inference

```
p_head           = sigmoid(head(z_src, z_tgt, relation))         # supervised signal
energy           = ||predictor(context_masked(tgt)) − target(tgt)|| / sqrt(d)
structural_score = exp(−energy / T)                              # JEPA consistency
jepa_score       = alpha * p_head + (1 − alpha) * structural_score
```

`jepa_flag` is `inconsistent` below `inconsistent_threshold`, `weak` below
`weak_threshold`, otherwise `ok` (see `config.ScoreConfig`). Edges with a
dangling endpoint or unknown relation are kept and flagged `inconsistent`.

---

## Data contract

Per-transcript and unified KGs share:

- nodes: `{id, text, type, evidence, turn_id}` (unified adds `occurrences` / `res_id`)
- edges: `{source_id, target_id, type, evidence, turn_id}` (+ `occurrences` / `res_id`)

The layer round-trips this exact shape and only **adds** `jepa_score` /
`jepa_flag` to edges. Extra top-level keys (`_method`, `_source`, …) are
preserved.

- 7 node types: `SYMPTOM, DIAGNOSIS, TREATMENT, PROCEDURE, LOCATION, MEDICAL_HISTORY, LAB_RESULT`
- 6 edge types: `CAUSES, INDICATES, LOCATED_AT, RULES_OUT, TAKEN_FOR, CONFIRMS`

---

## Install

```bash
pip install -r requirements.txt
```

`numpy` is enough for the `MockEncoder` + synthetic path. `torch` and
`torch-geometric` are required to train and score. `FlagEmbedding` is only
needed for the real BGE-M3 encoder (`--encoder bge`).

The package lives under `src/`, so run modules with `src` on the path:

```bash
export PYTHONPATH=src
```

---

## Usage

### Train on synthetic data (no model/data downloads)

```bash
PYTHONPATH=src python -m graph_jepa.train --data synthetic --out checkpoints/
```

Use the real encoder instead of the mock one:

```bash
PYTHONPATH=src python -m graph_jepa.train --data synthetic --encoder bge --out checkpoints/
```

### Score a pipeline KG

```bash
# single file
PYTHONPATH=src python -m graph_jepa.score \
    --input outputs/cooperative_20_enriched_v2/sub_kgs/RES0198_cooperative_multi_agent_enriched_v2.json \
    --checkpoint checkpoints/graph_jepa.pt \
    --output RES0198_jepa.json

# a whole directory of KGs
PYTHONPATH=src python -m graph_jepa.score \
    --input outputs/cooperative_20_enriched_v2/sub_kgs/ \
    --checkpoint checkpoints/graph_jepa.pt \
    --output outputs/cooperative_20_jepa/
```

Opt-in pruning (off by default):

```bash
... --prune-threshold 0.25
```

### Pipeline integration

`run_pipeline.sh` runs an optional **Step 4** when `GRAPH_JEPA_CKPT` is set:

```bash
GRAPH_JEPA_CKPT=checkpoints/graph_jepa.pt ./run_pipeline.sh RES0198
# optional pruning:
GRAPH_JEPA_CKPT=checkpoints/graph_jepa.pt GRAPH_JEPA_PRUNE=0.25 ./run_pipeline.sh
```

It writes `<unified>_jepa.json` next to the unified graph.

---

## MIMIC-IV plug-in contract (frozen)

`data.MimicGraphBuilder` is a documented stub. The interface is fixed now so
real MIMIC-IV data can be wired in later **without changing any other module**.
It raises a clear `NotImplementedError` until the tables are provided.

`MimicGraphBuilder.build()` must return a `list[PatientGraph]` (one per
admission, `hadm_id`) using this table → node-type mapping:

| MIMIC-IV table | Node type |
| --- | --- |
| `diagnoses_icd` (+ `d_icd_diagnoses`) | `DIAGNOSIS` |
| `prescriptions` | `TREATMENT` |
| `labevents` (+ `d_labitems`) | `LAB_RESULT` |
| `procedures_icd` (+ `d_icd_procedures`) | `PROCEDURE` |
| clinical notes (optional, `--mimic-notes`) | `SYMPTOM` / `MEDICAL_HISTORY` |

Edges are derived per admission from co-occurrence and `charttime` /
`starttime` temporal ordering, mapped onto the 6 relation types via
`data.RELATION_SCHEMA`. Because the encoder embeds `"{TYPE}: {text}"`, MIMIC and
pipeline nodes land in the same space — a model trained on MIMIC can score
pipeline graphs directly.

Swap training data with `--data mimic --mimic-root /path/to/mimic`.

---

## Verification

1. **No collapse:** train a few epochs on synthetic data; `jepa_inv` decreases
   while `latent_std` stays `> 0`.
2. **Round-trip:** score `RES0198`; node/edge counts are unchanged and every
   edge gains `jepa_score` / `jepa_flag`.
3. **MIMIC contract:** `--data mimic` raises a documented `NotImplementedError`
   with the interface unchanged.
