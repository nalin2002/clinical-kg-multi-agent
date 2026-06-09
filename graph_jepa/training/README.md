# graph_jepa / training

Builds the data and the world-model refiners.

## Stages

1. **`silver_reference_graphs/`** — LLM extracts a SILVER reference KG from each
   clinician note. Positives for training; references for tier-1 eval.
2. **`data_preparation/`** — wraps the EIR 13-agent extractor to produce the raw
   **LLM graphs** from transcripts (the `llm_only` condition / refinement input).
3. **World-model refiners** (pick one; all consume silver + LLM graphs):
   - **`edge_plausibility_classifier/`** — **DEFAULT.** Triple classifier with
     corruption-based negatives. CPU-only (sklearn), fast, interpretable.
   - `graph_contrastive_scorer/` — ablation: graph-level plausibility critic.
   - `masked_edge_prediction/` — ablation: JEPA-style masked-edge/relation
     prediction (edge completion).

## Inputs
- `data/aci_bench/transcripts/<PID>/<PID>.txt` and `<PID>_note.txt` (config `data.*`).

## Outputs (`training/outputs/`)
- `silver_reference_graphs/<PID>.json` — silver references (`source=clinician_note`)
- `llm_graphs/<PID>.json` — LLM graphs (`source=llm_transcript`)
- `<refiner>/model.*` + `metrics.json` (default) / `config_used.json` (ablations)
- `refined_graphs/<method>/<PID>.json` — refined graphs (`source=refined:<method>`),
  each tagged `extra.split = train|val|test`

## Commands
```bash
python -m graph_jepa.training.silver_reference_graphs.create_silver_graphs --config graph_jepa/config.yaml
python -m graph_jepa.training.data_preparation.create_llm_graphs            --config graph_jepa/config.yaml
python -m graph_jepa.training.edge_plausibility_classifier.train            --config graph_jepa/config.yaml
python -m graph_jepa.training.edge_plausibility_classifier.refine_graphs    --config graph_jepa/config.yaml --mode refine
python -m graph_jepa.training.edge_plausibility_classifier.refine_graphs    --config graph_jepa/config.yaml --mode fully_connected
```

## Training signal: corruption-based negatives
Silver edges are positives. Negatives corrupt them by: **relation replacement**,
**source-target swap**, **random target**, and **invalid clinical relation**
(violating the typed `RELATION_SCHEMA` in `common/graph_schema.py`). Mix and
count are configurable (`training.corruption_mix`, `training.negatives_per_positive`).

## Splitting (no leakage)
Patient-level split via `data.split` (seeded). The world model trains on the
**train** split only; refinement is applied to all patients but each graph is
stamped with its split so evaluation can report held-out numbers.

## Clinical-consistency violations metric
`RELATION_SCHEMA` defines type-plausible `(src_type, relation, tgt_type)`. The
fraction of edges violating it is a no-reference "clinical consistency" signal —
compare it before (LLM) vs after (refined) to report violation reduction.

## Paper interpretation
The world model is a **graph critic/refiner**: it learns what plausible clinical
structure looks like (from silver positives vs corruptions) and cleans the noisy
LLM graph. The default classifier is the headline method; the two ablations test
graph-level scoring and JEPA-style edge completion as alternative critics.
