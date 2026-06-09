# graph_jepa — Clinical KG Refinement with a Graph World Model

This module turns doctor–patient transcripts into clinical knowledge graphs with
the existing **EIR 13-agent LLM extractor**, then trains a **world model** that
acts as a **graph critic/refiner** to clean those graphs, and evaluates whether
refinement helps — both at the graph level and on a downstream QA task.

> **The world model here is a graph *critic/refiner*, not a future-state
> predictor.** It scores how plausible the existing graph structure is and
> prunes/relabels edges accordingly. It does not forecast future patient states
> or simulate interventions.

## The claim this module is built to test

**Strongest proposed method:** *LLM-extracted KG **+ world-model refinement*** produces
cleaner, more useful clinical graphs than the raw LLM KG.

**Training design = Setup B (self-supervised).** The world model trains on the
**EIR 13-agent transcript graphs** (`llm_graphs`) self-supervised; the **silver**
clinician-note graphs are used **only for evaluation**. Because silver is never
in training, there is **no train/eval leakage** and all patients can be scored
(no patient split needed).

Conditions compared throughout:

| Condition | Graph | Role |
|---|---|---|
| `llm_only` | raw EIR 13-agent graph | baseline (no refinement) |
| `masked_edge_prediction` | LLM graph → JEPA-style world-model refinement | **proposed method (headline)** |
| `graph_contrastive_scorer` | LLM graph → contrastive world-model refinement | proposed (alt world model) |
| `edge_plausibility_classifier` | trained on silver positives (Setup A) → refine | optional supervised baseline (needs a split) |
| `fully_connected_wm` | nodes only → candidates → WM pruning | ablation: recover relations from nodes alone |

> Why the headline is self-supervised, not the supervised classifier: a
> supervised critic trained with LLM edges as positives would learn that the
> LLM's *errors* are correct and prune nothing. Self-supervised training learns
> the corpus's dominant clinical structure and corrects minority inconsistencies
> — so it can train on the noisy LLM graphs and still denoise.

## Reference graphs are SILVER, not gold

Reference graphs are **silver-standard, clinician-note-derived** — an LLM extracts
them from each patient's clinician note. The note is human-written, but the graph
is LLM-extracted and **not** verified by a clinician. We never call them gold.
Treat tier-1 numbers as **relative** comparisons between conditions, not absolute
correctness against a human-curated truth.

## Pipeline

```
ACI-Bench transcript ──EIR 13-agent extractor (wrapped)──► LLM KG  ┐
clinician note ──LLM extraction──► SILVER reference KG             │
                                          │  (positives)           │
                          train world model (critic)               │
                                          ▼                         ▼
                              refine ◄────────────  LLM KG ──► refined KG
                                          │
                  ┌───────────────────────┴───────────────────────┐
            Tier-1 graph metrics                         QA-based evaluation
        (refined vs SILVER reference)            (answer Qs from graph only,
        Edge P/R/F1, Rel. Acc, GED                judge vs transcript gold)
```

## Three world-model training approaches

| Folder | Approach | Status |
|---|---|---|
| `training/edge_plausibility_classifier/` | scores `(src, relation, tgt, context)` triples; prunes/relabels edges | **DEFAULT / recommended** (most practical for a 5-day deadline; CPU-only sklearn) |
| `training/graph_contrastive_scorer/` | graph-level scorer: silver > corrupted; greedy plausibility-improving edits | ablation/prototype (torch) |
| `training/masked_edge_prediction/` | JEPA-style: mask edges, predict hidden relations from context; edge completion | ablation/prototype (torch) |

The **edge_plausibility_classifier is the default** because it is the most
practical and intuitive: a small classifier, corruption-based negatives, fast
CPU training, and a directly interpretable per-edge plausibility score. The other
two are implemented as ablations.

## Quickstart

```bash
pip install -r graph_jepa/requirements.txt
# Keys (only the providers you use): export ANTHROPIC_API_KEY=...  (silver graphs / QA / judge)

# --- Training (Setup B: self-supervised on the LLM transcript graphs) ---
python -m graph_jepa.training.silver_reference_graphs.create_silver_graphs --config graph_jepa/config.yaml  # eval reference
python -m graph_jepa.training.data_preparation.create_llm_graphs            --config graph_jepa/config.yaml  # training data (EIR)
python -m graph_jepa.training.masked_edge_prediction.train                  --config graph_jepa/config.yaml  # world model (headline)
python -m graph_jepa.training.masked_edge_prediction.refine_graphs          --config graph_jepa/config.yaml
python -m graph_jepa.training.graph_contrastive_scorer.train                --config graph_jepa/config.yaml  # alt world model
python -m graph_jepa.training.graph_contrastive_scorer.refine_graphs        --config graph_jepa/config.yaml

# --- Evaluation (no split: silver reserved for eval, all patients scored) ---
python -m graph_jepa.evaluation.tier1_graph_metrics.evaluate         --config graph_jepa/config.yaml --split all
python -m graph_jepa.evaluation.qa_evaluation.generate_qa_sets       --config graph_jepa/config.yaml
python -m graph_jepa.evaluation.qa_evaluation.answer_from_graphs     --config graph_jepa/config.yaml
python -m graph_jepa.evaluation.qa_evaluation.evaluate_answers       --config graph_jepa/config.yaml --split all
```

Or run the whole thing:

```bash
bash graph_jepa/scripts/run_all_training.sh
bash graph_jepa/scripts/run_all_evaluation.sh
```

> **No API keys / offline plumbing test:** set `llm.provider: mock` and
> `encoder.backend: hashing` in `config.yaml`. The full pipeline runs end-to-end
> with deterministic stub output (clearly logged as MOCK — never for results).

## Data splitting

**Setup B (headline, self-supervised): no patient split needed.** The world model
trains on the **LLM transcript graphs**; **silver is used only for evaluation**.
Silver is never seen in training, so scoring refined-vs-silver on **all** patients
is leak-free. Evaluate with `--split all` (the script default). A small graph-wise
holdout is used internally only for loss/accuracy monitoring.

**Setup A (optional supervised baseline `edge_plausibility_classifier`):** trains
on **silver positives**, so it DOES need a held-out split — evaluate it with
`--split test`. `train.py` saves the split; the evaluators read it. The one
combination to avoid there: training on all patients **and** reporting tier-1 vs
silver on all patients (circular).

Configure the training source with `training.train_graph_source`
(`llm_graphs` for Setup B, default).

## Primary paper metrics

| Metric | Where |
|---|---|
| Edge F1, Edge P/R | tier-1 |
| Relation Accuracy | tier-1 |
| Graph Edit Distance (approx) | tier-1 |
| QA Accuracy (judge), QA token-F1, exact match | QA eval |
| LLM-judge score (1–5), ensemble-capable | QA eval |
| Clinical-consistency violations (type-schema) reduced by refinement | derivable from `RELATION_SCHEMA` (see training/README) |

## Layout

```
graph_jepa/
  config.yaml            # all paths / models / hyperparameters (nothing hardcoded)
  requirements.txt
  common/                # schema, io, graph utils, encoders, llm client, EIR adapter
  training/
    silver_reference_graphs/   create_silver_graphs.py
    data_preparation/          create_llm_graphs.py    (EIR adapter)
    edge_plausibility_classifier/  featurize/model/train/refine_graphs  (DEFAULT)
    graph_contrastive_scorer/  (ablation)
    masked_edge_prediction/    (ablation)
    outputs/                   silver_reference_graphs/, llm_graphs/, refined_graphs/, <model dirs>
  evaluation/
    tier1_graph_metrics/   evaluate.py
    qa_evaluation/         generate_qa_sets / answer_from_graphs / evaluate_answers
    outputs/               qa_sets/, qa_answers/, qa_eval/, tier1/
  scripts/                 run_all_training.sh, run_all_evaluation.sh
```

## EIR integration

We **do not rewrite** the 13-agent extractor. `common/eir_adapter.py` loads
existing EIR per-patient KGs and converts them to the canonical schema; with
`eir.run_eir: true` it can shell out to the EIR entry script. To cover all 207
ACI-Bench patients, run EIR yourself and point `eir.existing_kg_dirs` at its raw
per-patient output. See `common/README.md`.
