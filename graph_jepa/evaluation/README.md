# graph_jepa / evaluation

Two complementary evaluations of whether world-model refinement helps.

## Tier 1 — graph-level (`tier1_graph_metrics/`)
Refined graphs vs **silver** clinician-note-derived references.
- **Metrics:** Edge Precision, Recall, F1; Relation Accuracy; Graph Edit
  Distance (approx). Node matching is fuzzy (`evaluation.node_match_threshold`).
- **Compares:** `llm_only`, `edge_plausibility_classifier`, `fully_connected_wm`
  (config `evaluation.methods`).
- **Split:** defaults to `--split test` (held-out; avoids the circularity of
  scoring against silver the model trained on). `--split all` = transductive.
- **Outputs (`outputs/tier1/`):** `per_patient.csv`, `aggregate.csv`,
  `summary.json`, `summary.md`.

```bash
python -m graph_jepa.evaluation.tier1_graph_metrics.evaluate --config graph_jepa/config.yaml --split test
```

## Tier 2 — QA-based (`qa_evaluation/`)
Tests graph **usefulness**: answer transcript-grounded questions using ONLY a
graph (no transcript), then score answers.

1. `generate_qa_sets` — LLM writes N transcript-answerable Q&As per patient →
   `outputs/qa_sets/<PID>.json` (`question/gold_answer/evidence/category`).
2. `answer_from_graphs` — for each method's graph, the answerer LLM answers from
   the serialized graph only → `outputs/qa_answers/<method>/<PID>.json`.
3. `evaluate_answers` — exact match, token-F1, and **LLM-as-judge** (0/1
   correctness + 1–5 score; ensemble-capable) → `outputs/qa_eval/`
   (`per_question.csv`, `per_patient.csv`, `aggregate.csv`, `summary.md/json`).

```bash
python -m graph_jepa.evaluation.qa_evaluation.generate_qa_sets   --config graph_jepa/config.yaml
python -m graph_jepa.evaluation.qa_evaluation.answer_from_graphs --config graph_jepa/config.yaml
python -m graph_jepa.evaluation.qa_evaluation.evaluate_answers   --config graph_jepa/config.yaml --split test
```

The judge abstraction (`common/llm_utils.py`) supports Claude / GPT / Gemini;
set `evaluation.qa.judge_provider` or `evaluation.qa.judge_ensemble: [anthropic, openai, gemini]`.

## Headline comparison
`llm_only` vs `edge_plausibility_classifier` (proposed) vs `fully_connected_wm`
(ablation), on both tiers.

## Paper interpretation
- **Tier 1** measures structural fidelity to a clinician-note reference; expect
  refinement to raise relation accuracy / lower GED by removing implausible
  edges (relative, not absolute, since references are silver).
- **Tier 2** measures whether a cleaner graph yields better answers — the
  downstream utility argument. QA gold is transcript-derived and independent of
  the silver graphs, so it is the more leakage-robust signal.
- **`fully_connected_wm`** isolates the world model's relational power: starting
  from nodes only, can it reconstruct useful relations? Lower-bound on the
  refiner as a standalone relation predictor.
