# tier1_graph_metrics

**Purpose.** Graph-level evaluation: each method's graph vs the **silver**
clinician-note-derived reference.

**Metrics.** Edge Precision / Recall / F1 (direction- & relation-sensitive,
fuzzy node matching), Relation Accuracy (correct relation among endpoint-matched
edges), Graph Edit Distance approximation (normalized edge insert+delete).

**Inputs.** `silver_reference_graphs/`; method graphs from `llm_graphs/` and
`refined_graphs/<method>/` (config `evaluation.methods`); split metadata from the
trained model's `metrics.json`.

**Outputs (`evaluation/outputs/tier1/`).** `per_patient.csv`, `aggregate.csv`,
`summary.json`, `summary.md`.

**Run.**
```bash
python -m graph_jepa.evaluation.tier1_graph_metrics.evaluate \
    --config graph_jepa/config.yaml --split test   # test|val|train|all
```

**Split.** Default `test` (held-out). Scoring on training patients vs silver is
circular (the refiner trained on those silver edges); `--split all` is a
transductive number only.

**Paper interpretation.** Structural fidelity to a clinician reference. Expect
the world model to raise relation accuracy and lower normalized GED relative to
`llm_only` by removing implausible edges. Absolute values are bounded by silver
quality — report deltas between conditions, not absolute correctness.
