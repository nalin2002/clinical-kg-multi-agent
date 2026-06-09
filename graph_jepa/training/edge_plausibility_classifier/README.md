# edge_plausibility_classifier  (DEFAULT world-model refiner)

**Purpose.** The recommended, most practical refiner. A classifier scores the
plausibility of a triple `(source_node, relation, target_node, context)` in
[0,1]; the score is used to **prune** implausible edges and **relabel** edges to
their best type-plausible relation. Recommended for a 5-day deadline: CPU-only
(sklearn), fast, interpretable.

**Files.** `featurize.py` (triple → feature vector), `model.py` (sklearn
classifier + F1-tuned threshold), `train.py`, `refine_graphs.py`.

**Training signal.**
- **Positives:** edges from silver reference graphs (train split).
- **Negatives:** corruptions — relation replacement, source-target swap, random
  target, invalid clinical relation (`training.corruption_mix`,
  `training.negatives_per_positive`).
- **Features:** `[emb(src), emb(tgt), emb(src)-emb(tgt), emb(src)*emb(tgt),
  onehot(src_type), onehot(tgt_type), onehot(relation)]`.

**Inputs.** `silver_reference_graphs/` (train), `llm_graphs/` (to refine).

**Outputs.**
- `training/outputs/edge_plausibility_classifier/model.pkl`, `metrics.json`
  (threshold, train/val P/R/F1, the patient split).
- `training/outputs/refined_graphs/edge_plausibility_classifier/<PID>.json`
  (`--mode refine`) and `.../fully_connected_wm/<PID>.json` (`--mode fully_connected`),
  each tagged `extra.split`.

**Run.**
```bash
python -m graph_jepa.training.edge_plausibility_classifier.train         --config graph_jepa/config.yaml
python -m graph_jepa.training.edge_plausibility_classifier.refine_graphs --config graph_jepa/config.yaml --mode refine
python -m graph_jepa.training.edge_plausibility_classifier.refine_graphs --config graph_jepa/config.yaml --mode fully_connected
```

**Threshold.** Tuned for F1 on the validation split unless
`training.edge_plausibility_classifier.threshold` is fixed.

**`fully_connected` mode.** Builds the type-plausible fully-connected candidate
set over the graph's nodes and keeps only above-threshold edges — the
FullyConnected+WM ablation (relation recovery from nodes alone).

**Paper interpretation.** The world model as an explicit **edge critic**. Pruning
low-plausibility edges should raise precision / relation accuracy and reduce
clinical-consistency violations vs the raw LLM graph, with a tunable
precision/recall trade-off (the threshold).
