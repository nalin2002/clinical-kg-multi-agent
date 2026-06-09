# graph_contrastive_scorer (ABLATION)

A **graph-level** plausibility critic. Instead of scoring edges one at a time,
it pools an entire graph into a single vector and emits one scalar plausibility
score, then uses that score to greedily pick beneficial single edits.

## Purpose

Tests whether a holistic, graph-level plausibility signal — trained only to
prefer real graphs over corrupted ones — is enough to drive useful refinement
of noisy LLM-extracted clinical knowledge graphs.

## How it works

* **Featurization** (`model.py`): each graph -> `[mean(node_emb), mean(edge_feat)]`
  where `edge_feat = [emb(src), emb(tgt), onehot(relation)]`. Node embeddings
  come from the shared `encoders.node_embeddings`.
* **Model**: a small MLP mapping the pooled vector to a scalar score.
* **Training** (`train.py`): for each TRAIN-split silver graph, several
  corrupted copies are produced with `graph_utils.corrupt_graph` (strategies:
  `edge_deletion`, `invalid_edge_addition`, `relation_label_replacement`,
  `direction_flip`). A **margin-ranking** loss enforces
  `score(real) - score(corrupt) > margin`. The VAL split reports ranking
  accuracy (fraction of pairs ordered correctly).
* **Refinement** (`refine_graphs.py`): greedy hill-climbing on each LLM graph.
  Candidate single edits = drop an edge, or relabel an edge to a type-plausible
  relation (`graph_utils.candidate_relations_for`). An edit is kept only if it
  strictly **increases** the whole-graph score. Up to `edits_per_graph` edits.

## Inputs

* Silver graphs: `paths.silver_graphs` (positive/training signal).
* LLM graphs: `paths.llm_graphs` (to be refined).
* Config block: `training.graph_contrastive_scorer`
  (`epochs`, `lr`, `hidden_dim`, `margin`, `edits_per_graph`).

## Outputs

* `paths.training_outputs/graph_contrastive_scorer/model.pt` — checkpoint.
* `paths.training_outputs/graph_contrastive_scorer/config_used.json` — run record.
* `paths.refined_graphs/graph_contrastive_scorer/<patient_id>.json` — refined
  graphs with `source = "refined:graph_contrastive_scorer"`.

## Run

```bash
# 1) train the scorer on TRAIN-split silver graphs
python -m graph_jepa.training.graph_contrastive_scorer.train --config graph_jepa/config.yaml

# 2) refine the LLM graphs with the trained scorer
python -m graph_jepa.training.graph_contrastive_scorer.refine_graphs --config graph_jepa/config.yaml
```

> Requires `torch` (`pip install torch`). The encoder falls back to a
> deterministic hashing encoder if `sentence-transformers` is unavailable.

## Paper interpretation

This is an **ablation**. It replaces the default per-edge plausibility
classifier with a coarse, graph-level critic that only ever sees whole-graph
plausibility. Because every candidate edit must be scored by re-pooling the
entire graph, refinement is **slower**, and the supervision signal is **weaker**
(a single scalar per graph rather than per-edge labels). The comparison
isolates the value of edge-level supervision: if the graph-level critic picks
substantially worse edits than the edge classifier, it shows that fine-grained,
per-edge plausibility judgements — not just a global plausibility prior — are
what drive effective KG refinement.
