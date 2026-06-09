# masked_edge_prediction (ABLATION / prototype)

A **JEPA-style** world model for clinical KGs: mask some of a graph's edges,
then predict the hidden edges' relations from their endpoints plus a context
summary of the *visible* graph. At refine time the same model recovers /
completes edges in noisy LLM graphs.

## Purpose

Tests whether a model trained purely to *predict hidden graph structure*
(masked-edge relation recovery) yields useful corrections — relabeling wrong
relations, dropping unsupported edges, and adding high-confidence missing ones.

## How it works

* **Featurization** (`model.py`): for a masked edge, the input is
  `[emb(src), emb(tgt), context]`, where `context = mean of node embeddings of
  the visible graph`. Node embeddings come from `encoders.node_embeddings`.
* **Model**: an MLP relation classifier with a softmax head over
  `RELATION_TYPES`.
* **Training** (`train.py`): for each TRAIN-split silver graph, mask
  `ceil(mask_ratio * n_edges)` edges, encode the visible graph into a context
  vector, and predict each masked edge's relation. Loss = cross-entropy. The
  VAL split reports masked-relation accuracy.
* **Refinement** (`refine_graphs.py`), two phases:
  1. **Relabel / drop**: for each existing edge, if `P(current relation) <
     completion_threshold`, relabel to the argmax relation — or drop the edge if
     even the argmax probability is below the threshold.
  2. **Add missing edges**: enumerate type-plausible node pairs with
     `graph_utils.fully_connected_candidates` and add any whose predicted
     relation probability exceeds a high threshold (0.9).

## Inputs

* Silver graphs: `paths.silver_graphs` (positive/training signal).
* LLM graphs: `paths.llm_graphs` (to be refined).
* Config block: `training.masked_edge_prediction`
  (`epochs`, `lr`, `hidden_dim`, `mask_ratio`, `completion_threshold`).

## Outputs

* `paths.training_outputs/masked_edge_prediction/model.pt` — checkpoint.
* `paths.training_outputs/masked_edge_prediction/config_used.json` — run record.
* `paths.refined_graphs/masked_edge_prediction/<patient_id>.json` — refined
  graphs with `source = "refined:masked_edge_prediction"`.

## Run

```bash
# 1) train the masked-edge predictor on TRAIN-split silver graphs
python -m graph_jepa.training.masked_edge_prediction.train --config graph_jepa/config.yaml

# 2) refine the LLM graphs (relabel / drop / add)
python -m graph_jepa.training.masked_edge_prediction.refine_graphs --config graph_jepa/config.yaml
```

> Requires `torch` (`pip install torch`). The encoder falls back to a
> deterministic hashing encoder if `sentence-transformers` is unavailable.

## Paper interpretation

This is an **ablation / prototype** and the approach closest to the
"world model predicts hidden structure" framing. By masking edges and learning
to reconstruct their relations from node content and visible-graph context, the
model behaves as a JEPA-style predictor over graph structure rather than an
explicit plausibility classifier. Refinement is reframed as **edge completion /
relation recovery**: the world model fills in what it expects the hidden edges
to be. Comparing it against the default edge-plausibility classifier isolates
whether *predicting* masked structure transfers to *correcting* real LLM
extraction errors as well as discriminative plausibility scoring does.
