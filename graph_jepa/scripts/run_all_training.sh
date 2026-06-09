#!/usr/bin/env bash
# Full graph_jepa training pipeline — SETUP B (self-supervised world model).
#   silver graphs (eval reference) + LLM transcript graphs (training data)
#   -> train self-supervised refiners on the LLM graphs -> refine the LLM graphs
# Silver is used ONLY for evaluation, so there is no train/eval leakage.
# Usage: bash graph_jepa/scripts/run_all_training.sh [path/to/config.yaml]
# Run from the repository root.
set -euo pipefail

CONFIG="${1:-graph_jepa/config.yaml}"
PY="${PYTHON:-python}"
export PYTHONPATH="${PYTHONPATH:-.}"

echo "=== [1/4] Silver reference graphs (from clinician notes; EVAL reference only) ==="
$PY -m graph_jepa.training.silver_reference_graphs.create_silver_graphs --config "$CONFIG"

echo "=== [2/4] LLM transcript graphs via EIR 13-agent extractor (TRAINING data) ==="
$PY -m graph_jepa.training.data_preparation.create_llm_graphs --config "$CONFIG"

echo "=== [3/4] Train self-supervised world models on the LLM graphs ==="
$PY -m graph_jepa.training.masked_edge_prediction.train  --config "$CONFIG"   # JEPA-style (headline)
$PY -m graph_jepa.training.graph_contrastive_scorer.train --config "$CONFIG"  # contrastive

echo "=== [4/4] Refine the LLM graphs with each world model ==="
$PY -m graph_jepa.training.masked_edge_prediction.refine_graphs  --config "$CONFIG"
$PY -m graph_jepa.training.graph_contrastive_scorer.refine_graphs --config "$CONFIG"

# --- Optional Setup-A supervised baseline (trains on SILVER, needs a split). ---
# $PY -m graph_jepa.training.edge_plausibility_classifier.train         --config "$CONFIG"
# $PY -m graph_jepa.training.edge_plausibility_classifier.refine_graphs --config "$CONFIG" --mode refine
# $PY -m graph_jepa.training.edge_plausibility_classifier.refine_graphs --config "$CONFIG" --mode fully_connected

echo "=== Training complete. Refined graphs under training/outputs/refined_graphs/ ==="
