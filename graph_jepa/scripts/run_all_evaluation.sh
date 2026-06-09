#!/usr/bin/env bash
# Run the full graph_jepa evaluation pipeline:
#   tier-1 graph metrics  +  QA generation -> answer-from-graph -> judge
# Usage: bash graph_jepa/scripts/run_all_evaluation.sh [path/to/config.yaml] [split]
#   split defaults to "all" — Setup B trains on LLM graphs and evaluates vs
#   silver, so there is no leakage and all patients can be scored. (Use "test"
#   only if you also ran the Setup-A supervised classifier, which trains on silver.)
# Run from the repository root, AFTER run_all_training.sh.
set -euo pipefail

CONFIG="${1:-graph_jepa/config.yaml}"
SPLIT="${2:-all}"
PY="${PYTHON:-python}"
export PYTHONPATH="${PYTHONPATH:-.}"

echo "=== [1/4] Tier-1 graph metrics (refined vs silver, split=$SPLIT) ==="
$PY -m graph_jepa.evaluation.tier1_graph_metrics.evaluate --config "$CONFIG" --split "$SPLIT"

echo "=== [2/4] Generate transcript-grounded QA sets ==="
$PY -m graph_jepa.evaluation.qa_evaluation.generate_qa_sets --config "$CONFIG"

echo "=== [3/4] Answer QA from each graph (no transcript access) ==="
$PY -m graph_jepa.evaluation.qa_evaluation.answer_from_graphs --config "$CONFIG"

echo "=== [4/4] Score answers (token-F1 + LLM judge), split=$SPLIT ==="
$PY -m graph_jepa.evaluation.qa_evaluation.evaluate_answers --config "$CONFIG" --split "$SPLIT"

echo "=== Evaluation complete. See evaluation/outputs/tier1/ and evaluation/outputs/qa_eval/ ==="
