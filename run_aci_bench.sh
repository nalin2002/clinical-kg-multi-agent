#!/usr/bin/env bash
# run_aci_bench.sh
# ================
# End-to-end ACI-Bench pipeline: pull -> multi-agent extract -> Graph-JEPA refine.
#
# Steps:
#   1. Pull encounters from Hugging Face (mkieffer/ACI-Bench) into bracketed
#      transcripts that the multi-agent extractor understands.
#   2. Run the cooperative multi-agent KG extractor over those transcripts.
#   3. Train a Graph-JEPA checkpoint on ACI-Bench KGs if one isn't supplied.
#   4. Annotate each per-transcript KG edge with jepa_score / jepa_flag.
#
# Usage:
#   ./run_aci_bench.sh                       # train split, all 3 subsets
#   ACI_LIMIT=5 ./run_aci_bench.sh           # first 5 encounters (smoke test)
#   ACI_SUBSETS="aci virtscribe" ./run_aci_bench.sh   # restrict subsets
#   ACI_IDS="D2N008 D2N018" ./run_aci_bench.sh
#   GRAPH_JEPA_CKPT=checkpoints/graph_jepa.pt ./run_aci_bench.sh   # reuse a ckpt
#   GRAPH_JEPA_MODULE=graph_jepa_v4 ./run_aci_bench.sh             # use v4
#   GRAPH_JEPA_TRAIN_ARGS="--encoder sapbert" ./run_aci_bench.sh   # extra train/pretrain args
#   GRAPH_JEPA_FINETUNE_ARGS="--epochs 75" GRAPH_JEPA_MODULE=graph_jepa_v4 ./run_aci_bench.sh
#   GRAPH_JEPA_PRUNE=0.25 ./run_aci_bench.sh # opt-in edge pruning
#
# ACI-Bench ships three subsets (aci, virtassist, virtscribe); all are pulled
# by default.
#
# Requires OPENROUTER_API_KEY in .env (or openrouter in api_keys.json).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
PYTHON="${PYTHON:-python}"

ACI_SPLIT="${ACI_SPLIT:-train}"
TRANSCRIPTS_DIR="$REPO_ROOT/data/aci_bench/transcripts"
EXTRACT_DIR="$REPO_ROOT/outputs/aci_bench/sub_kgs"
REFINED_DIR="$REPO_ROOT/outputs/aci_bench/sub_kgs_jepa"
GRAPH_JEPA_MODULE="${GRAPH_JEPA_MODULE:-graph_jepa}"
case "$GRAPH_JEPA_MODULE" in
    graph_jepa_v4)
        DEFAULT_CKPT="$REPO_ROOT/checkpoints/graph_jepa_v4.pt"
        ;;
    graph_jepa_v3)
        DEFAULT_CKPT="$REPO_ROOT/checkpoints/graph_jepa_v3.pt"
        ;;
    graph_jepa_v2)
        DEFAULT_CKPT="$REPO_ROOT/checkpoints/graph_jepa_v2.pt"
        ;;
    *)
        DEFAULT_CKPT="$REPO_ROOT/checkpoints/graph_jepa.pt"
        ;;
esac
CKPT="${GRAPH_JEPA_CKPT:-$DEFAULT_CKPT}"

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# ---- Step 1: Pull ACI-Bench transcripts ----
echo "=== Step 1: Pull ACI-Bench (mkieffer/ACI-Bench) ==="
PULL_ARGS=(--out "$TRANSCRIPTS_DIR")
if [ -n "${ACI_IDS:-}" ]; then
    # shellcheck disable=SC2206
    PULL_ARGS+=(--ids ${ACI_IDS})
else
    PULL_ARGS+=(--split "$ACI_SPLIT")
fi
if [ -n "${ACI_SUBSETS:-}" ]; then
    # shellcheck disable=SC2206
    PULL_ARGS+=(--subsets ${ACI_SUBSETS})
fi
if [ -n "${ACI_LIMIT:-}" ]; then
    PULL_ARGS+=(--limit "$ACI_LIMIT")
fi
"$PYTHON" -m aci_bench "${PULL_ARGS[@]}"

# ---- Step 2: Multi-agent KG extraction ----
echo ""
echo "=== Step 2: Multi-Agent KG Extraction ==="
"$PYTHON" "$REPO_ROOT/src/multi_agent_cooperative_kg.py" \
    --output "$EXTRACT_DIR" \
    --transcripts-dir "$TRANSCRIPTS_DIR"

# ---- Step 3: Ensure a Graph-JEPA checkpoint exists ----
echo ""
echo "=== Step 3: Graph-JEPA checkpoint ==="
if [ ! -f "$CKPT" ]; then
    echo "No checkpoint at $CKPT — training $GRAPH_JEPA_MODULE on ACI-Bench KGs."
    if [ "$GRAPH_JEPA_MODULE" = "graph_jepa_v4" ]; then
        PRETRAIN_CKPT="${GRAPH_JEPA_PRETRAIN_CKPT:-$(dirname "$CKPT")/graph_jepa_v4_pretrain.pt}"
        EXTRA_PRETRAIN_ARGS=()
        if [ -n "${GRAPH_JEPA_TRAIN_ARGS:-}" ]; then
            # shellcheck disable=SC2206
            EXTRA_PRETRAIN_ARGS+=($GRAPH_JEPA_TRAIN_ARGS)
        fi
        if [ -n "${GRAPH_JEPA_PRETRAIN_ARGS:-}" ]; then
            # shellcheck disable=SC2206
            EXTRA_PRETRAIN_ARGS+=($GRAPH_JEPA_PRETRAIN_ARGS)
        fi
        if [ ! -f "$PRETRAIN_CKPT" ]; then
            "$PYTHON" -m graph_jepa_v4.pretrain \
                --data aci-bench \
                --aci-kg-path "$EXTRACT_DIR" \
                --out "$(dirname "$PRETRAIN_CKPT")" \
                "${EXTRA_PRETRAIN_ARGS[@]}"
        else
            echo "Using existing v4 pretrain checkpoint: $PRETRAIN_CKPT"
        fi

        EXTRA_FINETUNE_ARGS=()
        if [ -n "${GRAPH_JEPA_FINETUNE_ARGS:-}" ]; then
            # shellcheck disable=SC2206
            EXTRA_FINETUNE_ARGS+=($GRAPH_JEPA_FINETUNE_ARGS)
        fi
        "$PYTHON" -m graph_jepa_v4.finetune \
            --data aci-bench \
            --aci-kg-path "$EXTRACT_DIR" \
            --checkpoint "$PRETRAIN_CKPT" \
            --out "$(dirname "$CKPT")" \
            "${EXTRA_FINETUNE_ARGS[@]}"
    else
        EXTRA_TRAIN_ARGS=()
        if [ -n "${GRAPH_JEPA_TRAIN_ARGS:-}" ]; then
            # shellcheck disable=SC2206
            EXTRA_TRAIN_ARGS+=($GRAPH_JEPA_TRAIN_ARGS)
        fi
        "$PYTHON" -m "$GRAPH_JEPA_MODULE.train" \
            --data aci-bench \
            --aci-kg-path "$EXTRACT_DIR" \
            --out "$(dirname "$CKPT")" \
            "${EXTRA_TRAIN_ARGS[@]}"
    fi
else
    echo "Using existing checkpoint: $CKPT"
fi

# ---- Step 4: Graph-JEPA refinement (annotate-only by default) ----
echo ""
echo "=== Step 4: Graph-JEPA Refinement ==="
PRUNE_ARGS=()
if [ -n "${GRAPH_JEPA_PRUNE:-}" ]; then
    PRUNE_ARGS+=(--prune-threshold "$GRAPH_JEPA_PRUNE")
fi
"$PYTHON" -m "$GRAPH_JEPA_MODULE.score" \
    --input "$EXTRACT_DIR" \
    --checkpoint "$CKPT" \
    --output "$REFINED_DIR" \
    "${PRUNE_ARGS[@]}"

echo ""
echo "Done!"
echo "  Transcripts : $TRANSCRIPTS_DIR"
echo "  Raw KGs     : $EXTRACT_DIR"
echo "  Refined KGs : $REFINED_DIR"
