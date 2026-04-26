#!/usr/bin/env bash
# run_v10_260426.sh — one-line entrypoint for the EIR_260426 submission
#
# Usage:
#   ./run_v10_260426.sh in-corpus      # 20 in-corpus transcripts, scored vs in-corpus curator
#   ./run_v10_260426.sh aci-bench      # 20 ACI-Bench transcripts, scored vs OoC ACI curator
#   ./run_v10_260426.sh in-corpus --workers 4
#   ./run_v10_260426.sh aci-bench --no-score
#
# The combined curated+synthetic KB is set as the default. To use the
# in-corpus-only 222-label KB, override with:
#   EIR_CURATED_KB_PATH=$(pwd)/curated_kb_260419.json ./run_v10_260426.sh ...
#
# To use any other KB:
#   EIR_CURATED_KB_PATH=/path/to/your_kb.json ./run_v10_260426.sh ...
#
# Created: 260426

set -euo pipefail

cd "$(dirname "$0")"

# ─── Default: combined curated+synthetic KB ───────────────────────────────────
export EIR_CURATED_KB_PATH="${EIR_CURATED_KB_PATH:-$(pwd)/curated_synthetic_kb_combined_260426.json}"

# ─── Logging ─────────────────────────────────────────────────────────────────
mkdir -p run_logs
TS=$(date +%y%m%d_%H%M%S)

cmd="${1:-help}"
shift || true

case "$cmd" in
    in-corpus)
        LOG="run_logs/v10_in_corpus_${TS}.log"
        echo "[run_v10] mode: in-corpus 20 patients"
        echo "[run_v10] KB:   $EIR_CURATED_KB_PATH"
        echo "[run_v10] log:  $LOG"
        python smoke_test_v10_260425.py "$@" 2>&1 | tee "$LOG"
        ;;
    aci-bench)
        LOG="run_logs/v10_aci_bench_${TS}.log"
        echo "[run_v10] mode: out-of-corpus ACI-Bench 20 patients"
        echo "[run_v10] KB:   $EIR_CURATED_KB_PATH"
        echo "[run_v10] log:  $LOG"
        # Auto-set the OoC baseline so the auto-scoring chain compares
        # against our hand-curated ACI graph, not the in-corpus one.
        python smoke_test_v10_aci_bench_260425.py \
            --baseline "$(pwd)/eir_aci_bench/unified_graph_curated_aci.json" \
            "$@" 2>&1 | tee "$LOG"
        ;;
    help|--help|-h|"")
        sed -n '2,22p' "$0"
        exit 0
        ;;
    *)
        echo "unknown mode: $cmd" >&2
        echo "valid modes: in-corpus | aci-bench | help" >&2
        exit 2
        ;;
esac
