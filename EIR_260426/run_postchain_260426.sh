#!/usr/bin/env bash
# run_postchain_260426.sh — robust convert / dump_graph / canonicalize / score chain
#
# Use this when:
#   (a) a wrapper run finished extraction but the in-line scorer hit the
#       RES_D2N filename underscore bug (Per-Patient Coverage 0.00%)
#   (b) you want to re-score an existing run dir with a different KB or
#       canon threshold without re-extracting
#   (c) you just want a clean post-chain you can copy-paste
#
# Usage:
#   ./run_postchain_260426.sh                      # auto-pick latest aci-bench run
#   ./run_postchain_260426.sh in-corpus            # latest in-corpus run
#   ./run_postchain_260426.sh aci-bench            # latest aci-bench run (default)
#   ./run_postchain_260426.sh aci-bench RUN_DIR    # specific run dir (timestamp subdir)
#   ./run_postchain_260426.sh in-corpus RUN_DIR --threshold 0.75
#
# The script does NOT extract — it operates on an existing run dir produced by
# smoke_test_v10_*.py. If the wrapper succeeded end-to-end you don't need this.
#
# Created: 260426

set -euo pipefail

# ─── Resolve script location and project layout ──────────────────────────────
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/../.." && pwd)"
EIR_ROOT="$(cd "$HERE/.." && pwd)"

PYTHON="${PYTHON:-/Users/amitlamba/miniconda3/bin/python}"
[ -x "$PYTHON" ] || PYTHON="$(command -v python3)"

# ─── Default KB (can override via EIR_CURATED_KB_PATH) ───────────────────────
KB_DEFAULT="$HERE/curated_synthetic_kb_combined_260426.json"
KB="${EIR_CURATED_KB_PATH:-$KB_DEFAULT}"
[ -f "$KB" ] || { echo "ERROR: KB not found at $KB" >&2; exit 1; }

# ─── Parse args ──────────────────────────────────────────────────────────────
MODE="${1:-aci-bench}"
shift || true

EXPLICIT_RUN=""
THRESHOLD="0.70"
while [ $# -gt 0 ]; do
    case "$1" in
        --threshold) THRESHOLD="$2"; shift 2 ;;
        --threshold=*) THRESHOLD="${1#*=}"; shift ;;
        -*)
            echo "Unknown flag: $1" >&2
            echo "Valid: --threshold N" >&2
            exit 2
            ;;
        *)
            EXPLICIT_RUN="$1"; shift
            ;;
    esac
done

case "$MODE" in
    in-corpus)
        EXTRACT_BASE="$EIR_ROOT/eir_results/smoke_test_glm47_specialized_parallel_v10"
        SCORE_BASE="$EIR_ROOT/eir_results/v10_for_organizer"
        BASELINE="$PROJECT_ROOT/data/human_curated/unified_graph_curated.json"
        TAG="v10"
        ;;
    aci-bench)
        EXTRACT_BASE="$EIR_ROOT/eir_results/smoke_test_glm47_specialized_parallel_v10_aci_bench"
        SCORE_BASE="$EIR_ROOT/eir_results/v10_aci_bench_for_organizer"
        BASELINE="$EIR_ROOT/eir_aci_bench/unified_graph_curated_aci.json"
        TAG="v10_aci_bench"
        ;;
    *)
        echo "Mode must be 'in-corpus' or 'aci-bench' (got: $MODE)" >&2
        exit 2
        ;;
esac

# ─── Pick run dir ────────────────────────────────────────────────────────────
if [ -n "$EXPLICIT_RUN" ]; then
    # Caller passed an explicit run-name; resolve under SCORE_BASE
    if [ -d "$EXPLICIT_RUN" ]; then
        SCORE_RUN_DIR="$(cd "$EXPLICIT_RUN" && pwd)"
    elif [ -d "$SCORE_BASE/$EXPLICIT_RUN" ]; then
        SCORE_RUN_DIR="$SCORE_BASE/$EXPLICIT_RUN"
    else
        echo "ERROR: run dir not found: $EXPLICIT_RUN" >&2
        echo "  tried: $SCORE_BASE/$EXPLICIT_RUN" >&2
        exit 1
    fi
else
    if [ ! -d "$SCORE_BASE" ]; then
        echo "ERROR: no scoring base dir at $SCORE_BASE" >&2
        echo "  Run extraction first (./run_v10_260426.sh $MODE)" >&2
        exit 1
    fi
    SCORE_RUN_DIR="$(ls -td "$SCORE_BASE"/*/ 2>/dev/null | head -1 | sed 's:/$::')"
    if [ -z "$SCORE_RUN_DIR" ]; then
        echo "ERROR: no timestamped run dirs under $SCORE_BASE" >&2
        echo "  Run extraction first (./run_v10_260426.sh $MODE)" >&2
        exit 1
    fi
fi

RUN_NAME="$(basename "$SCORE_RUN_DIR")"
NAME="${TAG}_${RUN_NAME}_fixed"
SUB_KGS="$SCORE_RUN_DIR/sub_kgs"

[ -d "$SUB_KGS" ] || { echo "ERROR: sub_kgs missing: $SUB_KGS" >&2; exit 1; }

echo "============================================================"
echo "  POST-CHAIN — convert / dump_graph / canon / score"
echo "============================================================"
echo "  mode      : $MODE"
echo "  run dir   : $SCORE_RUN_DIR"
echo "  sub_kgs   : $SUB_KGS"
echo "  KB        : $KB"
echo "  baseline  : $BASELINE"
echo "  threshold : $THRESHOLD"
echo "  output    : ${NAME}*"
echo "============================================================"

# ─── Step 1 — sanitize filenames (defensive against the RES_D2N bug) ─────────
# The convert script is now fixed at root, but this is defensive in case the
# user re-runs against an old sub_kgs dir produced before the fix.
echo ""
echo "[1/3] sanitize sub_kg filenames (RES_D2N* → RESD2N*)"
FIXED=0
for f in "$SUB_KGS"/RES_D2N*_v7.json; do
    [ -e "$f" ] || continue   # nothing to do; glob didn't match
    new="$(dirname "$f")/$(basename "$f" | sed 's/RES_D2N/RESD2N/')"
    mv "$f" "$new"
    FIXED=$((FIXED + 1))
done
echo "  renamed: $FIXED file(s)"

# ─── Step 2 — dump_graph (BGE-M3 entity resolution) ──────────────────────────
echo ""
echo "[2/3] dump_graph — BGE-M3 entity resolution"
( cd "$PROJECT_ROOT" && \
  PYTHONPATH="$PROJECT_ROOT/src" "$PYTHON" -m Clinical_KG_OS_LLM.dump_graph \
      --input  "$SUB_KGS" \
      --output "$SCORE_RUN_DIR" \
      --name   "$NAME" )

UNIFIED="$SCORE_RUN_DIR/unified_graph_${NAME}.json"
[ -f "$UNIFIED" ] || { echo "ERROR: dump_graph did not produce $UNIFIED" >&2; exit 1; }

# ─── Step 3a — canonicalize (BGE-M3 cosine ≥ threshold) ──────────────────────
CANON_SCRIPT="$HERE/canonicalize_to_curator_260426.py"
[ -f "$CANON_SCRIPT" ] || CANON_SCRIPT="$EIR_ROOT/canonicalize_to_curator_260426.py"
[ -f "$CANON_SCRIPT" ] || { echo "ERROR: canonicalize script not found" >&2; exit 1; }

CANON_OUT="$SCORE_RUN_DIR/unified_graph_${NAME}_canonicalized.json"
CANON_REPORT="$SCORE_RUN_DIR/canonicalize_report_${NAME}.json"

echo ""
echo "[3a] canonicalize @ cosine ≥ $THRESHOLD"
( cd "$PROJECT_ROOT" && \
  "$PYTHON" "$CANON_SCRIPT" \
      --unified   "$UNIFIED" \
      --kb        "$KB" \
      --output    "$CANON_OUT" \
      --report    "$CANON_REPORT" \
      --threshold "$THRESHOLD" )

[ -f "$CANON_OUT" ] || { echo "ERROR: canonicalize did not produce $CANON_OUT" >&2; exit 1; }

# ─── Step 3b — kg_similarity_scorer ──────────────────────────────────────────
SCORE_OUT="$SCORE_RUN_DIR/score_${NAME}_canonicalized.json"

echo ""
echo "[3b] kg_similarity_scorer"
( cd "$PROJECT_ROOT" && \
  PYTHONPATH="$PROJECT_ROOT/src" "$PYTHON" -m Clinical_KG_OS_LLM.kg_similarity_scorer \
      --student  "$CANON_OUT" \
      --baseline "$BASELINE" \
      --output   "$SCORE_OUT" )

echo ""
echo "============================================================"
echo "  DONE."
echo "  unified       : $UNIFIED"
echo "  canonicalized : $CANON_OUT"
echo "  score         : $SCORE_OUT"
echo "  canon report  : $CANON_REPORT"
echo "============================================================"
