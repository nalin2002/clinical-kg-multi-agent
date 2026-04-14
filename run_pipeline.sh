#!/usr/bin/env bash
# run_pipeline.sh
# ================
# End-to-end runner: extract → dump → score
#
# Usage:
#   ./run_pipeline.sh                        # all 20 transcripts
#   ./run_pipeline.sh RES0198 RES0199        # subset for testing
#   OPENROUTER_API_KEY=sk-or-... ./run_pipeline.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
SIBLING_REPO="$SCRIPT_DIR/../Clinical_KG_OS_LLM"
OUTPUT_DIR="$REPO_ROOT/my_kg_multi"
UNIFIED_DIR="$REPO_ROOT/my_kg_multi_unified"

# ---- Step 1: Extract KGs ----
echo "=== Step 1: Multi-Agent KG Extraction ==="
if [ $# -gt 0 ]; then
    python "$REPO_ROOT/multi_agent_kg_extraction.py" \
        --output "$OUTPUT_DIR" \
        --res-ids "$@"
else
    python "$REPO_ROOT/multi_agent_kg_extraction.py" \
        --output "$OUTPUT_DIR"
fi

# ---- Step 2: Dump / Entity Resolution ----
echo ""
echo "=== Step 2: Entity Resolution & Unification ==="
cd "$SIBLING_REPO"
uv run python -m Clinical_KG_OS_LLM.dump_graph \
    --input "$OUTPUT_DIR" \
    --output "$UNIFIED_DIR"

# ---- Step 3: Score ----
echo ""
echo "=== Step 3: Composite Score ==="
BASELINE="$SIBLING_REPO/data/human_curated/unified_graph_curated.json"
STUDENT_KG=$(find "$UNIFIED_DIR" -name "unified_graph_*.json" | head -1)

uv run python -m Clinical_KG_OS_LLM.kg_similarity_scorer \
    --student "$STUDENT_KG" \
    --baseline "$BASELINE"

echo ""
echo "Done! Your unified KG is at: $STUDENT_KG"
