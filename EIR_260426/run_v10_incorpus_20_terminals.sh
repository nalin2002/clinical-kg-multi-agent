#!/usr/bin/env bash
set -euo pipefail

# Run EIR v10 in-corpus as one transcript per Terminal window, then combine,
# convert, dump_graph, and score.
#
# Usage:
#   cd "/Users/nalinprabhath/VSCode Repos/clinical-kg-multi-agent/EIR_260426"
#   bash run_v10_incorpus_20_terminals.sh
#
# Optional env vars:
#   LAUNCH_MODE=terminal|background
#   RUN_NAME=eir_v10_incorpus_20_parallel
#   KG_ROOT="/Users/nalinprabhath/VSCode Repos/Clinical_KG_OS_LLM"
#   EIR_CURATED_KB_PATH=/path/to/curated_kb.json
#   RES_IDS_CSV=RES0198,RES0199
#   MERGE_THRESHOLD=0.85

EIR_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$EIR_ROOT/.." && pwd)"
KG_ROOT="${KG_ROOT:-/Users/nalinprabhath/VSCode Repos/Clinical_KG_OS_LLM}"

RUN_NAME="${RUN_NAME:-eir_v10_incorpus_20_parallel}"
TRANSCRIPT_DIR="$PROJECT_ROOT/data/transcripts"
BASELINE="${BASELINE:-$PROJECT_ROOT/data/human_curated/unified_graph_curated.json}"
MERGE_THRESHOLD="${MERGE_THRESHOLD:-0.85}"

OUTPUT_BASE="$EIR_ROOT/eir_results/$RUN_NAME"
RAW_RUNS_DIR="$OUTPUT_BASE/per_transcript_raw_runs"
COMBINED_RAW_DIR="$OUTPUT_BASE/combined_raw"
SUB_KGS_DIR="$OUTPUT_BASE/sub_kgs"
DONE_DIR="$OUTPUT_BASE/_done"
LOG_DIR="$OUTPUT_BASE/logs"
COMMAND_DIR="$OUTPUT_BASE/_commands"
PATHS_DIR="$OUTPUT_BASE/_run_paths"
UNIFIED_NAME="${UNIFIED_NAME:-${RUN_NAME}_all}"

if [[ "${LAUNCH_MODE:-}" == "" ]]; then
  if [[ "$(uname -s)" == "Darwin" ]]; then
    LAUNCH_MODE="terminal"
  else
    LAUNCH_MODE="background"
  fi
fi

mkdir -p "$RAW_RUNS_DIR" "$COMBINED_RAW_DIR" "$SUB_KGS_DIR" "$DONE_DIR" "$LOG_DIR" "$COMMAND_DIR" "$PATHS_DIR"
rm -f "$DONE_DIR"/*.ok "$DONE_DIR"/*.fail "$PATHS_DIR"/*.path

kg_python() {
  if [[ -x "$KG_ROOT/.venv/bin/python" ]]; then
    printf '%s\n' "$KG_ROOT/.venv/bin/python"
  else
    printf '%s\n' "python"
  fi
}

eir_python() {
  if [[ -x "$KG_ROOT/.venv/bin/python" ]]; then
    printf '%s\n' "$KG_ROOT/.venv/bin/python"
  else
    printf '%s\n' "python"
  fi
}

discover_transcripts() {
  local d id txt
  shopt -s nullglob
  for d in "$TRANSCRIPT_DIR"/RES*; do
    [[ -d "$d" ]] || continue
    id="$(basename "$d")"
    txt="$d/$id.txt"
    [[ -f "$txt" ]] || continue
    printf '%s\n' "$id"
  done
  shopt -u nullglob
}

build_transcript_list() {
  if [[ -n "${RES_IDS_CSV:-}" ]]; then
    echo "$RES_IDS_CSV" | tr ',' '\n' | sed '/^[[:space:]]*$/d'
  else
    discover_transcripts
  fi
}

count_files_with_extension() {
  local ext="$1"
  local files=()
  shopt -s nullglob
  files=("$DONE_DIR"/*."$ext")
  shopt -u nullglob
  echo "${#files[@]}"
}

create_job_script() {
  local res_id="$1"
  local script_path="$COMMAND_DIR/${res_id}.sh"
  local log_path="$LOG_DIR/${res_id}.log"
  local ok_path="$DONE_DIR/${res_id}.ok"
  local fail_path="$DONE_DIR/${res_id}.fail"
  local path_file="$PATHS_DIR/${res_id}.path"
  local job_output_root="$RAW_RUNS_DIR/$res_id"
  local py
  py="$(eir_python)"

  mkdir -p "$job_output_root"

  cat >"$script_path" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$EIR_ROOT"
trap 'touch "$fail_path"' ERR

before_file="\$(mktemp)"
after_file="\$(mktemp)"
find "$job_output_root" -mindepth 1 -maxdepth 1 -type d -print | sort > "\$before_file" || true

echo "=== EIR v10 $res_id ==="
echo "Output root: $job_output_root"
echo "Log: $log_path"
echo

"$py" smoke_test_v10_260425.py \\
  --no-score \\
  --output "$job_output_root" \\
  --res-ids "$res_id" \\
  --workers 1 2>&1 | tee "$log_path"

find "$job_output_root" -mindepth 1 -maxdepth 1 -type d -print | sort > "\$after_file" || true
run_dir="\$(comm -13 "\$before_file" "\$after_file" | tail -n 1)"
if [[ -z "\$run_dir" ]]; then
  run_dir="\$(find "$job_output_root" -mindepth 1 -maxdepth 1 -type d -print | sort | tail -n 1)"
fi
if [[ -z "\$run_dir" || ! -d "\$run_dir/$res_id" ]]; then
  echo "Could not find EIR raw patient output for $res_id under $job_output_root" >&2
  exit 1
fi

printf '%s\n' "\$run_dir" > "$path_file"
touch "$ok_path"
echo
echo "DONE: $res_id"
echo "Raw run dir: \$run_dir"
EOF

  chmod +x "$script_path"
  printf '%s\n' "$script_path"
}

launch_job() {
  local res_id="$1"
  local job_script
  job_script="$(create_job_script "$res_id")"

  if [[ "$LAUNCH_MODE" == "terminal" ]]; then
    if [[ "$(uname -s)" != "Darwin" ]]; then
      echo "LAUNCH_MODE=terminal requires macOS. Use LAUNCH_MODE=background." >&2
      exit 1
    fi
    osascript -e "tell application \"Terminal\" to do script \"bash \\\"$job_script\\\"\""
  elif [[ "$LAUNCH_MODE" == "background" ]]; then
    bash "$job_script" &
  else
    echo "Unsupported LAUNCH_MODE: $LAUNCH_MODE" >&2
    exit 1
  fi
}

wait_for_jobs() {
  local total="$1"
  local ok_count fail_count done_count
  while true; do
    ok_count="$(count_files_with_extension ok)"
    fail_count="$(count_files_with_extension fail)"
    done_count=$((ok_count + fail_count))
    printf '\rJobs complete: %s/%s (ok=%s failed=%s)' "$done_count" "$total" "$ok_count" "$fail_count"
    if [[ "$done_count" -ge "$total" ]]; then
      echo
      break
    fi
    sleep 5
  done
}

combine_raw_outputs() {
  local ids=("$@")
  local id run_dir

  rm -rf "$COMBINED_RAW_DIR"
  mkdir -p "$COMBINED_RAW_DIR"

  for id in "${ids[@]}"; do
    if [[ ! -f "$PATHS_DIR/${id}.path" ]]; then
      echo "Missing run path for $id" >&2
      exit 1
    fi
    run_dir="$(cat "$PATHS_DIR/${id}.path")"
    if [[ ! -d "$run_dir/$id" ]]; then
      echo "Missing raw patient dir: $run_dir/$id" >&2
      exit 1
    fi
    cp -R "$run_dir/$id" "$COMBINED_RAW_DIR/$id"
  done

  echo "Combined raw EIR outputs: $COMBINED_RAW_DIR"
}

print_score_summary() {
  local report="$1"
  local py
  py="$(kg_python)"
  "$py" - <<EOF
import json
from pathlib import Path
report = Path("$report")
data = json.loads(report.read_text())
print()
print("Score summary")
print("-------------")
print(f"composite_score: {data['composite_score']}")
for k, v in data["component_scores"].items():
    print(f"{k}: {v}")
EOF
}

main() {
  IDS=()
  while IFS= read -r id; do
    [[ -n "$id" ]] || continue
    IDS+=("$id")
  done < <(build_transcript_list)

  if [[ "${#IDS[@]}" -eq 0 ]]; then
    echo "No transcripts found in $TRANSCRIPT_DIR" >&2
    exit 1
  fi

  echo "EIR v10 in-corpus parallel run"
  echo "Run name: $RUN_NAME"
  echo "Transcripts: ${#IDS[@]}"
  echo "Launch mode: $LAUNCH_MODE"
  echo "EIR root: $EIR_ROOT"
  echo "KG root: $KG_ROOT"
  echo "Baseline: $BASELINE"
  echo "Merge threshold: $MERGE_THRESHOLD"
  echo

  local id
  for id in "${IDS[@]}"; do
    launch_job "$id"
  done

  wait_for_jobs "${#IDS[@]}"

  local fail_count
  fail_count="$(count_files_with_extension fail)"
  if [[ "$fail_count" -gt 0 ]]; then
    echo "One or more EIR v10 jobs failed. Check logs in: $LOG_DIR" >&2
    exit 1
  fi

  combine_raw_outputs "${IDS[@]}"

  echo
  echo "Converting EIR raw category outputs to KG JSON..."
  local py
  py="$(kg_python)"
  "$py" "$EIR_ROOT/convert_v7_to_kg_extraction_format_260425.py" \\
    --nodes-dir "$COMBINED_RAW_DIR" \\
    --edges-dir "$COMBINED_RAW_DIR" \\
    --output "$SUB_KGS_DIR"

  echo
  echo "Merging with dump_graph..."
  "$py" -m Clinical_KG_OS_LLM.dump_graph \\
    --input "$SUB_KGS_DIR" \\
    --output "$OUTPUT_BASE" \\
    --name "$UNIFIED_NAME" \\
    --threshold "$MERGE_THRESHOLD"

  local unified_graph score_report
  unified_graph="$OUTPUT_BASE/unified_graph_${UNIFIED_NAME}.json"
  score_report="$OUTPUT_BASE/score_report_${UNIFIED_NAME}.json"

  echo
  echo "Scoring..."
  "$py" -m Clinical_KG_OS_LLM.kg_similarity_scorer \\
    --student "$unified_graph" \\
    --baseline "$BASELINE" \\
    --output "$score_report"

  print_score_summary "$score_report"

  echo
  echo "Unified graph: $unified_graph"
  echo "Score report: $score_report"
}

main "$@"
