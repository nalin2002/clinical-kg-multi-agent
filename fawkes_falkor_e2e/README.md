# Fawkes Falkor E2E

Standalone pipeline for:

```text
raw HF admission JSONL -> admission KG JSONL -> optional note embeddings -> FalkorDB upload -> Falkor GraphRAG QA
```

The pipeline is resumable in the simplest possible way. After each completed
data stage, it uploads the stage JSONL directly to the configured Hugging Face
dataset repo. On the next run, the script checks for that JSONL and skips the
stage if it already exists. The local `--workdir` is only scratch space and can
disappear between HF Jobs.

## Stages

1. `normalize`
   - Downloads raw admission rows from `RAW_REPO/RAW_FILE`.
   - Selects eligible patients/admissions.
   - Writes `admissions_normalized.jsonl`.

2. `build_kg`
   - Builds nodes and deterministic backbone edges.
   - Optionally calls a chat-completions-compatible endpoint for inferred edges.
   - Preserves top-level `note` and `input_block`.
   - Writes `kg_raw.jsonl`.

3. `embed_notes` optional
  - Enabled with `--embed-notes` or `EMBED_NOTES=1`.
  - Embeds the top-level `note` field with `Simonlee711/Clinical_ModernBERT`
    by default.
  - Uses masked mean pooling over the model's last hidden state.
  - Writes `kg_embedded.jsonl` with `note_embedding`, `embed_model`, and
    `embed_dim` on each graph row.
  - Falkor uploads `kg_embedded.jsonl` instead of `kg_raw.jsonl` when this
    stage is enabled.

4. `upload_falkor`
   - Uploads the final KG JSONL to FalkorDB.
   - Creates `KGGraph`, `KGNode`, `CONTAINS_NODE`, and typed relation edges.
   - Writes `falkor_upload_manifest.json`.

## Local Dry Run

This parses/builds a tiny deterministic KG locally and does not call Falkor:

```bash
cd fawkes_falkor_e2e

cat > /tmp/fawkes_smoke_raw.jsonl <<'JSONL'
{"subject_id":1,"hadm_id":2,"symptoms":["cough"],"diagnoses":["Pneumonia","Hypertension","Diabetes"],"medical_history":[],"medications":[{"name":"Ceftriaxone"},{"name":"Lisinopril"},{"name":"Metformin"}],"procedures":["Chest x-ray"],"microbiology":[],"services":["MED"],"n_diagnoses":3,"n_medications":3,"n_services":1,"note":"[1] Patient had cough and pneumonia. [2] Ceftriaxone was used for pneumonia.","input_block":"smoke admission"}
JSONL

uv run fawkes_falkor_e2e.py \
  --local-only \
  --workdir /tmp/fawkes_smoke_workdir \
  --raw-local-file /tmp/fawkes_smoke_raw.jsonl \
  --llm-mode off \
  --dry-run-falkor
```

## Full Local Run

```bash
export HF_TOKEN=...
export OUTPUT_REPO=your-org/fawkes-falkor-e2e-artifacts
export FALKORDB_URL=redis://localhost:6379

uv run fawkes_falkor_e2e.py \
  --llm-mode required \
  --endpoint-url https://YOUR-ENDPOINT/v1-or-root \
  --embed-notes \
  --include-note-embedding \
  --falkor-graph fawkes_clinical_kg \
  --drop-existing
```

If the endpoint URL already expects `/v1/chat/completions` after its root, pass
the root only. The script appends `/v1/models` and `/v1/chat/completions`.

## Hugging Face Jobs

From the parent directory containing this folder:

```bash
hf jobs uv run -s HF_TOKEN --timeout 12h \
  -e OUTPUT_REPO=your-org/fawkes-falkor-e2e-artifacts \
  -e RAW_REPO=on1onmangoes/fawkes-mimic-megaflat-admission-260615 \
  -e RAW_FILE=fawkes_mimic_megaflat_admission_260615.jsonl \
  -e LLM_MODE=required \
  -e ENDPOINT_URL=https://YOUR-ENDPOINT \
  -e EMBED_NOTES=1 \
  -e INCLUDE_NOTE_EMBEDDING=1 \
  -e EMBED_DEVICE=cuda \
  -e FALKORDB_URL=redis://USER:PASSWORD@HOST:PORT \
  -e FALKORDB_GRAPH=fawkes_clinical_kg \
  -d fawkes_falkor_e2e/fawkes_falkor_e2e.py
```

KG construction calls a hosted inference endpoint. Note embedding runs inside
the job, so use a GPU flavor when `EMBED_NOTES=1` for practical runtime. If you
disable note embedding, no local GPU is required.

## Resume Behavior

The script checks these files before each data stage:

```text
<OUTPUT_REPO>/admissions_normalized.jsonl
<OUTPUT_REPO>/kg_raw.jsonl
<OUTPUT_REPO>/kg_embedded.jsonl   # only when --embed-notes is enabled
```

If the file exists, the stage is skipped and the JSONL is downloaded into the
local workdir. If the file does not exist, the stage runs and uploads it when
complete. `--force` ignores existing JSONLs and overwrites them.

The Falkor upload step always runs because it is a database side effect, not a
JSONL-producing stage. It uses `MERGE` for graph entities and writes
`falkor_upload_manifest.json` for bookkeeping.

## Important Options

- `--n-patients 1000`: select 1000 eligible patients. Use `0` for all eligible.
- `--raw-local-file path.jsonl`: use a local raw JSONL instead of HF download.
- `--include-noteless`: keep admissions without notes. Without this, note-less
  admissions are skipped.
- `--llm-mode off|auto|required`: `auto` uses the endpoint if configured.
- `--embed-notes`: create `kg_embedded.jsonl` with per-admission note vectors.
- `--embed-device auto|cuda|cpu|mps`: choose where note embedding runs.
- `--stop-after normalize|build-kg|embed-notes|upload-falkor`: stop after a stage.
- `--exclude-text-fields`: do not store `note`/`input_block` on `KGGraph`.
- `--include-note-embedding`: store top-level `note_embedding` arrays in Falkor.

## Falkor Graph Shape

Upload creates:

```text
(:KGGraph:Admission {graph_id, subject_id, hadm_id, note, input_block, ...})
(:KGNode:<TYPE> {kg_node_id, graph_id, source_node_id, text, node_type, ...})
(:KGGraph)-[:CONTAINS_NODE]->(:KGNode)
(:KGNode)-[:RELATION {kg_edge_id, graph_id, evidence, confidence, ...}]->(:KGNode)
```

This matches the existing Falkor-backed GraphRAG QA loader in the main repo.

## QA Against FalkorDB

After upload, use `fawkes_falkor_qa.py` to answer questions from the graph in
FalkorDB.

List graph IDs:

```bash
uv run fawkes_falkor_qa.py \
  --falkor-url redis://USER:PASSWORD@HOST:PORT \
  --falkor-graph fawkes_clinical_kg \
  --list-graphs \
  --limit 10
```

Ask an ad-hoc question for one admission graph:

```bash
uv run fawkes_falkor_qa.py \
  --falkor-url redis://USER:PASSWORD@HOST:PORT \
  --falkor-graph fawkes_clinical_kg \
  --graph-ids SUBJECT_ID:HADM_ID \
  --question "What are the main diagnoses and treatments documented for this admission?"
```

You can also resolve by `subject_id` or `hadm_id` with `--ids`, or pass a JSON
list of questions with `--questions-json questions.json`.

Generation providers:

- `--provider auto`: try `QA_ENDPOINT_URL`, then OpenRouter, then Ollama.
- `--provider endpoint`: use a chat-completions-compatible endpoint.
- `--provider openrouter`: requires `OPENROUTER_API_KEY`.
- `--provider ollama`: uses `OLLAMA_API_URL` and `OLLAMA_MODEL`.
- `--retrieve-only`: save retrieved context and prompts without LLM generation.

Retrieval is lexical by default so the QA script stays lightweight. For embedding
retrieval, install `sentence-transformers` and run with
`--retriever embedding`.
