# graph_jepa / common

Shared foundation imported by every training and evaluation module.

| File | Purpose |
|---|---|
| `graph_schema.py` | Canonical `Graph`/`Node`/`Edge`; 7 node types, 6 relations, typed `RELATION_SCHEMA`; converters from the EIR schema; `normalize_text`, `validate_graph`. |
| `io_utils.py` | `load_config` (+ path resolution), `discover_patients`, `split_patients` (seeded, leak-free), graph/JSON/JSONL I/O, logging. |
| `graph_utils.py` | corruptions (`sample_negatives`, `corrupt_graph`), `fully_connected_candidates`, and metrics (`edge_prf`, `relation_accuracy`, `graph_edit_distance_approx`). |
| `encoders.py` | text encoder for featurization — `sentence-transformers` (BGE-M3) with a deterministic **hashing fallback** so everything runs offline. |
| `llm_utils.py` | provider-agnostic `LLMClient` (anthropic/openai/gemini/openrouter/**mock**), robust JSON extraction, `judge_answer`. |
| `eir_adapter.py` | integration with the EIR 13-agent extractor (wrap, don't rewrite). |
| `prompts/` | prompt templates: silver extraction, QA generation, graph QA answering, judge. |

## Schema
Canonical on-disk form: `nodes[].{id,name,type,normalized_name,evidence,turn_id}`,
`edges[].{source,target,relation,evidence,confidence}`. EIR's
`nodes[].text` / `edges[].source_id,type` is auto-converted on load.

## EIR integration notes
`get_llm_graphs()` resolves LLM graphs by, in order:
1. scanning `eir.existing_kg_dirs` for already-extracted per-patient KGs
   (filenames are fuzzily matched to patient ids), then
2. if `eir.run_eir: true`, shelling out to `eir.entry_script` for the rest.

EIR is a heavy batch pipeline needing an OpenRouter key and its own transcript
dir. **Recommended:** run EIR yourself on `data/aci_bench`, then point
`eir.existing_kg_dirs` at its raw per-patient output. The bundled curated EIR KGs
(`EIR_260426/eir_aci_bench/transcripts`) are a convenient smoke-test stand-in but
are *curated*, not raw LLM output — don't use them as the `llm_only` baseline for
final numbers.

## Offline mode
`llm.provider: mock` + `encoder.backend: hashing` runs the entire pipeline with
no API keys or model downloads (deterministic stubs, logged as MOCK). For
plumbing/CI only — never for reported results.
