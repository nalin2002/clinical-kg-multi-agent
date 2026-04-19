# KARMA Pipeline — How to Run

A four-agent KG extraction pipeline that replaces the naive single-pass
implementation.  Drop-in compatible with `dump_graph.py` and
`kg_similarity_scorer.py`.

## What it does

| Agent | What it does | LLM? |
|-------|--------------|------|
| 1. Extractor | Broad base extraction with a prompt that explicitly asks for all 7 node types — especially the commonly-missed `LOCATION` and `LAB_RESULT`. | Yes (1 call) |
| 2. Completeness Enhancer | Second pass over the transcript + pass-1 KG. Targets missed body-part `LOCATION`s, vitals / lab `LAB_RESULT`s, `MEDICAL_HISTORY` risk factors, and missing edges. Only adds *new* items. | Yes (1 call) |
| 3. Schema Agent | Rule-based. Coerces node & edge types into the allowed taxonomy, fixes mis-labels (e.g. `MEDICATION → TREATMENT`), normalizes `turn_id`, drops invalid items. | No |
| 4. Entity + Conflict Resolver | Rule-based. Within-patient dedup, drops duplicate edges, self-loops, and dangling-id edges. | No |

Cross-patient entity resolution still happens downstream via `dump_graph.py`
(BGE-M3, 0.85 cosine), so the per-patient output just needs to be internally
clean.

## 1. Add your OpenRouter API key

```bash
cp api_keys_example.json api_keys.json
# then edit api_keys.json and paste your key
```

## 2. Quick iteration on a 3-patient subset

```bash
# fast dev loop: ~2 calls × 3 transcripts = 6 LLM calls
uv run python -m Clinical_KG_OS_LLM.karma_kg_extraction \
  --output ./my_kg_karma \
  --res-ids RES0198 RES0199 RES0200

# merge into a unified graph (uses BGE-M3 embedding ER)
uv run python -m Clinical_KG_OS_LLM.dump_graph \
  --input ./my_kg_karma \
  --output ./my_kg_karma_unified

# score
uv run python -m Clinical_KG_OS_LLM.kg_similarity_scorer \
  --student ./my_kg_karma_unified/unified_graph_my_kg_karma.json \
  --baseline ./data/human_curated/unified_graph_curated.json
```

Note: the 3-patient subset score won't be directly comparable to the 0.562
full-20 naive score — it's useful for seeing whether the agents are producing
sensible nodes/edges.

## 3. Full 20-patient run (submission)

```bash
uv run python -m Clinical_KG_OS_LLM.karma_kg_extraction --output ./my_kg_karma_full
uv run python -m Clinical_KG_OS_LLM.dump_graph  --input ./my_kg_karma_full --output ./submission
uv run python -m Clinical_KG_OS_LLM.kg_similarity_scorer \
  --student ./submission/unified_graph_my_kg_karma_full.json \
  --baseline ./data/human_curated/unified_graph_curated.json
```

The file `./submission/unified_graph_my_kg_karma_full.json` is what you
submit for organizer evaluation.

## 4. Cost controls

| Flag | Effect | Calls per transcript |
|------|--------|----------------------|
| (default) | Extractor + Enhancer + Schema + Conflict | 2 |
| `--no-enhancer` | Skip the Completeness Enhancer | 1 |
| `--model qwen/qwen3-14b` | Swap the model | 2 |

Total LLM calls for full run: 40 (default) or 20 (`--no-enhancer`).

## 5. Expected score impact

Baseline (naive, full 20):

```
Composite 0.562 │ Entity F1 0.428 │ Population 0.432 │ Relation 0.674 │ Schema 0.714
```

Where the naive falls short and how KARMA addresses it:

- **Schema Completeness 0.714** (5/7 types — missing `LOCATION` and `LAB_RESULT`).
  Agent 1's prompt explicitly targets these; Agent 2 specifically hunts for
  them.  Target: 1.0.
- **Population Completeness 0.432** (only 144 of ~222 baseline nodes).  The
  composite formula rewards up to 1.5× baseline (~330 nodes), so the Enhancer's
  additional nodes compound directly into this component.
- **Entity F1 0.428**.  More grounded, clinically-salient extractions lift both
  precision and recall.

See `validate_karma_offline.py` for a no-API-key sanity run against the
naive sub-KGs.

## 6. What's written

Per-patient sub-KG files follow the same schema as
`data/naive_results/sub_kgs/RES0198_naive.json` plus a `_meta` block:

```json
{
  "nodes": [...],
  "edges": [...],
  "_meta": {
    "pipeline": "karma",
    "model": "z-ai/glm-4.7-flash",
    "pass1": {"nodes": 22, "edges": 18},
    "pass2": {"nodes_added": 9, "edges_added": 6},
    "schema": {"node_type_remapped": 1, ...},
    "dedupe": {"duplicate_edges_dropped": 2, ...}
  },
  "_usage": {"prompt": 5421, "completion": 842}
}
```

`dump_graph.py` ignores `_meta` / `_usage` and merges the nodes/edges exactly
as before.
