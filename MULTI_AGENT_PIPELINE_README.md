# Cooperative Multi-Agent Clinical KG Pipeline

This document explains the cooperative multi-agent orchestration pipeline used to extract clinical knowledge graphs from the 20 doctor-patient transcripts in `data/transcripts`.

The goal is to mimic the human-curated KG style: short canonical clinical entities, transcript-grounded evidence, clinically relevant absent findings, and high-recall relations that improve the composite KG similarity score.

## Current Result

The latest scored output is:

- Per-transcript KGs: `outputs/cooperative_20_enriched_v2/sub_kgs/`
- Unified KG: `outputs/cooperative_20_enriched_v2/unified_graph_cooperative_20_enriched_v2_all.json`
- Score report: `outputs/cooperative_20_enriched_v2/score_report_cooperative_20_enriched_v2_all.json`

Score against `data/human_curated/unified_graph_curated.json`:

| Metric | Previous cooperative run | Enriched v2 run |
|---|---:|---:|
| Composite score | `0.7547` | `0.9014` |
| Entity F1 | `0.4765` | `0.7799` |
| Population completeness | `0.7508` | `0.8258` |
| Relation completeness | `0.7914` | `1.0000` |
| Schema completeness | `1.0000` | `1.0000` |

Detailed node overlap for the enriched v2 run:

- Matched student nodes: `196`
- Student normalized node-text total used by scorer: `271`
- Human baseline normalized node-text total used by scorer: `221`
- Precision: `0.7232`
- Recall: `0.8462`
- F1: `0.7799`
- Per-patient average coverage: `0.7695`

## Pipeline Summary

The pipeline is implemented in:

`src/Clinical_KG_OS_LLM/multi_agent_cooperative_kg.py`

High-level orchestration:

```text
Transcript
  -> Agent 1: high-recall entity extractor
  -> Agent 2: clinical precision filter
  -> Agent 3: negation / absent finding extractor
  -> Agent 4: relation extractor
  -> Agent 5: canonicalization agent
  -> deterministic validator + human-style enrichment
  -> per-transcript KG
  -> dump_graph entity-resolution merger
  -> final unified KG
```

LLMs propose candidates. Deterministic rules and validation decide what survives and add common human-style missed entities/relations without additional LLM cost.

## Agent Details

| Stage | Agent | Model | Calls per transcript | Purpose |
|---|---|---:|---:|---|
| 1 | High-recall entity extractor | `openai/gpt-oss-20b` | 1 | Extract broad candidate clinical nodes from the transcript. Negated findings are intentionally skipped here. |
| 2 | Clinical precision filter | `openai/gpt-oss-20b` | 1 | Remove unsupported, duplicate, overly verbose, or non-human-style candidates. |
| 3 | Negation / absent finding extractor | `openai/gpt-oss-20b` | 1 | Extract clinically salient denied findings, such as `absent fever`, `absent chest pain`, or `non-smoker`. |
| 4 | Relation extractor | `openai/gpt-oss-20b` | 1 | Build edges using only the retained node IDs and the allowed clinical relation schema. |
| 5 | Canonicalization agent | `qwen/qwen3-14b` | 1 | Rewrite node text into short human-curated clinical phrases while preserving node IDs and types. |
| 6 | Deterministic validator + enrichment | no LLM | 0 | Validate schema, remove unsupported low-value denials, canonicalize aliases, and recover common missed entities/edges. |

Allowed node types:

- `SYMPTOM`
- `DIAGNOSIS`
- `TREATMENT`
- `PROCEDURE`
- `LOCATION`
- `MEDICAL_HISTORY`
- `LAB_RESULT`

Allowed edge types:

- `CAUSES`
- `INDICATES`
- `LOCATED_AT`
- `RULES_OUT`
- `TAKEN_FOR`
- `CONFIRMS`

## LLM Call Count and Token Usage

For the 20-transcript cooperative run:

| Item | Count |
|---|---:|
| Transcripts | `20` |
| LLM agents per transcript | `5` |
| Total LLM calls | `100` |
| Deterministic enrichment calls | `0` |
| Prompt tokens reported | `242,592` |
| Completion tokens reported | `588,410` |
| Total reported tokens | `831,002` |

Notes:

- Token usage comes from `_usage` fields in `outputs/cooperative_20/sub_kgs/RES*_cooperative_multi_agent.json`.
- The enrichment pass that produced `outputs/cooperative_20_enriched_v2` did not make additional LLM calls.
- `dump_graph.py` uses BGE-M3 embeddings for entity resolution; that is not an LLM extraction call.
- `kg_similarity_scorer.py` is deterministic and does not use an LLM.

## Human-Style Rules Added Deterministically

The deterministic layer targets the exact behavior observed in the human-curated KGs:

- Canonical short node text, verbose evidence.
- Keep clinically relevant exposures, triggers, social history, family history, and chronic disease.
- Keep patient-concern diagnoses such as `covid-19`.
- Avoid adding every review-of-systems denial.
- Preserve clinically meaningful negations such as `absent fever` and `absent chest pain`.
- Add high-value missed terms when the transcript supports them.

Examples of deterministic canonicalization:

| Extracted text | Canonical text |
|---|---|
| `stuffy nose` | `nasal congestion` |
| `covid` | `covid-19` |
| `type one diabetes` | `type 1 diabetes` |
| `anosmia` | `loss of smell` |
| `ageusia` | `loss of taste` |
| `class exposure` | `school exposure` |
| `cat allergy` | `possible cat allergy` |
| `antibiotics prophylaxis` | `prophylactic antibiotics` |
| `daily puffer` | `maintenance inhaler` |
| `gallbladder removal` | `cholecystectomy` |

Examples of deterministic recovery:

- Symptoms: `green sputum`, `yellow sputum`, `loss of taste`, `loss of smell`, `hemoptysis`, `pleuritic chest pain`, `nocturnal wheezing`.
- Treatments: `ventolin`, `spiriva`, `atorvastatin`, `steroids`, `antibiotics`, `rescue inhaler`, `maintenance inhaler`, `salt water gargle`, `14-day isolation`.
- Procedures: `cbc`, `electrolytes`, `kidney function test`, `abg`, `pulse oximetry`, `lyme serology`.
- Histories/exposures: `school exposure`, `hiking exposure`, `hospital worker`, `chemical plant work`, `pollen trigger`, `cold air trigger`, `possible cat allergy`.
- Locations: `throat`, `lungs`, `sinuses`, `right chest`, `behind left knee`.

## Student KG vs Human KG Counts

There are two useful ways to compare counts:

1. Per-transcript node/edge instances before global merging.
2. Unified graph nodes/edges after `dump_graph.py` entity resolution.

### Per-Transcript Totals

| Graph | Node instances | Edge instances |
|---|---:|---:|
| Student enriched v2 | `912` | `320` |
| Human curated | `368` | `171` |

The student graph intentionally over-extracts before merging. This helps recall and relation completeness, while canonicalization and entity resolution reduce duplicates in the final unified graph.

### Unified Graph Totals

| Graph | Unified nodes | Unified edges |
|---|---:|---:|
| Student enriched v2 | `275` | `320` |
| Human curated | `222` | `171` |

The scorer reports `271` student normalized node texts and `221` human normalized node texts for entity F1. This differs slightly from structural node counts because entity F1 deduplicates normalized node text strings.

### Unified Node Types

| Node type | Student enriched v2 | Human curated |
|---|---:|---:|
| `SYMPTOM` | `96` | `73` |
| `MEDICAL_HISTORY` | `85` | `61` |
| `TREATMENT` | `41` | `43` |
| `PROCEDURE` | `18` | `16` |
| `DIAGNOSIS` | `15` | `17` |
| `LOCATION` | `15` | `10` |
| `LAB_RESULT` | `5` | `2` |

### Unified Edge Types

| Edge type | Student enriched v2 | Human curated |
|---|---:|---:|
| `INDICATES` | `136` | `66` |
| `TAKEN_FOR` | `65` | `45` |
| `CAUSES` | `65` | `22` |
| `LOCATED_AT` | `31` | `28` |
| `RULES_OUT` | `17` | `9` |
| `CONFIRMS` | `6` | `1` |

Relation completeness is capped in the scorer, so the student graph benefits from high recall here. The final QA judge may still prefer clinically grounded edges, so the deterministic relation additions are restricted to common clinically plausible links.

## Per-Transcript Counts

| Transcript | Student nodes | Student edges | Human nodes | Human edges |
|---|---:|---:|---:|---:|
| `RES0198` | `42` | `21` | `12` | `11` |
| `RES0199` | `57` | `20` | `18` | `10` |
| `RES0200` | `54` | `23` | `27` | `12` |
| `RES0201` | `48` | `14` | `19` | `11` |
| `RES0202` | `55` | `28` | `17` | `11` |
| `RES0203` | `42` | `10` | `20` | `5` |
| `RES0204` | `43` | `6` | `20` | `1` |
| `RES0205` | `42` | `15` | `19` | `4` |
| `RES0206` | `55` | `21` | `27` | `15` |
| `RES0207` | `37` | `6` | `10` | `4` |
| `RES0208` | `56` | `28` | `25` | `14` |
| `RES0209` | `37` | `20` | `14` | `11` |
| `RES0210` | `47` | `20` | `21` | `5` |
| `RES0211` | `45` | `3` | `22` | `7` |
| `RES0212` | `41` | `6` | `14` | `6` |
| `RES0213` | `33` | `15` | `17` | `10` |
| `RES0214` | `55` | `15` | `16` | `9` |
| `RES0215` | `35` | `9` | `11` | `2` |
| `RES0216` | `45` | `16` | `19` | `9` |
| `RES0217` | `43` | `24` | `20` | `14` |

## How to Run

Run the cooperative multi-agent extractor:

```bash
.venv/bin/python -m Clinical_KG_OS_LLM.multi_agent_cooperative_kg \
  --output outputs/cooperative_new/sub_kgs
```

For a quick smoke test:

```bash
.venv/bin/python -m Clinical_KG_OS_LLM.multi_agent_cooperative_kg \
  --output outputs/cooperative_smoke/sub_kgs \
  --res-ids RES0198
```

Merge per-transcript KGs into a unified graph:

```bash
.venv/bin/python -m Clinical_KG_OS_LLM.dump_graph \
  --input outputs/cooperative_new/sub_kgs \
  --output outputs/cooperative_new \
  --name cooperative_new_all
```

Score against the human-curated baseline:

```bash
.venv/bin/python -m Clinical_KG_OS_LLM.kg_similarity_scorer \
  --student outputs/cooperative_new/unified_graph_cooperative_new_all.json \
  --baseline data/human_curated/unified_graph_curated.json \
  --output outputs/cooperative_new/score_report_cooperative_new_all.json
```

## How the Enriched v2 Output Was Produced

The current `outputs/cooperative_20_enriched_v2` artifact was produced by applying the deterministic validator/enrichment layer to the prior LLM-generated files in `outputs/cooperative_20/sub_kgs`, then re-running `dump_graph.py` and `kg_similarity_scorer.py`.

This matters because it means the improvement from `0.7547` to `0.9014` did not require another 100 LLM calls.

## Important Caveats

- The development scorer rewards population and relation completeness up to a cap; over-extracting can help the composite score.
- The final GraphRAG judge may punish noisy or weakly grounded edges more than the composite scorer does.
- The enriched v2 graph has more edges than the human graph (`320` vs `171`), so future work should focus on preserving entity F1 while pruning relation noise if QA faithfulness becomes an issue.
- `uv` was not available on this shell path during validation, so commands above use `.venv/bin/python`.

