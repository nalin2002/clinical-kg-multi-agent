## Human-Curated KG Data Analysis

This analysis summarizes the **human-curated per-transcript KGs** (`data/transcripts/RES*/RES*_curated_kg.json`) and the merged curated graph (`data/human_curated/unified_graph_curated.json`) to help design a stronger extraction pipeline.

## 1) Dataset Size and Density (Per-Transcript Curated KGs)

- Transcripts analyzed: `20`
- Total node instances across transcripts: `368`
- Total edge instances across transcripts: `171`
- Avg nodes / transcript: `18.4` (range: `10` to `27`)
- Avg edges / transcript: `8.55` (range: `1` to `15`)

Per-transcript human-curated counts:

| Transcript | Nodes | Entities | Edges |
|---|---:|---:|---:|
| RES0198 | 12 | 12 | 11 |
| RES0199 | 18 | 18 | 10 |
| RES0200 | 27 | 27 | 12 |
| RES0201 | 19 | 19 | 11 |
| RES0202 | 17 | 17 | 11 |
| RES0203 | 20 | 20 | 5 |
| RES0204 | 20 | 20 | 1 |
| RES0205 | 19 | 19 | 4 |
| RES0206 | 27 | 27 | 15 |
| RES0207 | 10 | 10 | 4 |
| RES0208 | 25 | 25 | 14 |
| RES0209 | 14 | 14 | 11 |
| RES0210 | 21 | 21 | 5 |
| RES0211 | 22 | 22 | 7 |
| RES0212 | 14 | 14 | 6 |
| RES0213 | 17 | 17 | 10 |
| RES0214 | 16 | 16 | 9 |
| RES0215 | 11 | 11 | 2 |
| RES0216 | 19 | 19 | 9 |
| RES0217 | 20 | 20 | 14 |
| **Total** | **368** | **368** | **171** |

Note: in these curated per-transcript KGs, `Entities` equals `Nodes` because each node corresponds to one curated entity mention.

Interpretation:
- The target is moderately dense on entities, but relation density varies a lot by case.
- Some cases are symptom-heavy with sparse relations; others are tightly connected.

## 2) Node Type Distribution (Per-Transcript Curated Totals)

- `SYMPTOM`: `138` (largest)
- `MEDICAL_HISTORY`: `88`
- `TREATMENT`: `52`
- `PROCEDURE`: `34`
- `DIAGNOSIS`: `29`
- `LOCATION`: `25`
- `LAB_RESULT`: `2` (very rare)

Implications:
- Recall on `SYMPTOM` + `MEDICAL_HISTORY` dominates.
- `LAB_RESULT` contributes little volume but can still matter for precision.

## 3) Edge Type Distribution (Per-Transcript Curated Totals)

- `INDICATES`: `66` (largest)
- `TAKEN_FOR`: `45`
- `LOCATED_AT`: `28`
- `CAUSES`: `22`
- `RULES_OUT`: `9`
- `CONFIRMS`: `1` (very rare)

Implications:
- Most relation signal is in `INDICATES` and `TAKEN_FOR`.
- `CONFIRMS` is almost absent; over-predicting it will hurt precision.

## 4) Unified Curated Graph Snapshot

From `data/human_curated/unified_graph_curated.json`:

- Unified nodes: `222`
- Unified edges: `171`
- Total node occurrences stored in `occurrences`: `368`

Unified node type counts:
- `SYMPTOM` 73, `MEDICAL_HISTORY` 61, `TREATMENT` 43, `DIAGNOSIS` 17, `PROCEDURE` 16, `LOCATION` 10, `LAB_RESULT` 2

Unified edge relation counts:
- `INDICATES`: `66`
- `TAKEN_FOR`: `45`
- `LOCATED_AT`: `28`
- `CAUSES`: `22`
- `RULES_OUT`: `9`
- `CONFIRMS`: `1`

Relation schema present in curated unified KG:
- `INDICATES`, `TAKEN_FOR`, `LOCATED_AT`, `CAUSES`, `RULES_OUT`, `CONFIRMS`

The edge counts in unified match per-transcript totals (171), which is expected after merge + canonicalization.

## 5) Edge Relation by Entity-Type Pairs (Unified Curated KG)

This shows how each relation is distributed across `(source_type -> target_type)` in `data/human_curated/unified_graph_curated.json`.

- `INDICATES` (`66` total)
  - `SYMPTOM -> DIAGNOSIS`: `49`
  - `SYMPTOM -> MEDICAL_HISTORY`: `6`
  - `PROCEDURE -> DIAGNOSIS`: `5`
  - `MEDICAL_HISTORY -> DIAGNOSIS`: `4`
  - `DIAGNOSIS -> DIAGNOSIS`: `1`
  - `MEDICAL_HISTORY -> MEDICAL_HISTORY`: `1`

- `TAKEN_FOR` (`45` total)
  - `TREATMENT -> MEDICAL_HISTORY`: `24`
  - `TREATMENT -> DIAGNOSIS`: `13`
  - `TREATMENT -> SYMPTOM`: `8`

- `LOCATED_AT` (`28` total)
  - `SYMPTOM -> LOCATION`: `25`
  - `DIAGNOSIS -> LOCATION`: `1`
  - `MEDICAL_HISTORY -> LOCATION`: `1`
  - `PROCEDURE -> LOCATION`: `1`

- `CAUSES` (`22` total)
  - `MEDICAL_HISTORY -> DIAGNOSIS`: `7`
  - `MEDICAL_HISTORY -> MEDICAL_HISTORY`: `7`
  - `MEDICAL_HISTORY -> SYMPTOM`: `3`
  - `SYMPTOM -> SYMPTOM`: `3`
  - `DIAGNOSIS -> DIAGNOSIS`: `1`
  - `TREATMENT -> SYMPTOM`: `1`

- `RULES_OUT` (`9` total)
  - `PROCEDURE -> DIAGNOSIS`: `6`
  - `SYMPTOM -> DIAGNOSIS`: `2`
  - `SYMPTOM -> MEDICAL_HISTORY`: `1`

- `CONFIRMS` (`1` total)
  - `LAB_RESULT -> SYMPTOM`: `1`

Interpretation:
- Most strong structural signal is `SYMPTOM -> DIAGNOSIS` (`INDICATES`) and `TREATMENT -> {MEDICAL_HISTORY, DIAGNOSIS}` (`TAKEN_FOR`).
- `LOCATED_AT` is overwhelmingly `SYMPTOM -> LOCATION`.
- Rare relation/type-pair combos exist; overproducing those can hurt precision.

## 6) Most Frequent Curated Entities (Cross-Transcript)

Top repeated `(type, text)` in per-transcript curated KGs:

- `DIAGNOSIS`: `covid-19` (10)
- `PROCEDURE`: `covid swab` (10)
- `LOCATION`: `chest` (10)
- `SYMPTOM`: `dry cough` (9)
- `SYMPTOM`: `fatigue` (7)
- `SYMPTOM`: `shortness of breath` (7)
- `MEDICAL_HISTORY`: `non-smoker` (6)
- `SYMPTOM`: `fever` (6)
- `PROCEDURE`: `physical exam` (6)

Design note:
- Normalization quality for common respiratory terms has high leverage.

## 7) Most Frequent Curated Relations (Canonical Triples)

Top repeated `(source_text, relation, target_text)`:

- `covid swab` `RULES_OUT` `covid-19` (5)
- `self-isolation` `TAKEN_FOR` `covid-19` (4)
- `sore throat` `LOCATED_AT` `throat` (4)
- `dry cough` `INDICATES` `covid-19` (3)
- `loss of smell` `INDICATES` `covid-19` (3)
- `loss of taste` `INDICATES` `covid-19` (3)

Design note:
- Strongly suggests relation extraction should be conservative but robust for common diagnostic/treatment links.

## 8) Per-Transcript Variability (Why one prompt can fail)

High-node transcripts:
- `RES0200` (27 nodes, 12 edges)
- `RES0206` (27 nodes, 15 edges)
- `RES0208` (25 nodes, 14 edges)

Sparse-edge transcripts:
- `RES0204` (20 nodes, 1 edge)
- `RES0215` (11 nodes, 2 edges)

Takeaway:
- Cases are heterogeneous; a successful pipeline likely needs either:
  - role-specialized extraction (symptoms/history/plan), or
  - staged node-then-edge extraction with repair/enrichment.

