1. v8 and v9 are complete messes. Best approach would have been to just make v8 not use the same examples.

2. No-corpus run insight (260425): when entity prompts get an empty `<curated_labels>` block, both `matched` and `fine_grained` buckets should logically be empty (matched needs a list to match against; fine_grained is defined as "finer than entries in <curated_labels>" — no entries, no reference for "finer than"). Everything legitimately lands in `other`. The converter flattens all three buckets into flat nodes regardless, so the scorer's curator-alignment still works. Side effect to note: Stage 1 emits more entities into `other` when there's no curated anchor, which makes Stage 2 edge prompts larger (~15k chars vs ~11k), increases per-call latency, and amplifies upstream-deepseek 429 pressure.

## Hackathon-organizer documentation — scope (TODO; format like Astraeus, HTML, detailed)

- How `eir/curated_kb_260419.json` was built from the organizer's `data/human_curated/unified_graph_curated.json` + per-patient `data/transcripts/RES0XXX/RES0XXX_curated_kg.json` files
- The 7 entity prompts (SYMPTOM, DIAGNOSIS, TREATMENT, PROCEDURE, LOCATION, MEDICAL_HISTORY, LAB_RESULT) + 6 edge prompts (INDICATES, CAUSES, CONFIRMS, RULES_OUT, TAKEN_FOR, LOCATED_AT) — what each category targets and why split this way
- Stage 1 vs Stage 2: model choice (glm-4.7-flash for entities vs deepseek-r1 reasoning for edges), why a reasoning model for edges, why the cost/latency split
- Matched / Other / Fine Grained semantics: when each is populated, how they collapse in the converter, why the asymmetry surfaces on no-corpus runs
- Convert → dump_graph (BGE-M3 ER @ 0.85 cosine) → kg_similarity_scorer chain
- Run-by-run progression v7 → v8 → v9 → v10 with composite scores and what changed prompt-side vs runtime-side at each step
- How to swap the curated KB at runtime via the `EIR_CURATED_KB_PATH` env var:
    - default: `eir/curated_kb_260419.json` (in-corpus 222 labels mined from `data/human_curated/unified_graph_curated.json`)
    - alternative shipped: `eir/curated_synthetic_kb_combined_260426.json` (522 labels = 222 in-corpus + 300 ACI-bench synthetic)
    - schema: list of `{entity_id, label, entity_type, description, aliases, source, source_code}`; merge new KBs by `(label, entity_type)` to dedupe
    - how to construct your own KB from a `unified_graph_*.json` (any pipeline run can become a KB seed for the next iteration)
- The OoC eval bundle (`eir/eir_aci_bench/`): 20 ACI-Bench transcripts pulled from `mkieffer/ACI-Bench`, paired with hand-curated KGs (Claude-curated, 558 nodes / 261 edges total), plus the BGE-M3-merged unified graph used as the OoC scoring baseline (`unified_graph_curated_aci.json`)

## Why a fast model for entities (Stage 1) and a reasoning model for edges (Stage 2)

**Stage 1 — `z-ai/glm-4.7-flash` (entities).**
Entity extraction is mostly retrieval + light classification: read transcript, identify spans that name a clinical concept, normalize to a curated label or near-synonym, output JSON. Single forward pass per category. No multi-step inference. A fast/small model is sufficient and the right cost trade-off:
- 7 entity calls × 20 patients = 140 calls per batch, each emitting ~3k output tokens
- Reasoning-model pricing on this volume would dominate the budget for marginal quality gain
- Latency matters at this stage because Stage 2 can't start without it (Stage 2 needs Stage 1 entity lists as input)

**Stage 2 — `deepseek/deepseek-r1-distill-qwen-32b` (edges).**
Edge extraction is multi-step inference, not retrieval:
- Given two extracted entities, decide whether a clinical relation exists between them
- Disambiguate edge type when context allows multiple (covid swab → covid-19 could be RULES_OUT or CONFIRMS depending on the doctor's framing)
- Handle negation correctly (`absent X` doesn't INDICATE a diagnosis but may RULES_OUT it)
- Weigh competing differentials (same symptom may INDICATE two diagnoses with different strengths)
- Pick the right evidence quote across multiple candidate turns

These tasks benefit from a reasoning model's hidden-trace deliberation. Empirically: when we tried fast models on edges in earlier versions, we saw edge-type confusion (RULES_OUT covid swabs misclassified as INDICATES) and weak src/dst pairings. Reasoning fixed both.

Cost is bounded for Stage 2: 6 calls × 20 patients = 120 calls (fewer than Stage 1); inputs are smaller (entity list + transcript, no curated_labels block); the intellectual heavy lifting is concentrated here, where reasoning's quality lift maps directly to Relation Completeness (25% of composite).

**Asymmetric model choice = right tool for each job.** Fast model for high-volume retrieval; reasoning model for low-volume inference.

### Volume / cost / latency — Stage 1 vs Stage 2

Numbers from v10 baseline run (260425_181459, 20/20 OK, 22.2 min):

| | Stage 1 (entities) | Stage 2 (edges) |
|---|---|---|
| Model | `z-ai/glm-4.7-flash` | `deepseek/deepseek-r1-distill-qwen-32b` |
| Purpose | retrieval + classification | multi-step inference |
| Calls per batch | 140 (7 cats × 20 patients) | 120 (6 cats × 20 patients) |
| Streaming | yes (chunked) | no (full response) |
| Reasoning trace | none | ~3.2k hidden chars per call |
| Per-call in tokens | ~3.3k | ~3.7k |
| Per-call out tokens | ~3.2k (visible JSON) | ~1.0k visible + ~3.2k hidden reasoning |
| Per-call latency | ~58s avg | ~75s avg |
| Cost-per-token | low (~$0.10/M) | higher (~$0.50/M, 5× Stage 1) |
| Total volume per batch | higher token throughput | fewer calls but heavier compute |
| Wall-clock with parallelism | bounded by slowest cat per patient | bounded by slowest edge per patient + retry tail (deepseek 429 cascades) |

Net: Stage 1 has more calls but cheap per-token; Stage 2 has fewer calls but premium per-token. Roughly cost-balanced. Latency is dominated by Stage 2 — the 75 s/call reasoning latency is the critical path of every batch.

## Why multi-agent fan-out, not transcript-based routing

Every transcript runs all 13 categories (7 entity + 6 edge) — we deliberately do NOT route based on transcript content. Rationale:

1. **Routing requires reading the transcript first** — a pre-pass LLM call per patient just to classify which agents to skip. Adds latency, another failure point, another cost.
2. **Recall safety** — a router misclassification ("no labs in this transcript") permanently drops that category. With "run all", every LAB_RESULT call still gets a chance to find `BP 148/90` mentioned in passing. Routing trades recall for cost; we chose recall.
3. **Empty categories are cheap** — when a transcript truly has nothing for a category, the model returns empty buckets quickly. Saving ~4 such calls per patient × 20 patients ≈ 80 calls; the routing pre-pass costs ~20 calls. Net savings is small.
4. **Schema Completeness component (25% of organizer composite)** — missing any of the 7 entity types or 6 edge types in the unified graph penalizes us. Always running all 13 guarantees a chance at each.
5. **Parallel fan-out** — within a transcript the 13 categories run concurrently; wall-clock is bounded by the slowest single category, not the sum. Routing to fewer doesn't shrink the critical path much.
6. **Deterministic shape** — every patient yields the same 13-file fan-out, which simplifies the converter, dump_graph, scoring, and any downstream tooling.
7. **Prompt specialization** — each of the 13 prompts is tuned to its category's semantics (denials in SYMPTOM, family-history naming in MEDICAL_HISTORY, src/dst type patterns in edges). One omnibus prompt can't carry that specialization without bloating to ~80k tokens.

## Why split calls by category — even though it means 13× more calls per transcript than a single omnibus prompt

The split (7 entity + 6 edge categories per transcript) is intentional. It trades raw call count for quality, reliability, and modularity:

1. **Per-category prompt specialization.** Each prompt is tuned to its category's semantics — SYMPTOM has denial rules, MEDICAL_HISTORY has family-history naming, edge prompts have src/dst type patterns and OoC anchors specific to that relation. One omnibus prompt would either drop these specializations or balloon past 80k tokens.

2. **Forced category coverage.** With per-category prompts the model is explicitly asked to look for X. With one omnibus prompt, the model tends to extract whatever's most salient and skip the less obvious categories — populating the easy ones (SYMPTOM, DIAGNOSIS) and missing the rare ones (LAB_RESULT, CONFIRMS).

3. **Failure isolation.** If `INDICATES` parsing fails on RES0202, only that one cell goes empty (1/120 edges). With one omnibus prompt, a single failure wipes out all 13 categories for that patient — we saw exactly this in v8 workers=8 where 3 patients lost all 6 edge categories because one upstream call timed out.

4. **Independent retry granularity.** A 429 on TAKEN_FOR doesn't block CAUSES, RULES_OUT, etc. Each category retries on its own budget. With one big call, a single retry replays the entire 80k-token prompt.

5. **Parallel fan-out within a transcript.** 13 categories run concurrently per transcript, wall-clock = max(category latency), not sum. One omnibus prompt is serial — one massive call must finish before any extraction is usable.

6. **Asymmetric model choice depends on the split.** We can run entities on `glm-4.7-flash` and edges on `deepseek-r1` only because the calls are separable. One omnibus prompt forces one model for everything — and you'd pick the more expensive one to cover the harder edge logic, paying premium-per-token on retrieval-class tasks.

7. **Cleaner output contracts.** Each category has a focused JSON shape (matched / other / fine_grained with category-specific fields). One omnibus output would need a 13-section schema, harder to parse, harder to validate, more prone to JSON errors and partial outputs.

8. **Iterability.** When v9 INDICATES needed fixing, we edited only INDICATES_PROMPT and re-ran. With one omnibus prompt, every prompt edit risks regressing all 13 categories — slower iteration and higher rollback risk.

9. **Per-category telemetry.** We can see "INDICATES hit% = 45.5%, TAKEN_FOR hit% = 73.3%" because calls are tagged by category. With one omnibus call, error attribution becomes a parsing problem.

**Cost bound:** the extra calls are absorbed by parallelism (wall-clock unchanged) and by the cheap-model-for-high-volume choice (Stage 1 at ~$0.10/M tokens, not the reasoning model). The 260 calls per batch don't cost meaningfully more than a single omnibus prompt would, because each is small (~3-4k input tokens). The 13× call count is essentially free — and buys all the points above.

## Why we pass the whole transcript to each call (not a summary)

1. **Verbatim quote requirement.** Every emitted entity/edge must cite a `turn_id` and a `quote: "..."` taken verbatim from the transcript. A summary loses both — turn IDs disappear and exact phrasings get rewritten.
2. **Synonym bridging needs raw phrasing.** Patient says "felt really hot" → curator wants `subjective fever`. The whole point of synonym mapping is to bridge lay phrasings to curated terms. A summary normalizes "felt really hot" to "fever" upstream — the bridge it's supposed to test is already collapsed.
3. **Denials require originals.** "No fever" vs "fever" vs "I had a fever last year" — a summary often keeps one and drops the others, or fuses them into "denies fever." Our `absent X` denial form depends on seeing the raw P-N response in question context.
4. **Subjective/objective tie-breaks need the discriminator.** `pleuritic chest pain` vs bare `chest pain` is decided by whether the patient said "worse with deep breath." A summary like "patient has chest pain" drops the qualifier we need.
5. **Color priority, compound preservation, phonetic variants** — all our prompt rules operate on raw phrasings. "Green yellowy sputum" → first color = green. A summary would write "green-yellow sputum" or "discolored sputum" and the rule has nothing left to fire on.
6. **No single summary serves 13 categories.** Each prompt looks for different things. SYMPTOM cares about denials and qualifiers; MEDICAL_HISTORY cares about family-history phrasing and exposure descriptions; edge prompts care about doctor framing ("we should rule out X" vs "this confirms X"). A pre-summarizer would have to preserve everything for everyone — at which point it's not a summary anymore.
7. **Information bottleneck.** Summary is lossy and we can't undo it. A category's prompt may need a single phrase the summarizer dropped — and we'd never know.
8. **Cost is small.** Transcripts are ~5-7k chars (~1.5k tokens). 13 calls × 1.5k = 20k tokens of transcript per patient. A summary call would still cost ~3k tokens to produce + per-category summary tokens ≈ same volume, with an extra failure point and quality loss. The saving doesn't exist.
9. **Reasoning model (Stage 2) needs the doctor's exact framing.** deepseek-r1 disambiguates INDICATES vs RULES_OUT vs CONFIRMS by reading "we should rule out", "this is consistent with", "we'll do a swab to confirm." A summary normalizes those into "doctor considers diagnosis X" and the disambiguation signal is gone.
10. **Multi-turn evidence.** Some edges require the symptom from one turn (P-1 "shortness of breath") and the diagnostic framing from another (D-43 "COPD exacerbation"). A summary may collapse those into a single line and lose the turn-level provenance.

## Agentic design pattern — Parallelization (Sectioning) Workflow, two-stage

In Anthropic's "Building Effective Agents" taxonomy, Eir is the **parallelization workflow** with **sectioning**: decompose a task into independent subtasks that run concurrently, then aggregate. We chain two stages of it:

```
                     ┌─────────────────────────────────────────────┐
   transcript ──►    │  Stage 1 — fan out 7 entity specialists     │
                     │  (SYMPTOM, DIAGNOSIS, TREATMENT, PROCEDURE, │
                     │   LOCATION, MEDICAL_HISTORY, LAB_RESULT)    │
                     │  parallel, glm-4.7-flash                    │
                     └────────────────────┬────────────────────────┘
                                          │ entity list
                                          ▼
                     ┌─────────────────────────────────────────────┐
                     │  Stage 2 — fan out 6 edge specialists       │
                     │  (INDICATES, CAUSES, CONFIRMS, RULES_OUT,   │
                     │   TAKEN_FOR, LOCATED_AT)                    │
                     │  parallel, deepseek-r1 (reasoning)          │
                     └────────────────────┬────────────────────────┘
                                          │ per-patient sub-KG
                                          ▼
                     ┌─────────────────────────────────────────────┐
                     │  Aggregation — convert → dump_graph         │
                     │  (BGE-M3 entity resolution) → unified KG    │
                     └─────────────────────────────────────────────┘
```

Why this pattern, not the others:
- Not **Routing** — we don't classify the transcript first and dispatch to a subset; every transcript runs all 13 specialists.
- Not **Orchestrator-Workers** — the decomposition is static (13 fixed categories), not chosen at runtime by an LLM orchestrator.
- Not **Evaluator-Optimizer** — no critic loop refining outputs.
- Not **Reflection** — no self-critique step.
- Not **ReAct / Tool use** — just LLM completions, no external tool calls during extraction.

What it IS:
- **Parallelization (sectioning)** — each specialist handles a different aspect of the same input concurrently.
- **Pipeline / staged workflow** — Stage 1's output is Stage 2's input via a deterministic handoff.
- **Specialist ensemble** — each of the 13 prompts is a role-specialized "agent" with its own model, temperature, reasoning toggle, and prompt tuning.

The aggregation step (BGE-M3 ER inside `dump_graph`) is the classic **fan-in / merge** that completes the parallelization pattern — bringing the 13 parallel outputs back into one canonical KG.

### Why we didn't adopt the other agentic patterns

**Routing (transcript-based dispatch).** Covered above in "Why multi-agent fan-out, not transcript-based routing": pre-pass LLM call per patient, recall safety (a misclassification permanently drops a category), Schema Completeness penalty in the composite, parallelism makes the cost savings tiny, deterministic shape simplifies downstream tooling.

**Orchestrator-Workers (LLM decides decomposition at runtime).**
- The decomposition is fixed by the curator's ontology — 7 entity types, 6 edge types. Nothing for an orchestrator to decide.
- An orchestrator adds another LLM call per transcript that doesn't produce final output, and introduces run-to-run variance.
- Static decomposition already covers the full schema; the dynamic alternative buys nothing.

**Evaluator-Optimizer (critic loop).**
- The natural evaluator IS the organizer's `kg_similarity_scorer`, which sits downstream of extraction and uses a curator graph we don't have at runtime in the holdout setting. We can't call it from inside extraction.
- LLM-as-judge alternatives would double the call count, add another failure mode, and still need quality criteria we'd have to define from scratch.
- We optimize at prompt-engineering iteration time (v7 → v10), not per-call — cheaper, reproducible, auditable.

**Reflection (self-critique).**
- Reflection at call level (extract → critique → re-extract) doubles latency and cost per call.
- Our errors are mostly recall gaps (model missing things), not precision problems (model emitting wrong things). Reflection helps precision more than recall.
- Single-pass approximations of reflection — the v9 "comprehensive review" / "recall floor" prompt rules — dropped precision without lifting recall enough to justify them. We removed them in v10.
- True multi-call reflection would amplify the same problem at higher cost.

**ReAct / Tool use.**
- The transcript is a self-contained input. No databases to query, no calculations to perform, no APIs to call during extraction.
- Tool use would add infrastructure (tool definitions, execution layer, retry/fallback) for zero quality gain.
- The reasoning model (deepseek-r1) does multi-step inference internally — that's the right place for deliberation, not externalized into tool calls.

**Common thread:** clinical KG extraction with a fixed ontology, a single text input, and a downstream similarity scorer is naturally a static parallel pipeline. The fancier patterns (orchestrator/evaluator/reflection/ReAct) add cost and variance that buy nothing in this domain — they're solutions to problems we don't have.

## Dependency on the seeded curated corpus — zero-shot is not viable

Our approach assumes the curated label list is passed into entity prompts via the `<curated_labels>` block. To test how much that anchor matters, we ran v10 with `load_curated_by_category()` monkey-patched to return empty lists for every category — same prompts, same model, same workers, same scoring; only the curator hint removed.

### Result (run 260425_222145, v10_no_corpus)

```
COMPOSITE SCORE: 0.815   (Grade A on the organizer's scale,
                         but composite is misleading here —
                         Schema and Relation both come back at
                         1.0 because the model emits at least
                         one of every type, while node-level
                         metrics collapse)

Component Scores:
  Entity F1 (25%):              0.262
  Population Completeness (25%): 1.000
  Relation Completeness (25%):   1.000
  Schema Completeness (25%):     1.000

Detailed Node Overlap:
  Precision: 18.71% (110/588 student nodes matched)
  Recall:    43.44% (110/221 baseline nodes covered)
  F1 Score:  26.15%

Structural:
  Student:   622 nodes, 312 edges, density 0.50
  Baseline:  222 nodes, 171 edges, density 0.77

Per-Patient Average Coverage: 44.83%
```

### Side-by-side vs v10 with the seeded corpus

| | v10 with corpus (`260425_181459`) | v10 no corpus (`260425_222145`) | Δ |
|---|---|---|---|
| Composite | **0.930** | 0.815 | −0.115 |
| Entity F1 | **0.721** | 0.262 | **−0.459** |
| Population | 1.000 | 1.000 | 0 |
| Relation | 1.000 | 1.000 | 0 |
| Schema | 1.000 | 1.000 | 0 |
| Node precision | 61.6% | **18.7%** | −42.9 pp |
| Node recall | 86.9% | **43.4%** | −43.5 pp |
| Node F1 | 72.1% | 26.2% | −45.9 pp |
| Per-patient avg | 88.3% | 44.8% | −43.5 pp |
| Student nodes | 339 | **622** (+83%) | model over-emits without anchor |
| Curator nodes hit | 204 | 110 | −94 (lost almost half) |

### Interpretation

Without the curator's vocabulary in the prompt, the model goes wide — emits 622 nodes (vs 339 with corpus) — but only 18.7% of those align to a curator label. Recall drops from 87% to 43%. The model isn't lazy when given no list; it's *unaligned*. It extracts plenty of clinically-valid concepts using its own internal medical vocabulary, but those concepts don't match the curator's specific phrasings (`pleuritic chest pain` vs the model's `chest pain on inspiration`; `subjective fever` vs `feels warm`; `family history of lung cancer` vs `dad had lung cancer`).

The seeded corpus does two things that can't be replaced by prompt rules alone:
1. **Vocabulary alignment** — the curated labels tell the model exactly which phrasings are canonical for this curator.
2. **Categorical anchoring** — the labels per category tell the model what level of granularity is expected (e.g., that the curator wants `family history of asthma` as MEDICAL_HISTORY, not just `asthma`).

### Implication for the holdout test

If the organizer ships only transcripts (no curated labels per patient), our pipeline degrades from 0.930 → 0.815 composite, and from 72% to 26% on node F1. Population/Relation/Schema components mask the collapse because they're "≥1 of each type" indicators, not coverage measures.

To salvage zero-shot performance we would need either:
- A curator-supplied label list at runtime (best case — same pipeline)
- A *substitute* anchor list, e.g., SNOMED-CT or UMLS subsets that approximate the expected vocabulary
- A different scoring contract that isn't fuzzy-match-against-curator-strings (e.g., ontology-grounded equivalence)

**Bottom line:** v10's composite of 0.930 is a *seeded* number. The pipeline is not zero-shot ready, and zero-shot performance on the same transcripts is 0.815 (Grade A label is misleading — Entity F1 is 0.26).

Create submission Folder

## 260427 — Model A/B test: gpt-oss-20b + nemotron — trades F1 quality for Population count

### Setup

After CLI flags `--node-model` and `--edge-model` shipped (commit `f016adf`),
A/B-tested the default model pairing against the strongest alternative
pairing the published benchmarks suggested:

| Stage | Default | Alternative |
|---|---|---|
| Stage 1 (entities) | `z-ai/glm-4.7-flash` | `openai/gpt-oss-20b` (highest documented HealthBench, RLHF discipline) |
| Stage 2 (edges) | `deepseek/deepseek-r1-distill-qwen-32b` | `nvidia/nemotron-3-nano-30b-a3b` (top reasoning across MMLU-Pro / GPQA / MATH; ~2× faster than DeepSeek) |

Same 20 in-corpus transcripts, same 522-label combined KB, same prompts (4 edits + tight SYMPTOM/LOCATION + loose-other), canonicalization @ 0.70.

### Bug discovered after the first attempt

Per-call audit of the supposed first swap run (`260427_025015`) showed every Stage 1 entity call still used `z-ai/glm-4.7-flash`, not `openai/gpt-oss-20b`. Only Stage 2 actually swapped (to nemotron-3-nano). Reason: in v8, `OpenRouterClient.__init__(self, api_key, model=OPENROUTER_MODEL, ...)` captures the default value at class-definition time. Monkey-patching `v8.OPENROUTER_MODEL = node_model` after import does not change the captured default. v9's `InstrumentedOpenRouterClient` calls `super().__init__(api_key)` without an explicit `model=` argument, so the stale default kicks in.

Fix: in the wrapper, also patch `v8.OpenRouterClient.__init__.__defaults__` in-place with `inspect.signature` to identify the right tuple position. After the fix, the per-call log lines for the corrected swap run (`260427_082458`) show 140 calls to `openai/gpt-oss-20b` and 121 to `nvidia/nemotron-3-nano-30b-a3b` — zero stale calls.

### Results

| Run | Stage 1 actually called | Stage 2 actually called | Composite | F1 | Precision | Recall | Population | Nodes |
|---|---|---|---|---|---|---|---|---|
| 260427_005525 | glm-4.7-flash | deepseek-r1-distill | 0.925 | 0.699 | 58.5% | 86.9% | 1.000 | 374 |
| 260427_013403 | glm-4.7-flash | deepseek-r1-distill | 0.929 | 0.715 | 61.4% | 85.5% | 1.000 | 345 |
| 260427_025015 | glm-4.7-flash _(broken `--node-model`)_ | nemotron-3-nano | 0.927 | 0.709 | 61.1% | 84.6% | 1.000 | 343 |
| **260427_082458** | **gpt-oss-20b** | **nemotron-3-nano** | **0.891** | **0.803** | **77.9%** | 82.8% | **0.763** | **254** |

### Interpretation

The real swap moves the score significantly — **but in opposing directions on different components.**

- **gpt-oss-20b is dramatically more precise**: precision **+16.5 pp** (61.4% → 77.9%), recall **−2.7 pp**, Entity F1 **+8.8 pp** (0.715 → **0.803** — the highest in-corpus F1 we have ever recorded).
- **gpt-oss-20b is also more conservative**: 254 student nodes vs 345 for the default — 91 fewer total emissions. Population is `min(student/baseline, 1.5) / 1.5`, count-based and unforgiving — it drops from **1.000 → 0.763**.
- Net composite: F1 component +0.022, Population component −0.059 → **net composite −0.037** (matches observed −0.038).

The earlier "models do not move the score" conclusion (now removed from the HTML) was based on the broken swap and was wrong. The corrected finding: **model choice moves the score significantly; the composite scorer rewards quantity at the same weight as quality, so the smaller-but-better KG produced by gpt-oss-20b loses on composite despite winning on Entity F1.**

### Why the original prediction was wrong

Pre-swap I predicted +0.01 to +0.02 composite from the swap. The actual signed delta was −0.038. Three errors in the prediction:

1. **Did not anticipate the conservatism of an RLHF-disciplined model on this prompt.** gpt-oss-20b respects exclusion rules so well that it under-emits on the LOOSE prompts, not just on the TIGHT ones — the global node count drops by 91.

2. **Did not weigh Population as the binding component.** Composite is dominated by whichever component is furthest from 1.0. F1 is 0.715 in the default config (room to move up by +0.285) but Population is already 1.000 (room to move down by 1.0). A model that moves both in opposite directions hurts the composite even when it wins on F1.

3. **Stage 2 was confounded with Stage 1 in the broken-swap data.** With only Stage 2 actually swapped (and Relation already saturated), the 0.927 result looked like "no change" when in fact Stage 1 was unchanged. After the fix, the Stage 1 effect emerges as the −0.038 delta.

### Implications for the submission

The KG produced by gpt-oss-20b (F1 0.803, precision 77.9%) is clinically more defensible than the default's KG (F1 0.715, precision 61.4%) — it has fewer false-positive nodes and tighter conformance to the curator's surface forms.

But the composite scorer ranks the smaller-but-cleaner KG **lower** by 0.038. Two reasonable routes:

1. **Composite-optimized**: keep `z-ai/glm-4.7-flash` for Stage 1. Optionally switch Stage 2 to `nvidia/nemotron-3-nano-30b-a3b` for 2× speed at equal score (Stage 2 cannot move composite since Relation Completeness is saturated at 1.0).
2. **F1-optimized (better for downstream GraphRAG QA)**: switch to `openai/gpt-oss-20b` Stage 1, then aggressively loosen `other` / `fine_grained` emission rules across more categories (currently only 4 of 7 are loose; SYMPTOM and LOCATION stay TIGHT) to push student node count from 254 → 333+ and saturate Population. Should yield both F1 (~0.80) and Population (~1.0), with composite estimated ≈ 0.94+.

### Actual score levers (corrected ranking by measured impact)

1. **BGE-M3 canonicalization @ 0.70**: +0.04 composite OoC (+15.9 pp Entity F1)
2. **Model swap (Stage 1 → gpt-oss-20b)**: −0.038 composite in-corpus, but **+8.8 pp Entity F1 quality**
3. **4 prompt edits (SYMPTOM exam findings, MED_HIST activity/prior, LAB_RESULT outcome anchor, PROCEDURE bare-name)**: +0.024 composite OoC
4. **KB coverage** (in-corpus 222 → combined 522): +0.01 composite OoC
5. **Loose-other rules on TREATMENT/MEDICAL_HISTORY/PROCEDURE/LAB_RESULT (with SYMPTOM and LOCATION kept tight)**: +0.006 composite in-corpus
6. **Stage 2 model swap (deepseek → nemotron)**: 0 composite (saturated component); 2× speedup only

Stage 1 model choice IS a score lever — it moves both quality (F1) and quantity (Population) in opposite directions. The composite scorer's design (count-based Population + 25% weight) determines which side wins on the headline number.
