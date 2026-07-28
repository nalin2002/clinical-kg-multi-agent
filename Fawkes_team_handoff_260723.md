# Fawkes — Team Handoff

**A Graph-JEPA world model that refines LLM-drafted clinical knowledge graphs.**
Everything you need to reproduce our result is **one script**; the data and model already live in the `wmatbooth` HF org.

> **PRIVATE — MIMIC-IV-derived (PhysioNet DUA). Requires PhysioNet MIMIC-IV credentials and access to the `wmatbooth` org. Do not make public.**

---

## TL;DR

```bash
hf jobs uv run --flavor a100-large -s HF_TOKEN --timeout 4h \
  -d fawkes_trainer_jepa_entity_note_v16_wmatbooth_260723.py
```
~13–15 min → `[DONE] … LOO MRR=0.419`. That's the whole thing.

---

## What it does

An LLM drafts a clinical KG from a discharge note, but the **inferred** edges (which drug treats which diagnosis, which symptom indicates which condition) are noisy and unprovenanced. Fawkes **refines** the draft:

1. **Consolidated graph** per admission = deterministic MIMIC backbone + 4 LLM-inferred relations + an 18-signal evidence score per edge.
2. **Graph-JEPA world model** — self-supervised masked-latent pretraining (BYOL / EMA target), then a frozen-encoder edge-recovery readout (DistMult + InfoNCE).
3. **Note injection (the key finding):** the Clinical-ModernBERT note vector is placed **locally, on the entities the note grounds** — not as one global vector. That localization is what works.

## The result

Leave-one-out edge recovery (the honest refiner metric), same recipe, only the note differs:

| | Overall LOO MRR |
|---|---|
| No note (Option A) | 0.274 |
| Global note vector (diffuse) | 0.265 — *hurts* |
| **Entity-grounded note (Option B)** | **0.419** |

Every inferred relation improves (INDICATES 0.373 → 0.603). It is **deterministic** — a frozen rerun reproduces 0.419 to the third decimal.

## Options

- **Option A — note-free.** Uses no notes; covers every admission. `-e USE_NOTE=0`.
- **Option B — note-augmented (default).** Entity-grounded note; for the 64.4% of admissions that have a note. Falls back to A otherwise.

## What you need

1. `HF_TOKEN` with access to the **`wmatbooth`** org.
2. The **`hf` CLI** (`curl -LsSf https://hf.co/cli/install.sh | bash`). The script self-installs its Python deps via its `# /// script` header.

That's it — the dataset (`wmatbooth/fawkes-training-graph-embedded-260615`) and the output model repo (`wmatbooth/fawkes-graph-jepa-v16-paper-260723`) are baked in as defaults.

## Under the hood (for the curious)

- **Data:** 4,000 per-admission MIMIC-IV graphs (nodes + edges + v8 scores + 768-d Clinical-ModernBERT note embedding), read once from HF.
- **Encoder:** 2-layer / 4-head `TransformerConv`, hidden 128; ~2.6M params.
- **Determinism:** fixed seed 42, `CUBLAS_WORKSPACE_CONFIG`, `use_deterministic_algorithms`, leakage-free leave-one-out eval.
- **No fallback code** — unknown types/relations/dims fail loudly.

## More

- **Paper:** `fawkes_paper_v4_260616.tex` (6 pages, both options).
- **Full results log & design rationale:** `fawkes_v28_260616.html`.
- **Artifacts (wmatbooth org):** training data, the v16 model, the 1K longitudinal graphs, the ACI-Bench external eval set.

*Handoff generated 260723.*
