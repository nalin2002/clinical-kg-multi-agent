# silver_reference_graphs

**Purpose.** Build SILVER-STANDARD reference graphs by LLM-extracting a clinical
KG from each patient's **clinician note** (note only, not the transcript). These
are the positive training signal for the refiners and the references for tier-1
evaluation.

> Silver, **not gold**: the note is clinician-written but the graph is
> LLM-extracted and unverified. Use for relative comparison.

**Inputs.** `data/aci_bench/transcripts/<PID>/<PID>_note.txt` (config `data.*`);
LLM provider (`llm.provider`, default Claude).

**Outputs.** `training/outputs/silver_reference_graphs/<PID>.json` (canonical
schema, `source=clinician_note`) + `_manifest.json`.

**Run.**
```bash
python -m graph_jepa.training.silver_reference_graphs.create_silver_graphs \
    --config graph_jepa/config.yaml [--limit N] [--overwrite]
```

**Expected files.** One `<PID>.json` per patient with a note; nodes typed to the
7-type schema, edges to the 6-relation schema, each edge carrying `evidence`
(note span) and `confidence`.

**Paper interpretation.** Provides a clinician-anchored reference without
recruiting expert annotators — the practical substitute for a gold KG. Quality
should be spot-checked on a sample; report it as silver and lead with relative
(LLM-only vs refined) deltas.
