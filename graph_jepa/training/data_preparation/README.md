# data_preparation

**Purpose.** Produce the initial **LLM knowledge graphs** from transcripts using
the existing EIR 13-agent extractor (Stage-1: 7 entity agents, Stage-2: 6 edge
agents). This is the `llm_only` baseline and the input the world model refines.
We **wrap** EIR via `common/eir_adapter.py`; we do not reimplement it.

**Inputs.** ACI-Bench transcripts (`data.*`); existing EIR KGs
(`eir.existing_kg_dirs`) and/or a live EIR run (`eir.run_eir: true`).

**Outputs.** `training/outputs/llm_graphs/<PID>.json` (canonical schema,
`source=llm_transcript`) + `_manifest.json`.

**Run.**
```bash
python -m graph_jepa.training.data_preparation.create_llm_graphs \
    --config graph_jepa/config.yaml [--limit N]
```

**Coverage note.** Only patients with an available EIR extraction are written.
For all 207 ACI-Bench patients, run the EIR extractor on `data/aci_bench` and
point `eir.existing_kg_dirs` at its raw per-patient output, or set
`eir.run_eir: true` with an OpenRouter key. See `common/README.md`.

**Paper interpretation.** Defines the LLM-only condition. Its noise (implausible
or spurious edges) is exactly what the world model is asked to clean — so the
quality gap between this and the refined graphs is the headline result.
