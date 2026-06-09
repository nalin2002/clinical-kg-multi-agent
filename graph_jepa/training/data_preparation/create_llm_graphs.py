"""Generate the initial LLM knowledge graphs from transcripts via EIR.

This is the "LLM-only" condition and the input the world model refines. It does
NOT reimplement the EIR 13-agent extractor — it calls
:mod:`graph_jepa.common.eir_adapter`, which loads existing EIR per-patient KGs
(default) or optionally shells out to the EIR pipeline (``eir.run_eir: true``).

Run::

    python -m graph_jepa.training.data_preparation.create_llm_graphs \
        --config graph_jepa/config.yaml [--limit N]

Output: graph_jepa/training/outputs/llm_graphs/<patient_id>.json
        (canonical schema, source="llm_transcript")

NOTE on coverage: only patients with an available EIR extraction are written.
To cover all 207 ACI-Bench patients you must run the EIR extractor and point
`eir.existing_kg_dirs` (config.yaml) at its raw per-patient output dir, OR set
`eir.run_eir: true` with an OpenRouter key configured.
"""

from __future__ import annotations

import argparse

from graph_jepa.common.eir_adapter import get_llm_graphs
from graph_jepa.common.io_utils import (
    LOG,
    discover_patients,
    ensure_dir,
    load_config,
    save_graph,
    setup_logging,
    write_json,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Create LLM transcript graphs via the EIR adapter")
    ap.add_argument("--config", default="graph_jepa/config.yaml")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    setup_logging(args.log_level)
    cfg = load_config(args.config)
    out_dir = ensure_dir(cfg.path("paths.llm_graphs"))

    patients = discover_patients(cfg)
    if args.limit:
        patients = patients[: args.limit]

    graphs = get_llm_graphs(cfg, patients, out_dir)
    n = 0
    for pid, g in graphs.items():
        g.source = "llm_transcript"
        save_graph(out_dir, g)
        n += 1
        LOG.info("%s: %d nodes, %d edges", pid, len(g.nodes), len(g.edges))

    write_json(out_dir / "_manifest.json",
               {"created": n, "requested": len(patients),
                "run_eir": bool(cfg.get("eir.run_eir", False))})
    LOG.info("DONE LLM graphs: %d/%d patients -> %s", n, len(patients), out_dir)
    if n == 0:
        LOG.warning("No LLM graphs produced. Configure eir.existing_kg_dirs or eir.run_eir.")


if __name__ == "__main__":
    main()
