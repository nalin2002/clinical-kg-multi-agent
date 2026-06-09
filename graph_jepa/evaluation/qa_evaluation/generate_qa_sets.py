"""Generate a transcript-grounded QA set per patient (with a configurable LLM).

Questions must be answerable directly from the TRANSCRIPT. These QA sets are the
ground truth for the QA-based evaluation (graph usefulness), and are
intentionally independent of the silver graphs / world model.

Run::

    python -m graph_jepa.evaluation.qa_evaluation.generate_qa_sets \
        --config graph_jepa/config.yaml [--limit N] [--overwrite]

Output: graph_jepa/evaluation/outputs/qa_sets/<patient_id>.json

Schema::

    {"patient_id": "...",
     "questions": [{"question_id","question","gold_answer",
                    "evidence_from_transcript","category"}]}
"""

from __future__ import annotations

import argparse

from graph_jepa.common.io_utils import (
    LOG,
    discover_patients,
    ensure_dir,
    graph_path,
    load_config,
    setup_logging,
    write_json,
)
from graph_jepa.common.llm_utils import LLMClient, load_prompt

VALID_CATEGORIES = {"medication", "diagnosis", "symptom", "lab", "procedure", "plan", "other"}


def normalize_qa(patient_id: str, payload: dict, n_target: int) -> dict:
    questions = []
    for i, q in enumerate(payload.get("questions", []), 1):
        cat = str(q.get("category", "other")).lower()
        questions.append({
            "question_id": q.get("question_id") or f"Q{i}",
            "question": q.get("question", "").strip(),
            "gold_answer": q.get("gold_answer", "").strip(),
            "evidence_from_transcript": q.get("evidence_from_transcript", "").strip(),
            "category": cat if cat in VALID_CATEGORIES else "other",
        })
    questions = [q for q in questions if q["question"] and q["gold_answer"]]
    return {"patient_id": patient_id, "questions": questions}


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate transcript QA sets")
    ap.add_argument("--config", default="graph_jepa/config.yaml")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    setup_logging(args.log_level)
    cfg = load_config(args.config)
    out_dir = ensure_dir(cfg.path("paths.qa_sets"))
    n = int(cfg.get("evaluation.qa.questions_per_patient", 7))
    client = LLMClient(cfg, provider=cfg.get("evaluation.qa.answerer_provider") or cfg.get("llm.provider"))
    system = load_prompt("qa_generation").replace("{n}", str(n))

    patients = discover_patients(cfg)
    if args.limit:
        patients = patients[: args.limit]
    with_tx = [p for p in patients if p.has_transcript]
    LOG.info("QA generation: %d/%d patients have transcripts (provider=%s)",
             len(with_tx), len(patients), client.provider)

    n_ok = n_skip = 0
    for p in with_tx:
        dest = graph_path(out_dir, p.patient_id)
        if dest.exists() and not args.overwrite:
            n_skip += 1
            continue
        tx = p.read_transcript()
        try:
            payload = client.complete_json(system, f"=== TRANSCRIPT ===\n{tx}\n\nReturn ONLY the JSON object.")
            qa = normalize_qa(p.patient_id, payload, n)
        except Exception as exc:  # noqa: BLE001
            LOG.error("%s: QA generation failed: %s", p.patient_id, exc)
            continue
        write_json(dest, qa)
        n_ok += 1
        LOG.info("%s: %d questions", p.patient_id, len(qa["questions"]))

    LOG.info("DONE QA sets: %d created, %d skipped -> %s", n_ok, n_skip, out_dir)


if __name__ == "__main__":
    main()
