# qa_evaluation

**Purpose.** Measure graph **usefulness** downstream: answer transcript-grounded
questions using ONLY a graph (no transcript), then score the answers. Tests
whether world-model refinement yields graphs that better support QA.

**Three steps.**
1. `generate_qa_sets.py` — an LLM writes N transcript-answerable Q&As per patient
   (`question/gold_answer/evidence_from_transcript/category`).
2. `answer_from_graphs.py` — for each method's graph, the answerer LLM answers
   from the **serialized graph only** (transcript withheld).
3. `evaluate_answers.py` — scores predicted vs gold with **exact match**,
   **token-F1**, and **LLM-as-judge** (0/1 correctness + 1–5 score; ensemble via
   `evaluation.qa.judge_ensemble`).

**Inputs.** Transcripts (`data.*`); method graphs (`llm_graphs/`,
`refined_graphs/<method>/`).

**Outputs.**
- `evaluation/outputs/qa_sets/<PID>.json`
- `evaluation/outputs/qa_answers/<method>/<PID>.json`
- `evaluation/outputs/qa_eval/{per_question,per_patient,aggregate}.csv`, `summary.md/json`

**Run.**
```bash
python -m graph_jepa.evaluation.qa_evaluation.generate_qa_sets   --config graph_jepa/config.yaml
python -m graph_jepa.evaluation.qa_evaluation.answer_from_graphs --config graph_jepa/config.yaml
python -m graph_jepa.evaluation.qa_evaluation.evaluate_answers   --config graph_jepa/config.yaml --split test [--no-judge]
```

**Split.** QA gold is transcript-derived and independent of the silver graphs the
world model trained on, so leakage is weak. Report `--split test` as the clean
headline; `--split all` is an acceptable secondary (transductive) number.

**Judges.** `evaluation.qa.judge_provider` for a single judge, or
`evaluation.qa.judge_ensemble: [anthropic, openai, gemini]` for an ensemble
(scores averaged, correctness majority-voted). Validate the judge against a small
human-rated sample and report inter-judge agreement.

**Paper interpretation.** The downstream-utility argument: if refinement removes
spurious edges, graph-only answers should become more faithful/correct — visible
as higher judge accuracy / token-F1 for `edge_plausibility_classifier` vs
`llm_only`. `fully_connected_wm` tests whether nodes + world-model relations alone
suffice to answer.
