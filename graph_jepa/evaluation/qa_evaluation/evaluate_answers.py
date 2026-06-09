"""Score graph-based QA answers against gold answers.

Three signals per (patient, question, method):
  * exact_match   — normalized string equality (rarely 1 for long answers).
  * token_f1      — bag-of-tokens F1 vs gold (robust to phrasing).
  * judge_score   — LLM-as-judge correctness (0/1) + 1-5 score + explanation,
                    comparing predicted vs gold using transcript evidence.
                    Supports an ENSEMBLE of judges (config qa.judge_ensemble);
                    scores are averaged and correctness is majority-voted.

Outputs (graph_jepa/evaluation/outputs/qa_eval/):
  * per_question.csv     — one row per (patient, question, method)
  * per_patient.csv      — QA accuracy per (patient, method)
  * aggregate.csv        — per-method means
  * summary.md / summary.json — LLM-only vs LLM+WorldModel headline

Run::

    python -m graph_jepa.evaluation.qa_evaluation.evaluate_answers \
        --config graph_jepa/config.yaml [--split test|all] [--no-judge]
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics
from pathlib import Path
from typing import Dict, List

from graph_jepa.common.io_utils import (
    LOG,
    ensure_dir,
    load_config,
    read_json,
    setup_logging,
    write_json,
)
from graph_jepa.common.llm_utils import LLMClient, judge_answer

_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> List[str]:
    return _WORD.findall((text or "").lower())


def exact_match(pred: str, gold: str) -> int:
    return int(" ".join(_tokens(pred)) == " ".join(_tokens(gold)) and bool(_tokens(gold)))


def token_f1(pred: str, gold: str) -> float:
    p, g = _tokens(pred), _tokens(gold)
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    common: Dict[str, int] = {}
    gp = {}
    for t in g:
        gp[t] = gp.get(t, 0) + 1
    overlap = 0
    seen: Dict[str, int] = {}
    for t in p:
        seen[t] = seen.get(t, 0) + 1
        if seen[t] <= gp.get(t, 0):
            overlap += 1
    if overlap == 0:
        return 0.0
    prec = overlap / len(p)
    rec = overlap / len(g)
    return 2 * prec * rec / (prec + rec)


def _split_lookup(cfg) -> Dict[str, str]:
    mp = cfg.path("paths.training_outputs") / "edge_plausibility_classifier" / "metrics.json"
    if not mp.exists():
        return {}
    split = read_json(mp).get("split", {})
    return {pid: name for name, ids in split.items() for pid in ids}


def _ensemble_judge(clients: List[LLMClient], q: str, gold: str, pred: str, ev: str) -> dict:
    results = [judge_answer(c, q, gold, pred, ev) for c in clients]
    correctness = int(sum(r["correctness"] for r in results) * 2 >= len(results))  # majority
    score = statistics.mean(r["score"] for r in results)
    return {"correctness": correctness, "score": round(score, 3),
            "explanation": results[0]["explanation"], "n_judges": len(results)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Score graph QA answers vs gold")
    ap.add_argument("--config", default="graph_jepa/config.yaml")
    ap.add_argument("--split", default="test", choices=["test", "val", "train", "all"])
    ap.add_argument("--no-judge", action="store_true", help="skip LLM-as-judge (token-F1 only)")
    ap.add_argument("--methods", nargs="+", default=None)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    setup_logging(args.log_level)
    cfg = load_config(args.config)

    qa_dir = cfg.path("paths.qa_sets")
    qa_sets = {p.stem: read_json(p) for p in sorted(qa_dir.glob("*.json")) if not p.stem.startswith("_")}
    gold_lookup = {pid: {q["question_id"]: q for q in qa.get("questions", [])} for pid, qa in qa_sets.items()}

    split_map = _split_lookup(cfg)
    methods = args.methods or cfg.get("evaluation.methods", ["llm_only", "edge_plausibility_classifier"])
    answers_root = cfg.path("paths.eval_outputs") / "qa_answers"

    # Judges
    judges: List[LLMClient] = []
    if not args.no_judge:
        ensemble = cfg.get("evaluation.qa.judge_ensemble") or [cfg.get("evaluation.qa.judge_provider")
                                                               or cfg.get("llm.provider")]
        judges = [LLMClient(cfg, provider=p) for p in ensemble]
        LOG.info("judges: %s", [j.provider for j in judges])

    per_q: List[dict] = []
    for method in methods:
        mdir = answers_root / method
        if not mdir.exists():
            LOG.warning("no answers for method %s, skipping", method)
            continue
        for af in sorted(mdir.glob("*.json")):
            pid = af.stem
            if args.split != "all" and split_map and split_map.get(pid) != args.split:
                continue
            data = read_json(af)
            for a in data.get("answers", []):
                gold = gold_lookup.get(pid, {}).get(a["question_id"])
                if not gold:
                    continue
                pred = a.get("predicted_answer", "")
                row = {
                    "patient_id": pid, "method": method, "question_id": a["question_id"],
                    "category": gold.get("category", "other"),
                    "exact_match": exact_match(pred, gold["gold_answer"]),
                    "token_f1": round(token_f1(pred, gold["gold_answer"]), 4),
                    "judge_correct": "", "judge_score": "",
                }
                if judges:
                    j = _ensemble_judge(judges, gold["question"], gold["gold_answer"], pred,
                                        gold.get("evidence_from_transcript", ""))
                    row["judge_correct"] = j["correctness"]
                    row["judge_score"] = j["score"]
                per_q.append(row)
        LOG.info("scored method %s", method)

    if not per_q:
        LOG.error("No answers scored. Run answer_from_graphs (and check --split).")
        return

    # Aggregations
    out = ensure_dir(cfg.path("paths.eval_outputs") / "qa_eval")
    _write_csv(out / "per_question.csv", per_q)

    per_patient = _per_patient(per_q)
    _write_csv(out / "per_patient.csv", per_patient)

    agg = _aggregate(per_q)
    _write_csv(out / "aggregate.csv", agg)
    write_json(out / "summary.json", {"split": args.split, "judged": bool(judges), "aggregate": agg})
    md = _markdown(agg, args.split, bool(judges))
    (out / "summary.md").write_text(md, encoding="utf-8")
    LOG.info("DONE QA eval -> %s", out)
    print("\n" + md + "\n")


def _per_patient(per_q: List[dict]) -> List[dict]:
    groups: Dict[tuple, List[dict]] = {}
    for r in per_q:
        groups.setdefault((r["patient_id"], r["method"]), []).append(r)
    rows = []
    for (pid, method), rs in sorted(groups.items()):
        row = {"patient_id": pid, "method": method, "n_questions": len(rs),
               "mean_token_f1": round(statistics.mean(r["token_f1"] for r in rs), 4),
               "exact_match_rate": round(statistics.mean(r["exact_match"] for r in rs), 4)}
        jc = [r["judge_correct"] for r in rs if r["judge_correct"] != ""]
        js = [r["judge_score"] for r in rs if r["judge_score"] != ""]
        row["judge_accuracy"] = round(statistics.mean(jc), 4) if jc else ""
        row["mean_judge_score"] = round(statistics.mean(js), 4) if js else ""
        rows.append(row)
    return rows


def _aggregate(per_q: List[dict]) -> List[dict]:
    by_method: Dict[str, List[dict]] = {}
    for r in per_q:
        by_method.setdefault(r["method"], []).append(r)
    agg = []
    for method, rs in by_method.items():
        row = {"method": method, "n_questions": len(rs),
               "mean_token_f1": round(statistics.mean(r["token_f1"] for r in rs), 4),
               "exact_match_rate": round(statistics.mean(r["exact_match"] for r in rs), 4)}
        jc = [r["judge_correct"] for r in rs if r["judge_correct"] != ""]
        js = [r["judge_score"] for r in rs if r["judge_score"] != ""]
        row["judge_accuracy"] = round(statistics.mean(jc), 4) if jc else ""
        row["mean_judge_score"] = round(statistics.mean(js), 4) if js else ""
        agg.append(row)
    return agg


def _markdown(agg: List[dict], split: str, judged: bool) -> str:
    lines = [f"# QA Evaluation (split = `{split}`)", "",
             "Answers produced from each graph **without transcript access**, scored vs "
             "transcript-grounded gold answers.", "",
             "| Method | n_q | Token F1 | Exact | Judge Acc | Judge Score |",
             "|---|--:|--:|--:|--:|--:|"]
    for r in sorted(agg, key=lambda x: x["mean_token_f1"], reverse=True):
        ja = f"{r['judge_accuracy']:.3f}" if r["judge_accuracy"] != "" else "—"
        jsv = f"{r['mean_judge_score']:.2f}" if r["mean_judge_score"] != "" else "—"
        lines.append(f"| {r['method']} | {r['n_questions']} | {r['mean_token_f1']:.3f} | "
                     f"{r['exact_match_rate']:.3f} | {ja} | {jsv} |")
    if not judged:
        lines += ["", "_Judge columns blank: run without `--no-judge` and configure a judge provider._"]
    return "\n".join(lines)


def _write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    ensure_dir(path.parent)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
