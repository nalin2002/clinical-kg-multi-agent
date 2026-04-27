"""
smoke_test_v10_260425.py
========================

v9 runtime + V10 prompts (general + OoC examples, recall-pressure levers
removed) + automatic post-batch scoring chain.

Pipeline:
  1. Run the v9 extraction (workers=3 default, instrumented client,
     retry telemetry) using the prompts in prompts_v10_260425.
  2. Locate the new run dir under v9.OUT_DIR.
  3. Convert v10 entity/edge JSON → per-patient kg_extraction format
     via eir/convert_v7_to_kg_extraction_format_260425.py.
  4. Run Clinical_KG_OS_LLM.dump_graph (BGE-M3 entity resolution).
  5. Run canonicalize_to_curator_260426.py — deterministic BGE-M3 cosine
     post-step (≥ 0.70) that rewrites student node text to nearest curator
     KB label of the same entity_type. Skip with --no-canonicalize.
  6. Run Clinical_KG_OS_LLM.kg_similarity_scorer against the
     human-curated baseline. Stdout is streamed live so the
     COMPOSITE SCORE prints inline.

v9 itself is not modified; this script monkey-patches v8.CATEGORY_PROMPTS
with the v10 prompts before any worker thread runs.

Run:
    python smoke_test_v10_260425.py

Flags this script consumes (everything else passes through to v9.main()):
    --no-score              Skip the convert/dump_graph/canonicalize/scorer
                            chain.
    --no-canonicalize       Skip just the canonicalization step (still runs
                            convert + dump_graph + scorer).
    --canon-threshold T     BGE-M3 cosine threshold for the canonicalizer
                            (default 0.70). Higher = more conservative.
    --score-name NAME       Override the run name used in the unified-graph
                            and score JSON filenames.
    --baseline PATH         Path to the human-curated baseline JSON.

Created: 260425
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# Import the v8/v9 chain (which inherits prompts from v7).
import smoke_test_glm47_specialized_parallel_v8_optimized_260425 as v8  # noqa: E402
import smoke_test_glm47_specialized_parallel_v9_optimized_260425 as v9  # noqa: E402

# Import the v10 prompts module.
import prompts_v10_260425 as v10p  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# Monkey-patch BEFORE any pipeline call. v8.run_entity_category and
# v8.run_edge_category resolve CATEGORY_PROMPTS at call time from v8's
# module globals, so reassigning here makes every worker pick up the
# v10 prompt.
# ─────────────────────────────────────────────────────────────────────────────
v8.CATEGORY_PROMPTS = v10p.CATEGORY_PROMPTS

# Override the transcript directory to point at the ACI-Bench-derived folder.
# Same shape as data/transcripts/<RES_ID>/<RES_ID>.txt, but populated from
# the ACI-Bench Hugging Face dataset (mkieffer/ACI-Bench).
v8.TRANSCRIPT_DIR = v8.EIR_ROOT / "eir_aci_bench" / "transcripts"
v9.TRANSCRIPT_DIR = v8.TRANSCRIPT_DIR  # v9 has its own ref to v8.TRANSCRIPT_DIR

# ─────────────────────────────────────────────────────────────────────────────
# Curated KB override (team-friendly).
# Set EIR_CURATED_KB_PATH env var to point at any KB JSON file with the
# same shape as eir/curated_kb_260419.json. Useful for swapping in:
#   - eir/curated_synthetic_kb_combined_260426.json  (curated + ACI synthetic)
#   - your own KB built from a different unified-graph
# Default if unset: the in-corpus 222-label KB shipped with v7.
# ─────────────────────────────────────────────────────────────────────────────
_kb_override = os.environ.get("EIR_CURATED_KB_PATH")
if _kb_override:
    v8.CURATED_KB = Path(_kb_override).resolve()
    print(f"[v10-aci-bench] CURATED_KB overridden via env: {v8.CURATED_KB}", flush=True)

# Separate output dirs so the ACI-Bench run never mixes with the in-corpus runs.
v9.OUT_DIR = v8.EIR_ROOT / "eir_results" / "smoke_test_glm47_specialized_parallel_v10_aci_bench"

PROJECT_ROOT: Path = v8.PROJECT_ROOT
EIR_ROOT: Path = v8.EIR_ROOT

# Post-batch chain artefacts land here so scoring outputs don't pollute the
# extraction run dir. (Note: dump_graph + scorer expect a curator baseline;
# we leave them in place but the score won't be meaningful since the curator
# graph doesn't include this patient.)
SCORING_BASE: Path = EIR_ROOT / "eir_results" / "v10_aci_bench_for_organizer"

CONVERT_SCRIPT = EIR_ROOT / "convert_v7_to_kg_extraction_format_260425.py"
DEFAULT_BASELINE = PROJECT_ROOT / "data" / "human_curated" / "unified_graph_curated.json"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _consume_flag(name: str) -> bool:
    """Pop a boolean flag from sys.argv (so it doesn't leak to v9.main)."""
    if name in sys.argv:
        sys.argv.remove(name)
        return True
    return False


def _consume_value_flag(name: str) -> str | None:
    """Pop `--name VALUE` from sys.argv and return VALUE, or None."""
    if name not in sys.argv:
        return None
    i = sys.argv.index(name)
    if i + 1 >= len(sys.argv):
        sys.exit(f"ERROR: {name} requires a value")
    val = sys.argv[i + 1]
    del sys.argv[i:i + 2]
    return val


def _run(cmd: list, cwd: Path, env_extra: dict | None = None) -> None:
    """Run a subcommand, stream stdout/stderr live, exit on failure."""
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    pretty = " ".join(str(c) for c in cmd)
    print(f"\n[v10-aci-bench] $ {pretty}", flush=True)
    print(f"[v10-aci-bench]   (cwd: {cwd})", flush=True)
    result = subprocess.run(cmd, cwd=str(cwd), env=env)
    if result.returncode != 0:
        sys.exit(f"\n[v10-aci-bench] FAILED at: {pretty}\n"
                 f"[v10-aci-bench]   exit code: {result.returncode}")


def _find_new_run_dir(out_dir: Path, before: set[str]) -> Path | None:
    """Return the timestamped subdir created by v9.main() (or None)."""
    if not out_dir.exists():
        return None
    after = {p.name for p in out_dir.iterdir() if p.is_dir()}
    new = sorted(after - before)
    if not new:
        return None
    # If multiple new subdirs (shouldn't happen, but be safe), pick most recent
    return out_dir / new[-1]


def _score_run(run_dir: Path, score_name: str | None,
                baseline_path: Path,
                no_canonicalize: bool = False,
                canon_threshold: float = 0.70) -> None:
    """Run the convert → dump_graph → canonicalize → scorer chain on a v9 run dir."""
    run_name = run_dir.name  # e.g. "260425_120000"
    name = score_name or f"v10_aci_bench_{run_name}"
    score_dir = SCORING_BASE / run_name
    sub_kgs = score_dir / "sub_kgs"

    score_dir.mkdir(parents=True, exist_ok=True)
    sub_kgs.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 72, flush=True)
    print(f"[v10-aci-bench] POST-BATCH SCORING CHAIN", flush=True)
    print(f"  source run dir : {run_dir}", flush=True)
    print(f"  scoring dir    : {score_dir}", flush=True)
    print(f"  unified name   : {name}", flush=True)
    print(f"  baseline       : {baseline_path}", flush=True)
    print("=" * 72, flush=True)

    if not CONVERT_SCRIPT.exists():
        sys.exit(f"[v10-aci-bench] ERROR: convert script not found at {CONVERT_SCRIPT}")
    if not baseline_path.exists():
        sys.exit(f"[v10-aci-bench] ERROR: baseline not found at {baseline_path}\n"
                 f"  (override with --baseline PATH)")

    # ── Step 1/4 — convert v9 → kg_extraction per-patient format ────────────
    print("\n[v10-aci-bench] STEP 1/4 — convert", flush=True)
    _run(
        cmd=[
            sys.executable, str(CONVERT_SCRIPT),
            "--nodes-dir", str(run_dir),
            "--edges-dir", str(run_dir),
            "--output",    str(sub_kgs),
        ],
        cwd=PROJECT_ROOT,
    )

    # ── Step 2/4 — dump_graph (BGE-M3 entity resolution) ─────────────────────
    print("\n[v10-aci-bench] STEP 2/4 — dump_graph", flush=True)
    _run(
        cmd=[
            sys.executable, "-m", "Clinical_KG_OS_LLM.dump_graph",
            "--input",  str(sub_kgs),
            "--output", str(score_dir),
            "--name",   name,
        ],
        cwd=PROJECT_ROOT,
        env_extra={"PYTHONPATH": str(PROJECT_ROOT / "src")},
    )

    unified_path = score_dir / f"unified_graph_{name}.json"
    if not unified_path.exists():
        sys.exit(f"[v10-aci-bench] ERROR: dump_graph did not produce {unified_path}")

    # ── Step 3/4 — canonicalize student node text to curator labels ──────────
    # Deterministic BGE-M3 cosine post-step (≥ 0.70). See in-corpus wrapper for
    # rationale. Skip with --no-canonicalize.
    canon_path = unified_path
    if not no_canonicalize:
        canon_script = _HERE / "canonicalize_to_curator_260426.py"
        canon_kb = Path(v8.CURATED_KB).resolve()
        canon_path = score_dir / f"unified_graph_{name}_canonicalized.json"
        canon_report = score_dir / f"canonicalize_report_{name}.json"
        print("\n[v10-aci-bench] STEP 3/4 — canonicalize_to_curator (BGE-M3 @ 0.70)",
              flush=True)
        _run(
            cmd=[
                sys.executable, str(canon_script),
                "--unified",   str(unified_path),
                "--kb",        str(canon_kb),
                "--output",    str(canon_path),
                "--report",    str(canon_report),
                "--threshold", str(canon_threshold),
            ],
            cwd=PROJECT_ROOT,
        )
        if not canon_path.exists():
            sys.exit(f"[v10-aci-bench] ERROR: canonicalize did not produce {canon_path}")
    else:
        print("\n[v10-aci-bench] STEP 3/4 — canonicalize SKIPPED (--no-canonicalize)",
              flush=True)

    # ── Step 4/4 — composite score ───────────────────────────────────────────
    score_json = score_dir / f"score_{name}.json"
    print("\n[v10-aci-bench] STEP 4/4 — kg_similarity_scorer", flush=True)
    _run(
        cmd=[
            sys.executable, "-m", "Clinical_KG_OS_LLM.kg_similarity_scorer",
            "--student",  str(canon_path),
            "--baseline", str(baseline_path),
            "--output",   str(score_json),
        ],
        cwd=PROJECT_ROOT,
        env_extra={"PYTHONPATH": str(PROJECT_ROOT / "src")},
    )

    print("\n" + "=" * 72, flush=True)
    print(f"[v10-aci-bench] DONE", flush=True)
    print(f"  unified graph     : {unified_path}", flush=True)
    if canon_path != unified_path:
        print(f"  canonicalized     : {canon_path}", flush=True)
    print(f"  score json        : {score_json}", flush=True)
    print("=" * 72, flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    # Pull our own flags out of argv before v9.main() sees it.
    no_score = _consume_flag("--no-score")
    no_canonicalize = _consume_flag("--no-canonicalize")
    score_name = _consume_value_flag("--score-name")
    baseline_override = _consume_value_flag("--baseline")
    baseline_path = Path(baseline_override) if baseline_override else DEFAULT_BASELINE
    canon_threshold_str = _consume_value_flag("--canon-threshold")
    canon_threshold = float(canon_threshold_str) if canon_threshold_str else 0.70

    # Sanity: confirm the prompt swap took effect.
    # v10 prompts have no RECALL-ORIENTED / RECALL FLOOR language; verify by
    # absence of those tokens AND presence of OoC anchor (`polyuria`).
    sample_swap = v8.CATEGORY_PROMPTS.get("SYMPTOM", "")
    if "polyuria" not in sample_swap or "RECALL FLOOR" in sample_swap:
        sys.exit("ERROR: v10 prompt swap did not take effect — refusing to run.")
    print("[v10-aci-bench] prompts swapped to prompts_v10_260425", flush=True)

    # Default ACTIVE = all 13 categories ON. v7's ACTIVE has only SYMPTOM=True;
    # without this, a bare run would extract only SYMPTOM. v9.main() still
    # narrows correctly if the user passes --categories explicitly.
    for _cat in v8.ACTIVE:
        v8.ACTIVE[_cat] = True
    print("[v10-aci-bench] ACTIVE: all 13 categories ON by default "
          "(override with --categories)", flush=True)

    # Make sure both output roots exist before extraction starts so they're
    # visible immediately on the filesystem.
    v9.OUT_DIR.mkdir(parents=True, exist_ok=True)
    SCORING_BASE.mkdir(parents=True, exist_ok=True)

    print(f"[v10-aci-bench] extraction output dir: {v9.OUT_DIR}", flush=True)
    if no_score:
        print("[v10-aci-bench] --no-score: extraction only, no scoring chain",
              flush=True)
    else:
        print(f"[v10-aci-bench] scoring chain enabled — results land under "
              f"{SCORING_BASE}", flush=True)

    # Snapshot existing timestamped subdirs so we can identify the new one.
    before: set[str] = set()
    if v9.OUT_DIR.exists():
        before = {p.name for p in v9.OUT_DIR.iterdir() if p.is_dir()}

    # Run extraction.
    t0 = time.time()
    v9.main()
    extraction_ms = int((time.time() - t0) * 1000)
    print(f"\n[v10-aci-bench] extraction wall-clock: {extraction_ms/1000:.1f}s",
          flush=True)

    if no_score:
        return

    run_dir = _find_new_run_dir(v9.OUT_DIR, before)
    if run_dir is None:
        print("[v10-aci-bench] WARNING: could not locate a new run dir under "
              f"{v9.OUT_DIR} — skipping scoring chain.", flush=True)
        return

    _score_run(run_dir, score_name, baseline_path,
               no_canonicalize=no_canonicalize,
               canon_threshold=canon_threshold)


if __name__ == "__main__":
    main()
