"""
smoke_test_glm47_specialized_parallel_v9_optimized_260425.py
============================================================

v9 = v8 plumbing + the workers=3 sweet spot, made the new default.

Why v9 over v8
--------------
A 20-transcript / 13-category batch on v8 with --workers 8 produced a
Grade-A composite of 0.894 in 14.6 min, but with 19 LLM_FAIL/PARSE
casualties caused by simultaneous 429 hits overwhelming the retry
window. Re-running the same batch on v8 with --workers 3 produced:

    composite      0.901   (best run yet)
    entity F1      0.816   (+0.053 vs v7,  +0.018 vs v8 w8)
    node recall    86.9 %  (+4.5 pp vs v7)
    per-pt cov.    86.1 %  (+4.4 pp vs v7)
    LLM_FAIL/PARSE 0
    wall-clock     28 min  (vs v8 w8: 14.6 min, vs v7: 24.7 min)
    429 hits       11      — every one recovered on retry attempt 2/4

The trade-off is plain: 28 min beats 14.6 min when the longer run loses
nothing, because the shorter run loses 19 cells and the downstream
unified graph + scorer absorb that as a hit on Entity F1 / Population /
Per-Patient Coverage.

v9 therefore:
  - defaults --workers to 3 (was 8 in v8, was 4 in v7)
  - adds retry telemetry: counts of first-/second-/third-/fourth-attempt
    successes, plus 429-recovered counts, surfaced in the final
    _batch_summary.json so you can see at a glance whether retries are
    catching everything or whether the batch is starting to lose cells
  - lands runs in eir_results/smoke_test_glm47_specialized_parallel_v9/
    so v8 results are never co-mingled

Everything else is inherited from v8 (which inherits prompts from v7);
the prompt text and ACTIVE flags are byte-identical to v7.

Run:
    python smoke_test_glm47_specialized_parallel_v9_optimized_260425.py \
        --categories SYMPTOM DIAGNOSIS TREATMENT PROCEDURE LOCATION \
                     MEDICAL_HISTORY LAB_RESULT \
                     INDICATES CAUSES CONFIRMS RULES_OUT TAKEN_FOR LOCATED_AT

Created: 260425
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import random
import sys
import threading
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# Pull every prompt + helper from v8 (which pulls prompts from v7).
import smoke_test_glm47_specialized_parallel_v8_optimized_260425 as v8  # noqa: E402

# Re-export so external code that imports from v9 keeps working.
ENTITY_CATEGORIES = v8.ENTITY_CATEGORIES
EDGE_CATEGORIES   = v8.EDGE_CATEGORIES
ACTIVE            = v8.ACTIVE
OPENROUTER_MODEL  = v8.OPENROUTER_MODEL
EIR_ROOT          = v8.EIR_ROOT
PROJECT_ROOT      = v8.PROJECT_ROOT
TRANSCRIPT_DIR    = v8.TRANSCRIPT_DIR
CURATED_KB        = v8.CURATED_KB
API_KEYS_PATH     = v8.API_KEYS_PATH

OUT_DIR = EIR_ROOT / "eir_results" / "smoke_test_glm47_specialized_parallel_v9"

# v9 defaults — proven on a 20-transcript Grade-A run.
DEFAULT_WORKERS = 3
DEFAULT_MAX_ATTEMPTS = 4


# ─────────────────────────────────────────────────────────────────────────────
# RetryStats — thread-safe counters for per-attempt success/failure and
# 429-recovery, surfaced in _batch_summary.json at the end of the run.
# ─────────────────────────────────────────────────────────────────────────────
class RetryStats:
    def __init__(self, max_attempts: int = DEFAULT_MAX_ATTEMPTS) -> None:
        self.max_attempts = max_attempts
        self._lock = threading.Lock()
        self.success_on_attempt: list[int] = [0] * max_attempts
        self.failed_after_all_attempts = 0
        self.errors_429 = 0
        self.errors_429_recovered = 0
        self.errors_5xx = 0
        self.errors_5xx_recovered = 0
        self.errors_network = 0
        self.errors_network_recovered = 0
        self.errors_other_retriable = 0
        self.errors_nonretriable = 0
        self.empty_content_retries = 0

    def record_success(self, attempt: int, hit_429: bool, hit_5xx: bool,
                       hit_net: bool) -> None:
        with self._lock:
            idx = min(max(attempt, 0), self.max_attempts - 1)
            self.success_on_attempt[idx] += 1
            if attempt > 0:
                if hit_429: self.errors_429_recovered += 1
                if hit_5xx: self.errors_5xx_recovered += 1
                if hit_net: self.errors_network_recovered += 1

    def record_terminal_failure(self) -> None:
        with self._lock:
            self.failed_after_all_attempts += 1

    def record_error(self, kind: str) -> None:
        with self._lock:
            if kind == "429":
                self.errors_429 += 1
            elif kind == "5xx":
                self.errors_5xx += 1
            elif kind == "network":
                self.errors_network += 1
            elif kind == "retriable_other":
                self.errors_other_retriable += 1
            elif kind == "nonretriable":
                self.errors_nonretriable += 1
            elif kind == "empty_content":
                self.empty_content_retries += 1

    def as_dict(self) -> dict:
        with self._lock:
            total_calls = sum(self.success_on_attempt) + self.failed_after_all_attempts
            return {
                "total_calls": total_calls,
                "success_on_attempt": {
                    f"attempt_{i+1}": n
                    for i, n in enumerate(self.success_on_attempt)
                },
                "failed_after_all_attempts": self.failed_after_all_attempts,
                "errors": {
                    "429":               self.errors_429,
                    "429_recovered":     self.errors_429_recovered,
                    "5xx":               self.errors_5xx,
                    "5xx_recovered":     self.errors_5xx_recovered,
                    "network":           self.errors_network,
                    "network_recovered": self.errors_network_recovered,
                    "retriable_other":   self.errors_other_retriable,
                    "nonretriable":      self.errors_nonretriable,
                    "empty_content_retries": self.empty_content_retries,
                },
            }


# ─────────────────────────────────────────────────────────────────────────────
# Instrumented OpenRouterClient. Same wire behaviour as v8's client (same
# pool, same timeouts, same provider routing, same backoff); only adds
# RetryStats hooks at the right points in the attempt loop.
# ─────────────────────────────────────────────────────────────────────────────
class InstrumentedOpenRouterClient(v8.OpenRouterClient):
    def __init__(self, api_key: str, max_attempts: int = DEFAULT_MAX_ATTEMPTS,
                 stats: RetryStats | None = None) -> None:
        super().__init__(api_key, max_attempts=max_attempts)
        self.stats = stats or RetryStats(max_attempts=max_attempts)

    def generate(self, prompt: str, category: str) -> tuple:
        model = v8.MODEL_OVERRIDES_BY_CATEGORY.get(category, self.model)
        use_stream = category not in v8.NONSTREAM_CATEGORIES
        temperature = v8.TEMPERATURE_OVERRIDES_BY_CATEGORY.get(category)
        want_reasoning = category in v8.REASONING_TRACE_CATEGORIES

        # Track whether THIS call ever saw a particular error class, so the
        # success_on_attempt branch can credit the recovery to the right
        # bucket.
        hit_429 = hit_5xx = hit_net = False

        for attempt in range(self.max_attempts):
            t_send = time.time()
            v8._log(
                f"[{category}] attempt {attempt+1}/{self.max_attempts} "
                f"model={model} stream={use_stream} temp={temperature} "
                f"reasoning={want_reasoning} prompt_chars={len(prompt):,}"
            )
            try:
                call_kwargs: dict = dict(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    stream=use_stream,
                )
                extra: dict = {
                    "provider": {
                        "sort": "throughput",
                        "allow_fallbacks": True,
                    },
                }
                if want_reasoning:
                    extra["include_reasoning"] = True
                call_kwargs["extra_body"] = extra
                if temperature is not None:
                    call_kwargs["temperature"] = temperature
                if use_stream:
                    call_kwargs["stream_options"] = {"include_usage": True}

                # ── non-streaming ───────────────────────────────────────────
                if not use_stream:
                    response = self.client.chat.completions.create(**call_kwargs)
                    msg = response.choices[0].message
                    content = msg.content or ""
                    reasoning = getattr(msg, "reasoning", "") or ""
                    usage = None
                    if response.usage:
                        usage = {
                            "prompt_tokens": response.usage.prompt_tokens,
                            "completion_tokens": response.usage.completion_tokens,
                        }
                    dt = int((time.time() - t_send) * 1000)
                    v8._log(
                        f"[{category}] NONSTREAM ok ms={dt} "
                        f"content={len(content):,} reasoning={len(reasoning):,} "
                        + (f"in={usage['prompt_tokens']} out={usage['completion_tokens']}"
                           if usage else "usage=?")
                    )
                    if content:
                        self.stats.record_success(attempt, hit_429, hit_5xx, hit_net)
                        return content, usage, reasoning
                    self.stats.record_error("empty_content")
                    self._backoff_sleep(attempt, None)
                    continue

                # ── streaming ───────────────────────────────────────────────
                stream = self.client.chat.completions.create(**call_kwargs)
                content = ""
                reasoning = ""
                last_chunk = None
                n_chunks = 0
                t_first = None
                t_open = time.time()
                last_heartbeat = t_open
                for chunk in stream:
                    last_chunk = chunk
                    n_chunks += 1
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    if getattr(delta, "content", None):
                        content += delta.content
                        if t_first is None:
                            t_first = time.time()
                    rdelta = getattr(delta, "reasoning", None)
                    if rdelta:
                        reasoning += rdelta
                    if v8.VERBOSE:
                        now = time.time()
                        if now - last_heartbeat > 5.0:
                            v8._vlog(
                                f"[{category}] streaming… chunks={n_chunks} "
                                f"bytes={len(content)} elapsed={now-t_open:.1f}s"
                            )
                            last_heartbeat = now
                usage = None
                if last_chunk and getattr(last_chunk, "usage", None):
                    u = last_chunk.usage
                    usage = {
                        "prompt_tokens": u.prompt_tokens,
                        "completion_tokens": u.completion_tokens,
                    }
                dt = int((time.time() - t_send) * 1000)
                ttfc = int((t_first - t_send) * 1000) if t_first else -1
                v8._log(
                    f"[{category}] STREAM ok ms={dt} ttfc={ttfc} "
                    f"chunks={n_chunks} content={len(content):,} "
                    + (f"in={usage['prompt_tokens']} out={usage['completion_tokens']}"
                       if usage else "usage=?")
                )
                if content:
                    self.stats.record_success(attempt, hit_429, hit_5xx, hit_net)
                    return content, usage, reasoning
                self.stats.record_error("empty_content")
                self._backoff_sleep(attempt, None)
                continue

            except Exception as e:  # noqa: BLE001
                retriable, hint = self._retriable(e)
                # Classify the error class for stats.
                status = getattr(e, "status_code", None)
                if status is None:
                    resp = getattr(e, "response", None)
                    status = getattr(resp, "status_code", None)
                if isinstance(status, int) and status == 429:
                    hit_429 = True
                    self.stats.record_error("429")
                elif isinstance(status, int) and 500 <= status < 600:
                    hit_5xx = True
                    self.stats.record_error("5xx")
                else:
                    # Network / timeout class
                    try:
                        import httpx
                        if isinstance(e, (httpx.TimeoutException, httpx.NetworkError,
                                           httpx.RemoteProtocolError, httpx.PoolTimeout)):
                            hit_net = True
                            self.stats.record_error("network")
                        elif retriable:
                            self.stats.record_error("retriable_other")
                        else:
                            self.stats.record_error("nonretriable")
                    except ImportError:
                        if retriable:
                            self.stats.record_error("retriable_other")
                        else:
                            self.stats.record_error("nonretriable")

                v8._log(f"[{category}] ERROR {type(e).__name__}: {e!s:.200} "
                        f"retriable={retriable}")
                if not retriable or attempt == self.max_attempts - 1:
                    if attempt == self.max_attempts - 1:
                        v8._log(f"[{category}] retries exhausted")
                    self.stats.record_terminal_failure()
                    return "", None, ""
                self._backoff_sleep(attempt, hint)

        v8._log(f"[{category}] retries exhausted (no successful response)")
        self.stats.record_terminal_failure()
        return "", None, ""


# ─────────────────────────────────────────────────────────────────────────────
# Driver — same flow as v8.main(), but workers=3 default and stats injected.
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(
        description="v9 = v8 plumbing + workers=3 default + retry telemetry."
    )
    ap.add_argument("--res-ids", nargs="+", default=None,
                    help="Transcript ids (e.g. RES0200). If omitted, runs all.")
    ap.add_argument("--output", type=str, default=str(OUT_DIR))
    ap.add_argument("--categories", nargs="+", default=None,
                    help="Override ACTIVE — run only these categories.")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                    help=f"Number of transcripts to process in parallel "
                         f"(default {DEFAULT_WORKERS} in v9 — proven sweet "
                         f"spot for Grade-A clean runs).")
    ap.add_argument("--reuse-nodes-from", type=str, default=None,
                    help="Path to a previous run dir; if set, Stage 1 is "
                         "skipped and nodes are loaded from that run.")
    ap.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS,
                    help=f"Per-call retry cap (default {DEFAULT_MAX_ATTEMPTS}).")
    ap.add_argument("--verbose", action="store_true",
                    help="Restore v7-level chatty stdout.")
    args = ap.parse_args()

    v8.VERBOSE = bool(args.verbose)

    if args.categories:
        all_known = set(ENTITY_CATEGORIES) | set(EDGE_CATEGORIES)
        unknown = [c for c in args.categories if c not in all_known]
        if unknown:
            sys.exit(f"ERROR: unknown --categories: {unknown}. "
                     f"Allowed: {sorted(all_known)}")
        for c in ACTIVE:
            ACTIVE[c] = c in args.categories
        v8._log(f"[v9] --categories override: running only {args.categories}")

    t_batch = time.time()
    v8._log(f"[v9] batch start @ {time.strftime('%H:%M:%S')} "
            f"workers={args.workers} max_attempts={args.max_attempts} "
            f"verbose={v8.VERBOSE}")

    v8._log("[v9] loading OpenRouter API key …")
    key = v8.get_openrouter_key()
    v8._log("[v9] key loaded")

    v8._log("[v9] loading curated KB by category …")
    curated = v8.load_curated_by_category()
    for c in ENTITY_CATEGORIES:
        flag = "ON " if ACTIVE.get(c) else "off"
        v8._log(f"[v9]   {c:16} {flag}  labels={len(curated.get(c, []))}")
    for c in EDGE_CATEGORIES:
        flag = "ON " if ACTIVE.get(c) else "off"
        v8._log(f"[v9]   {c:16} {flag}")

    v8._log("[v9] constructing instrumented OpenRouter client …")
    stats = RetryStats(max_attempts=args.max_attempts)
    client = InstrumentedOpenRouterClient(
        key, max_attempts=args.max_attempts, stats=stats,
    )

    res_ids = args.res_ids if args.res_ids else v8.collect_all_res_ids()

    run_stamp = time.strftime("%y%m%d_%H%M%S")
    out_base = Path(args.output) / run_stamp
    out_base.mkdir(parents=True, exist_ok=True)
    v8._log(f"[v9] batch: {len(res_ids)} transcripts, workers={args.workers}, "
            f"output={out_base}")

    reuse_path = Path(args.reuse_nodes_from) if args.reuse_nodes_from else None
    if reuse_path:
        v8._log(f"[v9] reuse-nodes-from: {reuse_path}")

    per_run_summaries: list[dict] = []
    try:
        with cf.ThreadPoolExecutor(
            max_workers=max(1, args.workers),
            thread_name_prefix="batch",
        ) as pool:
            futs = {
                pool.submit(v8.process_transcript, rid, client, curated,
                            out_base, reuse_path): rid
                for rid in res_ids
            }
            for fut in cf.as_completed(futs):
                rid = futs[fut]
                try:
                    summary = fut.result()
                except Exception as e:
                    v8._log(f"[{rid}] FATAL {type(e).__name__}: {e}")
                    summary = {"res_id": rid,
                               "status": f"FATAL: {type(e).__name__}: {e}"}
                per_run_summaries.append(summary)
    finally:
        client.close()

    total_ms = int((time.time() - t_batch) * 1000)
    total_in = sum(s.get("total_prompt_tokens", 0) for s in per_run_summaries)
    total_out = sum(s.get("total_completion_tokens", 0) for s in per_run_summaries)
    total_nodes = sum(s.get("n_nodes", 0) for s in per_run_summaries)
    total_edges = sum(s.get("n_edges", 0) for s in per_run_summaries)

    retry_stats = stats.as_dict()

    agg = {
        "version": "v9_workers_tuned_260425",
        "model": OPENROUTER_MODEL,
        "n_transcripts": len(res_ids),
        "n_transcripts_ok": sum(1 for s in per_run_summaries if s.get("status") == "OK"),
        "n_transcripts_failed": sum(1 for s in per_run_summaries if s.get("status") != "OK"),
        "total_ms": total_ms,
        "total_prompt_tokens": total_in,
        "total_completion_tokens": total_out,
        "total_nodes": total_nodes,
        "total_edges": total_edges,
        "active_entities": [c for c in ENTITY_CATEGORIES if ACTIVE.get(c)],
        "active_edges":    [c for c in EDGE_CATEGORIES    if ACTIVE.get(c)],
        "per_transcript": per_run_summaries,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "runtime": {
            "workers": args.workers,
            "max_attempts": args.max_attempts,
            "http_max_connections": v8.HTTP_MAX_CONNECTIONS,
            "http_max_keepalive": v8.HTTP_MAX_KEEPALIVE,
            "http_read_timeout": v8.HTTP_READ_TIMEOUT,
            "verbose": v8.VERBOSE,
        },
        "retry_stats": retry_stats,
    }
    v8._atomic_write_text(out_base / "_batch_summary.json",
                          json.dumps(agg, indent=2, ensure_ascii=False))

    v8._log("=" * 72)
    v8._log(f"BATCH DONE  transcripts={len(res_ids)} "
            f"ok={agg['n_transcripts_ok']} failed={agg['n_transcripts_failed']}")
    v8._log(f"  total time: {total_ms}ms ({total_ms/1000:.1f}s)")
    v8._log(f"  tokens: in={total_in:,} out={total_out:,}")
    v8._log(f"  nodes total: {total_nodes}  edges total: {total_edges}")
    v8._log("  retry distribution:")
    for slot, n in retry_stats["success_on_attempt"].items():
        v8._log(f"    {slot}: {n}")
    v8._log(f"    failed_after_all_attempts: {retry_stats['failed_after_all_attempts']}")
    err = retry_stats["errors"]
    v8._log(f"    429 hits / recovered:      {err['429']} / {err['429_recovered']}")
    v8._log(f"    5xx hits / recovered:      {err['5xx']} / {err['5xx_recovered']}")
    v8._log(f"    network hits / recovered:  {err['network']} / {err['network_recovered']}")
    v8._log(f"    nonretriable errors:       {err['nonretriable']}")
    v8._log(f"  summary: {out_base / '_batch_summary.json'}")


if __name__ == "__main__":
    main()
