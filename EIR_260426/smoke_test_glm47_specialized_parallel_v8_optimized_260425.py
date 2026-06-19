"""
smoke_test_glm47_specialized_parallel_v8_optimized_260425.py
============================================================

OPTIMIZED runtime wrapper around the v7 prompts.

Identical pipeline shape, identical prompts, identical CLI surface and output
schema — but materially faster and more reliable. The original
`smoke_test_glm47_specialized_parallel_v7_stage2_batch_260421.py` is NOT
modified; this file imports its prompts/constants and replaces only the
runtime layer (HTTP client, retry, logging, IO).

What changed vs v7
------------------
1. Connection pool: a single shared httpx.Client tuned for high concurrency
   (max_connections=256, keepalive=128). Default httpx pool of 100 was the
   silent bottleneck for parallel transcripts × parallel categories.
2. Timeouts: connect=15s, read=300s (reasoning), write=30s, pool=15s. The
   v7 client had no timeout — a stalled stream could hang a worker forever.
3. Retry: exponential backoff WITH FULL JITTER, classifies HTTP errors so
   only retriable ones (429 / 5xx / network / timeout) burn retries.
   Honors `Retry-After` when the server provides it.
4. OpenRouter provider routing: extra_body provider={"sort":"throughput",
   "allow_fallbacks": True} — fastest available provider for the model is
   selected automatically, with auto-failover.
5. Streaming usage: stream_options={"include_usage": True} so token usage
   is reported even on streamed calls (v7 sometimes lost it).
6. JSON extraction: greedy `json.JSONDecoder.raw_decode` walk → handles
   models that prepend prose, append commentary, or wrap in fences.
7. Logging: compact one-line-per-call by default. The full-prompt and
   full-response dumps were ~50% of runtime in v7's stdout — gated behind
   --verbose.
8. Atomic writes: per-category JSON written via tmp + os.replace so a
   crash never leaves a half-written file in the output dir.
9. Workers: default --workers raised from 4 → 8 (each transcript still
   parallelizes its own categories internally).
10. CLI compatibility: every v7 flag still works. Two new flags:
      --verbose       restore v7-level chatty stdout.
      --max-attempts  cap retries per call (default 4).

Run (one transcript, all entities + all edges):
    python smoke_test_glm47_specialized_parallel_v8_optimized_260425.py \
        --res-ids RES0200 \
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
import re
import sys
import time
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Import prompts and shared config from the v7 file. We do NOT modify it.
# ─────────────────────────────────────────────────────────────────────────────
# Make sure `eir/` is on sys.path no matter where the user runs from.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from smoke_test_glm47_specialized_parallel_v7_stage2_batch_260421 import (  # noqa: E402
    ENTITY_CATEGORIES,
    EDGE_CATEGORIES,
    ACTIVE,
    CATEGORY_PROMPTS,
    OPENROUTER_MODEL,
    MODEL_OVERRIDES_BY_CATEGORY,
    NONSTREAM_CATEGORIES,
    TEMPERATURE_OVERRIDES_BY_CATEGORY,
    REASONING_TRACE_CATEGORIES,
    EIR_ROOT,
    PROJECT_ROOT,
    TRANSCRIPT_DIR,
    CURATED_KB,
    API_KEYS_PATH,
    get_openrouter_key,
    load_curated_by_category,
    assemble_nodes_from_stage1,
    turn_text_map,
)

# v8 has its own output root so v7 results are never overwritten.
OUT_DIR = EIR_ROOT / "eir_results" / "smoke_test_glm47_specialized_parallel_v8_optimized"

# ─────────────────────────────────────────────────────────────────────────────
# Runtime knobs (overridable via CLI / env)
# ─────────────────────────────────────────────────────────────────────────────
MAX_ATTEMPTS_DEFAULT = 4
HTTP_MAX_CONNECTIONS = int(os.environ.get("OR_MAX_CONNECTIONS", "256"))
HTTP_MAX_KEEPALIVE   = int(os.environ.get("OR_MAX_KEEPALIVE",   "128"))
HTTP_CONNECT_TIMEOUT = float(os.environ.get("OR_CONNECT_TIMEOUT", "15"))
HTTP_READ_TIMEOUT    = float(os.environ.get("OR_READ_TIMEOUT",    "300"))
HTTP_WRITE_TIMEOUT   = float(os.environ.get("OR_WRITE_TIMEOUT",   "30"))
HTTP_POOL_TIMEOUT    = float(os.environ.get("OR_POOL_TIMEOUT",    "15"))

# Verbose mode toggled by CLI flag — defaults to compact logging.
VERBOSE = False


def _vlog(msg: str) -> None:
    if VERBOSE:
        print(msg, flush=True)


def _log(msg: str) -> None:
    print(msg, flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Robust JSON extraction
#
# Models occasionally:
#   - wrap output in ```json … ``` fences
#   - prepend a sentence ("Here's the JSON:")
#   - append a closing remark ("Hope this helps!")
#   - emit a trailing comma
# We walk every '{' position with raw_decode and keep the largest valid
# object. This is strictly more permissive than v7's regex approach.
# ─────────────────────────────────────────────────────────────────────────────
_JSON_DEC = json.JSONDecoder()


def extract_json_object(text: str) -> dict | None:
    if not text:
        return None
    s = text.strip()
    # Fast path
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    # Strip code fences
    if s.startswith("```"):
        m = re.match(r"```(?:json)?\s*(.*?)\s*```", s, re.DOTALL)
        if m:
            inner = m.group(1).strip()
            try:
                return json.loads(inner)
            except json.JSONDecodeError:
                s = inner
    best: dict | None = None
    best_len = -1
    for i, ch in enumerate(s):
        if ch != "{":
            continue
        try:
            obj, end = _JSON_DEC.raw_decode(s[i:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and end > best_len:
            best, best_len = obj, end
    if best is not None:
        return best
    # Last-ditch: drop trailing commas before } or ]
    fixed = re.sub(r",(\s*[}\]])", r"\1", s)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Atomic file writes
# ─────────────────────────────────────────────────────────────────────────────
def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


# ─────────────────────────────────────────────────────────────────────────────
# Optimized OpenRouter client — connection pooling, jittered backoff,
# provider routing, streaming usage. Same return shape as v7's client:
# (content, usage, reasoning).
# ─────────────────────────────────────────────────────────────────────────────
class OpenRouterClient:
    def __init__(self, api_key: str, model: str = OPENROUTER_MODEL,
                 max_attempts: int = MAX_ATTEMPTS_DEFAULT):
        try:
            import httpx
        except ImportError as e:
            raise SystemExit(
                "httpx is required for the optimized client. "
                "Install with: pip install httpx"
            ) from e
        from openai import OpenAI

        self._http = httpx.Client(
            limits=httpx.Limits(
                max_connections=HTTP_MAX_CONNECTIONS,
                max_keepalive_connections=HTTP_MAX_KEEPALIVE,
            ),
            timeout=httpx.Timeout(
                connect=HTTP_CONNECT_TIMEOUT,
                read=HTTP_READ_TIMEOUT,
                write=HTTP_WRITE_TIMEOUT,
                pool=HTTP_POOL_TIMEOUT,
            ),
            transport=httpx.HTTPTransport(retries=0),  # we own retries
        )
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            http_client=self._http,
        )
        self.model = model
        self.max_attempts = max(1, int(max_attempts))

    # ── retry classification ────────────────────────────────────────────────
    @staticmethod
    def _retriable(err: Exception) -> tuple[bool, float | None]:
        """Return (should_retry, server_hint_seconds)."""
        # Network/timeout errors → retry.
        try:
            import httpx
            if isinstance(err, (httpx.TimeoutException, httpx.NetworkError,
                                 httpx.RemoteProtocolError, httpx.PoolTimeout)):
                return True, None
        except ImportError:
            pass
        # OpenAI SDK wraps HTTP errors with a `.status_code`/`.response`.
        status = getattr(err, "status_code", None)
        if status is None:
            resp = getattr(err, "response", None)
            status = getattr(resp, "status_code", None)
        if isinstance(status, int):
            if status == 408 or status == 429 or 500 <= status < 600:
                hint = None
                resp = getattr(err, "response", None)
                if resp is not None:
                    try:
                        ra = resp.headers.get("retry-after")
                        if ra is not None:
                            hint = float(ra)
                    except Exception:
                        pass
                return True, hint
            # Non-retriable 4xx (auth, schema, model-not-found, etc.)
            return False, None
        # Unknown error class — retry once (last attempt will surface it).
        return True, None

    @staticmethod
    def _backoff_sleep(attempt: int, hint: float | None) -> None:
        # Full jitter exponential backoff. attempt is 0-indexed.
        base = min(30.0, 2.0 * (2 ** attempt))
        wait = random.uniform(0.0, base)
        if hint is not None:
            wait = max(wait, hint)
        time.sleep(wait)

    # ── main entry point ────────────────────────────────────────────────────
    def generate(self, prompt: str, category: str) -> tuple:
        """Return (content, usage, reasoning). reasoning is '' unless the
        category is in REASONING_TRACE_CATEGORIES."""
        model = MODEL_OVERRIDES_BY_CATEGORY.get(category, self.model)
        use_stream = category not in NONSTREAM_CATEGORIES
        temperature = TEMPERATURE_OVERRIDES_BY_CATEGORY.get(category)
        want_reasoning = category in REASONING_TRACE_CATEGORIES

        for attempt in range(self.max_attempts):
            t_send = time.time()
            _log(f"[{category}] attempt {attempt+1}/{self.max_attempts} "
                 f"model={model} stream={use_stream} temp={temperature} "
                 f"reasoning={want_reasoning} prompt_chars={len(prompt):,}")

            try:
                call_kwargs: dict = dict(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    stream=use_stream,
                )
                # OpenRouter-specific routing for throughput + reliability.
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
                    # Ask OpenRouter to include usage in the final SSE chunk.
                    call_kwargs["stream_options"] = {"include_usage": True}

                # ── non-streaming branch ────────────────────────────────────
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
                    _log(f"[{category}] NONSTREAM ok ms={dt} "
                         f"content={len(content):,} reasoning={len(reasoning):,} "
                         + (f"in={usage['prompt_tokens']} out={usage['completion_tokens']}"
                            if usage else "usage=?"))
                    if content:
                        return content, usage, reasoning
                    _vlog(f"[{category}] empty content — retrying")
                    self._backoff_sleep(attempt, None)
                    continue

                # ── streaming branch ────────────────────────────────────────
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
                    # Some OR providers expose reasoning deltas on streams.
                    rdelta = getattr(delta, "reasoning", None)
                    if rdelta:
                        reasoning += rdelta
                    if VERBOSE:
                        now = time.time()
                        if now - last_heartbeat > 5.0:
                            _vlog(f"[{category}] streaming… chunks={n_chunks} "
                                  f"bytes={len(content)} elapsed={now-t_open:.1f}s")
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
                _log(f"[{category}] STREAM ok ms={dt} ttfc={ttfc} "
                     f"chunks={n_chunks} content={len(content):,} "
                     + (f"in={usage['prompt_tokens']} out={usage['completion_tokens']}"
                        if usage else "usage=?"))
                if content:
                    return content, usage, reasoning
                _vlog(f"[{category}] empty content — retrying")
                self._backoff_sleep(attempt, None)
                continue

            except Exception as e:  # noqa: BLE001 — we classify below
                retriable, hint = self._retriable(e)
                _log(f"[{category}] ERROR {type(e).__name__}: {e!s:.200} "
                     f"retriable={retriable}")
                if not retriable or attempt == self.max_attempts - 1:
                    if attempt == self.max_attempts - 1:
                        _log(f"[{category}] retries exhausted")
                    return "", None, ""
                self._backoff_sleep(attempt, hint)

        _log(f"[{category}] retries exhausted (no successful response)")
        return "", None, ""

    def close(self) -> None:
        try:
            self._http.close()
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Per-category workers (Stage 1 entities, Stage 2 edges)
# Same return shape as v7. Logging compacted; full prompt/response dumps
# only emitted under --verbose.
# ─────────────────────────────────────────────────────────────────────────────
def _write_category_file(out_dir: Path, cat: str, payload: dict) -> None:
    file = out_dir / f"{out_dir.name}_{cat}.json"
    _atomic_write_text(file, json.dumps(payload, indent=2, ensure_ascii=False))
    _vlog(f"[{cat}] saved {file}")


def run_entity_category(
    cat: str,
    client: OpenRouterClient,
    transcript: str,
    curated_labels: list[str],
    out_per_cat_dir: Path,
) -> dict:
    if not ACTIVE.get(cat):
        return {"category": cat, "status": "SKIPPED_INACTIVE"}
    prompt_template = CATEGORY_PROMPTS.get(cat)
    if prompt_template is None:
        raise RuntimeError(
            f"ACTIVE['{cat}']=True but CATEGORY_PROMPTS['{cat}'] is None."
        )
    t_cat = time.time()
    bulleted = "\n".join(f"- {l}" for l in curated_labels) if curated_labels else "(none)"
    prompt = prompt_template.format(
        curated_labels_bulleted=bulleted,
        transcript=transcript,
    )
    _log(f"[{cat}] BEGIN entity curated={len(curated_labels)} "
         f"prompt_chars={len(prompt):,}")
    if VERBOSE:
        _vlog(f"\n┌─── PROMPT ({cat}) ─── {len(prompt):,} chars ───")
        _vlog(prompt)
        _vlog(f"└─── END PROMPT ({cat}) ───\n")

    content, usage, _reasoning = client.generate(prompt, cat)
    call_ms = int((time.time() - t_cat) * 1000)

    if not content:
        _log(f"[{cat}] status=LLM_FAIL ms={call_ms}")
        payload = {"category": cat, "status": "LLM_FAIL",
                   "category_ms": call_ms, "usage": usage,
                   "matched": [], "other": [], "fine_grained": []}
        _write_category_file(out_per_cat_dir, cat, payload)
        return payload

    if VERBOSE:
        _vlog(f"\n┌─── RESPONSE ({cat}) ─── {len(content):,} chars ───")
        _vlog(content)
        _vlog(f"└─── END RESPONSE ({cat}) ───\n")

    parsed = extract_json_object(content)
    if parsed is None:
        _log(f"[{cat}] status=PARSE_FAIL ms={call_ms} content={len(content):,}")
        payload = {"category": cat, "status": "PARSE_FAIL",
                   "category_ms": call_ms, "usage": usage,
                   "raw_content": content,
                   "matched": [], "other": [], "fine_grained": []}
        _write_category_file(out_per_cat_dir, cat, payload)
        return payload

    matched = parsed.get("matched") or []
    other = parsed.get("other") or []
    fine_grained = parsed.get("fine_grained") or []
    payload = {"category": cat, "status": "OK",
               "category_ms": call_ms, "usage": usage,
               "matched": matched, "other": other,
               "fine_grained": fine_grained,
               "raw_content": content}
    _log(f"[{cat}] status=OK matched={len(matched)} other={len(other)} "
         f"fine={len(fine_grained)} ms={call_ms}")
    if VERBOSE:
        if matched:      _vlog(f"  MATCHED:      {matched}")
        if other:        _vlog(f"  OTHER:        {other}")
        if fine_grained: _vlog(f"  FINE_GRAINED: {fine_grained}")
    _write_category_file(out_per_cat_dir, cat, payload)
    return payload


def run_edge_category(
    cat: str,
    client: OpenRouterClient,
    transcript: str,
    turn_map: dict[str, str],
    nodes: list[dict],
    out_per_cat_dir: Path,
) -> dict:
    if not ACTIVE.get(cat):
        return {"category": cat, "status": "SKIPPED_INACTIVE", "edges": []}
    if not nodes:
        return {"category": cat, "status": "NO_NODES", "edges": []}
    prompt_template = CATEGORY_PROMPTS.get(cat)
    if prompt_template is None:
        raise RuntimeError(
            f"ACTIVE['{cat}']=True but CATEGORY_PROMPTS['{cat}'] is None."
        )
    t_cat = time.time()
    numbered_entities_list = "\n".join(
        f'  {i}. "{n["text"]}" ({n["type"]}, turn {n.get("turn_id","")})'
        for i, n in enumerate(nodes, 1)
    )
    prompt = prompt_template.format(
        numbered_entities_list=numbered_entities_list,
        transcript=transcript,
    )
    _log(f"[{cat}] BEGIN edge nodes={len(nodes)} prompt_chars={len(prompt):,}")
    if VERBOSE:
        _vlog(f"\n┌─── PROMPT ({cat}) ─── {len(prompt):,} chars ───")
        _vlog(prompt)
        _vlog(f"└─── END PROMPT ({cat}) ───\n")

    content, usage, reasoning = client.generate(prompt, cat)
    call_ms = int((time.time() - t_cat) * 1000)

    if reasoning:
        reasoning_path = out_per_cat_dir / f"{out_per_cat_dir.name}_{cat}_reasoning.txt"
        _atomic_write_text(reasoning_path, reasoning)
        _vlog(f"[{cat}] saved reasoning → {reasoning_path.name} "
              f"({len(reasoning):,} chars)")

    if not content:
        _log(f"[{cat}] status=LLM_FAIL ms={call_ms}")
        payload = {"category": cat, "status": "LLM_FAIL",
                   "category_ms": call_ms, "usage": usage,
                   "reasoning_chars": len(reasoning),
                   "matched": [], "other": [], "fine_grained": [], "edges": []}
        _write_category_file(out_per_cat_dir, cat, payload)
        return payload

    if VERBOSE:
        _vlog(f"\n┌─── RESPONSE ({cat}) ─── {len(content):,} chars ───")
        _vlog(content)
        _vlog(f"└─── END RESPONSE ({cat}) ───\n")

    parsed = extract_json_object(content)
    if parsed is None or not any(k in parsed for k in ("matched", "other", "fine_grained")):
        _log(f"[{cat}] status=PARSE_FAIL ms={call_ms} content={len(content):,}")
        payload = {"category": cat, "status": "PARSE_FAIL",
                   "category_ms": call_ms, "usage": usage,
                   "reasoning_chars": len(reasoning),
                   "raw_content": content,
                   "matched": [], "other": [], "fine_grained": [], "edges": []}
        _write_category_file(out_per_cat_dir, cat, payload)
        return payload

    def _passthrough(raw_list) -> list:
        out = []
        for e in raw_list or []:
            src = e.get("src", "")
            dst = e.get("dst", "")
            if not src or not dst or src == dst:
                continue
            out.append({
                "src": src, "dst": dst, "type": cat,
                "turn_id": e.get("turn_id", ""),
                "quote": (e.get("quote") or "")[:300],
            })
        return out

    matched = _passthrough(parsed.get("matched"))
    other = _passthrough(parsed.get("other"))
    fine_grained = _passthrough(parsed.get("fine_grained"))
    vstats = {"matched_kept": len(matched),
              "other_kept": len(other),
              "fine_grained_kept": len(fine_grained)}
    all_edges = matched + other + fine_grained

    payload = {"category": cat, "status": "OK",
               "category_ms": call_ms, "usage": usage,
               "reasoning_chars": len(reasoning),
               "matched": matched, "other": other, "fine_grained": fine_grained,
               "edges": all_edges, "validation": vstats,
               "raw_content": content}
    _log(f"[{cat}] status=OK matched={len(matched)} other={len(other)} "
         f"fine={len(fine_grained)} ms={call_ms}")
    if VERBOSE:
        for bucket_name, bucket in (("matched", matched), ("other", other),
                                     ("fine_grained", fine_grained)):
            for e in bucket:
                _vlog(f"  EDGE[{bucket_name}]: {e['src']} -[{e['type']}]-> "
                      f"{e['dst']}  turn={e['turn_id']}  quote={e['quote']!r}")
    _write_category_file(out_per_cat_dir, cat, payload)
    return payload


# ─────────────────────────────────────────────────────────────────────────────
# Per-transcript processor
# ─────────────────────────────────────────────────────────────────────────────
def process_transcript(
    res_id: str,
    client: OpenRouterClient,
    curated: dict[str, list[str]],
    out_base: Path,
    reuse_nodes_from: Path | None = None,
) -> dict:
    t_script = time.time()
    transcript_path = TRANSCRIPT_DIR / res_id / f"{res_id}.txt"
    if not transcript_path.exists():
        return {"res_id": res_id, "status": "TRANSCRIPT_MISSING",
                "path": str(transcript_path)}
    transcript = transcript_path.read_text()
    n_turns = len(re.findall(r"\[[DP]-\d+\]", transcript))
    t_map = turn_text_map(transcript)

    out_root = out_base / res_id
    out_root.mkdir(parents=True, exist_ok=True)

    active_entities = [c for c in ENTITY_CATEGORIES if ACTIVE.get(c)]
    active_edges    = [c for c in EDGE_CATEGORIES    if ACTIVE.get(c)]

    _log("=" * 72)
    _log(f"  {res_id}  turns={n_turns} chars={len(transcript):,}")
    _log(f"  active entities: {active_entities}")
    _log(f"  active edges:    {active_edges}")
    if reuse_nodes_from:
        _log(f"  reuse nodes from: {reuse_nodes_from}")
    _log("=" * 72)

    stage1_start = time.time()
    stage1_results: dict[str, dict] = {}
    nodes: list[dict] = []
    if reuse_nodes_from is not None:
        prev_all = reuse_nodes_from / res_id / f"{res_id}_all.json"
        if not prev_all.exists():
            _log(f"[{res_id}] REUSE FAIL — no {prev_all}")
            return {"res_id": res_id, "status": "REUSE_FAIL", "path": str(prev_all)}
        prev = json.loads(prev_all.read_text())
        nodes = prev.get("nodes", []) or []
        stage1_results = prev.get("stage1_results", {}) or {}
        _log(f"[{res_id}] REUSE: loaded {len(nodes)} nodes from {prev_all.name}")
        for cat in ENTITY_CATEGORIES:
            src = reuse_nodes_from / res_id / f"{res_id}_{cat}.json"
            if src.exists():
                _atomic_write_text(out_root / src.name, src.read_text())
    elif active_entities:
        _log(f"[{res_id}] Stage 1 — parallel entity extraction "
             f"({len(active_entities)} workers)")
        with cf.ThreadPoolExecutor(
            max_workers=max(1, len(active_entities)),
            thread_name_prefix=f"s1-{res_id}",
        ) as pool:
            futs = {
                pool.submit(run_entity_category, cat, client, transcript,
                            curated.get(cat, []), out_root): cat
                for cat in active_entities
            }
            for fut in cf.as_completed(futs):
                cat = futs[fut]
                try:
                    stage1_results[cat] = fut.result()
                except Exception as e:
                    _log(f"[{res_id}/{cat}] WORKER ERROR {type(e).__name__}: {e}")
                    stage1_results[cat] = {"category": cat,
                                           "status": f"WORKER_ERROR: {e}",
                                           "matched": [], "other": [],
                                           "fine_grained": []}
    stage1_ms = int((time.time() - stage1_start) * 1000)
    _log(f"[{res_id}] Stage 1 done  stage1_ms={stage1_ms}")

    if reuse_nodes_from is None:
        nodes = assemble_nodes_from_stage1(stage1_results)
    _log(f"[{res_id}] nodes for Stage 2: {len(nodes)}")

    stage2_start = time.time()
    stage2_results: dict[str, dict] = {}
    if active_edges and nodes:
        _log(f"[{res_id}] Stage 2 — parallel edge extraction "
             f"({len(active_edges)} workers)")
        with cf.ThreadPoolExecutor(
            max_workers=max(1, len(active_edges)),
            thread_name_prefix=f"s2-{res_id}",
        ) as pool:
            futs = {
                pool.submit(run_edge_category, cat, client, transcript,
                            t_map, nodes, out_root): cat
                for cat in active_edges
            }
            for fut in cf.as_completed(futs):
                cat = futs[fut]
                try:
                    stage2_results[cat] = fut.result()
                except Exception as e:
                    _log(f"[{res_id}/{cat}] WORKER ERROR {type(e).__name__}: {e}")
                    stage2_results[cat] = {"category": cat,
                                           "status": f"WORKER_ERROR: {e}",
                                           "edges": []}
    elif active_edges and not nodes:
        _log(f"[{res_id}] Stage 2 skipped — no Stage 1 nodes")
    stage2_ms = int((time.time() - stage2_start) * 1000)
    _log(f"[{res_id}] Stage 2 done  stage2_ms={stage2_ms}")

    total_in = sum((r.get("usage") or {}).get("prompt_tokens", 0)
                   for r in list(stage1_results.values()) + list(stage2_results.values()))
    total_out = sum((r.get("usage") or {}).get("completion_tokens", 0)
                    for r in list(stage1_results.values()) + list(stage2_results.values()))
    total_ms = int((time.time() - t_script) * 1000)

    roll_up = {
        "res_id": res_id,
        "status": "OK",
        "model": OPENROUTER_MODEL,
        "active_entities": active_entities,
        "active_edges":    active_edges,
        "n_turns": n_turns,
        "n_chars": len(transcript),
        "n_nodes": len(nodes),
        "n_edges": sum(len(r.get("edges", [])) for r in stage2_results.values()),
        "stage1_ms": stage1_ms,
        "stage2_ms": stage2_ms,
        "total_ms": total_ms,
        "total_prompt_tokens": total_in,
        "total_completion_tokens": total_out,
        "stage1_results": stage1_results,
        "stage2_results": stage2_results,
        "nodes": nodes,
    }
    _atomic_write_text(out_root / f"{res_id}_all.json",
                       json.dumps(roll_up, indent=2, ensure_ascii=False))
    _log(f"[{res_id}] total={total_ms}ms tokens in={total_in:,} out={total_out:,} "
         f"nodes={len(nodes)} edges={roll_up['n_edges']}")
    return roll_up


def collect_all_res_ids() -> list[str]:
    return sorted(d.name for d in TRANSCRIPT_DIR.glob("RES*") if d.is_dir())


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    global VERBOSE
    ap = argparse.ArgumentParser(
        description="v8 optimized runtime over v7 prompts. Same CLI as v7."
    )
    ap.add_argument("--res-ids", nargs="+", default=None,
                    help="Transcript ids (e.g. RES0200 RES0203). "
                         "If omitted, runs all.")
    ap.add_argument("--output", type=str, default=str(OUT_DIR))
    ap.add_argument("--categories", nargs="+", default=None,
                    help="Override ACTIVE — run only these categories. "
                         "Allowed: " + str(sorted(set(ENTITY_CATEGORIES) |
                                                  set(EDGE_CATEGORIES))))
    ap.add_argument("--workers", type=int, default=8,
                    help="Number of transcripts to process in parallel "
                         "(default 8 in v8; was 4 in v7).")
    ap.add_argument("--reuse-nodes-from", type=str, default=None,
                    help="Path to a previous run dir. If set, Stage 1 is "
                         "skipped and nodes are loaded from that run.")
    ap.add_argument("--max-attempts", type=int, default=MAX_ATTEMPTS_DEFAULT,
                    help=f"Per-call retry cap (default {MAX_ATTEMPTS_DEFAULT}).")
    ap.add_argument("--verbose", action="store_true",
                    help="Restore v7-level chatty stdout (full prompt+response "
                         "dumps, streaming heartbeats).")
    args = ap.parse_args()

    VERBOSE = bool(args.verbose)

    if args.categories:
        all_known = set(ENTITY_CATEGORIES) | set(EDGE_CATEGORIES)
        unknown = [c for c in args.categories if c not in all_known]
        if unknown:
            sys.exit(f"ERROR: unknown --categories: {unknown}. "
                     f"Allowed: {sorted(all_known)}")
        for c in ACTIVE:
            ACTIVE[c] = c in args.categories
        _log(f"[v8] --categories override: running only {args.categories}")

    t_batch = time.time()
    _log(f"[v8] batch start @ {time.strftime('%H:%M:%S')} workers={args.workers} "
         f"max_attempts={args.max_attempts} verbose={VERBOSE}")

    _log("[v8] loading OpenRouter API key …")
    key = get_openrouter_key()
    _log("[v8] key loaded")

    _log("[v8] loading curated KB by category …")
    curated = load_curated_by_category()
    for c in ENTITY_CATEGORIES:
        flag = "ON " if ACTIVE.get(c) else "off"
        _log(f"[v8]   {c:16} {flag}  labels={len(curated.get(c, []))}")
    for c in EDGE_CATEGORIES:
        flag = "ON " if ACTIVE.get(c) else "off"
        _log(f"[v8]   {c:16} {flag}")

    _log("[v8] constructing OpenRouter client (pooled httpx, jittered retry) …")
    client = OpenRouterClient(key, max_attempts=args.max_attempts)

    if args.res_ids:
        res_ids = args.res_ids
    else:
        res_ids = collect_all_res_ids()

    run_stamp = time.strftime("%y%m%d_%H%M%S")
    out_base = Path(args.output) / run_stamp
    out_base.mkdir(parents=True, exist_ok=True)
    _log(f"[v8] batch: {len(res_ids)} transcripts, workers={args.workers}, "
         f"output={out_base}")

    reuse_path = Path(args.reuse_nodes_from) if args.reuse_nodes_from else None
    if reuse_path:
        _log(f"[v8] reuse-nodes-from: {reuse_path}")

    per_run_summaries: list[dict] = []
    try:
        with cf.ThreadPoolExecutor(
            max_workers=max(1, args.workers),
            thread_name_prefix="batch",
        ) as pool:
            futs = {
                pool.submit(process_transcript, rid, client, curated, out_base,
                            reuse_path): rid
                for rid in res_ids
            }
            for fut in cf.as_completed(futs):
                rid = futs[fut]
                try:
                    summary = fut.result()
                except Exception as e:
                    _log(f"[{rid}] FATAL {type(e).__name__}: {e}")
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

    agg = {
        "version": "v8_optimized_260425",
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
            "http_max_connections": HTTP_MAX_CONNECTIONS,
            "http_max_keepalive": HTTP_MAX_KEEPALIVE,
            "http_read_timeout": HTTP_READ_TIMEOUT,
            "verbose": VERBOSE,
        },
    }
    _atomic_write_text(out_base / "_batch_summary.json",
                       json.dumps(agg, indent=2, ensure_ascii=False))

    _log("=" * 72)
    _log(f"BATCH DONE  transcripts={len(res_ids)} "
         f"ok={agg['n_transcripts_ok']} failed={agg['n_transcripts_failed']}")
    _log(f"  total time: {total_ms}ms ({total_ms/1000:.1f}s)")
    _log(f"  tokens: in={total_in:,} out={total_out:,}")
    _log(f"  nodes total: {total_nodes}  edges total: {total_edges}")
    _log(f"  summary: {out_base / '_batch_summary.json'}")


if __name__ == "__main__":
    main()
