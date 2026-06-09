"""Provider-agnostic LLM access for silver-graph extraction, QA generation, and
answer judging.

Design goals:
* One :class:`LLMClient` interface; pick the backend in ``config.yaml``
  (``anthropic`` | ``openai`` | ``gemini`` | ``openrouter`` | ``mock``).
* A ``mock`` provider returns deterministic stub output so the entire pipeline
  runs with no API keys (for plumbing/CI — never for reported numbers).
* Robust JSON extraction so a model that wraps JSON in prose still parses.
* Keys are read from environment variables named in config; never hardcoded.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import List, Optional

from .io_utils import LOG, Config

PROMPTS_DIR = Path(__file__).parent / "prompts"


def load_prompt(name: str) -> str:
    return (PROMPTS_DIR / f"{name}.txt").read_text(encoding="utf-8")


def extract_json(text: str):
    """Pull the first JSON object/array out of an LLM response."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Fallback: first balanced { ... } or [ ... ].
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"no JSON found in model output: {text[:200]!r}")


# --------------------------------------------------------------------------- #
# Provider backends
# --------------------------------------------------------------------------- #
class LLMClient:
    """Unified chat client. Call :meth:`complete` for text, :meth:`complete_json`
    for a parsed object."""

    def __init__(self, cfg: Config, provider: Optional[str] = None):
        self.cfg = cfg
        self.provider = provider or cfg.get("llm.provider", "anthropic")
        self.max_tokens = int(cfg.get("llm.max_tokens", 4096))
        self.temperature = float(cfg.get("llm.temperature", 0.0))
        self.max_retries = int(cfg.get("llm.max_retries", 4))
        pcfg = cfg.get(f"llm.{self.provider}", {}) or {}
        self.model = pcfg.get("model", "")
        self.api_key = os.environ.get(pcfg.get("api_key_env", ""), "") if self.provider != "mock" else ""
        if self.provider != "mock" and not self.api_key:
            LOG.warning("provider %s selected but env %s is empty; calls will fail",
                        self.provider, pcfg.get("api_key_env"))

    # ----- public API ------------------------------------------------------ #
    def complete(self, system: str, user: str) -> str:
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                return self._dispatch(system, user)
            except Exception as exc:  # noqa: BLE001 — provider-agnostic retry
                last_exc = exc
                wait = min(2 ** attempt, 20)
                LOG.warning("LLM call failed (%s); retry in %ss", exc, wait)
                time.sleep(wait)
        raise RuntimeError(f"LLM call failed after {self.max_retries} retries: {last_exc}")

    def complete_json(self, system: str, user: str):
        return extract_json(self.complete(system, user))

    # ----- dispatch -------------------------------------------------------- #
    def _dispatch(self, system: str, user: str) -> str:
        return getattr(self, f"_call_{self.provider}")(system, user)

    def _call_mock(self, system: str, user: str) -> str:
        """Deterministic stub. Emits minimal valid JSON for graph/QA prompts so
        downstream code runs; clearly not a real extraction."""
        if "nodes" in user.lower() or "knowledge graph" in (system + user).lower():
            return json.dumps({"nodes": [], "edges": []})
        if "questions" in (system + user).lower():
            return json.dumps({"questions": []})
        if "correctness" in (system + user).lower():
            return json.dumps({"correctness": 0, "score": 1, "explanation": "mock judge"})
        return "MOCK"

    def _call_anthropic(self, system: str, user: str) -> str:
        import anthropic

        client = anthropic.Anthropic(api_key=self.api_key)
        kwargs = dict(model=self.model, max_tokens=self.max_tokens,
                      system=system, messages=[{"role": "user", "content": user}])
        # Some newer models (e.g. claude-opus-4-8) deprecate `temperature`; send
        # it only if not yet rejected for this client, and drop it permanently
        # on the deprecation 400 instead of failing the call.
        if not getattr(self, "_drop_temperature", False):
            try:
                resp = client.messages.create(temperature=self.temperature, **kwargs)
                return resp.content[0].text
            except anthropic.BadRequestError as exc:
                if "temperature" not in str(exc):
                    raise
                LOG.info("model %s rejects `temperature`; retrying without it", self.model)
                self._drop_temperature = True
        resp = client.messages.create(**kwargs)
        return resp.content[0].text

    def _call_openai(self, system: str, user: str) -> str:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key)
        resp = client.chat.completions.create(
            model=self.model, temperature=self.temperature,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return resp.choices[0].message.content

    def _call_gemini(self, system: str, user: str) -> str:
        from google import genai

        client = genai.Client(api_key=self.api_key)
        resp = client.models.generate_content(model=self.model, contents=f"{system}\n\n{user}")
        return resp.text

    def _call_openrouter(self, system: str, user: str) -> str:
        import requests

        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "temperature": self.temperature,
                  "messages": [{"role": "system", "content": system},
                               {"role": "user", "content": user}]},
            timeout=int(self.cfg.get("llm.timeout_s", 120)),
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


# --------------------------------------------------------------------------- #
# Task helpers
# --------------------------------------------------------------------------- #
def judge_answer(client: LLMClient, question: str, gold: str, predicted: str, evidence: str) -> dict:
    """Score a predicted answer vs the gold answer. Returns
    ``{correctness: 0/1, score: 1-5, explanation: str}``."""
    system = load_prompt("judge")
    user = (
        f"## Question\n{question}\n\n## Transcript evidence\n{evidence}\n\n"
        f"## Gold answer\n{gold}\n\n## Predicted answer\n{predicted}\n\n"
        "Return ONLY the JSON object."
    )
    try:
        out = client.complete_json(system, user)
        return {
            "correctness": int(out.get("correctness", 0)),
            "score": int(out.get("score", 1)),
            "explanation": str(out.get("explanation", "")),
        }
    except Exception as exc:  # noqa: BLE001
        LOG.warning("judge failed: %s", exc)
        return {"correctness": 0, "score": 1, "explanation": f"judge_error: {exc}"}
