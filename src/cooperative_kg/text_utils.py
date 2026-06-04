"""JSON parsing, transcript normalization, and canonicalization."""

from __future__ import annotations

import json
import re

from .constants import CANONICAL_ALIASES


def extract_json(text: str):
    """Parse JSON from model output, including fenced or think-tagged output."""
    text = re.sub(r"<think>[\s\S]*?</think>", "", text or "", flags=re.I).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fenced:
        try:
            return json.loads(fenced.group(1).strip())
        except json.JSONDecodeError:
            pass

    candidate = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
    if candidate:
        value = re.sub(r",\s*([}\]])", r"\1", candidate.group(1))
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
    return None


def normalize_for_match(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def canonicalize_text(text: str) -> str:
    text = normalize_for_match(text)
    text = re.sub(r"^(the|a|an)\s+", "", text)
    text = re.sub(r"\s+", " ", text).strip(" .,:;")
    if text.startswith("no "):
        return text
    if text.startswith("non "):
        text = "non-" + text[4:]
    if text.startswith("absent "):
        rest = canonicalize_text(text[len("absent ") :])
        return f"absent {rest}" if rest else ""
    return CANONICAL_ALIASES.get(text, text)


def token_overlap_supported(evidence: str, transcript_norm: str) -> bool:
    ev = normalize_for_match(evidence)
    if not ev:
        return False
    if ev in transcript_norm:
        return True
    ev_tokens = re.findall(r"\b\w+\b", ev)
    if len(ev_tokens) <= 3:
        return True
    transcript_tokens = set(re.findall(r"\b\w+\b", transcript_norm))
    hits = sum(1 for token in ev_tokens if token in transcript_tokens)
    return hits / len(ev_tokens) >= 0.6
