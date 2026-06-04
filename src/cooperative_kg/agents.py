"""Cooperative multi-agent LLM extraction stages."""

from __future__ import annotations

from . import models
from .constants import VALID_EDGE_TYPES, VALID_NODE_TYPES
from .prompts import (
    build_agent1_prompt,
    build_agent2_prompt,
    build_agent3_prompt,
    build_agent4_prompt,
    build_agent5_prompt,
)
from .text_utils import canonicalize_text, extract_json


def agent1_high_recall_entities(transcript: str, client) -> tuple[list[dict], dict]:
    prompt = build_agent1_prompt(transcript)
    content, usage = client.generate(prompt, models.MODEL_RECALL)
    nodes = extract_json(content)
    return (nodes if isinstance(nodes, list) else []), usage


def agent2_precision_filter(
    candidates: list[dict], transcript: str, client
) -> tuple[list[dict], dict]:
    if not candidates:
        return [], {}
    prompt = build_agent2_prompt(candidates, transcript)
    content, usage = client.generate(prompt, models.MODEL_FILTER)
    decision = extract_json(content)
    if not isinstance(decision, dict):
        return candidates, usage
    keep_ids = set(decision.get("keep_ids") or [])
    if not keep_ids:
        return candidates, usage
    return [n for n in candidates if n.get("id") in keep_ids], usage


def agent3_negations(transcript: str, client) -> tuple[list[dict], dict]:
    prompt = build_agent3_prompt(transcript)
    content, usage = client.generate(prompt, models.MODEL_NEGATION)
    nodes = extract_json(content)
    return (nodes if isinstance(nodes, list) else []), usage


def merge_nodes(raw_nodes: list[dict]) -> list[dict]:
    merged: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for node in raw_nodes:
        if not isinstance(node, dict):
            continue
        ntype = str(node.get("type") or "").strip().upper()
        if ntype not in VALID_NODE_TYPES:
            continue
        text = canonicalize_text(str(node.get("text") or ""))
        if not text:
            continue
        key = (text.lower(), ntype)
        if key in seen:
            continue
        seen.add(key)
        merged.append(
            {
                "id": f"N_{len(merged) + 1:03d}",
                "text": text,
                "type": ntype,
                "evidence": str(node.get("evidence") or "").strip(),
                "turn_id": str(node.get("turn_id") or "").strip(),
            }
        )
    return merged


def agent4_relations(nodes: list[dict], transcript: str, client) -> tuple[list[dict], dict]:
    if not nodes:
        return [], {}
    prompt = build_agent4_prompt(nodes, transcript)
    content, usage = client.generate(prompt, models.MODEL_RELATION)
    edges = extract_json(content)
    return (edges if isinstance(edges, list) else []), usage


def apply_canonicalization(kg: dict, rewritten: list) -> dict:
    """Apply agent-5 rewriting results to a KG dict."""
    nodes = kg.get("nodes", [])
    id_to_text = {
        str(item.get("id")): canonicalize_text(str(item.get("text") or ""))
        for item in rewritten
        if isinstance(item, dict) and item.get("id")
    }
    remapped_nodes: list[dict] = []
    old_to_new: dict[str, str] = {}
    seen: dict[tuple[str, str], str] = {}

    for node in nodes:
        new_text = id_to_text.get(node["id"]) or canonicalize_text(node["text"])
        if not new_text:
            continue
        key = (new_text.lower(), node["type"])
        if key in seen:
            old_to_new[node["id"]] = seen[key]
            continue
        new_id = f"N_{len(remapped_nodes) + 1:03d}"
        seen[key] = new_id
        old_to_new[node["id"]] = new_id
        remapped_nodes.append({**node, "id": new_id, "text": new_text})

    remapped_edges: list[dict] = []
    edge_seen: set[tuple[str, str, str]] = set()
    for edge in kg.get("edges", []):
        src = old_to_new.get(edge.get("source_id"))
        tgt = old_to_new.get(edge.get("target_id"))
        etype = str(edge.get("type") or "").upper()
        if not src or not tgt or src == tgt or etype not in VALID_EDGE_TYPES:
            continue
        key = (src, tgt, etype)
        if key in edge_seen:
            continue
        edge_seen.add(key)
        remapped_edges.append({**edge, "source_id": src, "target_id": tgt, "type": etype})

    return {"nodes": remapped_nodes, "edges": remapped_edges}


def agent5_canonicalize(kg: dict, client) -> tuple[dict, dict]:
    nodes = kg.get("nodes", [])
    if not nodes:
        return kg, {}
    prompt = build_agent5_prompt(kg)
    content, usage = client.generate(prompt, models.MODEL_CANONICALIZE)
    rewritten = extract_json(content)
    if not isinstance(rewritten, list):
        return kg, usage
    return apply_canonicalization(kg, rewritten), usage
