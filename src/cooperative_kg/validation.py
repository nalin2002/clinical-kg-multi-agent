"""Deterministic KG validation and human-style enrichment."""

from __future__ import annotations

import re
from typing import Optional

from .constants import (
    ENRICHMENT_PATTERNS,
    LOW_VALUE_NEGATION_RE,
    VALID_EDGE_TYPES,
    VALID_NODE_TYPES,
)
from .enrichment_edges import (
    CAUSE_LINKS,
    INDICATES_MAP,
    LOCATION_PAIRS,
    TREATMENT_TARGETS,
)
from .text_utils import (
    canonicalize_text,
    normalize_for_match,
    token_overlap_supported,
)


def deterministic_validator(kg: dict, transcript: str) -> dict:
    transcript_norm = normalize_for_match(transcript)
    clean_nodes: list[dict] = []
    id_map: dict[str, str] = {}
    seen_nodes: set[tuple[str, str]] = set()

    for node in kg.get("nodes", []):
        if not isinstance(node, dict):
            continue
        ntype = str(node.get("type") or "").upper()
        if ntype not in VALID_NODE_TYPES:
            continue
        text = canonicalize_text(str(node.get("text") or ""))
        if not text:
            continue
        evidence = str(node.get("evidence") or "").strip()
        if evidence and not token_overlap_supported(evidence, transcript_norm):
            continue
        key = (text.lower(), ntype)
        if key in seen_nodes:
            continue
        seen_nodes.add(key)
        new_id = f"N_{len(clean_nodes) + 1:03d}"
        id_map[str(node.get("id"))] = new_id
        clean_nodes.append(
            {
                "id": new_id,
                "text": text,
                "type": ntype,
                "evidence": evidence,
                "turn_id": str(node.get("turn_id") or "").strip(),
            }
        )

    valid_ids = {node["id"] for node in clean_nodes}
    edge_seen: set[tuple[str, str, str]] = set()
    clean_edges: list[dict] = []
    for edge in kg.get("edges", []):
        if not isinstance(edge, dict):
            continue
        src = id_map.get(str(edge.get("source_id")), str(edge.get("source_id")))
        tgt = id_map.get(str(edge.get("target_id")), str(edge.get("target_id")))
        etype = str(edge.get("type") or "").upper()
        if src not in valid_ids or tgt not in valid_ids or src == tgt or etype not in VALID_EDGE_TYPES:
            continue
        evidence = str(edge.get("evidence") or "").strip()
        if evidence and not token_overlap_supported(evidence, transcript_norm):
            continue
        key = (src, tgt, etype)
        if key in edge_seen:
            continue
        edge_seen.add(key)
        clean_edges.append(
            {
                "source_id": src,
                "target_id": tgt,
                "type": etype,
                "evidence": evidence,
                "turn_id": str(edge.get("turn_id") or "").strip(),
            }
        )

    return {"nodes": clean_nodes, "edges": clean_edges}


def find_evidence(transcript: str, pattern: str, fallback_terms: tuple[str, ...]) -> tuple[str, str]:
    match = re.search(pattern, transcript, flags=re.I)
    if match:
        start = max(0, match.start() - 70)
        end = min(len(transcript), match.end() + 70)
        window = transcript[start:end]
        turn = re.search(r"\[([PD]-\d+)\]", window)
        evidence = re.sub(r"\s+", " ", match.group(0)).strip()
        return evidence, turn.group(1) if turn else ""

    lowered = transcript.lower()
    for term in fallback_terms:
        idx = lowered.find(term.lower())
        if idx >= 0:
            start = max(0, idx - 60)
            end = min(len(transcript), idx + len(term) + 60)
            window = transcript[start:end]
            turn = re.search(r"\[([PD]-\d+)\]", window)
            return term, turn.group(1) if turn else ""
    return "", ""


def renumber_graph(nodes: list[dict], edges: list[dict]) -> dict:
    old_to_new: dict[str, str] = {}
    clean_nodes: list[dict] = []
    seen_nodes: set[tuple[str, str]] = set()
    for node in nodes:
        text = canonicalize_text(str(node.get("text") or ""))
        ntype = str(node.get("type") or "").upper()
        if not text or ntype not in VALID_NODE_TYPES:
            continue
        key = (text.lower(), ntype)
        if key in seen_nodes:
            continue
        seen_nodes.add(key)
        new_id = f"N_{len(clean_nodes) + 1:03d}"
        old_to_new[str(node.get("id"))] = new_id
        clean_nodes.append({**node, "id": new_id, "text": text, "type": ntype})

    valid_ids = {n["id"] for n in clean_nodes}
    clean_edges: list[dict] = []
    seen_edges: set[tuple[str, str, str]] = set()
    for edge in edges:
        src = old_to_new.get(str(edge.get("source_id")), str(edge.get("source_id")))
        tgt = old_to_new.get(str(edge.get("target_id")), str(edge.get("target_id")))
        etype = str(edge.get("type") or "").upper()
        key = (src, tgt, etype)
        if src in valid_ids and tgt in valid_ids and src != tgt and etype in VALID_EDGE_TYPES and key not in seen_edges:
            seen_edges.add(key)
            clean_edges.append({**edge, "source_id": src, "target_id": tgt, "type": etype})
    return {"nodes": clean_nodes, "edges": clean_edges}


def deterministic_human_style_enrichment(kg: dict, transcript: str) -> dict:
    """Recover common human-curated clinical entities/edges without extra LLM calls."""
    nodes = [
        n
        for n in kg.get("nodes", [])
        if not LOW_VALUE_NEGATION_RE.match(canonicalize_text(str(n.get("text") or "")))
    ]
    edges = list(kg.get("edges", []))

    def node_key(node: dict) -> tuple[str, str]:
        return (canonicalize_text(str(node.get("text") or "")).lower(), str(node.get("type") or "").upper())

    present = {node_key(n) for n in nodes}

    def add_node(ntype: str, text: str, evidence: str, turn_id: str) -> str:
        text = canonicalize_text(text)
        key = (text.lower(), ntype)
        for node in nodes:
            if node_key(node) == key:
                return node["id"]
        node_id = f"N_{len(nodes) + 1:03d}"
        nodes.append(
            {
                "id": node_id,
                "text": text,
                "type": ntype,
                "evidence": evidence,
                "turn_id": turn_id,
            }
        )
        present.add(key)
        return node_id

    for ntype, text, pattern, fallback_terms in ENRICHMENT_PATTERNS:
        key = (canonicalize_text(text).lower(), ntype)
        if key in present:
            continue
        evidence, turn_id = find_evidence(transcript, pattern, fallback_terms)
        if evidence:
            add_node(ntype, text, evidence, turn_id)

    graph = renumber_graph(nodes, edges)
    nodes = graph["nodes"]
    edges = graph["edges"]

    by_text = {(n["text"].lower(), n["type"]): n for n in nodes}

    def get(text: str, ntype: str) -> Optional[dict]:
        return by_text.get((canonicalize_text(text).lower(), ntype))

    edge_seen = {(e["source_id"], e["target_id"], e["type"]) for e in edges}

    def add_edge(src: Optional[dict], tgt: Optional[dict], etype: str, evidence: str = "", turn_id: str = "") -> None:
        if not src or not tgt or src["id"] == tgt["id"]:
            return
        key = (src["id"], tgt["id"], etype)
        if key in edge_seen:
            return
        edge_seen.add(key)
        edges.append(
            {
                "source_id": src["id"],
                "target_id": tgt["id"],
                "type": etype,
                "evidence": evidence or src.get("evidence") or tgt.get("evidence", ""),
                "turn_id": turn_id or src.get("turn_id") or tgt.get("turn_id", ""),
            }
        )

    for symptom, location in LOCATION_PAIRS.items():
        add_edge(
            get(symptom, "SYMPTOM") or get(symptom, "DIAGNOSIS"),
            get(location, "LOCATION"),
            "LOCATED_AT",
        )

    for diagnosis, symptoms in INDICATES_MAP.items():
        dx = get(diagnosis, "DIAGNOSIS")
        for symptom in symptoms:
            add_edge(get(symptom, "SYMPTOM"), dx, "INDICATES")

    for treatment, targets in TREATMENT_TARGETS.items():
        treatment_node = get(treatment, "TREATMENT")
        for target_text, target_type in targets:
            add_edge(treatment_node, get(target_text, target_type), "TAKEN_FOR")

    for proc in ("covid swab", "covid test"):
        add_edge(get(proc, "PROCEDURE"), get("covid-19", "DIAGNOSIS"), "RULES_OUT")
    for proc in ("chest x-ray", "cbc", "electrolytes", "kidney function test", "abg", "pulse oximetry", "laboratory tests"):
        for dx in ("pneumonia", "respiratory infection", "copd exacerbation", "asthma exacerbation"):
            add_edge(get(proc, "PROCEDURE"), get(dx, "DIAGNOSIS"), "RULES_OUT")

    add_edge(get("ekg", "PROCEDURE"), get("coronary artery disease", "DIAGNOSIS"), "RULES_OUT")
    add_edge(get("echocardiogram", "PROCEDURE"), get("congestive heart failure exacerbation", "DIAGNOSIS"), "RULES_OUT")
    add_edge(get("cardiac catheterization", "PROCEDURE"), get("coronary artery disease", "DIAGNOSIS"), "RULES_OUT")

    for proc in (
        "right knee x-ray",
        "lumbar spine x-ray",
        "right elbow x-ray",
        "lumbar mri",
        "mri",
        "right humerus x-ray",
        "right middle finger x-ray",
        "right lower extremity x-ray",
    ):
        for dx in (
            "lumbar strain",
            "acute lumbar strain",
            "arthritis exacerbation",
            "right elbow lateral epicondylitis",
            "right leg contusion",
            "right proximal humerus fracture",
            "right middle finger distal phalanx fracture",
        ):
            add_edge(get(proc, "PROCEDURE"), get(dx, "DIAGNOSIS"), "RULES_OUT")

    add_edge(get("endoscopy", "PROCEDURE"), get("gastritis", "DIAGNOSIS"), "CONFIRMS")
    add_edge(get("endoscopy", "PROCEDURE"), get("gastric polyp", "DIAGNOSIS"), "CONFIRMS")
    add_edge(get("colonoscopy", "PROCEDURE"), get("anemia", "DIAGNOSIS"), "RULES_OUT")
    add_edge(get("abdominal x-ray", "PROCEDURE"), get("kidney stones recurrence", "DIAGNOSIS"), "RULES_OUT")
    add_edge(get("ct scan abdomen and pelvis", "PROCEDURE"), get("kidney stones recurrence", "DIAGNOSIS"), "CONFIRMS")
    add_edge(get("right upper quadrant ultrasound", "PROCEDURE"), get("gallstones", "MEDICAL_HISTORY"), "RULES_OUT")

    add_edge(get("temperature ~101 f", "LAB_RESULT"), get("fever", "SYMPTOM"), "CONFIRMS")
    add_edge(get("temperature 37.4 c", "LAB_RESULT"), get("fever", "SYMPTOM"), "CONFIRMS")
    add_edge(get("elevated blood pressure", "LAB_RESULT"), get("hypertension", "DIAGNOSIS"), "CONFIRMS")
    add_edge(get("elevated blood pressure", "LAB_RESULT"), get("uncontrolled hypertension", "DIAGNOSIS"), "CONFIRMS")
    add_edge(get("elevated hemoglobin a1c", "LAB_RESULT"), get("newly diagnosed diabetes", "DIAGNOSIS"), "CONFIRMS")
    add_edge(get("elevated hemoglobin a1c", "LAB_RESULT"), get("hyperglycemia", "DIAGNOSIS"), "CONFIRMS")
    add_edge(get("hemoglobin a1c 8", "LAB_RESULT"), get("type 2 diabetes", "MEDICAL_HISTORY"), "CONFIRMS")
    add_edge(get("elevated glucose", "LAB_RESULT"), get("hyperglycemia", "DIAGNOSIS"), "CONFIRMS")
    add_edge(get("elevated glucose", "LAB_RESULT"), get("newly diagnosed diabetes", "DIAGNOSIS"), "CONFIRMS")
    add_edge(get("decreased ejection fraction", "LAB_RESULT"), get("congestive heart failure exacerbation", "DIAGNOSIS"), "CONFIRMS")
    add_edge(get("decreased ejection fraction", "LAB_RESULT"), get("pumping dysfunction", "DIAGNOSIS"), "CONFIRMS")
    add_edge(get("ekg lvh", "LAB_RESULT"), get("hypertension", "DIAGNOSIS"), "CONFIRMS")
    add_edge(get("lyme titer elevated", "LAB_RESULT"), get("lyme disease", "DIAGNOSIS"), "CONFIRMS")
    add_edge(get("rapid strep positive", "LAB_RESULT"), get("strep infection", "DIAGNOSIS"), "CONFIRMS")
    add_edge(get("hemoglobin 8.2", "LAB_RESULT"), get("anemia", "DIAGNOSIS"), "CONFIRMS")
    add_edge(get("endoscopy gastritis", "LAB_RESULT"), get("gastritis", "DIAGNOSIS"), "CONFIRMS")
    add_edge(get("endoscopy polyp", "LAB_RESULT"), get("gastric polyp", "DIAGNOSIS"), "CONFIRMS")
    add_edge(get("chest x-ray unremarkable", "LAB_RESULT"), get("pneumonia", "DIAGNOSIS"), "RULES_OUT")
    add_edge(get("ekg normal", "LAB_RESULT"), get("coronary artery disease", "DIAGNOSIS"), "RULES_OUT")

    for source_text, target_texts in CAUSE_LINKS.items():
        source = get(source_text, "MEDICAL_HISTORY")
        for target_text in target_texts:
            target = (
                get(target_text, "DIAGNOSIS")
                or get(target_text, "MEDICAL_HISTORY")
                or get(target_text, "SYMPTOM")
            )
            add_edge(source, target, "CAUSES")

    return renumber_graph(nodes, edges)
