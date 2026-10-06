"""
KARMA-style Multi-Agent KG Extraction Pipeline
==============================================
Four-agent architecture for clinical knowledge graph extraction:

  Agent 1 - Extractor:            broad base extraction (LLM call)
  Agent 2 - Completeness Enhancer: targeted second pass for commonly-missed
                                   node types (LOCATION, LAB_RESULT,
                                   MEDICAL_HISTORY)  (LLM call)
  Agent 3 - Schema Agent:         coerces node/edge types into the allowed
                                   enum, fixes mis-labels (rule-based)
  Agent 4 - Entity + Conflict Resolver: within-patient dedup, drop invalid /
                                   duplicate edges (rule-based)

The output per-patient JSON schema matches what ``dump_graph.py`` expects, so
this module is a drop-in replacement for ``kg_extraction.py``.

Usage:
    python -m Clinical_KG_OS_LLM.karma_kg_extraction --output ./my_kg_karma
    python -m Clinical_KG_OS_LLM.karma_kg_extraction --output ./my_kg_karma \
        --res-ids RES0198 RES0199 RES0200
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

from Clinical_KG_OS_LLM.paths import transcripts_dir


# === Configuration ===
TRANSCRIPT_DIR = transcripts_dir()
MAX_RETRIES = 3
# OPENROUTER_MODEL = "openai/gpt-3.5-turbo"
#OPENROUTER_MODEL = "openai/gpt-4o-mini"
OPENROUTER_MODEL = "z-ai/glm-4.7-flash"
OUTPUT_SUFFIX = "karma"

# Allowed taxonomy (must match evaluator / human-curated graph).
ALLOWED_NODE_TYPES = {
    "SYMPTOM",
    "DIAGNOSIS",
    "TREATMENT",
    "PROCEDURE",
    "LOCATION",
    "MEDICAL_HISTORY",
    "LAB_RESULT",
}
ALLOWED_EDGE_TYPES = {
    "CAUSES",
    "INDICATES",
    "LOCATED_AT",
    "RULES_OUT",
    "TAKEN_FOR",
    "CONFIRMS",
}

# Common LLM hallucinations -> canonical.  Applied by the Schema Agent.
NODE_TYPE_ALIASES = {
    "MEDICATION": "TREATMENT",
    "DRUG": "TREATMENT",
    "MEDICINE": "TREATMENT",
    "THERAPY": "TREATMENT",
    "INTERVENTION": "TREATMENT",
    "CONDITION": "DIAGNOSIS",
    "DISEASE": "DIAGNOSIS",
    "DISORDER": "DIAGNOSIS",
    "FINDING": "SYMPTOM",
    "SIGN": "SYMPTOM",
    "COMPLAINT": "SYMPTOM",
    "TEST": "PROCEDURE",
    "EXAM": "PROCEDURE",
    "IMAGING": "PROCEDURE",
    "VITAL": "LAB_RESULT",
    "VITALS": "LAB_RESULT",
    "LAB": "LAB_RESULT",
    "HISTORY": "MEDICAL_HISTORY",
    "RISK_FACTOR": "MEDICAL_HISTORY",
    "SOCIAL_HISTORY": "MEDICAL_HISTORY",
    "FAMILY_HISTORY": "MEDICAL_HISTORY",
    "BODY_PART": "LOCATION",
    "ANATOMY": "LOCATION",
    "ANATOMICAL_LOCATION": "LOCATION",
}

EDGE_TYPE_ALIASES = {
    "CAUSED_BY": "CAUSES",
    "LEADS_TO": "CAUSES",
    "SUGGESTS": "INDICATES",
    "SYMPTOM_OF": "INDICATES",
    "PRESENTS_WITH": "INDICATES",
    "AT": "LOCATED_AT",
    "LOCATION_OF": "LOCATED_AT",
    "NEGATES": "RULES_OUT",
    "NEGATIVE_FOR": "RULES_OUT",
    "USED_FOR": "TAKEN_FOR",
    "TREATS": "TAKEN_FOR",
    "PRESCRIBED_FOR": "TAKEN_FOR",
    "DIAGNOSES": "CONFIRMS",
    "CONFIRMED_BY": "CONFIRMS",
}


# =========================================================================
# Prompts
# =========================================================================
EXTRACTION_PROMPT = """You are a clinical knowledge-graph extractor. Extract a
comprehensive knowledge graph from the clinical conversation below.

## ALLOWED NODE TYPES (use exactly these labels, uppercase)
- SYMPTOM: patient-reported or observed symptoms (chest pain, shortness of breath,
  nasal congestion, cough, fatigue, nausea)
- DIAGNOSIS: active or suspected conditions the clinician is considering
  (COPD exacerbation, pneumonia, viral URI)
- TREATMENT: medications, therapies, devices, behavioural interventions
  (Aspirin, Metformin, DASH diet, inhaler, physical therapy)
- PROCEDURE: tests, exams, imaging, surgeries (ECG, CT, stress test, biopsy,
  physical exam, nasal swab)
- LOCATION: body parts / anatomical locations referenced (chest, left arm,
  heart, throat, nose, right knee, lower back)
- MEDICAL_HISTORY: pre-existing conditions, risk factors, social history,
  family history (diabetes, smoking, hypertension, mother had breast cancer,
  prior MI)
- LAB_RESULT: lab values, vital signs, measurements (A1C 7.2%, BP 148/90,
  BNP elevated, temperature 101F, SpO2 92%)

## ALLOWED EDGE TYPES (use exactly these labels, uppercase)
- CAUSES:      risk factor causes a condition            (smoking CAUSES COPD)
- INDICATES:   symptom/finding points to a diagnosis     (chest pain INDICATES angina)
- LOCATED_AT:  symptom lives at a body location          (pain LOCATED_AT chest)
- RULES_OUT:   a test or finding negates a condition     (ECG RULES_OUT MI)
- TAKEN_FOR:   a treatment is prescribed for a condition (Aspirin TAKEN_FOR angina)
- CONFIRMS:    a lab/test confirms a diagnosis           (elevated BNP CONFIRMS heart failure)

## EXTRACTION GUIDELINES
1. Be **thorough**: extract every distinct symptom, history item, treatment,
   procedure, body location and lab/vital value mentioned. Aim for high recall.
2. Extract **LOCATION** nodes (body parts) whenever a symptom references a body
   region - even casually ("pain in my chest" -> LOCATION: chest).
3. Extract **LAB_RESULT** nodes for any numeric measurement or vital (BP,
   temperature, oxygen saturation, weight) as well as qualitative findings like
   "elevated", "normal", "positive".
4. Extract **MEDICAL_HISTORY** for any past condition, risk factor, family
   history, smoking/alcohol use, occupational exposure.
5. Extract one node per distinct concept. If the same concept is discussed in
   multiple turns, create one node and use the earliest turn_id.
6. Add as many edges as possible but only when the conversation actually
   supports the relation.
7. **Ground every node and edge in the transcript** - the ``evidence`` field
   must be a short verbatim quote from the transcript.

## FORMAT REQUIREMENTS
- Node id: "N_001", "N_002", ... (zero-padded, ascending).
- turn_id: string "P-<n>" for patient turns, "D-<n>" for doctor turns,
  matching the [P-<n>] / [D-<n>] tags in the transcript.
- ``evidence``: short (<= 20 words) verbatim quote from the transcript.
- Output ONLY valid JSON, no prose, no markdown fences.

## TRANSCRIPT
{transcript}

## OUTPUT (JSON)
{{
  "nodes": [
    {{"id": "N_001", "text": "...", "type": "SYMPTOM",
      "evidence": "...", "turn_id": "P-1"}}
  ],
  "edges": [
    {{"source_id": "N_001", "target_id": "N_002", "type": "LOCATED_AT",
      "evidence": "...", "turn_id": "P-1"}}
  ]
}}
"""


ENHANCER_PROMPT = """You are a clinical KG **completeness reviewer**. A first
pass extracted the knowledge graph below from this transcript. Your job is to
find **additional** entities and relations the first pass missed - focus on:

1. **LOCATION** nodes: every body part referenced ANYWHERE in the conversation
   (chest, nose, throat, back, knee, head, abdomen, arm, leg, heart, lungs,
   etc.) - even in small mentions like "my back hurts".
2. **LAB_RESULT** nodes: every numeric value, vital sign, or qualitative lab
   finding (BP, HR, temp, SpO2, A1C, BNP, "elevated", "normal", "positive").
3. **MEDICAL_HISTORY**: risk factors and past conditions. Include smoking,
   alcohol use, family history, occupational exposures, prior surgeries,
   allergies, prior episodes of the current complaint.
4. **Missed SYMPTOMs**: anything the patient reports but that wasn't captured
   - including denied / negative symptoms should NOT be extracted as symptoms
   (they're just "not present").
5. **Edges**: add LOCATED_AT, INDICATES, TAKEN_FOR, CAUSES, CONFIRMS, RULES_OUT
   edges that were missed, linking both existing and newly-added nodes.

## RULES
- ONLY add **new** items. Do NOT duplicate nodes already in the pass-1 KG.
- Use node ids starting at "N_501" to avoid collisions with pass 1.
- When a new edge references a pass-1 node, use that node's existing id
  ("N_001", "N_002", ...). When both endpoints are new, use the new ids.
- Use ONLY these node types: SYMPTOM, DIAGNOSIS, TREATMENT, PROCEDURE,
  LOCATION, MEDICAL_HISTORY, LAB_RESULT.
- Use ONLY these edge types: CAUSES, INDICATES, LOCATED_AT, RULES_OUT,
  TAKEN_FOR, CONFIRMS.
- Every node / edge needs ``evidence`` (<=20-word verbatim quote) and ``turn_id``
  ("P-<n>" or "D-<n>").
- Output ONLY valid JSON with the same schema as pass 1. If nothing was
  missed, return {{"nodes": [], "edges": []}}.

## TRANSCRIPT
{transcript}

## PASS-1 KNOWLEDGE GRAPH
{pass1_kg}

## ADDITIONAL NODES AND EDGES (JSON only)
"""

from openai import OpenAI

# =========================================================================
# Model client
# =========================================================================
class OpenRouterClient:
    """Thin wrapper around the OpenRouter chat.completions API."""

    def __init__(self, api_key: str, model: str = OPENROUTER_MODEL):
        self.client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key
)
  #      self.client = OpenAI(
  #          api_key=api_key
#)
      #  self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
        self.model = model

    def generate(self, prompt: str) -> tuple[str, Optional[dict]]:
        """Generate text. Returns (content, usage_dict or None)."""
        for attempt in range(MAX_RETRIES):
            try:
                stream = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    stream=True,
                )
                content = ""
                last_chunk = None
                for chunk in stream:
                    last_chunk = chunk
                    delta = chunk.choices[0].delta
                    if delta.content:
                        content += delta.content

                usage = None
                if last_chunk and getattr(last_chunk, "usage", None):
                    u = last_chunk.usage
                    usage = {
                        "prompt_tokens": u.prompt_tokens,
                        "completion_tokens": u.completion_tokens,
                    }

                if content:
                    return content, usage
            except Exception as exc:  # pragma: no cover - network failure path
                print(f"(error: {exc}, retry {attempt + 1})", end=" ", flush=True)
                time.sleep(2**attempt)

        return "", None


def get_client(api_keys: dict, model: str = OPENROUTER_MODEL) -> OpenRouterClient:
     key = api_keys.get("openrouter")
   # key = api_keys.get("openrouter")
     if not key:
          raise SystemExit('api_keys.json must contain a non-empty "openrouter" key')
     return OpenRouterClient(key, model)


# =========================================================================
# JSON helpers
# =========================================================================
def extract_json_from_response(response_text: str) -> Optional[dict]:
    """Best-effort JSON extraction from an LLM response."""
    if not response_text:
        return None

    # 1. direct parse
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass

    # 2. stripped markdown fence
    stripped = response_text.strip()
    if stripped.startswith("```"):
        parts = stripped.split("```")
        if len(parts) >= 2:
            inner = parts[1]
            if inner.startswith("json"):
                inner = inner[4:]
            inner = inner.strip()
            try:
                return json.loads(inner)
            except json.JSONDecodeError:
                pass

    # 3. first top-level object
    match = re.search(r"\{[\s\S]*\}", response_text)
    if match:
        candidate = match.group(0)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            # strip trailing commas
            fixed = re.sub(r",(\s*[}\]])", r"\1", candidate)
            try:
                return json.loads(fixed)
            except json.JSONDecodeError:
                pass

    return None


def read_transcript(path: Path) -> str:
    return path.read_text()


def get_transcript_files() -> list[Path]:
    files: list[Path] = []
    for res_dir in sorted(TRANSCRIPT_DIR.glob("RES*")):
        if res_dir.is_dir():
            txt = res_dir / f"{res_dir.name}.txt"
            if txt.exists():
                files.append(txt)
    return files


# =========================================================================
# Agent 1 - Extractor
# =========================================================================
def agent_extract(transcript: str, client: OpenRouterClient) -> tuple[Optional[dict], Optional[dict]]:
    prompt = EXTRACTION_PROMPT.format(transcript=transcript)
    content, usage = client.generate(prompt)
    if not content:
        return None, usage
    kg = extract_json_from_response(content)
    if kg and isinstance(kg, dict):
        kg.setdefault("nodes", [])
        kg.setdefault("edges", [])
    return kg, usage


# =========================================================================
# Agent 2 - Completeness Enhancer
# =========================================================================
def agent_enhance(
    transcript: str, pass1_kg: dict, client: OpenRouterClient
) -> tuple[Optional[dict], Optional[dict]]:
    # Keep pass-1 payload small: just id / text / type, no evidence.
    compact = {
        "nodes": [
            {"id": n.get("id"), "text": n.get("text"), "type": n.get("type")}
            for n in pass1_kg.get("nodes", [])
        ],
        "edges": [
            {
                "source_id": e.get("source_id"),
                "target_id": e.get("target_id"),
                "type": e.get("type"),
            }
            for e in pass1_kg.get("edges", [])
        ],
    }
    prompt = ENHANCER_PROMPT.format(
        transcript=transcript,
        pass1_kg=json.dumps(compact, ensure_ascii=False),
    )
    content, usage = client.generate(prompt)
    if not content:
        return None, usage
    add = extract_json_from_response(content)
    if add and isinstance(add, dict):
        add.setdefault("nodes", [])
        add.setdefault("edges", [])
    return add, usage


def merge_kgs(base: dict, addition: Optional[dict]) -> dict:
    """Merge enhancer output into pass-1 KG. Dedupe by (text_lower, type)."""
    if not addition:
        return base

    existing_keys = {
        (str(n.get("text", "")).strip().lower(), str(n.get("type", "")).upper())
        for n in base.get("nodes", [])
    }
    existing_ids = {n.get("id") for n in base.get("nodes", [])}

    for node in addition.get("nodes", []):
        text = str(node.get("text", "")).strip()
        ntype = str(node.get("type", "")).upper()
        key = (text.lower(), ntype)
        if not text:
            continue
        if key in existing_keys:
            continue
        existing_keys.add(key)
        # Give the new node a non-colliding id.
        new_id = node.get("id") or ""
        if new_id in existing_ids or not new_id:
            new_id = f"N_{len(base['nodes']) + 1:03d}"
        existing_ids.add(new_id)
        base["nodes"].append(
            {
                "id": new_id,
                "text": text,
                "type": ntype,
                "evidence": node.get("evidence", ""),
                "turn_id": node.get("turn_id", ""),
            }
        )

    # Add edges. They'll be validated by the schema / conflict agents below.
    existing_edge_keys = {
        (e.get("source_id"), e.get("target_id"), str(e.get("type", "")).upper())
        for e in base.get("edges", [])
    }
    for edge in addition.get("edges", []):
        key = (edge.get("source_id"), edge.get("target_id"), str(edge.get("type", "")).upper())
        if key in existing_edge_keys:
            continue
        existing_edge_keys.add(key)
        base["edges"].append(
            {
                "source_id": edge.get("source_id"),
                "target_id": edge.get("target_id"),
                "type": str(edge.get("type", "")).upper(),
                "evidence": edge.get("evidence", ""),
                "turn_id": edge.get("turn_id", ""),
            }
        )

    return base


# =========================================================================
# Agent 3 - Schema Agent (rule-based)
# =========================================================================
def schema_agent(kg: dict) -> tuple[dict, dict]:
    """Coerce node / edge types into the allowed enum; drop un-salvageable items.

    Returns (kg, report) where report tracks how many items were remapped/dropped.
    """
    report = {
        "node_type_remapped": 0,
        "node_dropped_unknown_type": 0,
        "edge_type_remapped": 0,
        "edge_dropped_unknown_type": 0,
        "turn_id_fixed": 0,
    }

    good_nodes = []
    for node in kg.get("nodes", []):
        raw_type = str(node.get("type", "")).strip().upper().replace(" ", "_")
        if raw_type in ALLOWED_NODE_TYPES:
            new_type = raw_type
        elif raw_type in NODE_TYPE_ALIASES:
            new_type = NODE_TYPE_ALIASES[raw_type]
            report["node_type_remapped"] += 1
        else:
            # Try last-resort substring match
            mapped = None
            for alias, canon in NODE_TYPE_ALIASES.items():
                if alias in raw_type:
                    mapped = canon
                    break
            if mapped is None:
                for canon in ALLOWED_NODE_TYPES:
                    if canon in raw_type:
                        mapped = canon
                        break
            if mapped is None:
                report["node_dropped_unknown_type"] += 1
                continue
            new_type = mapped
            report["node_type_remapped"] += 1

        # Normalize turn_id: extract the first P-<n> or D-<n> we can find.
        turn_id = str(node.get("turn_id", ""))
        m = re.search(r"([PD])\s*-\s*(\d+)", turn_id)
        if m:
            fixed_turn = f"{m.group(1)}-{m.group(2)}"
            if fixed_turn != turn_id:
                report["turn_id_fixed"] += 1
            turn_id = fixed_turn
        elif turn_id:
            # No parseable turn; keep as-is but count the issue.
            pass

        good_nodes.append(
            {
                "id": node.get("id"),
                "text": str(node.get("text", "")).strip(),
                "type": new_type,
                "evidence": node.get("evidence", ""),
                "turn_id": turn_id,
            }
        )

    good_node_ids = {n["id"] for n in good_nodes}

    good_edges = []
    for edge in kg.get("edges", []):
        raw_type = str(edge.get("type", "")).strip().upper().replace(" ", "_")
        if raw_type in ALLOWED_EDGE_TYPES:
            new_type = raw_type
        elif raw_type in EDGE_TYPE_ALIASES:
            new_type = EDGE_TYPE_ALIASES[raw_type]
            report["edge_type_remapped"] += 1
        else:
            mapped = None
            for alias, canon in EDGE_TYPE_ALIASES.items():
                if alias in raw_type:
                    mapped = canon
                    break
            if mapped is None:
                for canon in ALLOWED_EDGE_TYPES:
                    if canon in raw_type:
                        mapped = canon
                        break
            if mapped is None:
                report["edge_dropped_unknown_type"] += 1
                continue
            new_type = mapped
            report["edge_type_remapped"] += 1

        src = edge.get("source_id")
        tgt = edge.get("target_id")
        if src not in good_node_ids or tgt not in good_node_ids:
            continue
        if src == tgt:  # no self-loops
            continue

        turn_id = str(edge.get("turn_id", ""))
        m = re.search(r"([PD])\s*-\s*(\d+)", turn_id)
        if m:
            turn_id = f"{m.group(1)}-{m.group(2)}"

        good_edges.append(
            {
                "source_id": src,
                "target_id": tgt,
                "type": new_type,
                "evidence": edge.get("evidence", ""),
                "turn_id": turn_id,
            }
        )

    return {"nodes": good_nodes, "edges": good_edges}, report


# =========================================================================
# Agent 4 - Entity + Conflict Resolver (rule-based)
# =========================================================================
def entity_conflict_agent(kg: dict) -> tuple[dict, dict]:
    """Within-patient entity dedup + drop duplicate / dangling edges.

    Cross-patient entity resolution is handled later by ``dump_graph.py``; here
    we just dedupe within the same transcript so the per-patient KG is clean.
    """
    report = {
        "duplicate_nodes_merged": 0,
        "duplicate_edges_dropped": 0,
        "dangling_edges_dropped": 0,
    }

    # Dedup nodes by (text_lower, type).  Retain the first id; remap later.
    canonical: dict[tuple[str, str], str] = {}
    merged_nodes: list[dict] = []
    id_remap: dict[str, str] = {}

    for node in kg.get("nodes", []):
        text = str(node.get("text", "")).strip()
        ntype = str(node.get("type", "")).upper()
        if not text or not ntype:
            continue
        key = (text.lower(), ntype)
        if key in canonical:
            id_remap[node["id"]] = canonical[key]
            report["duplicate_nodes_merged"] += 1
            continue
        canonical[key] = node["id"]
        id_remap[node["id"]] = node["id"]
        merged_nodes.append(node)

    node_ids = {n["id"] for n in merged_nodes}

    # Remap + dedupe edges.
    seen_edges: set[tuple[str, str, str]] = set()
    clean_edges: list[dict] = []
    for edge in kg.get("edges", []):
        src = id_remap.get(edge.get("source_id"), edge.get("source_id"))
        tgt = id_remap.get(edge.get("target_id"), edge.get("target_id"))
        etype = str(edge.get("type", "")).upper()

        if src not in node_ids or tgt not in node_ids:
            report["dangling_edges_dropped"] += 1
            continue
        if src == tgt:
            report["dangling_edges_dropped"] += 1
            continue

        key = (src, tgt, etype)
        if key in seen_edges:
            report["duplicate_edges_dropped"] += 1
            continue
        seen_edges.add(key)

        clean_edges.append(
            {
                "source_id": src,
                "target_id": tgt,
                "type": etype,
                "evidence": edge.get("evidence", ""),
                "turn_id": edge.get("turn_id", ""),
            }
        )

    return {"nodes": merged_nodes, "edges": clean_edges}, report


# =========================================================================
# Orchestrator
# =========================================================================
def process_one(
    txt_path: Path,
    client: OpenRouterClient,
    output_dir: Path,
    suffix: str,
    skip_enhancer: bool,
) -> tuple[str, str, int, int, dict]:
    res_id = txt_path.stem
    output_file = output_dir / f"{res_id}_{suffix}.json"

    if output_file.exists():
        return res_id, "SKIP", 0, 0, {}

    transcript = read_transcript(txt_path)
    print(f"  {res_id}...", end=" ", flush=True)

    totals = {"prompt": 0, "completion": 0}

    # --- Agent 1: Extract ---
    pass1_kg, u1 = agent_extract(transcript, client)
    if u1:
        totals["prompt"] += u1.get("prompt_tokens", 0)
        totals["completion"] += u1.get("completion_tokens", 0)
    if not pass1_kg:
        print("FAILED (agent 1)")
        return res_id, "FAILED", 0, 0, totals

    pass1_stats = {
        "nodes": len(pass1_kg.get("nodes", [])),
        "edges": len(pass1_kg.get("edges", [])),
    }

    # --- Agent 2: Enhance ---
    if skip_enhancer:
        pass2_stats = {"nodes_added": 0, "edges_added": 0}
    else:
        addition, u2 = agent_enhance(transcript, pass1_kg, client)
        if u2:
            totals["prompt"] += u2.get("prompt_tokens", 0)
            totals["completion"] += u2.get("completion_tokens", 0)
        pre_n, pre_e = len(pass1_kg["nodes"]), len(pass1_kg["edges"])
        pass1_kg = merge_kgs(pass1_kg, addition)
        pass2_stats = {
            "nodes_added": len(pass1_kg["nodes"]) - pre_n,
            "edges_added": len(pass1_kg["edges"]) - pre_e,
        }

    # --- Agent 3: Schema coercion ---
    pass1_kg, schema_report = schema_agent(pass1_kg)

    # --- Agent 4: Entity + conflict resolution ---
    final_kg, dedupe_report = entity_conflict_agent(pass1_kg)

    # Attach meta.
    final_kg["_meta"] = {
        "pipeline": "karma",
        "model": client.model,
        "pass1": pass1_stats,
        "pass2": pass2_stats,
        "schema": schema_report,
        "dedupe": dedupe_report,
    }
    final_kg["_usage"] = totals

    n, e = len(final_kg["nodes"]), len(final_kg["edges"])
    output_file.write_text(json.dumps(final_kg, indent=2, ensure_ascii=False))
    print(f"pass1=({pass1_stats['nodes']}n/{pass1_stats['edges']}e) "
          f"final=({n}n/{e}e)")

    return res_id, "OK", n, e, totals


def main() -> None:
    parser = argparse.ArgumentParser(
        description="KARMA-style multi-agent KG extraction pipeline"
    )
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for sub-KG JSON files")
    parser.add_argument("--res-ids", nargs="+", default=None,
                        help="Optional list of patient IDs to process. "
                             "Processes all if omitted.")
    parser.add_argument("--model", type=str, default=OPENROUTER_MODEL,
                        help=f"OpenRouter model to use (default: {OPENROUTER_MODEL}). "
                             "Must be one of the hackathon-allowed models.")
    parser.add_argument("--no-enhancer", action="store_true",
                        help="Skip the completeness-enhancer agent "
                             "(1 LLM call per transcript instead of 2).")
    parser.add_argument("--api-keys", type=str, default="api_keys.json",
                        help="Path to api_keys.json (default: ./api_keys.json)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.api_keys) as f:
        api_keys = json.load(f)
    client = get_client(api_keys, args.model)

    transcript_files = get_transcript_files()
    if args.res_ids:
        transcript_files = [f for f in transcript_files if f.parent.name in args.res_ids]

    print("KARMA-style KG Extraction Pipeline")
    print(f"Model:      {args.model}")
    print(f"Output dir: {output_dir}")
    print(f"Enhancer:   {'OFF' if args.no_enhancer else 'ON'}")
    print(f"Transcripts: {len(transcript_files)}")
    print("=" * 60)

    success = failed = 0
    total_tokens = {"prompt": 0, "completion": 0}
    details = []

    for txt_path in transcript_files:
        res_id, status, nodes, edges, usage = process_one(
            txt_path, client, output_dir, OUTPUT_SUFFIX, args.no_enhancer
        )
        if status == "OK":
            success += 1
            total_tokens["prompt"] += usage.get("prompt", 0)
            total_tokens["completion"] += usage.get("completion", 0)
            details.append({"res_id": res_id, "nodes": nodes, "edges": edges, **usage})
        elif status == "SKIP":
            print(f"  {res_id}: SKIP (exists)")
            success += 1
        else:
            failed += 1
        time.sleep(0.3)

    stats_path = output_dir / "_stats.json"
    with open(stats_path, "w") as f:
        json.dump(
            {
                "method": "karma",
                "model": args.model,
                "enhancer_enabled": not args.no_enhancer,
                "total_tokens": total_tokens,
                "success": success,
                "failed": failed,
                "details": details,
            },
            f,
            indent=2,
        )

    print("=" * 60)
    print(f"Done.  Success: {success}  Failed: {failed}")
    print(f"Total tokens: {total_tokens['prompt'] + total_tokens['completion']} "
          f"(prompt={total_tokens['prompt']}, completion={total_tokens['completion']})")
    print(f"Sub-KGs written to: {output_dir}/")


if __name__ == "__main__":
    main()
