"""
Unified KG Extraction Pipeline (SC v4 — clean)
==============================================
Same architecture as the original SC v4 (Extractor × K self-consistency
+ Enhancer ADD-only + rule-based validation), but with hardcoded lookup
tables minimized.

What was removed and why:
  - NON_BODY_LOCATIONS set: dropped. The prompt now defines LOCATION strictly.
  - TYPE_REMAP dict: reduced from 16 entries to 0; invalid types are dropped.
    The prompt enumerates the 7 valid types and forbids variants.
  - TEXT_NORMALIZATIONS regex list: dropped. The prompt's NAMING RULES
    handle canonical text (Arabic numerals, lowercase, hyphenation).
  - Per-entity examples that read like baseline-specific terms: replaced
    with the organizer-provided base prompt's own examples.

What was kept (still rule-based, still principle-based):
  - VALID_NODE_TYPES / VALID_EDGE_TYPES: the floor schema from the README.
  - EDGE_RULES: derived from each edge type's definition in the base prompt.
  - normalize_text Principles 1-2: structural patterns, not word lookups.
  - Self-consistency voting + Enhancer ADD-only + edge direction swap +
    dedup: pure algorithms, no data dependencies.

Usage:
    python kg_extraction.py --output baseline_naive/sub_kgs

After extraction, merge with:
    python dump_graph.py --input baseline_naive/sub_kgs --output baseline_naive/
"""

import json
import re
import argparse
import time
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from Clinical_KG_OS_LLM.paths import transcripts_dir

import os

# === Configuration ===
TRANSCRIPT_DIR = transcripts_dir()
MAX_RETRIES = 3

# Per-agent model configuration. Override via env vars for quick experiments.
EXTRACTOR_MODEL = os.getenv("EXTRACTOR_MODEL", "z-ai/glm-4.7-flash")
ENHANCER_MODEL = os.getenv("ENHANCER_MODEL", "z-ai/glm-4.7-flash")

OPENROUTER_MODEL = EXTRACTOR_MODEL  # legacy alias

# Self-consistency
SELF_CONSISTENCY_K = int(os.getenv("SELF_CONSISTENCY_K", "3"))
VOTE_THRESHOLD = int(os.getenv("VOTE_THRESHOLD", "2"))
EXTRACTION_TEMPERATURE = float(os.getenv("EXTRACTION_TEMPERATURE", "0.7"))

OUTPUT_SUFFIX = "clean_v3"
MAX_WORKERS = 5

PIPELINE_MODE = "karma"

# Edge enhancement: optional second-pass focused on missing edges only.
# Set to "0" via env var to disable. Adds one extra LLM call per patient.
EDGE_ENHANCER_ENABLED = os.getenv("EDGE_ENHANCER_ENABLED", "1") == "1"

# ============================================================
# Schema (the floor required by the organizer)
# ============================================================
VALID_NODE_TYPES = {
    "SYMPTOM", "DIAGNOSIS", "TREATMENT", "PROCEDURE",
    "LOCATION", "MEDICAL_HISTORY", "LAB_RESULT",
}

VALID_EDGE_TYPES = {
    "LOCATED_AT", "INDICATES", "TAKEN_FOR",
    "CAUSES", "RULES_OUT", "CONFIRMS",
}

# Edge type -> (allowed source types, allowed target types).
# Each entry is read directly from the edge definition in the base prompt:
#   "LOCATED_AT: Symptom at body location (pain LOCATED_AT chest)"  -> SYMPTOM -> LOCATION
#   "INDICATES: Symptom indicates diagnosis"                         -> SYMPTOM -> DIAGNOSIS
#   "TAKEN_FOR: Treatment for condition (Aspirin TAKEN_FOR angina)"  -> TREATMENT -> DIAGNOSIS|SYMPTOM
#   "CAUSES: Risk factor causes condition"                           -> MEDICAL_HISTORY -> DIAGNOSIS
#   "RULES_OUT: Test rules out condition"                            -> PROCEDURE -> DIAGNOSIS
#   "CONFIRMS: Lab/test confirms diagnosis"                          -> LAB_RESULT -> DIAGNOSIS
EDGE_RULES = {
    "LOCATED_AT": ({"SYMPTOM"}, {"LOCATION"}),
    "INDICATES":  ({"SYMPTOM"}, {"DIAGNOSIS"}),
    "TAKEN_FOR":  ({"TREATMENT"}, {"SYMPTOM", "DIAGNOSIS"}),
    "CAUSES":     ({"MEDICAL_HISTORY"}, {"DIAGNOSIS"}),
    "RULES_OUT":  ({"PROCEDURE"}, {"DIAGNOSIS"}),
    "CONFIRMS":   ({"LAB_RESULT"}, {"DIAGNOSIS"}),
}


# ============================================================
# Prompts
# ============================================================
EXTRACTION_PROMPT = """Extract a clinical knowledge graph from the transcript below.

Return ONLY valid JSON:
{{
  "nodes": [{{"id": "N_001", "text": "...", "type": "...", "evidence": "...", "turn_id": "..."}}],
  "edges": [{{"source_id": "N_001", "target_id": "N_002", "type": "...", "evidence": "...", "turn_id": "..."}}]
}}

NODE TYPES — use exactly these 7 labels. Do not invent variants.

  SYMPTOM         What the patient feels or reports, including character
                  modifiers as separate nodes. Capture functional symptoms
                  and temporal qualifiers when the patient describes them.
                  Use medical terminology when the patient describes a
                  symptom informally ("hard to breathe when climbing stairs"
                  -> "dyspnea on exertion").
                  Examples: chest pain, shortness of breath, fatigue,
                            dry cough, nasal congestion, headache,
                            absent fever, absent chest pain, absent nausea.

  DIAGNOSIS       Conditions the doctor considers, suspects, confirms, or
                  rules out. Includes both confirmed and working diagnoses.
                  Examples: COPD exacerbation, pneumonia, viral infection,
                            common cold, hypertension, type 2 diabetes.

  TREATMENT       Any therapeutic measure: medications, supportive care,
                  lifestyle interventions. Each medication name is a separate
                  TREATMENT (never MEDICAL_HISTORY), even when listed as
                  current medications. Split compound mentions ("tylenol or
                  nsaids" -> two TREATMENT nodes).
                  Examples: aspirin, metformin, tylenol, insulin, nsaids,
                            decongestants, hydration, rest, isolation,
                            DASH diet.

  PROCEDURE       The ACT of testing or examining — diagnostic tests, scans,
                  exams, panels mentioned by name (with or without results).
                  If a test is named without a specific value, it is
                  PROCEDURE, not LAB_RESULT.
                  Examples: ecg, chest x-ray, cbc, stress test,
                            CT angiography, spirometry, pulse oximetry,
                            lung auscultation, throat exam.

  LAB_RESULT      Specific measured values, vital signs, or qualitative
                  abnormalities WITH a number or explicit finding. The
                  result, not the act of measuring.
                  Examples: A1C 7.2%, BP 148/90, BNP elevated, FEV1 65%,
                            temperature 38C, oxygen saturation 92%.

  LOCATION        Anatomical body parts referenced in connection with a
                  symptom, finding, or procedure. Be specific ("forehead"
                  not "head" when the patient says forehead).
                  Must be a body part — not a place. Do NOT extract
                  non-anatomical locations such as homes, hospitals,
                  rooms, cities, or workplaces.
                  Examples: chest, left arm, heart, lower back, throat,
                            forehead, abdomen.

  MEDICAL_HISTORY Pre-existing conditions, risk factors, social/exposure
                  history, family history. Background context, not active
                  complaints. Denied history is acceptable as a node when
                  the doctor explicitly elicits it (e.g., "any allergies?"
                  -> "no known allergies"); skip trivial denials about
                  unrelated topics.
                  Examples: diabetes, hypertension, smoking, non-smoker,
                            alcohol use, family history of MI,
                            no known allergies, recent travel,
                            sick contacts, occupational exposure.

If a clinical mention does not fit any of these 7 types, do not extract it.

EDGE TYPES — use exactly these 6, and respect direction:

  LOCATED_AT  SYMPTOM -> LOCATION         ("pain" LOCATED_AT "chest")
  INDICATES   SYMPTOM -> DIAGNOSIS        ("chest pain" INDICATES "angina")
  TAKEN_FOR   TREATMENT -> DIAGNOSIS or SYMPTOM   ("Aspirin" TAKEN_FOR "angina")
  CAUSES      MEDICAL_HISTORY -> DIAGNOSIS        ("smoking" CAUSES "heart disease")
  RULES_OUT   PROCEDURE -> DIAGNOSIS              ("ECG" RULES_OUT "arrhythmia")
  CONFIRMS    LAB_RESULT -> DIAGNOSIS             ("elevated BNP" CONFIRMS "heart failure")

A symptom may indicate multiple diagnoses. A treatment may apply to multiple
conditions.

NAMING RULES:
- All text lowercase ("covid-19" not "COVID-19", "ecg" not "ECG").
- Use medical-standard terms when present in dialogue (prefer "nasal
  congestion" over "stuffy nose" if both are spoken).
- Arabic numerals ("type 1 diabetes" not "type one diabetes").
- Use hyphenation that is standard in clinical writing ("covid-19", not
  "covid 19" or "covid").
- Atomic — do not combine concepts with "or"/"and".
- Symptoms denied during a Review of Systems should be extracted as
  SYMPTOM nodes prefixed with "absent ". Standard ROS categories:
  constitutional, ENT, respiratory, cardiac, GI, GU, MSK, neurological,
  skin, psychiatric. Be specific when a body part is mentioned:
    "no chest pain"      -> "absent chest pain"   (NOT "absent pain")
    "denies abdominal pain" -> "absent abdominal pain"
  Do NOT extract "absent X" for non-ROS items (exposures, tests, lifestyle,
  objects). Denied lifestyle/history belongs to MEDICAL_HISTORY.
- No PATIENT nodes.

TURN_ID RULES:
- Format: "P-N" (patient) or "D-N" (doctor), matching transcript brackets [P-1], [D-39].
- Nodes use the turn where the concept is first mentioned.
- Edges use the turn where the relationship is established (may differ from
  the turns of the endpoint nodes).

EXTRACTION STRATEGY:
Read the transcript carefully and extract:
  - every symptom (with character modifiers like "dry cough", temporal
    qualifiers like "worse at night", and severity descriptors)
  - every body location tied to a symptom
  - every diagnosis considered, suspected, confirmed, or ruled out
  - every treatment, including supportive measures and lifestyle interventions
  - every procedure mentioned by name (tests, scans, physical exam actions)
  - every medical history item including social, exposure, and family history
  - every lab value or vital sign with a specific number

Pay extra attention to:
  - LOCATION: pair every body part with the symptom it relates to
    (LOCATED_AT edge).
  - LAB_RESULT: capture every numeric vital or value mentioned (temp, BP,
    O2 sat, heart rate, etc.).
  - PROCEDURE: capture every test or exam mentioned by name, even when
    only ordered or referenced (cbc, ecg, chest x-ray, pulse oximetry,
    physical exam).

Aim for 15-25 nodes per patient with at least 1.0 edges per node.

TRANSCRIPT:
{transcript}

Output ONLY valid JSON."""


ENHANCER_PROMPT = """You are a clinical KG completeness enhancer. The transcript
has been partially extracted. Your job is to find entities and relationships
that were MISSED, specifically in these high-gap categories.

TARGET THESE MISSING ITEMS (in priority order):

1. LOCATION nodes that should be LOCATED_AT a symptom:
   - Any body part the patient touches, points to, or describes pain in.
   - If a symptom (e.g., "chest pain") lacks a location edge, extract the
     anatomical location it points to.
   - Body parts only — do NOT extract places, rooms, or cities.

2. LAB_RESULT nodes — every numeric vital or lab value:
   - Temperature readings (any number with a unit).
   - Blood pressure readings (e.g., "140/90").
   - Heart rate, respiratory rate, oxygen saturation if stated with a number.
   - Any lab value mentioned by name with a value.

3. PROCEDURE nodes — every test/scan/exam mentioned by name, even without
   a result. Include the act of physical exam (auscultation, palpation,
   throat exam, vital sign measurement) as well as imaging and lab orders.

4. MEDICAL_HISTORY risk factors often missed:
   - Smoking status with details (pack-years, quit date).
   - Family history with specific diseases.
   - Specific exposures (occupational, environmental, infectious contacts).
   - Allergies, vaccinations, prior surgeries.

5. Missing EDGES between entities already extracted:
   - SYMPTOM -[LOCATED_AT]-> LOCATION
   - PROCEDURE -[RULES_OUT]-> DIAGNOSIS
   - LAB_RESULT -[CONFIRMS]-> DIAGNOSIS
   - MEDICAL_HISTORY -[CAUSES]-> DIAGNOSIS

RULES:
- ONLY ADD. Never modify or remove anything already extracted.
- Do not duplicate entities already in the list below.
- Use new node IDs with prefix "N_NEW_001", "N_NEW_002", etc.
- For edges, you may reference either new N_NEW_ IDs or existing node text
  (use exact text from "ALREADY EXTRACTED" — resolution happens automatically).
- Use only the same 7 node types and 6 edge types as the original extraction.

TRANSCRIPT:
{transcript}

ALREADY EXTRACTED (do NOT duplicate):
{existing_entities}

Output ONLY valid JSON:
{{
  "nodes": [{{"id": "N_NEW_001", "text": "...", "type": "...", "evidence": "...", "turn_id": "..."}}],
  "edges": [{{"source_id": "...", "target_id": "...", "type": "...", "evidence": "...", "turn_id": "..."}}]
}}

Empty output is fine if nothing is missing: {{"nodes": [], "edges": []}}."""


EDGE_ENHANCER_PROMPT = """You are a clinical KG edge completeness reviewer. The
transcript has already been extracted into entities (listed below). Your sole
job is to find MISSING edges between those entities — relationships that the
transcript supports but were not yet captured.

Do NOT add new nodes. Use only the entity IDs provided.

EDGE TYPES — six total. For each, here is what to scan for:

  INDICATES   SYMPTOM -> DIAGNOSIS
              For every DIAGNOSIS in the entities, look for symptoms in the
              encounter that point to it. A single diagnosis often has 2-4
              symptoms supporting it (chief complaint + supporting findings +
              risk-context findings). Include classic textbook indications,
              compatible-but-non-specific findings, and contextually relevant
              comorbid features when the transcript discusses them together.

  CAUSES      MEDICAL_HISTORY -> DIAGNOSIS
              Risk factors that contribute to a diagnosis through standard
              clinical pathways. Common patterns:
                - Behaviors / exposures   (e.g., smoking -> COPD)
                - Chronic conditions      (e.g., hypertension -> heart failure)
                - Family history          (e.g., family history of MI ->
                                           ischemic heart disease)
              Only emit when the link is unambiguous to any practicing
              internist — no rare-condition or specialist-level reasoning.

  LOCATED_AT  SYMPTOM -> LOCATION
              For every SYMPTOM with a body-part location mentioned in the
              encounter, ensure the LOCATED_AT edge exists.

  TAKEN_FOR   TREATMENT -> SYMPTOM or DIAGNOSIS
              For every TREATMENT, identify what it is being given for, when
              the transcript makes the indication explicit.

  RULES_OUT   PROCEDURE -> DIAGNOSIS  (when result is negative / normal)
              For every PROCEDURE in the entities, check whether it was used
              to exclude a diagnosis.

  CONFIRMS    LAB_RESULT -> DIAGNOSIS  (when result is positive / abnormal)
              For every abnormal LAB_RESULT, check which diagnosis it confirms.

WHAT TO PRIORITIZE:
- Diagnoses with fewer than 2 INDICATES edges — they are likely under-linked.
- Risk factors in MEDICAL_HISTORY that have plausible causal links to a
  current diagnosis but no CAUSES edge yet.
- Symptoms with body-part locations mentioned but no LOCATED_AT edge.
- Treatments with no TAKEN_FOR edge.

RULES:
- ONLY ADD EDGES. Do not add or modify nodes.
- Only emit edges between entities listed below — do not invent new entities.
- Respect direction strictly:
    LOCATED_AT  source must be SYMPTOM, target must be LOCATION
    INDICATES   source must be SYMPTOM, target must be DIAGNOSIS
    TAKEN_FOR   source must be TREATMENT, target must be SYMPTOM or DIAGNOSIS
    CAUSES      source must be MEDICAL_HISTORY, target must be DIAGNOSIS
    RULES_OUT   source must be PROCEDURE, target must be DIAGNOSIS
    CONFIRMS    source must be LAB_RESULT, target must be DIAGNOSIS
- Cite a turn_id and a short evidence phrase from the transcript for each edge.
- If an edge is already obvious from existing edges (avoid duplicating exact
  source-target-type triples), skip it.

TRANSCRIPT:
{transcript}

ENTITIES (use these IDs as source_id / target_id):
{entities_with_ids}

EXISTING EDGES (do NOT duplicate exact source-target-type triples):
{existing_edges}

Output ONLY valid JSON:
{{
  "edges": [{{"source_id": "...", "target_id": "...", "type": "...", "evidence": "...", "turn_id": "..."}}]
}}

Empty output is fine if no edges are missing: {{"edges": []}}."""


# ============================================================
# OpenRouter client
# ============================================================
class OpenRouterClient:
    """Client for OpenRouter API."""

    def __init__(self, api_key: str, model: str = OPENROUTER_MODEL):
        from openai import OpenAI
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        self.model = model

    def generate(self, prompt: str, temperature: float = None) -> tuple:
        for attempt in range(MAX_RETRIES):
            try:
                kwargs = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": True,
                }
                if temperature is not None:
                    kwargs["temperature"] = temperature
                stream = self.client.chat.completions.create(**kwargs)

                content = ""
                last_chunk = None
                for chunk in stream:
                    last_chunk = chunk
                    delta = chunk.choices[0].delta
                    if delta.content:
                        content += delta.content

                usage = None
                if last_chunk and hasattr(last_chunk, "usage") and last_chunk.usage:
                    u = last_chunk.usage
                    usage = {
                        "prompt_tokens": u.prompt_tokens,
                        "completion_tokens": u.completion_tokens,
                    }

                if content:
                    return content, usage

            except Exception as e:
                print(f"(error: {e}, retry {attempt + 1})", end=" ", flush=True)
                time.sleep(2 ** attempt)

        return "", None


def get_client(api_keys: dict, model: str = None) -> OpenRouterClient:
    key = api_keys.get("openrouter")
    if not key:
        raise SystemExit('api_keys.json must contain a non-empty "openrouter" key')
    return OpenRouterClient(key, model or EXTRACTOR_MODEL)


def get_agent_clients(api_keys: dict) -> dict:
    clients = {"extractor": get_client(api_keys, EXTRACTOR_MODEL)}
    if ENHANCER_MODEL == EXTRACTOR_MODEL:
        clients["enhancer"] = clients["extractor"]
    else:
        clients["enhancer"] = get_client(api_keys, ENHANCER_MODEL)
    return clients


# ============================================================
# Utilities
# ============================================================
def read_transcript(file_path: Path) -> str:
    with open(file_path, "r") as f:
        return f.read()


def extract_json_from_response(response_text: str) -> dict:
    """Parse JSON from LLM output, tolerating fences and trailing commas."""
    if not response_text:
        return None

    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass

    if response_text.strip().startswith("```"):
        parts = response_text.split("```")
        if len(parts) >= 2:
            inner = parts[1]
            if inner.startswith("json"):
                inner = inner[4:]
            inner = inner.strip()
            try:
                return json.loads(inner)
            except json.JSONDecodeError:
                pass

    match = re.search(r"\{[\s\S]*\}", response_text)
    if match:
        json_str = match.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            fixed = re.sub(r",(\s*[}\]])", r"\1", json_str)
            try:
                return json.loads(fixed)
            except json.JSONDecodeError:
                pass

    return None


# ============================================================
# Text normalization — principle-based, no word lookups
# ============================================================
def normalize_text(text: str, node_type: str = None) -> str:
    """Normalize node text via structural rules only.

    Two principles:
      1. Negation prefix unification: "no X" / "not X" / "denied X" /
         "negative X" -> "absent X".
      2. Past-tense lifestyle unification: "quit X" / "stopped X" /
         "former X" / "used to X" -> "past X".

    No per-word synonym mappings. Casing, punctuation, and whitespace
    cleanup are also performed.
    """
    t = text.lower().strip()
    t = t.rstrip(".,;:")
    t = re.sub(r"\s+", " ", t)

    # Principle 1: negation prefix
    neg_match = re.match(r"^(no|not|denied|negative)\s+(.+)$", t)
    if neg_match:
        rest = neg_match.group(2).strip()
        if not rest.startswith("absent"):
            t = f"absent {rest}"

    # Principle 2: past-tense lifestyle
    past_match = re.match(r"^(quit|stopped|used to|former)\s+(.+)$", t)
    if past_match:
        rest = past_match.group(2).strip()
        # Light handling: "used to smoke" -> "past smoking"
        if past_match.group(1) == "used to":
            if rest == "smoke":
                rest = "smoking"
            elif rest == "drink":
                rest = "drinking"
            elif rest == "vape":
                rest = "vaping"
        t = f"past {rest}"

    return t


# ============================================================
# Validation — schema rules only, no word/type lookup tables
# ============================================================
def validate_knowledge_graph(kg: dict) -> dict:
    """Validate and normalize KG.

    Rules applied:
      - Normalize node text (negation/tense principles).
      - Reject nodes with types outside the schema (no remapping).
      - Drop nodes with empty text.
      - Verify edge type is in schema; reject otherwise.
      - Verify edge endpoints exist.
      - Auto-swap reversed edges when types match the reverse direction.
      - Reject edges whose endpoint types violate the rule entirely.
      - Deduplicate nodes by (text, type) and edges by (src, type, tgt).
    """
    if not kg or "nodes" not in kg or "edges" not in kg:
        return kg

    valid_nodes = []
    node_type_map = {}
    normalized_count = 0
    dropped_type = 0

    for n in kg.get("nodes", []):
        original_text = n.get("text", "")
        normalized = normalize_text(original_text)
        if normalized != original_text.lower().strip():
            normalized_count += 1
        n["text"] = normalized

        t = n.get("type", "").upper().replace(" ", "_")
        if t not in VALID_NODE_TYPES:
            dropped_type += 1
            continue
        n["type"] = t

        if not n["text"]:
            continue

        valid_nodes.append(n)
        node_type_map[n["id"]] = t

    node_ids = set(node_type_map.keys())
    valid_edges = []
    swapped = 0
    dropped_edge = 0

    for edge in kg.get("edges", []):
        src_id = edge.get("source_id")
        tgt_id = edge.get("target_id")
        etype = edge.get("type", "").upper().replace(" ", "_")

        if etype not in VALID_EDGE_TYPES:
            dropped_edge += 1
            continue
        if src_id not in node_ids or tgt_id not in node_ids:
            dropped_edge += 1
            continue

        src_type = node_type_map[src_id]
        tgt_type = node_type_map[tgt_id]
        expected_src, expected_tgt = EDGE_RULES[etype]

        if src_type in expected_src and tgt_type in expected_tgt:
            edge["type"] = etype
            valid_edges.append(edge)
        elif tgt_type in expected_src and src_type in expected_tgt:
            edge["source_id"], edge["target_id"] = tgt_id, src_id
            edge["type"] = etype
            valid_edges.append(edge)
            swapped += 1
        else:
            dropped_edge += 1

    # Dedup nodes by (text, type)
    seen = {}
    deduplicated_nodes = []
    id_remap = {}
    for n in valid_nodes:
        key = (n["text"], n["type"])
        if key in seen:
            id_remap[n["id"]] = seen[key]
        else:
            seen[key] = n["id"]
            deduplicated_nodes.append(n)

    # Apply remap to edges and dedup edges
    final_edges = []
    seen_edges = set()
    dedup_edge_count = 0
    for e in valid_edges:
        e["source_id"] = id_remap.get(e["source_id"], e["source_id"])
        e["target_id"] = id_remap.get(e["target_id"], e["target_id"])
        edge_key = (e["source_id"], e["target_id"], e["type"])
        if edge_key in seen_edges:
            dedup_edge_count += 1
            continue
        seen_edges.add(edge_key)
        final_edges.append(e)

    if normalized_count:
        print(f"    Normalized {normalized_count} node texts")
    if dropped_type:
        print(f"    Dropped {dropped_type} invalid-type nodes")
    if swapped:
        print(f"    Swapped {swapped} edge directions")
    if dropped_edge:
        print(f"    Dropped {dropped_edge} invalid edges")
    dedup_node_count = len(valid_nodes) - len(deduplicated_nodes)
    if dedup_node_count:
        print(f"    Deduplicated {dedup_node_count} nodes")
    if dedup_edge_count:
        print(f"    Deduplicated {dedup_edge_count} edges")

    kg["nodes"] = deduplicated_nodes
    kg["edges"] = final_edges
    return kg


# ============================================================
# Pipeline stages (unchanged from SC v4)
# ============================================================
def get_transcript_files():
    files = []
    for res_dir in sorted(TRANSCRIPT_DIR.glob("RES*")):
        if res_dir.is_dir():
            txt_file = res_dir / f"{res_dir.name}.txt"
            if txt_file.exists():
                files.append(txt_file)
    return files


def extract_naive(transcript: str, client: OpenRouterClient) -> tuple:
    prompt = EXTRACTION_PROMPT.replace("{transcript}", transcript)
    content, usage = client.generate(prompt)
    if content:
        kg = extract_json_from_response(content)
        if kg:
            kg = validate_knowledge_graph(kg)
        return kg, usage
    return None, usage


def _accumulate_usage(total: dict, usage: dict) -> None:
    if not usage:
        return
    total["prompt_tokens"] = total.get("prompt_tokens", 0) + usage.get("prompt_tokens", 0)
    total["completion_tokens"] = total.get("completion_tokens", 0) + usage.get("completion_tokens", 0)


def _run_enhancer(transcript: str, kg: dict, client: OpenRouterClient) -> tuple:
    existing_entities = (
        "\n".join(
            f"  [{n.get('type', 'UNK')}] {n['text']}"
            for n in kg.get("nodes", [])
        )
        if kg and kg.get("nodes")
        else "  (none)"
    )
    prompt = (
        ENHANCER_PROMPT.replace("{transcript}", transcript).replace(
            "{existing_entities}", existing_entities
        )
    )
    content, usage = client.generate(prompt)
    additions = {"nodes": [], "edges": []}
    if content:
        parsed = extract_json_from_response(content)
        if parsed:
            additions["nodes"] = parsed.get("nodes", []) or []
            additions["edges"] = parsed.get("edges", []) or []
    return additions, usage


def _run_edge_enhancer(transcript: str, kg: dict, client: OpenRouterClient) -> tuple:
    """Edge-focused second pass. Adds missing edges only — never new nodes.

    Sees the transcript, the full entity list (with IDs), and the existing
    edges. Asked to find edges that should exist but don't yet.
    """
    nodes = kg.get("nodes", []) if kg else []
    edges = kg.get("edges", []) if kg else []

    if not nodes:
        return {"edges": []}, None

    # Format entities with IDs so the LLM can reference them by ID.
    entities_with_ids = "\n".join(
        f"  {n['id']} [{n.get('type', 'UNK')}] {n['text']}"
        for n in nodes
    )
    existing_edges = (
        "\n".join(
            f"  {e['source_id']} -[{e.get('type', 'UNK')}]-> {e['target_id']}"
            for e in edges
        )
        if edges
        else "  (none)"
    )

    prompt = (
        EDGE_ENHANCER_PROMPT
        .replace("{transcript}", transcript)
        .replace("{entities_with_ids}", entities_with_ids)
        .replace("{existing_edges}", existing_edges)
    )

    content, usage = client.generate(prompt)
    additions = {"edges": []}
    if content:
        parsed = extract_json_from_response(content)
        if parsed:
            additions["edges"] = parsed.get("edges", []) or []
    return additions, usage


def _merge_edge_additions(kg: dict, additions: dict) -> tuple:
    """Merge new edges only. Drops references to non-existent nodes silently."""
    new_edges = additions.get("edges", []) or []
    if not new_edges:
        return kg, 0

    existing_ids = {n["id"] for n in kg.get("nodes", [])}
    existing_edge_keys = {
        (e["source_id"], e["target_id"], e.get("type", "").upper())
        for e in kg.get("edges", [])
    }

    added = 0
    for e in new_edges:
        src = e.get("source_id", "")
        tgt = e.get("target_id", "")
        etype = (e.get("type", "") or "").upper()
        if not src or not tgt or not etype:
            continue
        if src not in existing_ids or tgt not in existing_ids:
            continue  # silently drop refs to unknown nodes
        key = (src, tgt, etype)
        if key in existing_edge_keys:
            continue  # avoid exact duplicates
        kg.setdefault("edges", []).append(e)
        existing_edge_keys.add(key)
        added += 1

    return kg, added


def _merge_enhancer_additions(kg: dict, additions: dict):
    if not additions.get("nodes") and not additions.get("edges"):
        return kg, 0, 0

    text_to_id = {n["text"].lower().strip(): n["id"] for n in kg.get("nodes", [])}
    existing_ids = {n["id"] for n in kg.get("nodes", [])}
    existing_texts = set(text_to_id.keys())

    added_nodes = 0
    for n in additions.get("nodes", []):
        node_id = n.get("id", "")
        node_text = n.get("text", "").lower().strip()
        if not node_id or not node_text:
            continue
        if node_id in existing_ids:
            continue
        if node_text in existing_texts:
            existing_id_for_text = text_to_id[node_text]
            text_to_id[node_id.lower().strip()] = existing_id_for_text
            continue
        kg.setdefault("nodes", []).append(n)
        existing_ids.add(node_id)
        existing_texts.add(node_text)
        text_to_id[node_text] = node_id
        added_nodes += 1

    added_edges = 0
    for e in additions.get("edges", []):
        src = e.get("source_id", "")
        tgt = e.get("target_id", "")
        if not src or not tgt:
            continue
        if src not in existing_ids:
            src_key = src.lower().strip()
            src = text_to_id.get(src_key, src)
        if tgt not in existing_ids:
            tgt_key = tgt.lower().strip()
            tgt = text_to_id.get(tgt_key, tgt)
        if src not in existing_ids or tgt not in existing_ids:
            continue
        e["source_id"] = src
        e["target_id"] = tgt
        kg.setdefault("edges", []).append(e)
        added_edges += 1

    return kg, added_nodes, added_edges


def _run_extractor_once(transcript: str, client: OpenRouterClient) -> tuple:
    prompt = EXTRACTION_PROMPT.replace("{transcript}", transcript)
    content, usage = client.generate(prompt, temperature=EXTRACTION_TEMPERATURE)
    if not content:
        return None, usage
    kg = extract_json_from_response(content)
    return kg, usage


def _vote_merge_kgs(kgs: list, threshold: int) -> dict:
    """Self-consistency voting: keep (text, type) pairs seen in >= threshold runs."""
    node_key_to_runs = {}
    for run_idx, kg in enumerate(kgs):
        if not kg:
            continue
        seen_this_run = set()
        for n in kg.get("nodes", []):
            text = (n.get("text") or "").lower().strip()
            ntype = (n.get("type") or "").upper()
            if not text or not ntype:
                continue
            key = (text, ntype)
            if key in seen_this_run:
                continue
            seen_this_run.add(key)
            node_key_to_runs.setdefault(key, set()).add(run_idx)

    winning_keys = {k for k, runs in node_key_to_runs.items() if len(runs) >= threshold}

    merged_nodes = []
    new_id_by_key = {}
    old_id_to_new_id = {}
    node_counter = 1

    for run_idx, kg in enumerate(kgs):
        if not kg:
            continue
        for n in kg.get("nodes", []):
            text = (n.get("text") or "").lower().strip()
            ntype = (n.get("type") or "").upper()
            if not text or not ntype:
                continue
            key = (text, ntype)
            if key not in winning_keys:
                continue
            if key not in new_id_by_key:
                new_id = f"N_{node_counter:03d}"
                node_counter += 1
                new_id_by_key[key] = new_id
                merged_nodes.append({
                    "id": new_id,
                    "text": text,
                    "type": ntype,
                    "evidence": n.get("evidence", ""),
                    "turn_id": n.get("turn_id", ""),
                })
            old_id_to_new_id[(run_idx, n.get("id", ""))] = new_id_by_key[key]

    merged_edges = []
    edge_key_set = set()
    for run_idx, kg in enumerate(kgs):
        if not kg:
            continue
        for e in kg.get("edges", []):
            src_old = e.get("source_id", "")
            tgt_old = e.get("target_id", "")
            src_new = old_id_to_new_id.get((run_idx, src_old))
            tgt_new = old_id_to_new_id.get((run_idx, tgt_old))
            if not src_new or not tgt_new:
                continue
            etype = (e.get("type") or "").upper()
            edge_key = (src_new, tgt_new, etype)
            if edge_key in edge_key_set:
                continue
            edge_key_set.add(edge_key)
            merged_edges.append({
                "source_id": src_new,
                "target_id": tgt_new,
                "type": etype,
                "evidence": e.get("evidence", ""),
                "turn_id": e.get("turn_id", ""),
            })

    return {"nodes": merged_nodes, "edges": merged_edges}


def extract_with_self_consistency(transcript: str, client: OpenRouterClient,
                                   k: int, threshold: int) -> tuple:
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0}
    runs = []
    for _ in range(k):
        kg, usage = _run_extractor_once(transcript, client)
        _accumulate_usage(total_usage, usage)
        runs.append(kg)

    valid_runs = [r for r in runs if r is not None]
    n_valid = len(valid_runs)
    if n_valid == 0:
        return None, total_usage

    effective_threshold = min(threshold, n_valid)
    merged = _vote_merge_kgs(runs, effective_threshold)

    total_nodes_before = sum(len(r.get("nodes", [])) for r in valid_runs)
    nodes_after = len(merged.get("nodes", []))
    print(f"[SC:{n_valid}runs,thresh={effective_threshold} {total_nodes_before}->{nodes_after}n]",
          end=" ", flush=True)

    return merged, total_usage


def extract_karma(transcript: str, clients: dict) -> tuple:
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0}

    if SELF_CONSISTENCY_K > 1:
        kg, usage = extract_with_self_consistency(
            transcript, clients["extractor"], SELF_CONSISTENCY_K, VOTE_THRESHOLD
        )
        _accumulate_usage(total_usage, usage)
    else:
        kg, usage = _run_extractor_once(transcript, clients["extractor"])
        _accumulate_usage(total_usage, usage)

    if not kg:
        return None, total_usage

    initial_n = len(kg.get("nodes", []))
    initial_e = len(kg.get("edges", []))

    added_n, added_e = 0, 0
    try:
        additions, usage = _run_enhancer(transcript, kg, clients["enhancer"])
        _accumulate_usage(total_usage, usage)
        kg, added_n, added_e = _merge_enhancer_additions(kg, additions)
    except Exception as ex:
        print(f"(enhancer error: {ex})", end=" ", flush=True)

    # Stage 2b: Optional edge-focused second pass. Disabled via env var
    # (EDGE_ENHANCER_ENABLED=0) when isolating other changes.
    edge_added = 0
    if EDGE_ENHANCER_ENABLED:
        try:
            edge_additions, usage = _run_edge_enhancer(transcript, kg, clients["enhancer"])
            _accumulate_usage(total_usage, usage)
            kg, edge_added = _merge_edge_additions(kg, edge_additions)
        except Exception as ex:
            print(f"(edge enhancer error: {ex})", end=" ", flush=True)

    kg = validate_knowledge_graph(kg)

    final_n = len(kg.get("nodes", []))
    final_e = len(kg.get("edges", []))

    print(f"[n:{initial_n}+{added_n}->{final_n} e:{initial_e}+{added_e}+{edge_added}->{final_e}]",
          end=" ", flush=True)

    return kg, total_usage


def extract(transcript: str, clients: dict) -> tuple:
    if PIPELINE_MODE == "karma":
        return extract_karma(transcript, clients)
    return extract_naive(transcript, clients["extractor"])


def process_one(txt_path: Path, clients: dict, output_dir: Path, suffix: str) -> tuple:
    res_id = txt_path.stem
    output_file = output_dir / f"{res_id}_{suffix}.json"

    if output_file.exists():
        return res_id, "SKIP", 0, 0, None

    try:
        transcript = read_transcript(txt_path)
        print(f"  {res_id}...", end=" ", flush=True)
        kg, usage = extract(transcript, clients)

        if not kg:
            print("FAILED")
            return res_id, "FAILED", 0, 0, None

        n, e = len(kg.get("nodes", [])), len(kg.get("edges", []))
        kg["_usage"] = usage

        with open(output_file, "w") as f:
            json.dump(kg, f, indent=2, ensure_ascii=False)

        print(f"({n}n/{e}e)")
        return res_id, "OK", n, e, usage

    except Exception as ex:
        print(f"ERROR: {ex}")
        return res_id, f"ERROR: {ex}", 0, 0, None


def main():
    parser = argparse.ArgumentParser(description="KG Extraction Pipeline (SC v4 — clean)")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--res-ids", nargs="+", default=None)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open("api_keys.json") as f:
        api_keys = json.load(f)

    clients = get_agent_clients(api_keys)
    suffix = OUTPUT_SUFFIX

    transcript_files = get_transcript_files()
    if args.res_ids:
        transcript_files = [f for f in transcript_files if f.parent.name in args.res_ids]

    print("KG Extraction Pipeline (SC v4 — clean)")
    print(f"Method: {PIPELINE_MODE}")
    print(f"  Extractor: {EXTRACTOR_MODEL}")
    if PIPELINE_MODE == "karma":
        print(f"  Enhancer : {ENHANCER_MODEL}")
        if SELF_CONSISTENCY_K > 1:
            print(f"  Self-Consistency: K={SELF_CONSISTENCY_K} runs, vote >= {VOTE_THRESHOLD}")
            print(f"  Extraction temperature: {EXTRACTION_TEMPERATURE}")
        print(f"  Edge Enhancer: {'ON' if EDGE_ENHANCER_ENABLED else 'OFF'}")
    print(f"Output: {output_dir}")
    print(f"Processing {len(transcript_files)} transcripts")
    print(f"Using {MAX_WORKERS} parallel workers...")
    print("=" * 60)

    success = 0
    failed = 0
    total_tokens = {"prompt": 0, "completion": 0}
    all_stats = []
    stats_lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_path = {
            executor.submit(process_one, txt_path, clients, output_dir, suffix): txt_path
            for txt_path in transcript_files
        }
        for future in as_completed(future_to_path):
            try:
                res_id, status, nodes, edges, usage = future.result()
                with stats_lock:
                    if status == "OK":
                        success += 1
                        if usage:
                            total_tokens["prompt"] += usage.get("prompt_tokens", 0)
                            total_tokens["completion"] += usage.get("completion_tokens", 0)
                            all_stats.append({"res_id": res_id, "nodes": nodes, "edges": edges, **usage})
                    elif status == "SKIP":
                        print(f"  {res_id}: SKIP (exists)")
                        success += 1
                    else:
                        failed += 1
            except Exception as e:
                txt_path = future_to_path[future]
                print(f"  ERROR in {txt_path.stem}: {e}")
                with stats_lock:
                    failed += 1

    stats_file = output_dir / "_stats.json"
    with open(stats_file, "w") as f:
        json.dump({
            "method": PIPELINE_MODE,
            "extractor_model": EXTRACTOR_MODEL,
            "enhancer_model": ENHANCER_MODEL if PIPELINE_MODE == "karma" else None,
            "self_consistency_k": SELF_CONSISTENCY_K,
            "vote_threshold": VOTE_THRESHOLD,
            "extraction_temperature": EXTRACTION_TEMPERATURE,
            "total_tokens": total_tokens,
            "success": success,
            "failed": failed,
            "details": all_stats,
        }, f, indent=2)

    print("=" * 60)
    print(f"Done! Success: {success}, Failed: {failed}")
    print(f"Total tokens: {total_tokens['prompt'] + total_tokens['completion']}")
    print(f"Output: {output_dir}/")


if __name__ == "__main__":
    main()