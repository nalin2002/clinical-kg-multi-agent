"""
Multi-Agent KG Extraction Pipeline
====================================
3-agent pipeline for clinical knowledge graph extraction from medical transcripts.

Architecture:
  Agent 1 (Primary Extractor):  deepseek/deepseek-r1-distill-qwen-32b  (reasoning model)
  Agent 2 (Gap Finder):         qwen/qwen3-14b                          (completeness pass)
  Merge:                        Python deduplication + edge validation   (no extra LLM cost)

Output files are named RES{ID}_multi_agent.json and are compatible with
Clinical_KG_OS_LLM dump_graph.py for entity resolution and unification.

Usage:
  python multi_agent_kg_extraction.py --output ./my_kg_multi
  python multi_agent_kg_extraction.py --output ./my_kg_multi --res-ids RES0198 RES0199
  python multi_agent_kg_extraction.py --output ./my_kg_multi --api-key sk-or-...
  python multi_agent_kg_extraction.py --output ./my_kg_multi --skip-agent2

Requires:
  pip install openai
"""

import json
import re
import argparse
import time
import os
from pathlib import Path
from openai import OpenAI

# ==================== CONFIGURATION ====================

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

AGENT1_MODEL = "deepseek/deepseek-r1-distill-qwen-32b"
AGENT2_MODEL = "qwen/qwen3-14b"
AGENT1_FALLBACK_MODEL = "qwen/qwen3-14b"   # used if Agent 1 returns unparseable JSON

OUTPUT_SUFFIX = "multi_agent"
MAX_RETRIES = 3

# Relative to this script: ../Clinical_KG_OS_LLM/data/transcripts
TRANSCRIPT_DIR = Path(__file__).parent.parent / "Clinical_KG_OS_LLM" / "data" / "transcripts"

VALID_NODE_TYPES = {"SYMPTOM", "DIAGNOSIS", "TREATMENT", "PROCEDURE", "LOCATION", "MEDICAL_HISTORY", "LAB_RESULT"}
VALID_EDGE_TYPES = {"CAUSES", "INDICATES", "LOCATED_AT", "RULES_OUT", "TAKEN_FOR", "CONFIRMS"}


# ==================== PROMPTS ====================

AGENT1_SYSTEM = (
    "You are an expert clinical knowledge graph extraction specialist with deep medical knowledge. "
    "Extract comprehensive, accurate structured information from doctor-patient transcripts. "
    "Be thorough — extract every clinically relevant entity and relationship."
)

AGENT1_PROMPT = """Extract a comprehensive clinical knowledge graph from this doctor-patient consultation transcript.

## NODE TYPES — Extract ALL that apply, be exhaustive:

**SYMPTOM** — Patient-reported or observed symptoms. Include BOTH present AND denied symptoms.
  Examples: chest pain, shortness of breath, nasal congestion, fatigue, fever, cough, nausea, vomiting

**DIAGNOSIS** — Active diagnoses, suspected conditions, differential diagnoses, ruled-out conditions.
  Examples: viral infection, COVID-19, type 1 diabetes, hypertension, pneumonia, common cold

**TREATMENT** — All medications, therapies, dietary changes, lifestyle interventions.
  Examples: Aspirin, insulin, decongestants, Tylenol Cold, DASH diet, antibiotics, hydration, bed rest

**PROCEDURE** — Tests, examinations, imaging, procedures ordered, done, or recommended.
  Examples: COVID swab, blood test, ECG, chest X-ray, stress test, physical exam, urine culture

**LOCATION** — Anatomical body parts and locations explicitly mentioned.
  Examples: chest, nose, left arm, heart, lungs, abdomen, throat, back

**MEDICAL_HISTORY** — Pre-existing conditions, past diagnoses, risk factors, family history, and social history.
  Social history includes: smoking, alcohol use, recreational drug use, occupation, living situation, recent exposures.
  Examples: type 1 diabetes, smoking history, family heart disease, daycare exposure, lives alone

**LAB_RESULT** — Specific lab values, vital signs, or test result numbers mentioned.
  Examples: A1C 7.2%, BP 148/90, elevated BNP, SpO2 95%, temperature 38.5°C

## EDGE TYPES — direction matters:
- **CAUSES**: (MEDICAL_HISTORY/risk factor) → (DIAGNOSIS/SYMPTOM). e.g., smoking CAUSES COPD; daycare exposure CAUSES viral infection
- **INDICATES**: (SYMPTOM) → (DIAGNOSIS). e.g., chest pain INDICATES angina; nasal congestion INDICATES viral infection
- **LOCATED_AT**: (SYMPTOM) → (LOCATION). e.g., pain LOCATED_AT chest
- **RULES_OUT**: (PROCEDURE) → (DIAGNOSIS). e.g., COVID swab RULES_OUT COVID-19; ECG RULES_OUT arrhythmia
- **TAKEN_FOR**: (TREATMENT) → (DIAGNOSIS/SYMPTOM). e.g., Aspirin TAKEN_FOR angina; insulin TAKEN_FOR diabetes
- **CONFIRMS**: (LAB_RESULT/PROCEDURE) → (DIAGNOSIS). e.g., elevated BNP CONFIRMS heart failure

## EXTRACTION RULES:
1. Extract EVERY entity mentioned, even briefly or in passing.
2. Include DENIED/ABSENT symptoms (e.g., "no fever" → add SYMPTOM node text="fever", evidence="no fever").
3. Extract ALL treatments — both patient-reported AND doctor-recommended/suggested medications.
4. For TREATMENT: include OTC meds, prescribed drugs, dietary advice, supportive care recommendations.
5. For MEDICAL_HISTORY: capture all social history (smoking, alcohol, drugs, occupation, housing, recent contacts/exposures).
6. For INDICATES edges: the SYMPTOM is always the source, the DIAGNOSIS is always the target.
7. For TAKEN_FOR edges: the TREATMENT is always the source, the condition it treats is the target.
8. Create edges for ALL relationships you can identify between extracted nodes.
9. Use exact short quotes from the transcript for evidence fields.
10. Turn ID format: "P-X" for patient turns [P-X], "D-X" for doctor turns [D-X].

## TRANSCRIPT:
{transcript}

## OUTPUT:
Return a single JSON object with two arrays:
- "nodes": [{{ "id": "N_001", "text": "<concise clinical term>", "type": "<NODE_TYPE>", "evidence": "<direct quote>", "turn_id": "<P-X or D-X>" }}, ...]
- "edges": [{{ "source_id": "N_001", "target_id": "N_002", "type": "<EDGE_TYPE>", "evidence": "<direct quote>", "turn_id": "<P-X or D-X>" }}, ...]

Output ONLY valid JSON. No markdown, no code blocks, no explanation."""


AGENT2_PROMPT = """You are doing a completeness audit of a clinical knowledge graph. A first extraction pass was done but may have missed entities and relationships.

## TRANSCRIPT:
{transcript}

## ALREADY EXTRACTED ENTITIES (do NOT re-extract these — only add what is MISSING):
{existing_entities}

## YOUR TASK: find what is missing from the above list.

Go through the transcript line by line and look for:
1. **SYMPTOMS** not yet listed — including briefly mentioned or denied symptoms (e.g., "no fever" → add SYMPTOM "fever")
2. **DIAGNOSES** not yet listed — differentials, suspected, or ruled-out conditions the doctor mentions
3. **TREATMENTS** not yet listed — any medication, OTC drug, prescribed drug, or therapy MENTIONED OR RECOMMENDED in the transcript
   (Important: check doctor's recommendations at the end of the visit — they often suggest medications not yet listed)
4. **PROCEDURES** not yet listed — any test, exam, or procedure ordered or referenced
5. **LOCATIONS** not yet listed — any body part mentioned
6. **MEDICAL_HISTORY** not yet listed — social history (smoking, alcohol, drugs, occupation, living situation, recent exposures/contacts), family history, past conditions
7. **MISSING EDGES** — additional relationships between already-extracted entities

## EDGE DIRECTION RULES:
- INDICATES: (SYMPTOM) → (DIAGNOSIS)  — symptom is source
- TAKEN_FOR: (TREATMENT) → (DIAGNOSIS or SYMPTOM) — treatment is source
- CAUSES: (MEDICAL_HISTORY or risk factor) → (DIAGNOSIS or SYMPTOM) — risk factor is source
- RULES_OUT: (PROCEDURE) → (DIAGNOSIS) — procedure is source
- LOCATED_AT: (SYMPTOM) → (LOCATION) — symptom is source
- CONFIRMS: (LAB_RESULT or PROCEDURE) → (DIAGNOSIS) — lab/procedure is source

## CRITICAL ID RULES:
- New node IDs must start from N_{start_id:03d} (avoid conflicts with Agent 1 IDs above)
- For edges between EXISTING nodes: use their EXACT IDs from the list above (e.g., N_001, N_002...)
- For edges involving new nodes: use new IDs (N_{start_id:03d}+)

Include evidence (exact transcript quotes) and turn_id for everything.
Output ONLY a JSON object with "nodes" and "edges" arrays for NEW/MISSING items only.
Output ONLY valid JSON."""


# ==================== OPENROUTER CLIENT ====================

class OpenRouterClient:
    """Thin wrapper around the OpenAI-compatible OpenRouter API."""

    def __init__(self, api_key: str, model: str):
        self.client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)
        self.model = model

    def generate(self, prompt: str, system: str = None) -> tuple[str, dict | None]:
        """Call the model. Returns (content, usage_dict)."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    stream=False,
                    max_tokens=8192,
                )
                content = response.choices[0].message.content or ""
                usage = None
                if response.usage:
                    usage = {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                    }
                return content, usage

            except Exception as e:
                wait = 2 ** attempt
                print(f"    [retry {attempt + 1}/{MAX_RETRIES}] {e} (wait {wait}s)", flush=True)
                time.sleep(wait)

        return "", None


# ==================== JSON EXTRACTION ====================

def strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks produced by reasoning models like deepseek-r1."""
    return re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE).strip()


def extract_json(text: str) -> dict | None:
    """Robustly extract a JSON object from an LLM response."""
    text = strip_think_tags(text)

    # Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strip markdown code fence
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Find first {...} block
    brace_match = re.search(r"\{[\s\S]*\}", text)
    if brace_match:
        candidate = brace_match.group(0)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            # Remove trailing commas before ] or }
            fixed = re.sub(r",\s*([}\]])", r"\1", candidate)
            try:
                return json.loads(fixed)
            except json.JSONDecodeError:
                pass

    return None


# ==================== KG MERGE & VALIDATION ====================

def normalize(text: str) -> str:
    return text.strip().lower()


def merge_kgs(kg1: dict, kg2: dict) -> dict:
    """
    Merge two KG dicts (nodes + edges), deduplicate nodes by (text, type),
    remap edge IDs, and drop edges with missing endpoints.
    """
    all_nodes = list(kg1.get("nodes", [])) + list(kg2.get("nodes", []))
    all_edges = list(kg1.get("edges", [])) + list(kg2.get("edges", []))

    # --- Deduplicate nodes: keep first occurrence per (text_lower, type) ---
    seen_keys: dict[tuple, str] = {}   # (text_norm, type) -> new canonical ID
    unique_nodes: list[dict] = []
    _counter = [1]  # mutable container to avoid nonlocal

    def process_nodes(nodes: list) -> dict:
        """Return mapping old_id -> new_id for this set of nodes."""
        id_map: dict[str, str] = {}
        for node in nodes:
            text = node.get("text", "").strip()
            if len(text) < 2:
                continue  # drop empty or single-char nodes
            key = (normalize(text), node.get("type", "").upper())
            if key in seen_keys:
                id_map[node["id"]] = seen_keys[key]
            else:
                new_id = f"N_{_counter[0]:03d}"
                node_type = node.get("type", "").upper()
                if node_type not in VALID_NODE_TYPES:
                    node_type = "SYMPTOM"  # fallback
                new_node = {
                    "id": new_id,
                    "text": text,
                    "type": node_type,
                    "evidence": node.get("evidence", ""),
                    "turn_id": node.get("turn_id", ""),
                }
                unique_nodes.append(new_node)
                seen_keys[key] = new_id
                id_map[node["id"]] = new_id
                _counter[0] += 1
        return id_map

    map1 = process_nodes(kg1.get("nodes", []))
    map2 = process_nodes(kg2.get("nodes", []))
    # Agent 2 may reference Agent 1 node IDs in its edges, so use the combined map
    combined_map = {**map1, **map2}

    # --- Remap edge IDs and deduplicate edges ---
    seen_edge_keys: set[tuple] = set()
    unique_edges: list[dict] = []

    def process_edges(edges: list, id_map: dict) -> None:
        for edge in edges:
            src = id_map.get(edge.get("source_id", ""))
            tgt = id_map.get(edge.get("target_id", ""))
            etype = edge.get("type", "").upper()
            if not src or not tgt:
                continue
            if etype not in VALID_EDGE_TYPES:
                continue
            key = (src, tgt, etype)
            if key in seen_edge_keys:
                continue
            seen_edge_keys.add(key)
            unique_edges.append({
                "source_id": src,
                "target_id": tgt,
                "type": etype,
                "evidence": edge.get("evidence", ""),
                "turn_id": edge.get("turn_id", ""),
            })

    process_edges(kg1.get("edges", []), map1)
    process_edges(kg2.get("edges", []), combined_map)  # combined so Agent 2 can ref Agent 1 nodes

    return {"nodes": unique_nodes, "edges": unique_edges}


def validate_kg(kg: dict) -> dict:
    """Final safety check: remove edges referencing non-existent nodes."""
    node_ids = {n["id"] for n in kg.get("nodes", [])}
    valid_edges = [
        e for e in kg.get("edges", [])
        if e["source_id"] in node_ids and e["target_id"] in node_ids
    ]
    removed = len(kg.get("edges", [])) - len(valid_edges)
    if removed:
        print(f"    Removed {removed} dangling edges", flush=True)
    return {"nodes": kg["nodes"], "edges": valid_edges}


# ==================== PIPELINE ====================

def format_existing_entities(kg: dict) -> str:
    """Format Agent 1's nodes as a readable list for the gap-finding prompt."""
    lines = []
    for node in kg.get("nodes", []):
        lines.append(f"  {node['id']}: [{node['type']}] {node['text']!r} (turn {node.get('turn_id', '?')})")
    return "\n".join(lines) if lines else "(none)"


def extract_for_transcript(
    transcript: str,
    agent1_client: OpenRouterClient,
    agent2_client: OpenRouterClient,
    api_key: str,
    skip_agent2: bool = False,
) -> tuple[dict | None, dict]:
    """
    Run the full 3-stage pipeline for a single transcript.
    Returns (merged_kg, usage_stats).
    """
    usage = {"agent1": None, "agent2": None}

    # --- Agent 1: Primary deep extraction ---
    print(f"    Agent 1 ({agent1_client.model.split('/')[-1]})...", end=" ", flush=True)
    prompt1 = AGENT1_PROMPT.format(transcript=transcript)
    content1, usage["agent1"] = agent1_client.generate(prompt1, system=AGENT1_SYSTEM)

    kg1 = None
    if content1:
        kg1 = extract_json(content1)

    # Fallback to qwen3-14b if primary model returns bad JSON
    if not kg1 or "nodes" not in kg1:
        if content1:
            print(f"WARN (bad JSON, falling back to {AGENT1_FALLBACK_MODEL.split('/')[-1]})...", end=" ", flush=True)
        else:
            print(f"WARN (no response, falling back to {AGENT1_FALLBACK_MODEL.split('/')[-1]})...", end=" ", flush=True)
        fallback_client = OpenRouterClient(api_key, AGENT1_FALLBACK_MODEL)
        content1, usage["agent1"] = fallback_client.generate(prompt1, system=AGENT1_SYSTEM)
        if content1:
            kg1 = extract_json(content1)

    if not kg1 or "nodes" not in kg1:
        print("FAILED")
        return None, usage

    n1, e1 = len(kg1.get("nodes", [])), len(kg1.get("edges", []))
    print(f"OK ({n1}n/{e1}e)", flush=True)

    if skip_agent2:
        kg1 = validate_kg(kg1)
        return kg1, usage

    # --- Agent 2: Gap finding ---
    next_id = n1 + 1
    existing_text = format_existing_entities(kg1)
    print("    Agent 2 (qwen3-14b)...", end=" ", flush=True)
    prompt2 = AGENT2_PROMPT.format(
        transcript=transcript,
        existing_entities=existing_text,
        start_id=next_id,
    )
    content2, usage["agent2"] = agent2_client.generate(prompt2)

    kg2: dict = {"nodes": [], "edges": []}
    if content2:
        parsed2 = extract_json(content2)
        if parsed2 and "nodes" in parsed2:
            kg2 = parsed2
            n2, e2 = len(kg2.get("nodes", [])), len(kg2.get("edges", []))
            print(f"OK (+{n2}n/+{e2}e)", flush=True)
        else:
            print("WARN (invalid JSON — using Agent 1 only)", flush=True)
    else:
        print("WARN (no response — using Agent 1 only)", flush=True)

    # --- Merge ---
    merged = merge_kgs(kg1, kg2)
    merged = validate_kg(merged)

    return merged, usage


def get_transcript_files(res_ids: list[str] | None = None) -> list[Path]:
    """Return sorted list of transcript .txt files, optionally filtered by res_ids."""
    files = []
    for res_dir in sorted(TRANSCRIPT_DIR.glob("RES*")):
        if not res_dir.is_dir():
            continue
        if res_ids and res_dir.name not in res_ids:
            continue
        txt_file = res_dir / f"{res_dir.name}.txt"
        if txt_file.exists():
            files.append(txt_file)
    return files


def load_api_key(cli_key: str | None) -> str:
    """Resolve API key: CLI arg > env var > api_keys.json."""
    if cli_key:
        return cli_key

    env_key = os.environ.get("OPENROUTER_API_KEY", "")
    if env_key:
        return env_key

    # Try api_keys.json in the sibling Clinical_KG_OS_LLM directory
    candidates = [
        Path(__file__).parent.parent / "Clinical_KG_OS_LLM" / "api_keys.json",
        Path(__file__).parent / "api_keys.json",
    ]
    for path in candidates:
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            key = data.get("openrouter", "")
            if key and not key.startswith("sk-or-v1-your"):
                print(f"  Using API key from {path}")
                return key

    raise SystemExit(
        "No OpenRouter API key found.\n"
        "Provide it via --api-key, OPENROUTER_API_KEY env var, or api_keys.json."
    )


# ==================== MAIN ====================

def main():
    parser = argparse.ArgumentParser(description="Multi-Agent KG Extraction Pipeline")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for per-patient KG JSON files")
    parser.add_argument("--res-ids", nargs="+", default=None,
                        help="Process only these patient IDs (e.g. RES0198 RES0199)")
    parser.add_argument("--api-key", type=str, default=None,
                        help="OpenRouter API key (overrides env var and api_keys.json)")
    parser.add_argument("--skip-agent2", action="store_true",
                        help="Run only Agent 1 (faster, lower quality)")
    parser.add_argument("--agent1-model", type=str, default=AGENT1_MODEL,
                        help=f"Agent 1 model (default: {AGENT1_MODEL})")
    parser.add_argument("--agent2-model", type=str, default=AGENT2_MODEL,
                        help=f"Agent 2 model (default: {AGENT2_MODEL})")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    api_key = load_api_key(args.api_key)

    agent1_client = OpenRouterClient(api_key, args.agent1_model)
    agent2_client = OpenRouterClient(api_key, args.agent2_model)

    transcript_files = get_transcript_files(args.res_ids)
    if not transcript_files:
        raise SystemExit(f"No transcripts found in {TRANSCRIPT_DIR}")

    print("=" * 65)
    print("Multi-Agent KG Extraction Pipeline")
    print(f"  Agent 1: {args.agent1_model}")
    if not args.skip_agent2:
        print(f"  Agent 2: {args.agent2_model}")
    else:
        print("  Agent 2: SKIPPED")
    print(f"  Output:  {output_dir}")
    print(f"  Transcripts: {len(transcript_files)}")
    print("=" * 65)

    success = failed = skipped = 0
    total_tokens = {"prompt": 0, "completion": 0}
    all_stats = []

    for txt_path in transcript_files:
        res_id = txt_path.stem
        output_file = output_dir / f"{res_id}_{OUTPUT_SUFFIX}.json"

        if output_file.exists():
            print(f"  {res_id}: SKIP (already exists)")
            skipped += 1
            continue

        print(f"  {res_id}:")
        transcript = txt_path.read_text(encoding="utf-8")

        kg, usage = extract_for_transcript(
            transcript, agent1_client, agent2_client,
            api_key=api_key, skip_agent2=args.skip_agent2,
        )

        if not kg:
            print(f"    => FAILED")
            failed += 1
            time.sleep(1)
            continue

        n, e = len(kg.get("nodes", [])), len(kg.get("edges", []))

        # Accumulate token counts
        for agent_usage in usage.values():
            if agent_usage:
                total_tokens["prompt"] += agent_usage.get("prompt_tokens", 0)
                total_tokens["completion"] += agent_usage.get("completion_tokens", 0)

        # Attach metadata
        kg["_meta"] = {
            "method": "multi_agent",
            "agents": {
                "agent1": args.agent1_model,
                "agent2": args.agent2_model if not args.skip_agent2 else None,
            },
            "pass1": {"nodes": n, "edges": e},  # post-merge totals
            "usage": usage,
        }

        output_file.write_text(json.dumps(kg, indent=2, ensure_ascii=False))
        print(f"    => OK ({n} nodes, {e} edges)", flush=True)

        all_stats.append({"res_id": res_id, "nodes": n, "edges": e, "usage": usage})
        success += 1
        time.sleep(0.5)  # polite rate-limit pause

    # Save run stats
    stats_file = output_dir / "_stats.json"
    stats_file.write_text(json.dumps({
        "method": "multi_agent",
        "agent1_model": args.agent1_model,
        "agent2_model": args.agent2_model if not args.skip_agent2 else None,
        "success": success,
        "failed": failed,
        "skipped": skipped,
        "total_tokens": total_tokens,
        "details": all_stats,
    }, indent=2))

    print("=" * 65)
    print(f"Done.  Success: {success}  Failed: {failed}  Skipped: {skipped}")
    print(f"Total tokens used: {total_tokens['prompt'] + total_tokens['completion']:,}")
    print(f"Output directory: {output_dir}/")
    print()
    print("Next step — merge into unified graph:")
    print(f"  cd ../Clinical_KG_OS_LLM")
    print(f"  uv run python -m Clinical_KG_OS_LLM.dump_graph \\")
    print(f"    --input {output_dir.resolve()} \\")
    print(f"    --output ./my_kg_multi_unified")
    print()
    print("Then score:")
    print(f"  uv run python -m Clinical_KG_OS_LLM.kg_similarity_scorer \\")
    print(f"    --student ./my_kg_multi_unified/unified_graph_my_kg_multi_unified.json \\")
    print(f"    --baseline ./data/human_curated/unified_graph_curated.json")


if __name__ == "__main__":
    main()
