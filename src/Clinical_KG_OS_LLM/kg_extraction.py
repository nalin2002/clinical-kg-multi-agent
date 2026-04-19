"""
Unified KG Extraction Pipeline
==============================
Extract clinical knowledge graphs from transcripts using a single-pass LLM call
(OpenRouter `z-ai/glm-4.7-flash`).

Usage:
    python kg_extraction.py --output baseline_naive/sub_kgs

After extraction, merge with:
    python dump_graph.py --input baseline_naive/sub_kgs --output baseline_naive/
"""

import json
import re
import argparse
import time
from pathlib import Path

from Clinical_KG_OS_LLM.paths import transcripts_dir

# === Configuration ===
TRANSCRIPT_DIR = transcripts_dir()
MAX_RETRIES = 3
OPENROUTER_MODEL = "z-ai/glm-4.7-flash"
OUTPUT_SUFFIX = "naive_glm"

# === Prompts ===
EXTRACTION_PROMPT = """Extract clinical knowledge graph from transcript.

## NODE TYPES:
- SYMPTOM: Patient-reported or observed symptoms (chest pain, shortness of breath)
- DIAGNOSIS: Active or suspected conditions (COPD exacerbation, pneumonia)
- TREATMENT: Medications, therapies, interventions (Aspirin, Metformin, DASH diet)
- PROCEDURE: Tests, exams, surgeries (ECG, stress test, CT angiography)
- LOCATION: Body parts and anatomical locations (chest, left arm, heart)
- MEDICAL_HISTORY: Pre-existing conditions, risk factors (diabetes, smoking)
- LAB_RESULT: Lab values and vital signs (A1C 7.2%, BP 148/90, BNP elevated)

## EDGE TYPES:
- CAUSES: Risk factor causes condition (smoking CAUSES heart disease)
- INDICATES: Symptom indicates diagnosis (chest pain INDICATES angina)
- LOCATED_AT: Symptom at body location (pain LOCATED_AT chest)
- RULES_OUT: Test rules out condition (ECG RULES_OUT arrhythmia)
- TAKEN_FOR: Treatment for condition (Aspirin TAKEN_FOR angina)
- CONFIRMS: Lab/test confirms diagnosis (elevated BNP CONFIRMS heart failure)

TRANSCRIPT:
{transcript}

## FORMAT REQUIREMENTS:
- Node IDs: "N_001", "N_002", etc.
- turn_id: String format "P-X" or "D-X" (P=Patient, D=Doctor, X=turn number)
  Example: "P-1", "D-39" (from [P-1], [D-39] in transcript)

Output JSON with nodes (id, text, type, evidence, turn_id) and edges (source_id, target_id, type, evidence, turn_id).
Output ONLY valid JSON."""


# === Model Client ===
class OpenRouterClient:
    """Client for OpenRouter API (GLM, etc.)"""

    def __init__(self, api_key: str, model: str = OPENROUTER_MODEL):
        from openai import OpenAI
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
        self.model = model

    def generate(self, prompt: str) -> tuple:
        """Generate response. Returns (content, usage_dict)."""
        for attempt in range(MAX_RETRIES):
            try:
                stream = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    stream=True
                )

                content = ""
                last_chunk = None
                for chunk in stream:
                    last_chunk = chunk
                    delta = chunk.choices[0].delta
                    if delta.content:
                        content += delta.content

                usage = None
                if last_chunk and hasattr(last_chunk, 'usage') and last_chunk.usage:
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


def get_client(api_keys: dict) -> OpenRouterClient:
    key = api_keys.get("openrouter")
    if not key:
        raise SystemExit("api_keys.json must contain a non-empty \"openrouter\" key")
    return OpenRouterClient(key, OPENROUTER_MODEL)


# === Utilities ===
def read_transcript(file_path: Path) -> str:
    with open(file_path, 'r') as f:
        return f.read()


def extract_json_from_response(response_text: str) -> dict:
    """Extract JSON from LLM response."""
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass

    if response_text.strip().startswith('```'):
        parts = response_text.split('```')
        if len(parts) >= 2:
            inner = parts[1]
            if inner.startswith('json'):
                inner = inner[4:]
            inner = inner.strip()
            try:
                return json.loads(inner)
            except json.JSONDecodeError:
                pass

    json_match = re.search(r'\{[\s\S]*\}', response_text)
    if json_match:
        json_str = json_match.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            fixed_json = re.sub(r',(\s*[}\]])', r'\1', json_str)
            try:
                return json.loads(fixed_json)
            except json.JSONDecodeError:
                pass

    return None


def validate_knowledge_graph(kg: dict) -> dict:
    """Validate and fix knowledge graph integrity."""
    if not kg or 'nodes' not in kg or 'edges' not in kg:
        return kg

    node_ids = {node['id'] for node in kg.get('nodes', [])}

    valid_edges = []
    invalid_count = 0

    for edge in kg.get('edges', []):
        source = edge.get('source_id')
        target = edge.get('target_id')
        if source in node_ids and target in node_ids:
            valid_edges.append(edge)
        else:
            invalid_count += 1

    if invalid_count > 0:
        print(f"    Removed {invalid_count} invalid edges")
        kg['edges'] = valid_edges

    return kg


def get_transcript_files():
    """Get all transcript files."""
    files = []
    for res_dir in sorted(TRANSCRIPT_DIR.glob("RES*")):
        if res_dir.is_dir():
            txt_file = res_dir / f"{res_dir.name}.txt"
            if txt_file.exists():
                files.append(txt_file)
    return files


def extract_naive(transcript: str, client: OpenRouterClient) -> tuple:
    """Single-pass naive extraction."""
    prompt = EXTRACTION_PROMPT.format(transcript=transcript)
    content, usage = client.generate(prompt)

    if content:
        kg = extract_json_from_response(content)
        if kg:
            kg = validate_knowledge_graph(kg)
        return kg, usage
    return None, usage


def process_one(txt_path: Path, client: OpenRouterClient, output_dir: Path, suffix: str) -> tuple:
    """Process single transcript."""
    res_id = txt_path.stem
    output_file = output_dir / f"{res_id}_{suffix}.json"

    # Skip if already exists
    if output_file.exists():
        return res_id, "SKIP", 0, 0, None

    try:
        transcript = read_transcript(txt_path)
        print(f"  {res_id}...", end=" ", flush=True)
        kg, usage = extract_naive(transcript, client)

        if not kg:
            print("FAILED")
            return res_id, "FAILED", 0, 0, None

        n, e = len(kg.get('nodes', [])), len(kg.get('edges', []))

        kg['_usage'] = usage

        with open(output_file, 'w') as f:
            json.dump(kg, f, indent=2, ensure_ascii=False)

        print(f"({n}n/{e}e)")
        return res_id, "OK", n, e, usage

    except Exception as ex:
        print(f"ERROR: {ex}")
        return res_id, f"ERROR: {ex}", 0, 0, None


def main():
    parser = argparse.ArgumentParser(description="KG Extraction Pipeline (naive, GLM via OpenRouter)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for sub-KG JSON files")
    parser.add_argument("--res-ids", nargs="+", default=None,
                        help="Optional list of patient IDs to process (e.g. RES0198 RES0199). Processes all if omitted.")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open("api_keys.json") as f:
        api_keys = json.load(f)

    client = get_client(api_keys)
    suffix = OUTPUT_SUFFIX

    transcript_files = get_transcript_files()
    if args.res_ids:
        transcript_files = [f for f in transcript_files if f.parent.name in args.res_ids]
    print("KG Extraction Pipeline")
    print(f"Method: naive ({OPENROUTER_MODEL})")
    print(f"Output: {output_dir}")
    print(f"Processing {len(transcript_files)} transcripts")
    print("=" * 60)

    success = 0
    failed = 0
    total_tokens = {"prompt": 0, "completion": 0}
    all_stats = []

    for txt_path in transcript_files:
        res_id, status, nodes, edges, usage = process_one(
            txt_path, client, output_dir, suffix
        )
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
        time.sleep(0.3)

    stats_file = output_dir / "_stats.json"
    with open(stats_file, 'w') as f:
        json.dump({
            "method": "naive",
            "model": OPENROUTER_MODEL,
            "total_tokens": total_tokens,
            "success": success,
            "failed": failed,
            "details": all_stats
        }, f, indent=2)

    print("=" * 60)
    print(f"Done! Success: {success}, Failed: {failed}")
    print(f"Total tokens: {total_tokens['prompt'] + total_tokens['completion']}")
    print(f"Output: {output_dir}/")


if __name__ == "__main__":
    main()
