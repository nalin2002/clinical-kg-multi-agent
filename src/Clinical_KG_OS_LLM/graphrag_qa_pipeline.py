"""
GraphRAG-QA Fixed Evaluation Pipeline
======================================
Accepts any unified KG in minimal interchange format, runs GraphRAG QA
on all gold questions, saves results per patient.

Usage:
    python graphrag_qa_pipeline.py --kg path/to/big_kg.json
    python graphrag_qa_pipeline.py --kg big_kg.json --bundle path/to/evaluation_bundle/
    python graphrag_qa_pipeline.py --kg big_kg.json --res-ids RES0198 --questions Q1 Q2

    Default bundle directory is repo ``data/transcripts`` (see ``Clinical_KG_OS_LLM.paths.transcripts_dir``).

Output:
    results_<kg_stem>/
    ├── RES0198/
    │   └── RES0198_answers.json   (7 QAs)
    └── ...

Minimal Interchange Format:
{
  "nodes": [
    {"id": "E_0001", "text": "nasal congestion",
     "occurrences": [{"res_id": "RES0198", "turn_id": "P-1"}]}
  ],
  "edges": [
    {"source_id": "E_0001", "target_id": "E_0004",
     "res_id": "RES0198", "turn_id": "P-6"}
  ]
}

Required node fields : id, text, occurrences (list of {res_id, turn_id})
Required edge fields : source_id, target_id, res_id, turn_id
Optional everywhere  : type, evidence
"""

import json
import re
import argparse
import time
import requests
import numpy as np
from pathlib import Path
from collections import defaultdict

from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from Clinical_KG_OS_LLM.paths import transcripts_dir

# ── Defaults ──────────────────────────────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "z-ai/glm-4.7-flash"
LLM_MODEL = "glm-4.7-flash"
OPENROUTER_TIMEOUT = 120
DIRECT_API_TIMEOUT = 300
SIMILARITY_TOP_K = 10
EMBEDDINGS_CACHE_DIR = Path("./embeddings_cache")

# Load OpenRouter API key
API_KEYS_PATH = Path("api_keys.json")
OPENROUTER_KEY = None
if API_KEYS_PATH.exists():
    with open(API_KEYS_PATH) as f:
        OPENROUTER_KEY = json.load(f).get("openrouter")


# ── Helpers ───────────────────────────────────────────────────────────
def parse_transcript(txt_path: str) -> dict[str, str]:
    turn_dict = {}
    text = Path(txt_path).read_text(encoding="utf-8")
    for m in re.finditer(
        r"\[(D-\d+|P-\d+)\]\s*[DP]:\s*(.*?)(?=\n\[|\n*$)", text, re.DOTALL
    ):
        turn_dict[m.group(1)] = m.group(2).strip()
    return turn_dict


def turn_sort_key(tid):
    """Sort key for turn IDs. Handles 'D-3', '[D-3]', int, and other formats."""
    if isinstance(tid, int):
        return ("", tid)
    if tid is None:
        return ("", 0)

    # Convert to string and clean up brackets/whitespace
    tid_str = str(tid).strip().strip("[]")

    # Try to parse as "X-N" format (e.g., "D-3", "P-5")
    parts = tid_str.split("-")
    if len(parts) == 2:
        try:
            return (parts[0], int(parts[1]))
        except ValueError:
            pass

    # Try to extract just numbers
    import re
    nums = re.findall(r'\d+', tid_str)
    if nums:
        return ("", int(nums[0]))

    return ("", 0)


# ── KG loader + validator ─────────────────────────────────────────────
def load_kg(kg_path: str) -> dict:
    with open(kg_path, encoding="utf-8") as f:
        kg = json.load(f)

    nodes = kg.get("nodes", [])
    edges = kg.get("edges", [])

    # Validate minimal required fields
    for i, n in enumerate(nodes):
        if "id" not in n or "text" not in n:
            raise ValueError(f"Node {i} missing 'id' or 'text': {n}")
        if "occurrences" not in n or not n["occurrences"]:
            raise ValueError(f"Node {i} (id={n['id']}) missing 'occurrences'")
        for occ in n["occurrences"]:
            if "res_id" not in occ:
                raise ValueError(
                    f"Node {n['id']} occurrence missing 'res_id': {occ}"
                )

    for i, e in enumerate(edges):
        if "source_id" not in e or "target_id" not in e:
            raise ValueError(f"Edge {i} missing 'source_id' or 'target_id': {e}")
        if "res_id" not in e:
            raise ValueError(f"Edge {i} missing 'res_id': {e}")

    return kg


# ── Embedding Cache Functions ─────────────────────────────────────────
def load_or_compute_embeddings(kg: dict, kg_path: Path, embed_model) -> tuple:
    """Load cached embeddings or compute and cache them."""
    cache_path = EMBEDDINGS_CACHE_DIR / f"{kg_path.stem}.npz"
    texts = [n["text"] for n in kg["nodes"]]

    if cache_path.exists():
        print(f"  Loading cached embeddings from {cache_path}...")
        data = np.load(cache_path, allow_pickle=True)
        embeddings = data["embeddings"]
        cached_texts = list(data["texts"])
        # Verify cache matches current KG
        if cached_texts == texts:
            print(f"  Loaded {len(texts)} embeddings")
            return embeddings, texts
        else:
            print(f"  Cache mismatch, recomputing...")

    print(f"  Computing embeddings for {len(texts)} nodes...")
    embeddings = embed_model.get_text_embedding_batch(texts, show_progress=True)
    embeddings = np.array(embeddings)

    # Save cache
    EMBEDDINGS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, embeddings=embeddings, texts=texts)
    print(f"  Cached embeddings to {cache_path}")

    return embeddings, texts


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# ── Retrieve context scoped to one patient ────────────────────────────
def retrieve_context(
    question: str,
    res_id: str,
    turn_dict: dict[str, str],
    kg: dict,
    embeddings: np.ndarray,
    texts: list,
    embed_model,
) -> str:
    """
    Deterministic retrieval using precomputed embeddings.

    Patient-scoped retrieval: filter by patient first, then take top-k from that patient's entities.
    This ensures each patient gets relevant entities without being crowded out by other patients.

    Steps:
    1. Find all node indices for this patient
    2. Compute query embedding
    3. Calculate similarity only for patient-related nodes
    4. Take top-k most similar patient nodes
    5. Extract related triples and transcript segments from KG
    """
    # Build lookup tables
    node_by_text = {n["text"]: n for n in kg["nodes"]}
    node_by_id = {n["id"]: n for n in kg["nodes"]}

    # Step 1: Find indices of nodes that have occurrences for this patient
    patient_node_indices = set()
    for i, node in enumerate(kg["nodes"]):
        for occ in node.get("occurrences", []):
            if occ["res_id"] == res_id:
                patient_node_indices.add(i)
                break

    # Compute query embedding
    query_emb = np.array(embed_model.get_text_embedding(question))

    # Compute similarities ONLY with patient-relevant nodes
    similarities = []
    for i in patient_node_indices:
        sim = cosine_similarity(query_emb, embeddings[i])
        similarities.append((i, sim))

    # Sort by similarity (descending) and take top-k
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_indices = [idx for idx, sim in similarities[:SIMILARITY_TOP_K]]

    # Extract retrieved entities
    retrieved_entities = set()
    for idx in top_indices:
        retrieved_entities.add(str(texts[idx]))

    # Find KG triples and turn_ids for this patient based on retrieved entities
    kg_lines = []
    turn_ids = set()

    # Check node occurrences for retrieved entities
    for entity_text in retrieved_entities:
        node = node_by_text.get(entity_text)
        if node:
            for occ in node.get("occurrences", []):
                if occ["res_id"] == res_id:
                    if occ.get("turn_id"):
                        turn_ids.add(occ["turn_id"])

    # Check edges for retrieved entities
    for e in kg["edges"]:
        if e.get("res_id") != res_id:
            continue
        src_node = node_by_id.get(e["source_id"])
        tgt_node = node_by_id.get(e["target_id"])
        if not src_node or not tgt_node:
            continue
        src_text = src_node["text"]
        tgt_text = tgt_node["text"]
        # Include edge if either endpoint was retrieved
        if src_text in retrieved_entities or tgt_text in retrieved_entities:
            edge_type = e.get("type", "RELATED_TO")
            src_type = src_node.get("type", "ENTITY")
            tgt_type = tgt_node.get("type", "ENTITY")
            kg_lines.append(f"[{src_type}] {src_text} --[{edge_type}]--> [{tgt_type}] {tgt_text}")
            if e.get("turn_id"):
                turn_ids.add(e["turn_id"])

    # Pull transcript segments
    transcript_lines = []
    for tid in sorted(turn_ids, key=turn_sort_key):
        if tid in turn_dict:
            speaker = "Doctor" if tid.startswith("D") else "Patient"
            transcript_lines.append(f"[{tid}] {speaker}: {turn_dict[tid]}")

    kg_ctx = "\n".join(kg_lines) if kg_lines else "(No KG triples retrieved)"
    tx_ctx = (
        "\n".join(transcript_lines)
        if transcript_lines
        else "(No transcript segments)"
    )

    return (
        "=== Knowledge Graph Information ===\n"
        f"{kg_ctx}\n\n"
        "=== Relevant Transcript Segments ===\n"
        f"{tx_ctx}"
    )


# ── Generate answer ───────────────────────────────────────────────────
def build_prompt(question: str, context: str) -> str:
    return (
        "You are a clinical documentation expert. Based on the following clinical "
        "information extracted from a doctor-patient consultation, answer the question "
        "thoroughly and accurately. Reference specific turn IDs (e.g., [P-1], [D-39]) "
        "when citing evidence from the transcript.\n\n"
        f"{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


def generate_answer_openrouter(prompt: str) -> str:
    """OpenRouter API call (primary, fastest)."""
    if not OPENROUTER_KEY:
        raise ValueError("OpenRouter API key not found")

    resp = requests.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {OPENROUTER_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": OPENROUTER_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
        },
        timeout=OPENROUTER_TIMEOUT,
        stream=True,
    )
    resp.raise_for_status()

    # Streaming response
    content = ""
    for line in resp.iter_lines():
        if line:
            line = line.decode("utf-8")
            if line.startswith("data: "):
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    if "content" in delta:
                        content += delta["content"]
                except json.JSONDecodeError:
                    pass
    return content.strip()


def generate_answer_llama_index(prompt: str) -> str:
    """Try llama-index call (fallback 1)."""
    from llama_index.llms.ollama import Ollama
    llm = Ollama(model=LLM_MODEL, request_timeout=120.0)
    return str(llm.complete(prompt)).strip()


def generate_answer_direct_api(prompt: str) -> str:
    """Direct Ollama API call (fallback 2)."""
    resp = requests.post(
        OLLAMA_API_URL,
        json={"model": LLM_MODEL, "prompt": prompt, "stream": False},
        timeout=DIRECT_API_TIMEOUT
    )
    return resp.json().get("response", "").strip()


# ── Main ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="GraphRAG-QA Fixed Evaluation Pipeline"
    )
    parser.add_argument(
        "--kg", required=True, help="Path to unified KG (interchange format)"
    )
    parser.add_argument(
        "--bundle",
        default=None,
        help=(
            "Path to evaluation bundle directory "
            "(default: repository data/transcripts)"
        ),
    )
    parser.add_argument("--output", default=None, help="Output directory")
    parser.add_argument("--res-ids", nargs="+", default=None)
    parser.add_argument("--questions", nargs="+", default=None)
    args = parser.parse_args()

    kg_path = Path(args.kg)
    bundle = Path(args.bundle) if args.bundle else transcripts_dir()
    out_dir = Path(args.output) if args.output else kg_path.parent / f"results_{kg_path.stem}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== GraphRAG-QA Pipeline ===")
    print(f"  KG     : {kg_path}")
    print(f"  Bundle : {bundle}")
    print(f"  Output : {out_dir}")

    # ── Models ──
    print("\nLoading embedding model (BAAI/bge-m3)...")
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")

    # ── Load KG + embeddings ──
    print(f"\nLoading KG ({kg_path.name})...")
    kg = load_kg(str(kg_path))
    print(f"  {len(kg['nodes'])} nodes, {len(kg['edges'])} edges")

    # Extract all_res_ids from KG
    all_res_ids = set()
    for n in kg["nodes"]:
        for occ in n["occurrences"]:
            all_res_ids.add(occ["res_id"])
    all_res_ids = sorted(all_res_ids)

    # Load or compute embeddings (deterministic)
    embeddings, texts = load_or_compute_embeddings(kg, kg_path, embed_model)

    # ── Determine scope ──
    query_res_ids = args.res_ids if args.res_ids else all_res_ids

    # ── Run QA ──
    total_start = time.time()
    all_results = {}

    for res_id in query_res_ids:
        tx_path = bundle / res_id / f"{res_id}.txt"
        qa_path = bundle / res_id / f"{res_id}_standard_answer.json"

        if not tx_path.exists():
            print(f"\n  SKIP {res_id}: no transcript")
            continue
        if not qa_path.exists():
            print(f"\n  SKIP {res_id}: no QA file")
            continue

        turn_dict = parse_transcript(str(tx_path))
        with open(qa_path, encoding="utf-8") as f:
            questions = json.load(f)

        if args.questions:
            questions = [q for q in questions if q["id"] in args.questions]

        print(f"\n{'='*60}")
        print(f"  {res_id}  ({len(questions)} questions)")
        print(f"{'='*60}")

        # Resume: skip if already completed
        res_dir = out_dir / res_id
        out_path = res_dir / f"{res_id}_answers.json"
        if out_path.exists():
            with open(out_path, encoding="utf-8") as f:
                existing = json.load(f)
            if len(existing) == len(questions):
                print(f"  Already complete ({len(existing)} answers), skipping")
                all_results[res_id] = existing
                continue

        results = []
        for q in questions:
            qid, qtxt = q["id"], q["question"]
            print(f"\n  [{qid}] {qtxt[:80]}...", flush=True)

            # Retrieve context with error handling
            try:
                t0 = time.time()
                context = retrieve_context(qtxt, res_id, turn_dict, kg, embeddings, texts, embed_model)
                t_ret = time.time() - t0
                print(f"    retrieval  {t_ret:.1f}s", flush=True)
            except Exception as e:
                print(f"    RETRIEVAL ERROR ({type(e).__name__}), using minimal context...", flush=True)
                # Build minimal context from KG directly
                turn_ids = set()
                for n in kg["nodes"]:
                    for occ in n["occurrences"]:
                        if occ["res_id"] == res_id and occ.get("turn_id"):
                            turn_ids.add(occ["turn_id"])
                transcript_lines = [f"[{tid}] {'Doctor' if tid.startswith('D') else 'Patient'}: {turn_dict.get(tid, '')}"
                                    for tid in sorted(turn_ids, key=turn_sort_key) if tid in turn_dict]
                context = "=== Relevant Transcript Segments ===\n" + "\n".join(transcript_lines)
                t_ret = 0

            # Try OpenRouter first, then llama-index, then direct Ollama API
            prompt = build_prompt(qtxt, context)
            method_used = "openrouter"
            answer = None
            t_gen = 0

            # Priority 1: OpenRouter (fastest)
            if OPENROUTER_KEY:
                try:
                    t0 = time.time()
                    answer = generate_answer_openrouter(prompt)
                    t_gen = time.time() - t0
                    print(f"    generation {t_gen:.1f}s (openrouter)", flush=True)
                except Exception as e:
                    print(f"    OPENROUTER FAILED ({type(e).__name__}), trying llama-index...", flush=True)
                    answer = None

            # Fallback 1: llama-index
            if answer is None:
                method_used = "llama-index"
                try:
                    t0 = time.time()
                    answer = generate_answer_llama_index(prompt)
                    t_gen = time.time() - t0
                    print(f"    generation {t_gen:.1f}s (llama-index)", flush=True)
                except Exception as e:
                    print(f"    LLAMA-INDEX FAILED ({type(e).__name__}), trying direct API...", flush=True)
                    answer = None

            # Fallback 2: direct Ollama API
            if answer is None:
                method_used = "direct-api"
                try:
                    t0 = time.time()
                    answer = generate_answer_direct_api(prompt)
                    t_gen = time.time() - t0
                    print(f"    generation {t_gen:.1f}s (direct-api)", flush=True)
                except Exception as e2:
                    print(f"    ALL METHODS FAILED: {e2}", flush=True)
                    answer = f"[ERROR: all generation methods failed]"
                    t_gen = 0
                    method_used = "FAILED"

            # ── GUARD: Empty answer detection and retry ──
            if not answer or not answer.strip() or answer.startswith("[ERROR"):
                print(f"    ⚠ EMPTY/ERROR ANSWER DETECTED, retrying once...", flush=True)
                # Retry with primary method
                if OPENROUTER_KEY:
                    try:
                        t0 = time.time()
                        answer = generate_answer_openrouter(prompt)
                        t_gen = time.time() - t0
                        print(f"    retry generation {t_gen:.1f}s (openrouter)", flush=True)
                    except Exception as e:
                        print(f"    RETRY FAILED: {e}", flush=True)

                # If still empty, use explicit error message
                if not answer or not answer.strip():
                    answer = "[ERROR: Answer generation failed - empty response after retry]"
                    print(f"    ⚠ FINAL: Empty answer persists, marking as error", flush=True)

            print(f"    preview    {answer[:120]}...", flush=True)

            results.append({
                "id": qid,
                "type": q.get("type", ""),
                "question": qtxt,
                "answer": answer,
                "prompt": prompt,
                "retrieval_time_s": round(t_ret, 2),
                "generation_time_s": round(t_gen, 2),
                "model": LLM_MODEL,
                "method": method_used,
            })

        # Save after each patient (crash-safe)
        res_dir.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n  Saved → {out_path}", flush=True)
        all_results[res_id] = results

    # ── Summary ──
    total_time = time.time() - total_start
    total_qs = sum(len(r) for r in all_results.values())

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  KG         : {kg_path.name}")
    print(f"  Transcripts: {len(all_results)}")
    print(f"  Questions  : {total_qs}")
    print(f"  Total time : {total_time:.1f}s")

    for rid, results in all_results.items():
        avg_r = sum(r["retrieval_time_s"] for r in results) / len(results)
        avg_g = sum(r["generation_time_s"] for r in results) / len(results)
        print(f"    {rid}: avg retrieve {avg_r:.1f}s, avg generate {avg_g:.1f}s")

    # Save summary
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "kg": kg_path.name,
            "total_transcripts": len(all_results),
            "total_questions": total_qs,
            "total_time_s": round(total_time, 2),
            "per_patient": {
                rid: {
                    "num_questions": len(res),
                    "avg_retrieval_s": round(sum(r["retrieval_time_s"] for r in res) / len(res), 2),
                    "avg_generation_s": round(sum(r["generation_time_s"] for r in res) / len(res), 2),
                }
                for rid, res in all_results.items()
            },
        }, f, indent=2)
    print(f"\n  Summary → {summary_path}")


if __name__ == "__main__":
    main()
