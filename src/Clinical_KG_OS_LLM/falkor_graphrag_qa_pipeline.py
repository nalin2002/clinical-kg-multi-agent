#!/usr/bin/env python3
"""
FalkorDB-backed GraphRAG QA pipeline.

Runs the same downstream QA flow as ``graphrag_qa_pipeline.py``, but reads
uploaded graph content from FalkorDB instead of loading a local unified KG JSON.

Examples:
    PYTHONPATH=src python -m Clinical_KG_OS_LLM.falkor_graphrag_qa_pipeline \\
        --graph-name fawkes_training_kg --graph-ids 10834547:26896259 \\
        --question "What are the main diagnoses and treatments?"

    PYTHONPATH=src python -m Clinical_KG_OS_LLM.falkor_graphrag_qa_pipeline \\
        --graph-name fawkes_aci_eval --res-ids RES0200 --questions Q1 Q2

The Falkor graph is expected to have been created by
``src/scripts/upload_kg_to_falkor.py``.  The script accepts Falkor ``graph_id``
values directly, or clinical/evaluation IDs such as ``res_id`` when those were
stored as KGGraph properties during upload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from Clinical_KG_OS_LLM.paths import transcripts_dir
from scripts.upload_kg_to_falkor import (
    DEFAULT_GRAPH_NAME,
    FalkorConfig,
    FalkorUploader,
)


OLLAMA_API_URL = "http://localhost:11434/api/generate"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "z-ai/glm-4.7-flash"
LLM_MODEL = "glm-4.7-flash"
OPENROUTER_TIMEOUT = 120
DIRECT_API_TIMEOUT = 300
SIMILARITY_TOP_K = 10
EDGE_LIMIT = 40
MAX_SOURCE_CHARS = 6000
EMBEDDINGS_CACHE_DIR = Path("./falkor_embeddings_cache")
RES_ID_RE = re.compile(r"(RES(?:_D2N)?\d+)", re.IGNORECASE)


def load_openrouter_key() -> str | None:
    env_key = os.getenv("OPENROUTER_API_KEY")
    if env_key:
        return env_key
    api_keys_path = Path("api_keys.json")
    if not api_keys_path.exists():
        return None
    with api_keys_path.open(encoding="utf-8") as handle:
        return json.load(handle).get("openrouter")


OPENROUTER_KEY = load_openrouter_key()


@dataclass(frozen=True)
class GraphRef:
    graph_id: str
    res_id: str | None = None
    encounter_id: str | None = None
    source_name: str | None = None
    requested_id: str | None = None


@dataclass(frozen=True)
class FalkorNode:
    kg_node_id: str
    source_node_id: str
    text: str
    node_type: str
    turn_id: str | None = None


@dataclass(frozen=True)
class FalkorEdge:
    kg_edge_id: str | None
    source_kg_node_id: str
    source_text: str
    source_type: str
    relation: str
    target_kg_node_id: str
    target_text: str
    target_type: str
    turn_id: str | None = None
    evidence: str | None = None


@dataclass(frozen=True)
class FalkorGraph:
    ref: GraphRef
    transcript: str | None
    note: str | None
    input_block: str | None
    nodes: list[FalkorNode]
    edges: list[FalkorEdge]


def _rows(result: Any) -> list[Any]:
    """Normalize FalkorDB result objects and simple test doubles to row lists."""
    if result is None:
        return []
    if hasattr(result, "result_set"):
        return list(result.result_set)
    if hasattr(result, "data"):
        data = result.data
        return data() if callable(data) else list(data)
    if isinstance(result, list):
        return result
    return []


def _cell(row: Any, idx: int, key: str | None = None) -> Any:
    if isinstance(row, dict) and key is not None:
        return row.get(key)
    if isinstance(row, (list, tuple)) and idx < len(row):
        return row[idx]
    return None


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    text = str(value)
    return text if text else None


def _restore_text_property(value: Any) -> str | None:
    text = _string_or_none(value)
    if text is None:
        return None
    return text.replace("\\n", "\n")


def _one_line(value: str, max_len: int = 180) -> str:
    text = re.sub(r"\s+", " ", value).strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."


class FalkorGraphStore:
    def __init__(self, uploader: FalkorUploader):
        self.uploader = uploader

    def query(self, query: str, params: dict[str, Any] | None = None) -> list[Any]:
        return _rows(self.uploader.query(query, params=params))

    def list_graphs(self, limit: int | None = None) -> list[GraphRef]:
        limit_clause = f" LIMIT {int(limit)}" if limit is not None else ""
        rows = self.query(
            """
            MATCH (g:KGGraph)
            RETURN g.graph_id AS graph_id,
                   g.res_id AS res_id,
                   g.encounter_id AS encounter_id,
                   g.source_name AS source_name
            ORDER BY g.graph_id
            """
            + limit_clause
        )
        return [_graph_ref_from_row(row) for row in rows]

    def resolve_graph(self, identifier: str) -> GraphRef | None:
        rows = self.query(
            """
            MATCH (g:KGGraph)
            WHERE g.graph_id = $identifier
               OR g.res_id = $identifier
               OR g.encounter_id = $identifier
               OR g.source_name = $identifier
            RETURN g.graph_id AS graph_id,
                   g.res_id AS res_id,
                   g.encounter_id AS encounter_id,
                   g.source_name AS source_name
            ORDER BY g.graph_id
            LIMIT 1
            """,
            {"identifier": identifier},
        )
        if not rows:
            return None
        ref = _graph_ref_from_row(rows[0])
        return GraphRef(
            graph_id=ref.graph_id,
            res_id=ref.res_id,
            encounter_id=ref.encounter_id,
            source_name=ref.source_name,
            requested_id=identifier,
        )

    def load_graph(self, ref: GraphRef) -> FalkorGraph:
        meta_rows = self.query(
            """
            MATCH (g:KGGraph {graph_id: $graph_id})
            RETURN g.graph_id AS graph_id,
                   g.res_id AS res_id,
                   g.encounter_id AS encounter_id,
                   g.source_name AS source_name,
                   g.transcript AS transcript,
                   g.note AS note,
                   g.input_block AS input_block
            LIMIT 1
            """,
            {"graph_id": ref.graph_id},
        )
        if not meta_rows:
            raise ValueError(f"Falkor graph_id not found: {ref.graph_id}")

        meta = meta_rows[0]
        graph_ref = GraphRef(
            graph_id=_string_or_none(_cell(meta, 0, "graph_id")) or ref.graph_id,
            res_id=_string_or_none(_cell(meta, 1, "res_id")) or ref.res_id,
            encounter_id=_string_or_none(_cell(meta, 2, "encounter_id")) or ref.encounter_id,
            source_name=_string_or_none(_cell(meta, 3, "source_name")) or ref.source_name,
            requested_id=ref.requested_id,
        )
        nodes = self._load_nodes(ref.graph_id)
        edges = self._load_edges(ref.graph_id)
        return FalkorGraph(
            ref=graph_ref,
            transcript=_restore_text_property(_cell(meta, 4, "transcript")),
            note=_restore_text_property(_cell(meta, 5, "note")),
            input_block=_restore_text_property(_cell(meta, 6, "input_block")),
            nodes=nodes,
            edges=edges,
        )

    def _load_nodes(self, graph_id: str) -> list[FalkorNode]:
        rows = self.query(
            """
            MATCH (n:KGNode)
            WHERE n.graph_id = $graph_id
            RETURN n.kg_node_id AS kg_node_id,
                   n.source_node_id AS source_node_id,
                   n.text AS text,
                   n.node_type AS node_type,
                   n.turn_id AS turn_id
            ORDER BY n.source_node_id
            """,
            {"graph_id": graph_id},
        )
        nodes = []
        for row in rows:
            kg_node_id = _string_or_none(_cell(row, 0, "kg_node_id"))
            if not kg_node_id:
                continue
            source_node_id = _string_or_none(_cell(row, 1, "source_node_id")) or kg_node_id
            text = _string_or_none(_cell(row, 2, "text")) or source_node_id
            node_type = _string_or_none(_cell(row, 3, "node_type")) or "ENTITY"
            nodes.append(
                FalkorNode(
                    kg_node_id=kg_node_id,
                    source_node_id=source_node_id,
                    text=text,
                    node_type=node_type,
                    turn_id=_string_or_none(_cell(row, 4, "turn_id")),
                )
            )
        return nodes

    def _load_edges(self, graph_id: str) -> list[FalkorEdge]:
        rows = self.query(
            """
            MATCH (source:KGNode)-[r]->(target:KGNode)
            WHERE r.graph_id = $graph_id
              AND source.graph_id = $graph_id
              AND target.graph_id = $graph_id
            RETURN r.kg_edge_id AS kg_edge_id,
                   source.kg_node_id AS source_kg_node_id,
                   source.text AS source_text,
                   source.node_type AS source_type,
                   type(r) AS relation_type,
                   r.relation AS relation,
                   r.turn_id AS turn_id,
                   r.evidence AS evidence,
                   target.kg_node_id AS target_kg_node_id,
                   target.text AS target_text,
                   target.node_type AS target_type
            ORDER BY r.kg_edge_id
            """,
            {"graph_id": graph_id},
        )
        edges = []
        for row in rows:
            source_id = _string_or_none(_cell(row, 1, "source_kg_node_id"))
            target_id = _string_or_none(_cell(row, 8, "target_kg_node_id"))
            if not source_id or not target_id:
                continue
            relation = (
                _string_or_none(_cell(row, 5, "relation"))
                or _string_or_none(_cell(row, 4, "relation_type"))
                or "RELATED_TO"
            )
            edges.append(
                FalkorEdge(
                    kg_edge_id=_string_or_none(_cell(row, 0, "kg_edge_id")),
                    source_kg_node_id=source_id,
                    source_text=_string_or_none(_cell(row, 2, "source_text")) or source_id,
                    source_type=_string_or_none(_cell(row, 3, "source_type")) or "ENTITY",
                    relation=relation,
                    target_kg_node_id=target_id,
                    target_text=_string_or_none(_cell(row, 9, "target_text")) or target_id,
                    target_type=_string_or_none(_cell(row, 10, "target_type")) or "ENTITY",
                    turn_id=_string_or_none(_cell(row, 6, "turn_id")),
                    evidence=_string_or_none(_cell(row, 7, "evidence")),
                )
            )
        return edges


def _graph_ref_from_row(row: Any) -> GraphRef:
    graph_id = _string_or_none(_cell(row, 0, "graph_id"))
    if graph_id is None:
        raise ValueError(f"FalkorDB returned a KGGraph row without graph_id: {row}")
    return GraphRef(
        graph_id=graph_id,
        res_id=_string_or_none(_cell(row, 1, "res_id")),
        encounter_id=_string_or_none(_cell(row, 2, "encounter_id")),
        source_name=_string_or_none(_cell(row, 3, "source_name")),
    )


def parse_transcript_text(text: str) -> dict[str, str]:
    turn_dict = {}
    for match in re.finditer(
        r"\[(D-\d+|P-\d+)\]\s*[DP]:\s*(.*?)(?=\n\[|\n*$)", text, re.DOTALL
    ):
        turn_dict[match.group(1)] = match.group(2).strip()
    return turn_dict


def parse_transcript(path: Path) -> dict[str, str]:
    return parse_transcript_text(path.read_text(encoding="utf-8"))


def turn_sort_key(turn_id: Any) -> tuple[str, int]:
    if isinstance(turn_id, int):
        return ("", turn_id)
    if turn_id is None:
        return ("", 0)
    tid = str(turn_id).strip().strip("[]")
    parts = tid.split("-")
    if len(parts) == 2:
        try:
            return (parts[0], int(parts[1]))
        except ValueError:
            pass
    nums = re.findall(r"\d+", tid)
    if nums:
        return ("", int(nums[0]))
    return ("", 0)


def extract_bundle_id(value: str | None) -> str | None:
    if not value:
        return None
    match = RES_ID_RE.search(value)
    return match.group(1) if match else None


def candidate_bundle_ids(ref: GraphRef) -> list[str]:
    candidates = [
        ref.requested_id,
        ref.res_id,
        ref.encounter_id,
        extract_bundle_id(ref.graph_id),
        extract_bundle_id(ref.source_name),
        ref.graph_id,
    ]
    out = []
    seen = set()
    for candidate in candidates:
        if candidate and candidate not in seen:
            out.append(candidate)
            seen.add(candidate)
    return out


def resolve_bundle_files(
    bundle: Path,
    ref: GraphRef,
) -> tuple[str | None, Path | None, Path | None]:
    for bundle_id in candidate_bundle_ids(ref):
        tx_path = bundle / bundle_id / f"{bundle_id}.txt"
        qa_path = bundle / bundle_id / f"{bundle_id}_standard_answer.json"
        if tx_path.exists() or qa_path.exists():
            return bundle_id, tx_path if tx_path.exists() else None, qa_path if qa_path.exists() else None
    return None, None, None


def normalize_questions(data: Any) -> list[dict[str, str]]:
    if isinstance(data, dict):
        data = data.get("questions", data.get("items", []))
    if not isinstance(data, list):
        raise ValueError("questions JSON must be a list or an object with a questions/items list")
    questions = []
    for idx, item in enumerate(data, start=1):
        if isinstance(item, str):
            questions.append({"id": f"Q{idx}", "question": item})
            continue
        if not isinstance(item, dict) or "question" not in item:
            raise ValueError(f"question item {idx} must be a string or object with 'question'")
        questions.append(
            {
                "id": str(item.get("id") or f"Q{idx}"),
                "type": str(item.get("type") or ""),
                "question": str(item["question"]),
            }
        )
    return questions


def load_questions(
    *,
    args: argparse.Namespace,
    qa_path: Path | None,
) -> list[dict[str, str]]:
    if args.question:
        return [
            {"id": f"Q{idx}", "type": "ad_hoc", "question": text}
            for idx, text in enumerate(args.question, start=1)
        ]
    if args.questions_json:
        return normalize_questions(json.loads(Path(args.questions_json).read_text(encoding="utf-8")))
    if qa_path is None:
        return []
    questions = normalize_questions(json.loads(qa_path.read_text(encoding="utf-8")))
    if args.questions:
        allowed = set(args.questions)
        questions = [question for question in questions if question["id"] in allowed]
    return questions


def load_turn_dict(graph: FalkorGraph, tx_path: Path | None) -> dict[str, str]:
    if tx_path is not None:
        return parse_transcript(tx_path)
    if graph.transcript:
        return parse_transcript_text(graph.transcript)
    return {}


def source_excerpt(graph: FalkorGraph, max_chars: int) -> str | None:
    for value in (graph.transcript, graph.note, graph.input_block):
        if value:
            return _one_line(value, max_chars)
    return None


def safe_path_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return safe or hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]


def cache_path_for_graph(graph_name: str, graph_id: str) -> Path:
    digest = hashlib.sha1(f"{graph_name}:{graph_id}".encode("utf-8")).hexdigest()[:16]
    return EMBEDDINGS_CACHE_DIR / f"{safe_path_name(graph_name)}_{digest}.npz"


def load_or_compute_embeddings(
    graph: FalkorGraph,
    *,
    graph_name: str,
    embed_model: Any,
) -> tuple[Any, list[str]]:
    import numpy as np

    texts = [node.text for node in graph.nodes]
    node_ids = [node.kg_node_id for node in graph.nodes]
    cache_path = cache_path_for_graph(graph_name, graph.ref.graph_id)
    if cache_path.exists():
        try:
            data = np.load(cache_path, allow_pickle=True)
            if list(data["texts"]) == texts and list(data["node_ids"]) == node_ids:
                return data["embeddings"], texts
        except (KeyError, ValueError):
            pass

    if not texts:
        return np.array([]), []

    embeddings = np.array(
        embed_model.get_text_embedding_batch(texts, show_progress=True)
    )
    EMBEDDINGS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, embeddings=embeddings, texts=texts, node_ids=node_ids)
    return embeddings, texts


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    a_vals = [float(value) for value in a]
    b_vals = [float(value) for value in b]
    dot = sum(left * right for left, right in zip(a_vals, b_vals))
    denom = math.sqrt(sum(value * value for value in a_vals)) * math.sqrt(
        sum(value * value for value in b_vals)
    )
    if denom == 0:
        return 0.0
    return float(dot / denom)


def tokenize_for_retrieval(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[A-Za-z0-9]+", text.lower())
        if len(token) > 2
    }


def lexical_similarity(question: str, node: FalkorNode) -> float:
    query_tokens = tokenize_for_retrieval(question)
    node_tokens = tokenize_for_retrieval(f"{node.text} {node.node_type}")
    if not query_tokens or not node_tokens:
        return 0.0
    overlap = len(query_tokens & node_tokens)
    return overlap / math.sqrt(len(query_tokens) * len(node_tokens))


def score_nodes(
    *,
    question: str,
    graph: FalkorGraph,
    embeddings: Sequence[Sequence[float]] | None,
    embed_model: Any | None,
) -> list[tuple[int, float]]:
    if embeddings is not None and embed_model is not None:
        query_emb = embed_model.get_text_embedding(question)
        scored = [
            (idx, cosine_similarity(query_emb, embeddings[idx]))
            for idx in range(len(graph.nodes))
        ]
    else:
        scored = [
            (idx, lexical_similarity(question, node))
            for idx, node in enumerate(graph.nodes)
        ]
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored


def retrieve_context(
    *,
    question: str,
    graph: FalkorGraph,
    embeddings: Sequence[Sequence[float]] | None,
    embed_model: Any | None,
    turn_dict: dict[str, str],
    top_k: int,
    edge_limit: int,
    max_source_chars: int,
) -> str:
    if not graph.nodes:
        return (
            "=== Retrieved Entities ===\n"
            "(No KG nodes found)\n\n"
            "=== Knowledge Graph Relationships ===\n"
            "(No KG relationships found)\n\n"
            "=== Relevant Source Text ===\n"
            "(No transcript or note text available)"
        )

    node_by_id = {node.kg_node_id: node for node in graph.nodes}
    scored = score_nodes(
        question=question,
        graph=graph,
        embeddings=embeddings,
        embed_model=embed_model,
    )
    top = scored[:top_k]
    top_node_ids = {graph.nodes[idx].kg_node_id for idx, _ in top}

    turn_ids = set()
    entity_lines = []
    for idx, score in top:
        node = graph.nodes[idx]
        if node.turn_id:
            turn_ids.add(node.turn_id)
        suffix = f" (turn {node.turn_id})" if node.turn_id else ""
        entity_lines.append(
            f"[{node.node_type}] {node.text}{suffix} | score={score:.3f}"
        )

    edge_lines = []
    for edge in graph.edges:
        if (
            edge.source_kg_node_id not in top_node_ids
            and edge.target_kg_node_id not in top_node_ids
        ):
            continue
        if edge.turn_id:
            turn_ids.add(edge.turn_id)
        source = node_by_id.get(edge.source_kg_node_id)
        target = node_by_id.get(edge.target_kg_node_id)
        src_type = source.node_type if source else edge.source_type
        src_text = source.text if source else edge.source_text
        tgt_type = target.node_type if target else edge.target_type
        tgt_text = target.text if target else edge.target_text
        qualifiers = []
        if edge.turn_id:
            qualifiers.append(f"turn {edge.turn_id}")
        if edge.evidence:
            qualifiers.append(f"evidence: {_one_line(edge.evidence)}")
        qualifier_text = f" ({'; '.join(qualifiers)})" if qualifiers else ""
        edge_lines.append(
            f"[{src_type}] {src_text} --[{edge.relation}]--> "
            f"[{tgt_type}] {tgt_text}{qualifier_text}"
        )
        if len(edge_lines) >= edge_limit:
            break

    transcript_lines = []
    for turn_id in sorted(turn_ids, key=turn_sort_key):
        if turn_id not in turn_dict:
            continue
        speaker = "Doctor" if str(turn_id).startswith("D") else "Patient"
        transcript_lines.append(f"[{turn_id}] {speaker}: {turn_dict[turn_id]}")

    if transcript_lines:
        source_lines = transcript_lines
    else:
        excerpt = source_excerpt(graph, max_source_chars)
        source_lines = [excerpt] if excerpt else ["(No transcript or note text available)"]

    return (
        "=== Retrieved Entities ===\n"
        f"{chr(10).join(entity_lines) if entity_lines else '(No KG nodes retrieved)'}\n\n"
        "=== Knowledge Graph Relationships ===\n"
        f"{chr(10).join(edge_lines) if edge_lines else '(No KG relationships retrieved)'}\n\n"
        "=== Relevant Source Text ===\n"
        f"{chr(10).join(source_lines)}"
    )


def build_prompt(question: str, context: str) -> str:
    return (
        "You are a clinical documentation expert. Based on the following clinical "
        "information extracted from a doctor-patient consultation or admission graph, "
        "answer the question thoroughly and accurately. Reference turn IDs when they "
        "are present in the provided evidence.\n\n"
        f"{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


def generate_answer_openrouter(prompt: str) -> str:
    if not OPENROUTER_KEY:
        raise ValueError("OpenRouter API key not found")

    import requests

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

    content = ""
    for line in resp.iter_lines():
        if not line:
            continue
        line = line.decode("utf-8")
        if not line.startswith("data: "):
            continue
        data = line[6:]
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            continue
        delta = chunk.get("choices", [{}])[0].get("delta", {})
        if "content" in delta:
            content += delta["content"]
    return content.strip()


def generate_answer_llama_index(prompt: str) -> str:
    from llama_index.llms.ollama import Ollama

    llm = Ollama(model=LLM_MODEL, request_timeout=120.0)
    return str(llm.complete(prompt)).strip()


def generate_answer_direct_api(prompt: str) -> str:
    import requests

    resp = requests.post(
        OLLAMA_API_URL,
        json={"model": LLM_MODEL, "prompt": prompt, "stream": False},
        timeout=DIRECT_API_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json().get("response", "").strip()


def generate_answer(prompt: str) -> tuple[str, str, float]:
    attempts = []
    if OPENROUTER_KEY:
        attempts.append(("openrouter", generate_answer_openrouter))
    attempts.extend(
        [
            ("llama-index", generate_answer_llama_index),
            ("direct-api", generate_answer_direct_api),
        ]
    )
    last_error = None
    for method, func in attempts:
        try:
            start = time.time()
            answer = func(prompt)
            elapsed = time.time() - start
            if answer.strip():
                return answer, method, elapsed
        except Exception as exc:  # pragma: no cover - external service failure
            last_error = exc
    return f"[ERROR: answer generation failed: {last_error}]", "FAILED", 0.0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--graph-name",
        default=os.getenv("FALKORDB_GRAPH", DEFAULT_GRAPH_NAME),
        help=f"FalkorDB graph name (default: {DEFAULT_GRAPH_NAME})",
    )
    parser.add_argument(
        "--host",
        default=os.getenv("FALKORDB_HOST", "localhost"),
        help="FalkorDB host (default: localhost)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("FALKORDB_PORT", "6379")),
        help="FalkorDB port (default: 6379)",
    )
    parser.add_argument("--username", default=os.getenv("FALKORDB_USERNAME"))
    parser.add_argument("--password", default=os.getenv("FALKORDB_PASSWORD"))
    parser.add_argument(
        "--url",
        default=os.getenv("FALKORDB_URL"),
        help="Optional FalkorDB URL; overrides host/port/user/password",
    )
    parser.add_argument("--ssl", action="store_true", help="Use TLS for host/port mode")
    parser.add_argument("--timeout-ms", type=int, default=None)
    parser.add_argument("--connect-timeout-sec", type=float, default=10.0)
    parser.add_argument("--socket-timeout-sec", type=float, default=30.0)
    parser.add_argument(
        "--graph-ids",
        nargs="+",
        default=None,
        help="Falkor KGGraph.graph_id values to answer against",
    )
    parser.add_argument(
        "--res-ids",
        nargs="+",
        default=None,
        help="Clinical/evaluation IDs to resolve via KGGraph.res_id, encounter_id, or graph_id",
    )
    parser.add_argument(
        "--bundle",
        default=None,
        help="Evaluation bundle directory (default: repository data/transcripts)",
    )
    parser.add_argument("--output", default=None, help="Output directory")
    parser.add_argument(
        "--questions",
        nargs="+",
        default=None,
        help="Question IDs from the bundle QA file to run",
    )
    parser.add_argument(
        "--question",
        action="append",
        default=None,
        help="Ad-hoc question text. May be passed more than once.",
    )
    parser.add_argument(
        "--questions-json",
        default=None,
        help="JSON list of question strings or objects with id/question fields",
    )
    parser.add_argument("--limit", type=int, default=None, help="Max graphs when no IDs are passed")
    parser.add_argument("--top-k", type=int, default=SIMILARITY_TOP_K)
    parser.add_argument("--edge-limit", type=int, default=EDGE_LIMIT)
    parser.add_argument("--max-source-chars", type=int, default=MAX_SOURCE_CHARS)
    parser.add_argument(
        "--retriever",
        choices=("auto", "embedding", "lexical"),
        default="auto",
        help=(
            "Node ranking method. 'auto' uses BGE embeddings when llama_index is "
            "installed, otherwise lexical token overlap (default: auto)."
        ),
    )
    parser.add_argument(
        "--retrieve-only",
        action="store_true",
        help="Write retrieved context and prompt without calling an LLM",
    )
    return parser


def connect_store(args: argparse.Namespace) -> FalkorGraphStore:
    config = FalkorConfig(
        graph_name=args.graph_name,
        host=args.host,
        port=args.port,
        username=args.username,
        password=args.password,
        url=args.url,
        ssl=args.ssl,
        connect_timeout_sec=args.connect_timeout_sec,
        socket_timeout_sec=args.socket_timeout_sec,
        timeout_ms=args.timeout_ms,
    )
    return FalkorGraphStore(FalkorUploader.connect(config))


def load_embedding_model(args: argparse.Namespace) -> Any | None:
    if args.retriever == "lexical":
        print("\nUsing lexical retriever (no embedding model).")
        return None
    try:
        print("\nLoading embedding model (BAAI/bge-m3)...")
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        return HuggingFaceEmbedding(model_name="BAAI/bge-m3")
    except ModuleNotFoundError as exc:
        if args.retriever == "embedding":
            raise SystemExit(
                "Missing llama_index. Install it, or rerun with "
                "`--retriever lexical` for a dependency-light retrieval check."
            ) from exc
        print("\nllama_index is not installed; using lexical retriever.")
        return None


def resolve_graph_refs(store: FalkorGraphStore, args: argparse.Namespace) -> list[GraphRef]:
    identifiers = []
    if args.graph_ids:
        identifiers.extend(args.graph_ids)
    if args.res_ids:
        identifiers.extend(args.res_ids)
    if identifiers:
        refs = []
        for identifier in identifiers:
            ref = store.resolve_graph(identifier)
            if ref is None:
                print(f"  SKIP {identifier}: no matching KGGraph in FalkorDB")
                continue
            refs.append(ref)
        return refs
    return store.list_graphs(limit=args.limit)


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if args.top_k <= 0:
        raise SystemExit("--top-k must be positive")
    if args.edge_limit <= 0:
        raise SystemExit("--edge-limit must be positive")

    bundle = Path(args.bundle) if args.bundle else transcripts_dir()
    out_dir = Path(args.output) if args.output else Path(f"results_falkor_{args.graph_name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== FalkorDB GraphRAG-QA Pipeline ===")
    print(f"  Falkor graph : {args.graph_name}")
    print(f"  Bundle       : {bundle}")
    print(f"  Output       : {out_dir}")

    store = connect_store(args)
    refs = resolve_graph_refs(store, args)
    if not refs:
        raise SystemExit("No Falkor graphs selected")

    embed_model = load_embedding_model(args)

    total_start = time.time()
    all_results: dict[str, list[dict[str, Any]]] = {}

    for ref in refs:
        graph = store.load_graph(ref)
        bundle_id, tx_path, qa_path = resolve_bundle_files(bundle, graph.ref)
        output_id = bundle_id or graph.ref.res_id or graph.ref.encounter_id or graph.ref.graph_id
        output_slug = safe_path_name(output_id)
        questions = load_questions(args=args, qa_path=qa_path)

        if not questions:
            print(f"\n  SKIP {output_id}: no questions provided or found in bundle")
            continue

        print(f"\n{'=' * 60}")
        print(f"  {output_id}  graph_id={graph.ref.graph_id}  ({len(questions)} questions)")
        print(f"{'=' * 60}")
        print(f"  Nodes={len(graph.nodes)} Edges={len(graph.edges)}")

        turn_dict = load_turn_dict(graph, tx_path)
        embeddings = None
        if embed_model is not None:
            embeddings, _ = load_or_compute_embeddings(
                graph,
                graph_name=args.graph_name,
                embed_model=embed_model,
            )

        res_dir = out_dir / output_slug
        out_path = res_dir / f"{output_slug}_answers.json"
        if out_path.exists():
            existing = json.loads(out_path.read_text(encoding="utf-8"))
            if len(existing) == len(questions):
                print(f"  Already complete ({len(existing)} answers), skipping")
                all_results[output_id] = existing
                continue

        results = []
        for question in questions:
            qid = question["id"]
            qtxt = question["question"]
            print(f"\n  [{qid}] {qtxt[:80]}...", flush=True)

            start = time.time()
            context = retrieve_context(
                question=qtxt,
                graph=graph,
                embeddings=embeddings,
                embed_model=embed_model,
                turn_dict=turn_dict,
                top_k=args.top_k,
                edge_limit=args.edge_limit,
                max_source_chars=args.max_source_chars,
            )
            retrieval_time = time.time() - start
            print(f"    retrieval  {retrieval_time:.1f}s", flush=True)

            prompt = build_prompt(qtxt, context)
            if args.retrieve_only:
                answer = "[RETRIEVE_ONLY]"
                method = "retrieve-only"
                generation_time = 0.0
            else:
                answer, method, generation_time = generate_answer(prompt)
                print(f"    generation {generation_time:.1f}s ({method})", flush=True)

            print(f"    preview    {answer[:120]}...", flush=True)
            results.append(
                {
                    "id": qid,
                    "type": question.get("type", ""),
                    "question": qtxt,
                    "answer": answer,
                    "prompt": prompt,
                    "context": context,
                    "retrieval_time_s": round(retrieval_time, 2),
                    "generation_time_s": round(generation_time, 2),
                    "model": LLM_MODEL,
                    "method": method,
                    "graph_id": graph.ref.graph_id,
                    "bundle_id": bundle_id,
                }
            )

        res_dir.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n  Saved -> {out_path}", flush=True)
        all_results[output_id] = results

    total_time = time.time() - total_start
    total_qs = sum(len(results) for results in all_results.values())
    summary = {
        "falkor_graph": args.graph_name,
        "total_graphs": len(all_results),
        "total_questions": total_qs,
        "total_time_s": round(total_time, 2),
        "per_graph": {
            graph_id: {
                "num_questions": len(results),
                "avg_retrieval_s": round(
                    sum(item["retrieval_time_s"] for item in results) / len(results),
                    2,
                ),
                "avg_generation_s": round(
                    sum(item["generation_time_s"] for item in results) / len(results),
                    2,
                ),
            }
            for graph_id, results in all_results.items()
            if results
        },
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Falkor graph : {args.graph_name}")
    print(f"  Graphs       : {len(all_results)}")
    print(f"  Questions    : {total_qs}")
    print(f"  Total time   : {total_time:.1f}s")
    print(f"  Summary      : {summary_path}")


if __name__ == "__main__":
    main()
