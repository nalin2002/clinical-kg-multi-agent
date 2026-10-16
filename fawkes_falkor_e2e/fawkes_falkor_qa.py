#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "falkordb>=1.0.0",
#   "requests>=2.32.0",
# ]
# ///
"""FalkorDB-backed GraphRAG QA for Fawkes admission KGs.

This script is standalone so it can travel with ``fawkes_falkor_e2e``. It reads
the KG uploaded by ``fawkes_falkor_e2e.py`` from FalkorDB, retrieves relevant
nodes/edges/source note snippets, and answers ad-hoc clinical questions.

Examples:
  uv run fawkes_falkor_qa.py --list-graphs --limit 10

  uv run fawkes_falkor_qa.py \\
    --falkor-url redis://USER:PASSWORD@HOST:PORT \\
    --falkor-graph fawkes_clinical_kg \\
    --graph-ids 10000032:22595853 \\
    --question "What diagnoses and medications are documented?"

  uv run fawkes_falkor_qa.py \\
    --graph-ids 10000032:22595853 \\
    --questions-json questions.json \\
    --retrieve-only
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


DEFAULT_FALKOR_GRAPH = "fawkes_clinical_kg"
DEFAULT_QA_MODEL = "z-ai/glm-4.7-flash"
DEFAULT_OLLAMA_MODEL = "glm-4.7-flash"
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
TOP_K = 10
EDGE_LIMIT = 50
MAX_SOURCE_CHARS = 7000
NOTE_CHUNKS = 6


@dataclass(frozen=True)
class GraphRef:
    graph_id: str
    subject_id: str | None = None
    hadm_id: str | None = None
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
    confidence: float | None = None
    sentence: str | None = None
    quote: str | None = None


@dataclass(frozen=True)
class FalkorGraph:
    ref: GraphRef
    note: str | None
    input_block: str | None
    transcript: str | None
    nodes: list[FalkorNode]
    edges: list[FalkorEdge]


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def safe_path_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return safe or hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]


def _rows(result: Any) -> list[Any]:
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


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _restore_text_property(value: Any) -> str | None:
    text = _string_or_none(value)
    if text is None:
        return None
    return text.replace("\\n", "\n")


def _one_line(value: str, max_len: int = 220) -> str:
    text = re.sub(r"\s+", " ", value).strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."


class FalkorGraphStore:
    def __init__(self, graph: Any, timeout_ms: int | None = None):
        self.graph = graph
        self.timeout_ms = timeout_ms

    @classmethod
    def connect(cls, args: argparse.Namespace) -> "FalkorGraphStore":
        from falkordb import FalkorDB

        if args.falkor_url:
            db = FalkorDB.from_url(args.falkor_url)
        else:
            db = FalkorDB(
                host=args.falkor_host,
                port=args.falkor_port,
                username=args.falkor_username,
                password=args.falkor_password,
                ssl=args.falkor_ssl,
                socket_connect_timeout=args.connect_timeout_sec,
                socket_timeout=args.socket_timeout_sec,
            )
        return cls(db.select_graph(args.falkor_graph), timeout_ms=args.timeout_ms)

    def query(self, query: str, params: dict[str, Any] | None = None) -> list[Any]:
        return _rows(self.graph.query(query, params=dict(params or {}), timeout=self.timeout_ms))

    def list_graphs(self, limit: int | None = None) -> list[GraphRef]:
        limit_clause = f" LIMIT {int(limit)}" if limit is not None else ""
        rows = self.query(
            """
            MATCH (g:KGGraph)
            RETURN g.graph_id AS graph_id,
                   g.subject_id AS subject_id,
                   g.hadm_id AS hadm_id,
                   g.res_id AS res_id,
                   g.encounter_id AS encounter_id,
                   g.source_name AS source_name
            ORDER BY g.graph_id
            """
            + limit_clause
        )
        return [_graph_ref_from_row(row) for row in rows]

    def resolve_graph(self, identifier: str) -> GraphRef | None:
        numeric_identifier = _int_or_none(identifier)
        rows = self.query(
            """
            MATCH (g:KGGraph)
            WHERE g.graph_id = $identifier
               OR g.res_id = $identifier
               OR g.encounter_id = $identifier
               OR g.source_name = $identifier
               OR g.subject_id = $identifier
               OR g.hadm_id = $identifier
               OR g.subject_id = $numeric_identifier
               OR g.hadm_id = $numeric_identifier
            RETURN g.graph_id AS graph_id,
                   g.subject_id AS subject_id,
                   g.hadm_id AS hadm_id,
                   g.res_id AS res_id,
                   g.encounter_id AS encounter_id,
                   g.source_name AS source_name
            ORDER BY g.graph_id
            LIMIT 1
            """,
            {"identifier": identifier, "numeric_identifier": numeric_identifier},
        )
        if not rows:
            return None
        ref = _graph_ref_from_row(rows[0])
        return GraphRef(
            graph_id=ref.graph_id,
            subject_id=ref.subject_id,
            hadm_id=ref.hadm_id,
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
                   g.subject_id AS subject_id,
                   g.hadm_id AS hadm_id,
                   g.res_id AS res_id,
                   g.encounter_id AS encounter_id,
                   g.source_name AS source_name,
                   g.note AS note,
                   g.input_block AS input_block,
                   g.transcript AS transcript
            LIMIT 1
            """,
            {"graph_id": ref.graph_id},
        )
        if not meta_rows:
            raise RuntimeError(f"Falkor KGGraph not found: {ref.graph_id}")

        meta = meta_rows[0]
        graph_ref = GraphRef(
            graph_id=_string_or_none(_cell(meta, 0, "graph_id")) or ref.graph_id,
            subject_id=_string_or_none(_cell(meta, 1, "subject_id")) or ref.subject_id,
            hadm_id=_string_or_none(_cell(meta, 2, "hadm_id")) or ref.hadm_id,
            res_id=_string_or_none(_cell(meta, 3, "res_id")) or ref.res_id,
            encounter_id=_string_or_none(_cell(meta, 4, "encounter_id")) or ref.encounter_id,
            source_name=_string_or_none(_cell(meta, 5, "source_name")) or ref.source_name,
            requested_id=ref.requested_id,
        )
        return FalkorGraph(
            ref=graph_ref,
            note=_restore_text_property(_cell(meta, 6, "note")),
            input_block=_restore_text_property(_cell(meta, 7, "input_block")),
            transcript=_restore_text_property(_cell(meta, 8, "transcript")),
            nodes=self._load_nodes(ref.graph_id),
            edges=self._load_edges(ref.graph_id),
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
                   r.confidence AS confidence,
                   r.sentence AS sentence,
                   r.quote AS quote,
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
            target_id = _string_or_none(_cell(row, 11, "target_kg_node_id"))
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
                    target_text=_string_or_none(_cell(row, 12, "target_text")) or target_id,
                    target_type=_string_or_none(_cell(row, 13, "target_type")) or "ENTITY",
                    turn_id=_string_or_none(_cell(row, 6, "turn_id")),
                    evidence=_string_or_none(_cell(row, 7, "evidence")),
                    confidence=_float_or_none(_cell(row, 8, "confidence")),
                    sentence=_string_or_none(_cell(row, 9, "sentence")),
                    quote=_string_or_none(_cell(row, 10, "quote")),
                )
            )
        return edges


def _graph_ref_from_row(row: Any) -> GraphRef:
    graph_id = _string_or_none(_cell(row, 0, "graph_id"))
    if graph_id is None:
        raise RuntimeError(f"FalkorDB returned KGGraph row without graph_id: {row}")
    return GraphRef(
        graph_id=graph_id,
        subject_id=_string_or_none(_cell(row, 1, "subject_id")),
        hadm_id=_string_or_none(_cell(row, 2, "hadm_id")),
        res_id=_string_or_none(_cell(row, 3, "res_id")),
        encounter_id=_string_or_none(_cell(row, 4, "encounter_id")),
        source_name=_string_or_none(_cell(row, 5, "source_name")),
    )


def _int_or_none(value: Any) -> int | None:
    try:
        text = str(value).strip()
        if not re.fullmatch(r"-?\d+", text):
            return None
        return int(text)
    except (TypeError, ValueError):
        return None


def normalize_questions(data: Any) -> list[dict[str, str]]:
    if isinstance(data, dict):
        data = data.get("questions", data.get("items", []))
    if not isinstance(data, list):
        raise ValueError("questions JSON must be a list or an object with questions/items")
    questions = []
    for idx, item in enumerate(data, start=1):
        if isinstance(item, str):
            questions.append({"id": f"Q{idx}", "type": "ad_hoc", "question": item})
            continue
        if not isinstance(item, dict) or "question" not in item:
            raise ValueError(f"question item {idx} must be a string or object with question")
        questions.append(
            {
                "id": str(item.get("id") or f"Q{idx}"),
                "type": str(item.get("type") or "ad_hoc"),
                "question": str(item["question"]),
            }
        )
    return questions


def load_questions(args: argparse.Namespace) -> list[dict[str, str]]:
    questions = []
    if args.question:
        questions.extend(
            {"id": f"Q{idx}", "type": "ad_hoc", "question": text}
            for idx, text in enumerate(args.question, start=1)
        )
    if args.questions_json:
        questions.extend(
            normalize_questions(json.loads(Path(args.questions_json).read_text(encoding="utf-8")))
        )
    return questions


def parse_transcript_text(text: str) -> dict[str, str]:
    turns = {}
    for match in re.finditer(
        r"\[(D-\d+|P-\d+)\]\s*[DP]:\s*(.*?)(?=\n\[|\n*$)", text, re.DOTALL
    ):
        turns[match.group(1)] = match.group(2).strip()
    return turns


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


def tokenize(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[A-Za-z0-9]+", text.lower())
        if len(token) > 2
    }


def lexical_score(query: str, text: str) -> float:
    query_tokens = tokenize(query)
    text_tokens = tokenize(text)
    if not query_tokens or not text_tokens:
        return 0.0
    return len(query_tokens & text_tokens) / math.sqrt(len(query_tokens) * len(text_tokens))


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    dot = sum(float(a) * float(b) for a, b in zip(left, right))
    denom = math.sqrt(sum(float(v) ** 2 for v in left)) * math.sqrt(
        sum(float(v) ** 2 for v in right)
    )
    return float(dot / denom) if denom else 0.0


class EmbeddingRanker:
    def __init__(self, model_name: str, cache_dir: Path):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.cache_dir = cache_dir
        self.model = SentenceTransformer(model_name)

    def load_or_compute(self, graph: FalkorGraph) -> list[list[float]]:
        texts = [f"{node.node_type}: {node.text}" for node in graph.nodes]
        node_ids = [node.kg_node_id for node in graph.nodes]
        cache_path = self.cache_dir / f"{safe_path_name(graph.ref.graph_id)}.json"
        if cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("model") == self.model_name and cached.get("node_ids") == node_ids:
                return cached["embeddings"]
        embeddings = self.model.encode(texts, normalize_embeddings=True).tolist()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(
                {
                    "model": self.model_name,
                    "graph_id": graph.ref.graph_id,
                    "node_ids": node_ids,
                    "texts": texts,
                    "embeddings": embeddings,
                }
            ),
            encoding="utf-8",
        )
        return embeddings

    def score(self, question: str, graph: FalkorGraph, embeddings: list[list[float]]) -> list[tuple[int, float]]:
        query_embedding = self.model.encode([question], normalize_embeddings=True)[0].tolist()
        scored = [
            (idx, cosine_similarity(query_embedding, embeddings[idx]))
            for idx in range(len(graph.nodes))
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored


def load_embedding_ranker(args: argparse.Namespace) -> EmbeddingRanker | None:
    if args.retriever == "lexical":
        return None
    try:
        return EmbeddingRanker(
            model_name=args.embedding_model,
            cache_dir=Path(args.embedding_cache_dir),
        )
    except ModuleNotFoundError as exc:
        if args.retriever == "embedding":
            raise SystemExit(
                "Embedding retrieval requires sentence-transformers. Install it "
                "or run with --retriever lexical."
            ) from exc
        print("sentence-transformers not installed; using lexical retriever.")
        return None


def score_nodes(
    question: str,
    graph: FalkorGraph,
    ranker: EmbeddingRanker | None,
    embeddings: list[list[float]] | None,
) -> list[tuple[int, float]]:
    if ranker is not None and embeddings is not None:
        return ranker.score(question, graph, embeddings)
    scored = [
        (idx, lexical_score(question, f"{node.node_type} {node.text}"))
        for idx, node in enumerate(graph.nodes)
    ]
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored


def source_text(graph: FalkorGraph) -> str:
    return "\n\n".join(
        value.strip()
        for value in (graph.note, graph.input_block, graph.transcript)
        if value and value.strip()
    )


def source_chunks(text: str, max_chunk_chars: int = 900) -> list[str]:
    parts = [
        part.strip()
        for part in re.split(r"(?=\[\d+\]\s)|(?<=[.!?])\s+(?=[A-Z\[])", text)
        if part.strip()
    ]
    chunks = []
    current = ""
    for part in parts:
        if not current:
            current = part
        elif len(current) + len(part) + 1 <= max_chunk_chars:
            current = f"{current} {part}"
        else:
            chunks.append(current)
            current = part
    if current:
        chunks.append(current)
    if chunks:
        return chunks
    return [text[:max_chunk_chars]] if text else []


def select_source_snippets(
    question: str,
    graph: FalkorGraph,
    turn_dict: dict[str, str],
    turn_ids: Iterable[str],
    max_chars: int,
) -> list[str]:
    transcript_lines = []
    for turn_id in sorted(set(turn_ids), key=turn_sort_key):
        if turn_id not in turn_dict:
            continue
        speaker = "Doctor" if str(turn_id).startswith("D") else "Patient"
        transcript_lines.append(f"[{turn_id}] {speaker}: {turn_dict[turn_id]}")
    if transcript_lines:
        return _cap_lines(transcript_lines, max_chars)

    text = source_text(graph)
    if not text:
        return ["(No note, input_block, or transcript text available)"]
    chunks = source_chunks(text)
    scored = [(chunk, lexical_score(question, chunk)) for chunk in chunks]
    scored.sort(key=lambda item: item[1], reverse=True)
    selected = [chunk for chunk, score in scored[:NOTE_CHUNKS] if score > 0]
    if not selected:
        selected = chunks[:NOTE_CHUNKS]
    return _cap_lines([_one_line(chunk, max_chars) for chunk in selected], max_chars)


def _cap_lines(lines: list[str], max_chars: int) -> list[str]:
    capped = []
    used = 0
    for line in lines:
        add = len(line) + 1
        if capped and used + add > max_chars:
            break
        capped.append(line[: max(0, max_chars - used)])
        used += add
        if used >= max_chars:
            break
    return capped


def retrieve_context(
    *,
    question: str,
    graph: FalkorGraph,
    ranker: EmbeddingRanker | None,
    embeddings: list[list[float]] | None,
    top_k: int,
    edge_limit: int,
    max_source_chars: int,
) -> str:
    turn_dict = parse_transcript_text(graph.transcript or "")
    node_by_id = {node.kg_node_id: node for node in graph.nodes}
    scored = score_nodes(question, graph, ranker, embeddings)
    top = scored[:top_k]
    top_node_ids = {graph.nodes[idx].kg_node_id for idx, _ in top}

    turn_ids = set()
    entity_lines = []
    for idx, score in top:
        node = graph.nodes[idx]
        if node.turn_id:
            turn_ids.add(node.turn_id)
        suffix = f" turn={node.turn_id}" if node.turn_id else ""
        entity_lines.append(f"[{node.node_type}] {node.text}{suffix} score={score:.3f}")

    edge_lines = []
    for edge in graph.edges:
        if edge.source_kg_node_id not in top_node_ids and edge.target_kg_node_id not in top_node_ids:
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
        if edge.confidence is not None:
            qualifiers.append(f"confidence={edge.confidence:.2f}")
        if edge.evidence:
            qualifiers.append(f"evidence={edge.evidence}")
        if edge.sentence:
            qualifiers.append(f"sentence={edge.sentence}")
        if edge.quote:
            qualifiers.append(f"quote={_one_line(edge.quote)}")
        suffix = f" ({'; '.join(qualifiers)})" if qualifiers else ""
        edge_lines.append(
            f"[{src_type}] {src_text} --[{edge.relation}]--> [{tgt_type}] {tgt_text}{suffix}"
        )
        if len(edge_lines) >= edge_limit:
            break

    source_lines = select_source_snippets(
        question,
        graph,
        turn_dict,
        turn_ids,
        max_chars=max_source_chars,
    )
    meta_lines = [
        f"graph_id: {graph.ref.graph_id}",
        f"subject_id: {graph.ref.subject_id or ''}",
        f"hadm_id: {graph.ref.hadm_id or ''}",
    ]
    return (
        "=== Graph Metadata ===\n"
        f"{chr(10).join(meta_lines)}\n\n"
        "=== Retrieved Entities ===\n"
        f"{chr(10).join(entity_lines) if entity_lines else '(No KG nodes retrieved)'}\n\n"
        "=== Knowledge Graph Relationships ===\n"
        f"{chr(10).join(edge_lines) if edge_lines else '(No KG relationships retrieved)'}\n\n"
        "=== Relevant Source Text ===\n"
        f"{chr(10).join(source_lines)}"
    )


def build_prompt(question: str, context: str) -> str:
    return (
        "You are a clinical documentation expert answering questions from a "
        "patient admission knowledge graph. Use only the provided graph and "
        "source-text context. If the answer is not supported, say that the "
        "provided evidence is insufficient. Be concise but clinically precise. "
        "Cite graph entities, relationships, note snippets, or turn IDs when "
        "available.\n\n"
        f"{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


def endpoint_chat_completion(args: argparse.Namespace, prompt: str) -> str:
    import requests

    token = os.getenv(args.qa_endpoint_token_env) or os.getenv("HF_TOKEN") or ""
    if not args.qa_endpoint_url or not token:
        raise RuntimeError("QA endpoint URL or token is missing")
    base_url = args.qa_endpoint_url.rstrip("/")
    if base_url.endswith("/v1/chat/completions"):
        url = base_url
    elif base_url.endswith("/v1"):
        url = base_url + "/chat/completions"
    else:
        url = base_url + "/v1/chat/completions"
    response = requests.post(
        url,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json={
            "model": args.qa_model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": args.qa_max_tokens,
            "temperature": args.qa_temperature,
        },
        timeout=args.qa_timeout_sec,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


def openrouter_chat_completion(args: argparse.Namespace, prompt: str) -> str:
    import requests

    key = os.getenv("OPENROUTER_API_KEY") or ""
    if not key:
        api_keys_path = Path("api_keys.json")
        if api_keys_path.exists():
            key = json.loads(api_keys_path.read_text(encoding="utf-8")).get("openrouter", "")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is missing")
    response = requests.post(
        DEFAULT_OPENROUTER_URL,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model": args.openrouter_model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": args.qa_max_tokens,
            "temperature": args.qa_temperature,
        },
        timeout=args.qa_timeout_sec,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


def ollama_completion(args: argparse.Namespace, prompt: str) -> str:
    import requests

    response = requests.post(
        args.ollama_url,
        json={"model": args.ollama_model, "prompt": prompt, "stream": False},
        timeout=args.qa_timeout_sec,
    )
    response.raise_for_status()
    return response.json().get("response", "").strip()


def generate_answer(args: argparse.Namespace, prompt: str) -> tuple[str, str, float]:
    if args.retrieve_only or args.provider == "none":
        return "[RETRIEVE_ONLY]", "retrieve-only", 0.0

    providers: list[tuple[str, Any]]
    if args.provider == "auto":
        providers = [
            ("endpoint", endpoint_chat_completion),
            ("openrouter", openrouter_chat_completion),
            ("ollama", ollama_completion),
        ]
    else:
        provider_map = {
            "endpoint": endpoint_chat_completion,
            "openrouter": openrouter_chat_completion,
            "ollama": ollama_completion,
        }
        providers = [(args.provider, provider_map[args.provider])]

    last_error = None
    for name, func in providers:
        try:
            start = time.time()
            answer = func(args, prompt)
            elapsed = time.time() - start
            if answer.strip():
                return answer, name, elapsed
        except Exception as exc:  # pragma: no cover - depends on external services
            last_error = exc
            if args.provider != "auto":
                break
    return f"[ERROR: answer generation failed: {last_error}]", "FAILED", 0.0


def resolve_graph_refs(store: FalkorGraphStore, args: argparse.Namespace) -> list[GraphRef]:
    identifiers = []
    if args.graph_ids:
        identifiers.extend(args.graph_ids)
    if args.ids:
        identifiers.extend(args.ids)

    if identifiers:
        refs = []
        for identifier in identifiers:
            ref = store.resolve_graph(identifier)
            if ref is None:
                print(f"SKIP {identifier}: no matching KGGraph in FalkorDB")
                continue
            refs.append(ref)
        return refs

    if args.limit is None and not args.list_graphs:
        raise SystemExit("Pass --graph-ids/--ids, or pass --limit N to run over the first N graphs.")
    return store.list_graphs(limit=args.limit)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def model_name_for_method(args: argparse.Namespace, method: str) -> str:
    if method == "endpoint":
        return args.qa_model
    if method == "openrouter":
        return args.openrouter_model
    if method == "ollama":
        return args.ollama_model
    return ""


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--falkor-graph",
        "--graph-name",
        default=os.getenv("FALKORDB_GRAPH", DEFAULT_FALKOR_GRAPH),
        help="FalkorDB graph name",
    )
    parser.add_argument("--falkor-url", default=os.getenv("FALKORDB_URL"))
    parser.add_argument("--falkor-host", default=os.getenv("FALKORDB_HOST", "localhost"))
    parser.add_argument("--falkor-port", type=int, default=int(os.getenv("FALKORDB_PORT", "6379")))
    parser.add_argument("--falkor-username", default=os.getenv("FALKORDB_USERNAME"))
    parser.add_argument("--falkor-password", default=os.getenv("FALKORDB_PASSWORD"))
    parser.add_argument("--falkor-ssl", action="store_true")
    parser.add_argument("--timeout-ms", type=int, default=None)
    parser.add_argument("--connect-timeout-sec", type=float, default=10.0)
    parser.add_argument("--socket-timeout-sec", type=float, default=30.0)

    parser.add_argument("--list-graphs", action="store_true", help="List selected KGGraph IDs and exit")
    parser.add_argument("--graph-ids", nargs="+", default=None, help="KGGraph.graph_id values")
    parser.add_argument(
        "--ids",
        nargs="+",
        default=None,
        help="IDs to resolve via graph_id, res_id, encounter_id, source_name, subject_id, or hadm_id",
    )
    parser.add_argument("--limit", type=int, default=None, help="Max graphs when no explicit IDs are passed")

    parser.add_argument("--question", action="append", default=None, help="Ad-hoc question text")
    parser.add_argument("--questions-json", default=None, help="JSON list of question strings or objects")
    parser.add_argument("--output", default="qa_results", help="Output directory")

    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--edge-limit", type=int, default=EDGE_LIMIT)
    parser.add_argument("--max-source-chars", type=int, default=MAX_SOURCE_CHARS)
    parser.add_argument(
        "--retriever",
        choices=("auto", "embedding", "lexical"),
        default=os.getenv("QA_RETRIEVER", "lexical"),
        help="Use lexical retrieval by default. 'embedding' requires sentence-transformers.",
    )
    parser.add_argument("--embedding-model", default=os.getenv("QA_EMBEDDING_MODEL", "BAAI/bge-m3"))
    parser.add_argument("--embedding-cache-dir", default=os.getenv("QA_EMBEDDING_CACHE_DIR", ".qa_embedding_cache"))

    parser.add_argument(
        "--provider",
        choices=("auto", "endpoint", "openrouter", "ollama", "none"),
        default=os.getenv("QA_PROVIDER", "auto"),
    )
    parser.add_argument("--retrieve-only", action="store_true")
    parser.add_argument("--qa-endpoint-url", default=os.getenv("QA_ENDPOINT_URL", os.getenv("ENDPOINT_URL", "")))
    parser.add_argument("--qa-endpoint-token-env", default=os.getenv("QA_ENDPOINT_TOKEN_ENV", "HF_TOKEN"))
    parser.add_argument("--qa-model", default=os.getenv("QA_MODEL", DEFAULT_QA_MODEL))
    parser.add_argument("--qa-max-tokens", type=int, default=int(os.getenv("QA_MAX_TOKENS", "1200")))
    parser.add_argument("--qa-temperature", type=float, default=float(os.getenv("QA_TEMPERATURE", "0")))
    parser.add_argument("--qa-timeout-sec", type=float, default=float(os.getenv("QA_TIMEOUT_SEC", "180")))
    parser.add_argument("--openrouter-model", default=os.getenv("OPENROUTER_MODEL", DEFAULT_QA_MODEL))
    parser.add_argument("--ollama-url", default=os.getenv("OLLAMA_API_URL", DEFAULT_OLLAMA_URL))
    parser.add_argument("--ollama-model", default=os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL))
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.top_k <= 0:
        raise SystemExit("--top-k must be positive")
    if args.edge_limit <= 0:
        raise SystemExit("--edge-limit must be positive")
    if args.max_source_chars <= 0:
        raise SystemExit("--max-source-chars must be positive")
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")
    if args.provider == "none":
        args.retrieve_only = True


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    validate_args(args)

    store = FalkorGraphStore.connect(args)
    refs = resolve_graph_refs(store, args)
    if not refs:
        raise SystemExit("No Falkor graphs selected")

    if args.list_graphs:
        for ref in refs:
            print(
                f"{ref.graph_id}"
                f"\tsubject_id={ref.subject_id or ''}"
                f"\thadm_id={ref.hadm_id or ''}"
                f"\tres_id={ref.res_id or ''}"
            )
        return

    questions = load_questions(args)
    if not questions:
        raise SystemExit("Pass --question or --questions-json for QA.")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    ranker = load_embedding_ranker(args)

    print("=== Fawkes Falkor GraphRAG QA ===")
    print(f"  Falkor graph : {args.falkor_graph}")
    print(f"  Graphs       : {len(refs)}")
    print(f"  Questions    : {len(questions)}")
    print(f"  Output       : {out_dir}")

    summary: dict[str, Any] = {
        "falkor_graph": args.falkor_graph,
        "started_at_utc": utc_now(),
        "graphs": {},
    }

    for ref in refs:
        graph = store.load_graph(ref)
        output_id = graph.ref.requested_id or graph.ref.graph_id
        output_slug = safe_path_name(output_id)
        graph_dir = out_dir / output_slug
        out_path = graph_dir / f"{output_slug}_answers.json"

        print(f"\n{'=' * 60}")
        print(f"  graph_id={graph.ref.graph_id}  nodes={len(graph.nodes)}  edges={len(graph.edges)}")
        print(f"{'=' * 60}")

        embeddings = ranker.load_or_compute(graph) if ranker is not None else None
        results = []
        for question in questions:
            qid = question["id"]
            qtxt = question["question"]
            print(f"\n  [{qid}] {qtxt[:100]}...", flush=True)

            start = time.time()
            context = retrieve_context(
                question=qtxt,
                graph=graph,
                ranker=ranker,
                embeddings=embeddings,
                top_k=args.top_k,
                edge_limit=args.edge_limit,
                max_source_chars=args.max_source_chars,
            )
            retrieval_time = time.time() - start
            prompt = build_prompt(qtxt, context)
            answer, method, generation_time = generate_answer(args, prompt)

            print(f"    retrieval  {retrieval_time:.1f}s", flush=True)
            print(f"    generation {generation_time:.1f}s ({method})", flush=True)
            print("\n" + answer + "\n", flush=True)

            results.append(
                {
                    "id": qid,
                    "type": question.get("type", ""),
                    "question": qtxt,
                    "answer": answer,
                    "method": method,
                    "model": model_name_for_method(args, method),
                    "graph_id": graph.ref.graph_id,
                    "subject_id": graph.ref.subject_id,
                    "hadm_id": graph.ref.hadm_id,
                    "retrieval_time_s": round(retrieval_time, 2),
                    "generation_time_s": round(generation_time, 2),
                    "context": context,
                    "prompt": prompt,
                }
            )

        write_json(out_path, results)
        summary["graphs"][graph.ref.graph_id] = {
            "answers_file": str(out_path),
            "questions": len(results),
            "nodes": len(graph.nodes),
            "edges": len(graph.edges),
        }
        print(f"  Saved -> {out_path}")

    summary["completed_at_utc"] = utc_now()
    write_json(out_dir / "summary.json", summary)
    print(f"\nSummary -> {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
