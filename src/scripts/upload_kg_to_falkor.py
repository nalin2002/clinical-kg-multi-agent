#!/usr/bin/env python3
"""Upload JSON knowledge graphs to FalkorDB.

The default input is the embedded Fawkes training JSONL under
``data/fawkes-training-graph-embedded-260615``.  The importer is split into
small pieces so a different graph export can reuse the FalkorDB upload layer by
providing another ``GraphAdapter``.

Example:
    python src/scripts/upload_kg_to_falkor.py --dry-run

    python src/scripts/upload_kg_to_falkor.py \\
        --graph-name fawkes_training_kg \\
        --host localhost --port 6379 \\
        --drop-existing
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Protocol


DEFAULT_INPUT = Path(
    "data/fawkes-training-graph-embedded-260615/"
    "fawkes_training_graph_full_embedded_260615.jsonl"
)
DEFAULT_GRAPH_NAME = "fawkes_training_kg"
TEXT_FIELD_KEYS = {"note", "input_block"}
GRAPH_PAYLOAD_KEYS = {"nodes", "edges"}
IDENTIFIER_RE = re.compile(r"[^A-Za-z0-9_]+")
PROPERTY_RE = re.compile(r"[^A-Za-z0-9_]+")
LOGGER = logging.getLogger("falkor_upload")


@dataclass(frozen=True)
class GraphRecord:
    """A source graph ready for normalization before upload."""

    graph_id: str
    source_name: str
    properties: dict[str, Any]
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]


class GraphAdapter(Protocol):
    """Adapter boundary for graph sources."""

    def iter_records(self, limit: int | None = None) -> Iterator[GraphRecord]:
        """Yield graph records from a source."""


@dataclass
class UploadOptions:
    """Controls how source records are projected into FalkorDB."""

    include_note_embedding: bool = False
    include_text_fields: bool = False
    link_graphs: bool = True
    source_tag: str = ""


@dataclass
class UploadStats:
    records: int = 0
    nodes: int = 0
    edges: int = 0
    graph_links: int = 0
    skipped_edges: int = 0
    node_labels: Counter[str] = field(default_factory=Counter)
    edge_relations: Counter[str] = field(default_factory=Counter)

    def add(self, other: "UploadStats") -> None:
        self.records += other.records
        self.nodes += other.nodes
        self.edges += other.edges
        self.graph_links += other.graph_links
        self.skipped_edges += other.skipped_edges
        self.node_labels.update(other.node_labels)
        self.edge_relations.update(other.edge_relations)


@dataclass
class UploadRows:
    admissions: list[dict[str, Any]] = field(default_factory=list)
    nodes_by_label: dict[str, list[dict[str, Any]]] = field(
        default_factory=lambda: defaultdict(list)
    )
    graph_links: list[dict[str, Any]] = field(default_factory=list)
    edges_by_relation: dict[str, list[dict[str, Any]]] = field(
        default_factory=lambda: defaultdict(list)
    )
    stats: UploadStats = field(default_factory=UploadStats)


def sanitize_identifier(value: Any, default: str = "IDENTIFIER") -> str:
    """Return a Cypher-safe label or relationship type identifier."""
    raw = str(value or "").strip()
    if not raw:
        raw = default
    ident = IDENTIFIER_RE.sub("_", raw.upper()).strip("_")
    if not ident:
        ident = default
    if ident[0].isdigit():
        ident = f"_{ident}"
    return ident


def sanitize_property_key(value: Any, default: str = "property") -> str:
    """Return a Cypher-safe property key."""
    raw = str(value or "").strip()
    if not raw:
        raw = default
    key = PROPERTY_RE.sub("_", raw).strip("_").lower()
    if not key:
        key = default
    if key[0].isdigit():
        key = f"_{key}"
    return key


def _dedup_key(existing: Mapping[str, Any], key: str) -> str:
    if key not in existing:
        return key
    suffix = 2
    while f"{key}_{suffix}" in existing:
        suffix += 1
    return f"{key}_{suffix}"


def escape_multiline_string(value: str) -> str:
    """Avoid raw newlines in FalkorDB parameter serialization."""
    return value.replace("\r\n", "\\n").replace("\r", "\\n").replace("\n", "\\n")


def _json_string(value: Any) -> str:
    return escape_multiline_string(
        json.dumps(value, ensure_ascii=False, default=str)
    )


def _coerce_scalar(value: Any) -> Any | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return escape_multiline_string(value)
    return None


def coerce_property_value(value: Any) -> Any | None:
    """Coerce source values to FalkorDB property-friendly values."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    scalar = _coerce_scalar(value)
    if scalar is not None:
        return scalar
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        items = []
        for item in value:
            item_scalar = _coerce_scalar(item)
            if item_scalar is None and item is not None:
                return _json_string(value)
            if item_scalar is not None:
                items.append(item_scalar)
        return items
    if isinstance(value, dict):
        return _json_string(value)
    return _json_string(value)


def flatten_properties(
    raw: Mapping[str, Any],
    *,
    prefix: str = "",
    max_depth: int = 1,
) -> dict[str, Any]:
    """Flatten one level of nested maps and coerce values for FalkorDB."""
    out: dict[str, Any] = {}
    for key, value in raw.items():
        safe_key = sanitize_property_key(key)
        full_key = f"{prefix}_{safe_key}" if prefix else safe_key
        if isinstance(value, dict) and max_depth > 0:
            nested = flatten_properties(value, prefix=full_key, max_depth=max_depth - 1)
            for nested_key, nested_value in nested.items():
                if nested_value is None:
                    continue
                out[_dedup_key(out, nested_key)] = nested_value
            continue
        coerced = coerce_property_value(value)
        if coerced is None:
            continue
        out[_dedup_key(out, full_key)] = coerced
    return out


def _graph_id_from_data(data: Mapping[str, Any], source_name: str) -> str:
    if data.get("graph_id"):
        return str(data["graph_id"])
    subject_id = data.get("subject_id")
    hadm_id = data.get("hadm_id")
    if subject_id not in (None, "") and hadm_id not in (None, ""):
        return f"{subject_id}:{hadm_id}"
    if hadm_id not in (None, ""):
        return str(hadm_id)
    return source_name


def graph_record_from_data(data: Mapping[str, Any], source_name: str) -> GraphRecord:
    """Build a normalized record wrapper from raw graph JSON."""
    nodes = data.get("nodes")
    edges = data.get("edges")
    if not isinstance(nodes, list) or not isinstance(edges, list):
        raise ValueError(f"{source_name} must contain list-valued nodes and edges")
    properties = {str(k): v for k, v in data.items() if k not in GRAPH_PAYLOAD_KEYS}
    return GraphRecord(
        graph_id=_graph_id_from_data(data, source_name),
        source_name=source_name,
        properties=properties,
        nodes=[dict(node) for node in nodes],
        edges=[dict(edge) for edge in edges],
    )


class JsonlGraphAdapter:
    """Read one graph per line from a JSONL file."""

    def __init__(self, path: Path):
        self.path = path

    def iter_records(self, limit: int | None = None) -> Iterator[GraphRecord]:
        emitted = 0
        with self.path.open(encoding="utf-8") as handle:
            for lineno, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                source_name = f"{self.path.name}:{lineno}"
                yield graph_record_from_data(json.loads(stripped), source_name)
                emitted += 1
                if limit is not None and emitted >= limit:
                    break


class JsonGraphAdapter:
    """Read a single JSON graph or a list of graph objects."""

    def __init__(self, path: Path):
        self.path = path

    def iter_records(self, limit: int | None = None) -> Iterator[GraphRecord]:
        data = json.loads(self.path.read_text(encoding="utf-8"))
        records = data if isinstance(data, list) else [data]
        for idx, item in enumerate(records, start=1):
            if limit is not None and idx > limit:
                break
            if not isinstance(item, Mapping):
                raise ValueError(f"{self.path}:{idx} is not a JSON object")
            source_name = self.path.name if len(records) == 1 else f"{self.path.name}:{idx}"
            yield graph_record_from_data(item, source_name)


class DirectoryGraphAdapter:
    """Read every non-manifest JSON graph under a directory."""

    def __init__(self, path: Path):
        self.path = path

    def iter_records(self, limit: int | None = None) -> Iterator[GraphRecord]:
        emitted = 0
        files = sorted(
            file
            for file in self.path.rglob("*.json")
            if not file.name.startswith("_")
        )
        for file in files:
            if limit is not None and emitted >= limit:
                break
            adapter = JsonGraphAdapter(file)
            for record in adapter.iter_records(limit=1):
                yield record
                emitted += 1
                break


def build_adapter(path: Path) -> GraphAdapter:
    if path.is_dir():
        return DirectoryGraphAdapter(path)
    if path.suffix.lower() == ".jsonl":
        return JsonlGraphAdapter(path)
    if path.suffix.lower() == ".json":
        return JsonGraphAdapter(path)
    raise ValueError(f"Unsupported graph input: {path}")


def _node_local_id(node: Mapping[str, Any], idx: int) -> str:
    return str(node.get("id") or node.get("node_id") or f"N_{idx:05d}")


def _node_text(node: Mapping[str, Any], local_id: str) -> str:
    return str(
        node.get("text")
        or node.get("name")
        or node.get("normalized_name")
        or local_id
    )


def _edge_source(edge: Mapping[str, Any]) -> str:
    return str(edge.get("source") or edge.get("source_id") or "")


def _edge_target(edge: Mapping[str, Any]) -> str:
    return str(edge.get("target") or edge.get("target_id") or "")


def _edge_relation(edge: Mapping[str, Any]) -> str:
    return str(edge.get("relation") or edge.get("type") or "RELATED_TO")


def _kg_node_id(graph_id: str, local_id: str) -> str:
    return f"{graph_id}:{local_id}"


def admission_properties(record: GraphRecord, options: UploadOptions) -> dict[str, Any]:
    raw = {}
    for key, value in record.properties.items():
        if key == "note_embedding" and not options.include_note_embedding:
            continue
        if key in TEXT_FIELD_KEYS and not options.include_text_fields:
            continue
        raw[key] = value

    props = flatten_properties(raw, max_depth=1)
    props["graph_id"] = record.graph_id
    props["source_name"] = record.source_name
    if options.source_tag:
        props["source_tag"] = options.source_tag
    return props


def build_upload_rows(
    records: Iterable[GraphRecord],
    options: UploadOptions,
) -> UploadRows:
    rows = UploadRows()
    for record in records:
        rows.stats.records += 1
        rows.admissions.append(
            {
                "graph_id": record.graph_id,
                "properties": admission_properties(record, options),
            }
        )

        known_nodes: set[str] = set()
        for idx, node in enumerate(record.nodes):
            local_id = _node_local_id(node, idx)
            known_nodes.add(local_id)
            node_type = sanitize_identifier(node.get("type"), "ENTITY")
            kg_node_id = _kg_node_id(record.graph_id, local_id)
            props = flatten_properties(node, max_depth=1)
            props.update(
                {
                    "kg_node_id": kg_node_id,
                    "graph_id": record.graph_id,
                    "source_node_id": local_id,
                    "node_type": node_type,
                    "text": escape_multiline_string(_node_text(node, local_id)),
                }
            )
            rows.nodes_by_label[node_type].append(
                {"kg_node_id": kg_node_id, "properties": props}
            )
            rows.stats.nodes += 1
            rows.stats.node_labels[node_type] += 1
            if options.link_graphs:
                rows.graph_links.append(
                    {"graph_id": record.graph_id, "kg_node_id": kg_node_id}
                )
                rows.stats.graph_links += 1

        for idx, edge in enumerate(record.edges):
            source = _edge_source(edge)
            target = _edge_target(edge)
            if not source or not target or source not in known_nodes or target not in known_nodes:
                rows.stats.skipped_edges += 1
                continue
            relation = sanitize_identifier(_edge_relation(edge), "RELATED_TO")
            kg_edge_id = f"{record.graph_id}:E_{idx:05d}"
            props = flatten_properties(edge, max_depth=1)
            props.update(
                {
                    "kg_edge_id": kg_edge_id,
                    "graph_id": record.graph_id,
                    "source_node_id": source,
                    "target_node_id": target,
                    "relation": relation,
                }
            )
            rows.edges_by_relation[relation].append(
                {
                    "kg_edge_id": kg_edge_id,
                    "source_id": _kg_node_id(record.graph_id, source),
                    "target_id": _kg_node_id(record.graph_id, target),
                    "properties": props,
                }
            )
            rows.stats.edges += 1
            rows.stats.edge_relations[relation] += 1
    return rows


def batched(records: Iterable[GraphRecord], batch_size: int) -> Iterator[list[GraphRecord]]:
    batch: list[GraphRecord] = []
    for record in records:
        batch.append(record)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


@dataclass
class FalkorConfig:
    graph_name: str
    host: str = "localhost"
    port: int = 6379
    username: str | None = None
    password: str | None = None
    url: str | None = None
    ssl: bool = False
    connect_timeout_sec: float | None = 10.0
    socket_timeout_sec: float | None = 30.0
    timeout_ms: int | None = None


class FalkorUploader:
    """Small wrapper around the FalkorDB Python graph client."""

    def __init__(self, graph: Any, timeout_ms: int | None = None):
        self.graph = graph
        self.timeout_ms = timeout_ms

    @classmethod
    def connect(cls, config: FalkorConfig) -> "FalkorUploader":
        try:
            from falkordb import FalkorDB
        except ImportError as exc:
            raise SystemExit(
                "Missing FalkorDB Python client. Install dependencies with "
                "`pip install -r requirements.txt` or `pip install FalkorDB`."
            ) from exc

        if config.url:
            LOGGER.info("connecting to FalkorDB url=%s graph=%s", config.url, config.graph_name)
            db = FalkorDB.from_url(config.url)
        else:
            LOGGER.info(
                "connecting to FalkorDB host=%s port=%s ssl=%s username=%s graph=%s",
                config.host,
                config.port,
                config.ssl,
                config.username or "",
                config.graph_name,
            )
            db = FalkorDB(
                host=config.host,
                port=config.port,
                username=config.username,
                password=config.password,
                ssl=config.ssl,
                socket_connect_timeout=config.connect_timeout_sec,
                socket_timeout=config.socket_timeout_sec,
            )
        LOGGER.info("connected to FalkorDB graph=%s", config.graph_name)
        return cls(db.select_graph(config.graph_name), timeout_ms=config.timeout_ms)

    def query(self, q: str, params: dict[str, Any] | None = None) -> Any:
        return self.graph.query(q, params=params, timeout=self.timeout_ms)

    def drop_existing(self) -> None:
        self.graph.delete()

    def ensure_indexes(self) -> None:
        for label, properties in (
            ("KGGraph", ("graph_id",)),
            ("KGNode", ("kg_node_id",)),
            ("KGNode", ("graph_id",)),
        ):
            try:
                self.graph.create_node_range_index(label, *properties)
            except Exception as exc:  # pragma: no cover - DB-version dependent
                print(
                    f"[falkor-upload] warning: could not create index "
                    f"{label}{properties}: {exc}"
                )

    def upload_rows(self, rows: UploadRows, batch_no: int | None = None) -> None:
        batch_tag = f"batch={batch_no} " if batch_no is not None else ""
        if rows.admissions:
            LOGGER.info("%suploading %s KGGraph/Admission rows", batch_tag, len(rows.admissions))
            self.query(
                """
                UNWIND $items AS item
                MERGE (g:KGGraph {graph_id: item.graph_id})
                SET g += item.properties
                SET g:Admission
                """,
                {"items": rows.admissions},
            )
            LOGGER.info("%suploaded %s KGGraph/Admission rows", batch_tag, len(rows.admissions))

        for label, nodes in sorted(rows.nodes_by_label.items()):
            LOGGER.info(
                "%suploading %s KGNode rows with label=%s",
                batch_tag,
                len(nodes),
                label,
            )
            self.query(
                f"""
                UNWIND $items AS item
                MERGE (n:KGNode {{kg_node_id: item.kg_node_id}})
                SET n += item.properties
                SET n:{label}
                """,
                {"items": nodes},
            )
            LOGGER.info(
                "%suploaded %s KGNode rows with label=%s",
                batch_tag,
                len(nodes),
                label,
            )

        if rows.graph_links:
            LOGGER.info(
                "%suploading %s KGGraph->KGNode links",
                batch_tag,
                len(rows.graph_links),
            )
            self.query(
                """
                UNWIND $items AS item
                MATCH (g:KGGraph {graph_id: item.graph_id})
                MATCH (n:KGNode {kg_node_id: item.kg_node_id})
                MERGE (g)-[:CONTAINS_NODE]->(n)
                """,
                {"items": rows.graph_links},
            )
            LOGGER.info(
                "%suploaded %s KGGraph->KGNode links",
                batch_tag,
                len(rows.graph_links),
            )

        for relation, edges in sorted(rows.edges_by_relation.items()):
            LOGGER.info(
                "%suploading %s edges with relation=%s",
                batch_tag,
                len(edges),
                relation,
            )
            self.query(
                f"""
                UNWIND $items AS item
                MATCH (source:KGNode {{kg_node_id: item.source_id}})
                MATCH (target:KGNode {{kg_node_id: item.target_id}})
                MERGE (source)-[r:{relation} {{kg_edge_id: item.kg_edge_id}}]->(target)
                SET r += item.properties
                """,
                {"items": edges},
            )
            LOGGER.info(
                "%suploaded %s edges with relation=%s",
                batch_tag,
                len(edges),
                relation,
            )


def run_upload(
    adapter: GraphAdapter,
    *,
    options: UploadOptions,
    uploader: FalkorUploader | None,
    batch_size: int,
    limit: int | None,
) -> UploadStats:
    total = UploadStats()
    for batch_no, batch in enumerate(
        batched(adapter.iter_records(limit=limit), batch_size), start=1
    ):
        LOGGER.info("batch=%s preparing %s source graph records", batch_no, len(batch))
        rows = build_upload_rows(batch, options)
        LOGGER.info(
            "batch=%s prepared records=%s nodes=%s edges=%s graph_links=%s skipped_edges=%s",
            batch_no,
            rows.stats.records,
            rows.stats.nodes,
            rows.stats.edges,
            rows.stats.graph_links,
            rows.stats.skipped_edges,
        )
        if uploader is not None:
            uploader.upload_rows(rows, batch_no=batch_no)
        total.add(rows.stats)
        LOGGER.info(
            "batch=%s complete totals: records=%s nodes=%s edges=%s skipped_edges=%s",
            batch_no,
            total.records,
            total.nodes,
            total.edges,
            total.skipped_edges,
        )
    return total


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help=f"Graph JSONL, JSON, or directory to upload (default: {DEFAULT_INPUT})",
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
    parser.add_argument(
        "--username",
        default=os.getenv("FALKORDB_USERNAME"),
        help="Optional FalkorDB username, or FALKORDB_USERNAME",
    )
    parser.add_argument(
        "--password",
        default=os.getenv("FALKORDB_PASSWORD"),
        help="Optional FalkorDB password, or FALKORDB_PASSWORD",
    )
    parser.add_argument(
        "--url",
        default=os.getenv("FALKORDB_URL"),
        help="Optional FalkorDB URL; overrides host/port/user/password",
    )
    parser.add_argument("--ssl", action="store_true", help="Use TLS for host/port mode")
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=None,
        help="Optional FalkorDB query timeout in milliseconds",
    )
    parser.add_argument(
        "--connect-timeout-sec",
        type=float,
        default=10.0,
        help="Socket connect timeout in seconds (default: 10)",
    )
    parser.add_argument(
        "--socket-timeout-sec",
        type=float,
        default=30.0,
        help="Socket read/write timeout in seconds (default: 30)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of source graphs to upload per batch (default: 50)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional graph limit")
    parser.add_argument(
        "--drop-existing",
        action="store_true",
        help="Delete the target FalkorDB graph before upload",
    )
    parser.add_argument(
        "--skip-indexes",
        action="store_true",
        help="Skip helper indexes on KGGraph.graph_id and KGNode ids",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and count rows without connecting to FalkorDB",
    )
    parser.add_argument(
        "--include-note-embedding",
        action="store_true",
        help="Store top-level note_embedding arrays on KGGraph nodes",
    )
    parser.add_argument(
        "--include-text-fields",
        action="store_true",
        help="Store top-level note/input_block text on KGGraph nodes",
    )
    parser.add_argument(
        "--no-graph-links",
        action="store_true",
        help="Do not create KGGraph-[:CONTAINS_NODE]->KGNode links",
    )
    parser.add_argument(
        "--source-tag",
        default="",
        help="Optional tag stored on every KGGraph node",
    )
    parser.add_argument(
        "--log-level",
        default=os.getenv("FALKOR_UPLOAD_LOG_LEVEL", "INFO"),
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        help="Logging verbosity (default: INFO, or FALKOR_UPLOAD_LOG_LEVEL)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input does not exist: {input_path}")

    adapter = build_adapter(input_path)
    options = UploadOptions(
        include_note_embedding=args.include_note_embedding,
        include_text_fields=args.include_text_fields,
        link_graphs=not args.no_graph_links,
        source_tag=args.source_tag,
    )

    uploader = None
    if not args.dry_run:
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
        try:
            uploader = FalkorUploader.connect(config)
        except Exception as exc:
            LOGGER.error("failed to connect to FalkorDB: %s", exc)
            raise SystemExit(1)
        if args.drop_existing:
            print(f"[falkor-upload] deleting existing graph {args.graph_name!r}")
            uploader.drop_existing()
        if not args.skip_indexes:
            uploader.ensure_indexes()

    try:
        stats = run_upload(
            adapter,
            options=options,
            uploader=uploader,
            batch_size=args.batch_size,
            limit=args.limit,
        )
    except Exception as exc:
        LOGGER.error("upload failed: %s", exc)
        raise SystemExit(1)
    mode = "dry-run" if args.dry_run else "uploaded"
    LOGGER.info(
        "%s: records=%s nodes=%s edges=%s graph_links=%s skipped_edges=%s",
        mode,
        stats.records,
        stats.nodes,
        stats.edges,
        stats.graph_links,
        stats.skipped_edges,
    )
    LOGGER.info(
        "node_labels=%s",
        dict(sorted(stats.node_labels.items())),
    )
    LOGGER.info(
        "edge_relations=%s",
        dict(sorted(stats.edge_relations.items())),
    )


if __name__ == "__main__":
    main()
