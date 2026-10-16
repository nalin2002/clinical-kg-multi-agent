#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "huggingface_hub>=0.25",
#   "hf_transfer",
#   "falkordb>=1.0.0",
#   "torch==2.4.1",
#   "transformers>=4.48",
# ]
# ///
"""End-to-end Fawkes admission KG builder and FalkorDB uploader.

This script is intentionally standalone so the whole ``fawkes_falkor_e2e``
folder can be copied out of this repository and run elsewhere with ``uv`` or
Hugging Face Jobs.

Pipeline:
  1. Download raw admission JSONL from Hugging Face and normalize/select rows.
  2. Build one admission KG per row, preserving note/input_block on the graph.
  3. Optionally embed each top-level note into ``note_embedding``.
  4. Upload each completed artifact to a Hugging Face dataset repo.
  5. Upload the final KG JSONL to FalkorDB for retrieval/QA.

Resume is controlled by the stage JSONL outputs in the output repo. If
``admissions_normalized.jsonl``, ``kg_raw.jsonl``, or ``kg_embedded.jsonl``
already exists, that stage is skipped unless ``--force`` is set.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import logging
import math
import os
import random
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping


os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:  # pragma: no cover - old Python fallback
    pass

LOG = logging.getLogger("fawkes_falkor_e2e")


# --------------------------------------------------------------------------- #
# Clinical edge extraction prompt and KG builders
# --------------------------------------------------------------------------- #

SPEC_TMPL = """<role>
You are an EXPERT {persona} extracting {REL} edges from ONE hospital discharge note. You receive the full NODE list (the patient's structured items) and the numbered note. Emit edges in two buckets.
</role>

<nodes>
{nodes}
</nodes>

<note>
{note}
</note>

<task>
A {REL} edge means: {definition}
Valid src -> dst type patterns:
{patterns}
Only emit pairs matching one of these type patterns. Arrow direction is src -> dst. Never reverse.
{extra}
Err toward INCLUSION on MATCHED (both endpoints are <nodes> entries); precision is handled downstream.

Walk every plausible (src, dst) pair drawn from <nodes>:
  Step 1 - CANDIDATE WALK. For each pair decide ACCEPT / REJECT / UNCERTAIN for "{question}". Cite the note sentence number.
  Step 2 - RESOLVE every UNCERTAIN to ACCEPT or REJECT.
  Step 3 - VERIFY LABELS. MATCHED endpoints are character-for-character copies of a <nodes> name.
  Step 4 - ANTI-COLLAPSE. One src may link to several dst and vice-versa; include each.

Two buckets:
  MATCHED - both src and dst are <nodes> names (use the EXACT name string).
  OTHER   - at least one endpoint is named/implied in the note but NOT a <nodes> entry.

NEGATION BLOCKS EMISSION: if the note denies/excludes the link, do not emit it. DEDUPLICATE within each bucket.
</task>

<output>
Emit ONE JSON object, nothing else:
{{"rationale": "<=60 words", "matched": [{{"src":"...","dst":"...","sentence":"[n]","quote":"...","confidence":0.0}}], "other": [{{"src":"...","dst":"...","sentence":"[n]","quote":"...","confidence":0.0}}]}}
Set "confidence" to your ACTUAL certainty THIS specific link is correct - a graded value 0.30-0.99, and VARY it.
Any bucket may be []. Both keys must be present.
</output>"""


SPECIALISTS = {
    "MANAGED_FOR": (
        "clinical pharmacist",
        "a MEDICATION is given to treat / manage / control a DIAGNOSIS "
        "(active problem or chronic condition). The most valuable edge - a "
        "medication paired with the problem it addresses.",
        "  - MEDICATION -> DIAGNOSIS",
        "a medication is taken for a diagnosis",
        "Use standard clinical knowledge to link a listed MEDICATION to the "
        "listed DIAGNOSIS it treats. Supportive fluids/flushes/vaccines have "
        "no MANAGED_FOR target - skip them. If a medication has NO diagnosis "
        "in the node list that it plausibly treats, emit NOTHING for it. A "
        "shared medication-list sentence is co-location, not shared purpose.",
    ),
    "CONFIRMS": (
        "clinical internist",
        "a PROCEDURE established / was diagnostic-of a DIAGNOSIS.",
        "  - PROCEDURE -> DIAGNOSIS",
        "a procedure confirmed a diagnosis",
        "Only emit when the note ties the procedure to the diagnosis it established.",
    ),
    "COMPLICATED_BY": (
        "clinical internist",
        "the acute course / primary DIAGNOSIS was COMPLICATED BY another "
        "DIAGNOSIS that arose this stay. NOT every comorbidity.",
        "  - DIAGNOSIS -> DIAGNOSIS",
        "the first was complicated by the second this admission",
        "Do NOT link chronic comorbidities as complications.",
    ),
    "INDICATES": (
        "clinical internist",
        "a presenting SYMPTOM/finding points to a DIAGNOSIS worked up this stay.",
        "  - DIAGNOSIS(PRESENTING) -> DIAGNOSIS",
        "the presenting feature points to the diagnosis",
        "STRICT: src MUST be a (PRESENTING) node. Never link two established "
        "diagnoses; never emit a reverse pair. No (PRESENTING) node -> emit nothing.",
    ),
}


def norm_name(value: Any) -> str:
    return re.sub(
        r"\s+",
        " ",
        re.sub(r"[^a-z0-9 ]", " ", str(value or "").lower()),
    ).strip()


_JSON_DEC = json.JSONDecoder()


def extract_json(text: str | None) -> dict[str, Any]:
    """Extract a JSON object from endpoint output without silently inventing data."""

    s = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()
    if not s:
        return {}
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    if s.startswith("```"):
        match = re.match(r"```(?:json)?\s*(.*?)\s*```", s, re.DOTALL)
        if match:
            s = match.group(1).strip()
            try:
                return json.loads(s)
            except json.JSONDecodeError:
                pass
    best, best_len = None, -1
    for idx, ch in enumerate(s):
        if ch != "{":
            continue
        try:
            obj, end = _JSON_DEC.raw_decode(s[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and end > best_len:
            best, best_len = obj, end
    if best is not None:
        return best
    try:
        return json.loads(re.sub(r",(\s*[}\]])", r"\1", s))
    except json.JSONDecodeError:
        return {}


def medication_name(value: Any) -> str:
    if isinstance(value, Mapping):
        return str(value.get("name") or "").strip()
    return str(value or "").strip()


def microbiology_name(value: Any) -> str:
    if isinstance(value, Mapping):
        return str(value.get("organism") or "").strip()
    return str(value or "").strip()


def graph_id_for(record: Mapping[str, Any], fallback: str = "") -> str:
    if record.get("graph_id"):
        return str(record["graph_id"])
    subject_id = record.get("subject_id")
    hadm_id = record.get("hadm_id")
    if subject_id not in (None, "") and hadm_id not in (None, ""):
        return f"{subject_id}:{hadm_id}"
    if hadm_id not in (None, ""):
        return str(hadm_id)
    return fallback or hashlib.sha1(json.dumps(record, sort_keys=True, default=str).encode()).hexdigest()


def has_note(record: Mapping[str, Any]) -> bool:
    return bool(str(record.get("note") or "").strip())


def build_nodes(record: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    nodes: list[dict[str, Any]] = []
    index: dict[tuple[str, str], dict[str, Any]] = {}

    def add(node_type: str, name: Any, *, presenting: bool = False) -> None:
        text = str(name or "").strip()
        if not text:
            return
        key = (node_type, text.lower())
        if key in index:
            if presenting:
                index[key]["presenting"] = True
            return
        node = {
            "id": f"N_{len(nodes):03d}",
            "type": node_type,
            "name": text,
            "text": text,
            "normalized_name": text.lower(),
        }
        if presenting:
            node["presenting"] = True
        nodes.append(node)
        index[key] = node

    add("PATIENT", f"patient_{record['subject_id']}")
    for symptom in record.get("symptoms", []) or []:
        add("DIAGNOSIS", symptom, presenting=True)
    for diagnosis in record.get("diagnoses", []) or []:
        add("DIAGNOSIS", diagnosis)
    for history in record.get("medical_history", []) or []:
        add("DIAGNOSIS", history)
    for medication in record.get("medications", []) or []:
        add("MEDICATION", medication_name(medication))
    for procedure in record.get("procedures", []) or []:
        add("PROCEDURE", procedure)
    for micro in record.get("microbiology", []) or []:
        add("MICROBIOLOGY", microbiology_name(micro))
    for service in record.get("services", []) or []:
        add("SERVICE", service)
    for lab in record.get("lab_tests", []) or []:
        add("LAB_TEST", lab)
    return nodes, index


def node_id(index: Mapping[tuple[str, str], Mapping[str, Any]], node_type: str, name: Any) -> str | None:
    node = index.get((node_type, str(name or "").strip().lower()))
    return str(node["id"]) if node else None


def deterministic_edges(record: Mapping[str, Any], index: Mapping[tuple[str, str], Mapping[str, Any]]) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    patient_id = node_id(index, "PATIENT", f"patient_{record['subject_id']}")

    def add(source: str | None, relation: str, target: str | None) -> None:
        if source and target and source != target:
            edges.append(
                {
                    "source": source,
                    "target": target,
                    "relation": relation,
                    "confidence": 1.0,
                    "evidence": "structured",
                }
            )

    for diagnosis in (
        list(record.get("symptoms", []) or [])
        + list(record.get("diagnoses", []) or [])
        + list(record.get("medical_history", []) or [])
    ):
        add(patient_id, "HAS_DIAGNOSIS", node_id(index, "DIAGNOSIS", diagnosis))
    for medication in record.get("medications", []) or []:
        add(patient_id, "TAKES_MEDICATION", node_id(index, "MEDICATION", medication_name(medication)))
    for procedure in record.get("procedures", []) or []:
        add(patient_id, "UNDERWENT_PROCEDURE", node_id(index, "PROCEDURE", procedure))
    for service in record.get("services", []) or []:
        add(patient_id, "MANAGED_BY_SERVICE", node_id(index, "SERVICE", service))
    for micro in record.get("microbiology", []) or []:
        organism = microbiology_name(micro)
        organism_id = node_id(index, "MICROBIOLOGY", organism)
        add(patient_id, "HAD_MICROBIOLOGY", organism_id)
        if isinstance(micro, Mapping):
            for susceptibility in micro.get("susceptibilities", []) or []:
                if susceptibility.get("interpretation") == "S":
                    med_id = node_id(index, "MEDICATION", susceptibility.get("antibiotic", ""))
                    if med_id:
                        add(med_id, "TARGETS_ORGANISM", organism_id)
    for lab in record.get("lab_tests", []) or []:
        add(patient_id, "HAD_LAB_TEST", node_id(index, "LAB_TEST", lab))
    return edges


def make_prompt(relation: str, nodes: list[dict[str, Any]], note: str) -> str:
    persona, definition, patterns, question, extra = SPECIALISTS[relation]
    node_lines = "\n".join(
        f'  {idx}. "{node["name"]}" ({node["type"]}{", PRESENTING" if node.get("presenting") else ""})'
        for idx, node in enumerate(nodes, start=1)
    )
    return SPEC_TMPL.format(
        persona=persona,
        REL=relation,
        definition=definition,
        patterns=patterns,
        question=question,
        extra=extra,
        nodes=node_lines,
        note=note,
    )


def resolve_endpoint_name(name: Any, nodes: list[dict[str, Any]]) -> str | None:
    query = norm_name(name)
    for node in nodes:
        if norm_name(node["name"]) == query:
            return str(node["id"])
    for node in nodes:
        node_name = norm_name(node["name"])
        if query and (query in node_name or node_name in query):
            return str(node["id"])
    return None


def assemble_graph(
    record: Mapping[str, Any],
    nodes: list[dict[str, Any]],
    det_edges: list[dict[str, Any]],
    parsed_by_relation: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    llm_edges: list[dict[str, Any]] = []
    for relation, parsed in parsed_by_relation.items():
        for match in parsed.get("matched") or []:
            source = resolve_endpoint_name(match.get("src"), nodes)
            target = resolve_endpoint_name(match.get("dst"), nodes)
            if not source or not target or source == target:
                continue
            raw_conf = match.get("confidence")
            if isinstance(raw_conf, (int, float)) and 0 < float(raw_conf) <= 1:
                confidence = round(float(raw_conf), 2)
            else:
                confidence = 0.5
            llm_edges.append(
                {
                    "source": source,
                    "target": target,
                    "relation": relation,
                    "confidence": confidence,
                    "evidence": "llm",
                    "sentence": str(match.get("sentence") or ""),
                    "quote": str(match.get("quote") or "")[:300],
                }
            )

    seen: set[tuple[str, str, str]] = set()
    edges: list[dict[str, Any]] = []
    for edge in det_edges + llm_edges:
        key = (str(edge["source"]), str(edge["relation"]), str(edge["target"]))
        if key in seen:
            continue
        seen.add(key)
        edges.append(edge)

    graph_id = graph_id_for(record)
    graph = {
        "graph_id": graph_id,
        "subject_id": record.get("subject_id"),
        "hadm_id": record.get("hadm_id"),
        "kg_stage": "raw",
        "has_note": has_note(record),
        "note": record.get("note") or "",
        "input_block": record.get("input_block") or "",
        "nodes": nodes,
        "edges": edges,
        "n_nodes": len(nodes),
        "n_edges": len(edges),
    }
    for key in (
        "gender",
        "anchor_age",
        "race",
        "admission_type",
        "admission_location",
        "discharge_location",
        "died_in_hospital",
    ):
        if key in record:
            graph[key] = record[key]
    return graph


# --------------------------------------------------------------------------- #
# Hugging Face artifact store
# --------------------------------------------------------------------------- #


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def jsonl_count(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid JSONL at {path}:{lineno}") from exc


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    tmp_path.replace(path)


@dataclass
class ArtifactStore:
    repo_id: str | None
    workdir: Path
    private: bool = True
    local_only: bool = False
    _files: set[str] | None = field(default=None, init=False)

    def rel(self, name: str) -> str:
        return name

    def local(self, name: str) -> Path:
        return self.workdir / name

    def _api(self):
        from huggingface_hub import HfApi

        return HfApi()

    def ensure_repo(self) -> None:
        if self.local_only:
            return
        if not self.repo_id:
            raise RuntimeError("--output-repo is required unless --local-only is set")
        self._api().create_repo(
            self.repo_id,
            repo_type="dataset",
            private=self.private,
            exist_ok=True,
        )

    def refresh(self) -> None:
        if self.local_only or not self.repo_id:
            self._files = set()
            return
        from huggingface_hub import list_repo_files

        try:
            self._files = set(list_repo_files(self.repo_id, repo_type="dataset"))
        except Exception:
            self._files = set()

    def exists(self, name: str) -> bool:
        if self.local_only:
            return self.local(name).exists()
        if self._files is None:
            self.refresh()
        return self.rel(name) in (self._files or set())

    def upload(self, local_path: Path, name: str) -> None:
        if self.local_only:
            return
        self.ensure_repo()
        self._api().upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=self.rel(name),
            repo_id=self.repo_id,
            repo_type="dataset",
        )
        if self._files is not None:
            self._files.add(self.rel(name))

    def download(self, name: str, local_path: Path) -> bool:
        if self.local_only:
            return local_path.exists()
        if not self.repo_id or not self.exists(name):
            return False
        from huggingface_hub import hf_hub_download

        downloaded = hf_hub_download(
            self.repo_id,
            self.rel(name),
            repo_type="dataset",
        )
        local_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(downloaded, local_path)
        return True


# --------------------------------------------------------------------------- #
# Normalize/select raw records
# --------------------------------------------------------------------------- #


def eligible_record(record: Mapping[str, Any], args: argparse.Namespace) -> bool:
    if args.no_eligibility_filter:
        return True
    return (
        args.min_dx <= int(record.get("n_diagnoses", 0) or 0) <= args.max_dx
        and args.min_med <= int(record.get("n_medications", 0) or 0) <= args.max_med
        and int(record.get("n_services", 0) or 0) >= args.min_svc
    )


def normalize_stage(args: argparse.Namespace, store: ArtifactStore) -> Path:
    output = store.local("admissions_normalized.jsonl")
    manifest_path = store.local("admissions_manifest.json")

    if not args.force and store.download("admissions_normalized.jsonl", output):
        LOG.info("[RESUME] normalize found admissions_normalized.jsonl")
        store.download("admissions_manifest.json", manifest_path)
        return output

    if args.raw_local_file:
        raw_path = str(Path(args.raw_local_file).expanduser())
        if not Path(raw_path).exists():
            raise RuntimeError(f"--raw-local-file does not exist: {raw_path}")
    else:
        from huggingface_hub import hf_hub_download

        raw_path = hf_hub_download(args.raw_repo, args.raw_file, repo_type="dataset")
    raw_records = list(iter_jsonl(Path(raw_path)))
    if not raw_records:
        raise RuntimeError(f"0 raw records in {args.raw_repo}/{args.raw_file}")

    patient_to_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    eligible_patients: set[str] = set()
    for record in raw_records:
        subject_id = str(record.get("subject_id") or "")
        if not subject_id:
            continue
        patient_to_records[subject_id].append(record)
        if eligible_record(record, args):
            eligible_patients.add(subject_id)

    selected_patients = sorted(eligible_patients)
    random.Random(args.seed).shuffle(selected_patients)
    if args.n_patients > 0:
        selected_patients = selected_patients[: args.n_patients]
    selected_patient_set = set(selected_patients)

    selected: list[dict[str, Any]] = []
    for subject_id in selected_patients:
        selected.extend(patient_to_records[subject_id])
    if not args.include_noteless:
        selected = [record for record in selected if has_note(record)]
    if args.limit_admissions is not None:
        selected = selected[: args.limit_admissions]
    if not selected:
        raise RuntimeError("normalization selected 0 admissions")

    normalized = []
    note_count = 0
    for idx, record in enumerate(selected, start=1):
        item = dict(record)
        item["graph_id"] = graph_id_for(item, fallback=f"record:{idx}")
        item["has_note"] = has_note(item)
        if item["has_note"]:
            note_count += 1
        normalized.append(item)

    write_jsonl(output, normalized)
    manifest = {
        "stage": "normalize",
        "raw_repo": args.raw_repo,
        "raw_file": args.raw_file,
        "raw_local_file": args.raw_local_file,
        "records_in_raw": len(raw_records),
        "eligible_patients": len(eligible_patients),
        "selected_patients": len(selected_patient_set),
        "admissions": len(normalized),
        "admissions_with_note": note_count,
        "admissions_without_note": len(normalized) - note_count,
        "include_noteless": args.include_noteless,
        "n_patients": args.n_patients,
        "seed": args.seed,
        "sha256": sha256_file(output),
    }
    write_json(manifest_path, manifest)
    store.upload(output, "admissions_normalized.jsonl")
    store.upload(manifest_path, "admissions_manifest.json")
    LOG.info(
        "[DONE] normalize admissions=%s noted=%s -> %s",
        len(normalized),
        note_count,
        output,
    )
    return output


# --------------------------------------------------------------------------- #
# Endpoint client and KG build stage
# --------------------------------------------------------------------------- #


class EndpointClient:
    def __init__(
        self,
        *,
        endpoint_url: str,
        model: str,
        token: str,
        max_new_tokens: int,
    ):
        self.endpoint_url = endpoint_url.rstrip("/")
        self.model = model
        self.token = token
        self.max_new_tokens = max_new_tokens
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

    def resolve_model(self) -> None:
        try:
            req = urllib.request.Request(
                f"{self.endpoint_url}/v1/models",
                headers=self.headers,
            )
            with urllib.request.urlopen(req, timeout=30) as response:
                served = json.loads(response.read())["data"][0]["id"]
            if served and served != self.model:
                LOG.info("[MODEL] %r -> served %r", self.model, served)
                self.model = served
        except Exception as exc:
            LOG.warning("[MODEL] /v1/models probe failed (%s); using %r", type(exc).__name__, self.model)

    def call(self, prompt: str) -> str:
        body = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": self.max_new_tokens,
                "temperature": 0,
            }
        ).encode()
        url = f"{self.endpoint_url}/v1/chat/completions"
        for attempt in range(4):
            try:
                req = urllib.request.Request(url, data=body, headers=self.headers)
                with urllib.request.urlopen(req, timeout=600) as response:
                    payload = json.loads(response.read())
                return payload["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as exc:
                if exc.code in (408, 429, 503) or 500 <= exc.code < 600:
                    time.sleep(2 * (2**attempt))
                    continue
                LOG.warning("[ENDPOINT] non-retryable HTTP %s", exc.code)
                return ""
            except Exception as exc:
                LOG.warning("[ENDPOINT] attempt %s failed: %s", attempt + 1, exc)
                time.sleep(2 * (2**attempt))
        return ""


def endpoint_client_from_args(args: argparse.Namespace) -> EndpointClient | None:
    if args.llm_mode == "off":
        return None
    token = os.getenv(args.endpoint_token_env) or os.getenv("HF_TOKEN") or ""
    ready = bool(args.endpoint_url and token)
    if not ready:
        if args.llm_mode == "required":
            raise RuntimeError(
                "--llm-mode required needs --endpoint-url and an endpoint token "
                f"in {args.endpoint_token_env} or HF_TOKEN"
            )
        LOG.warning("[LLM] disabled: endpoint URL or token is missing")
        return None
    client = EndpointClient(
        endpoint_url=args.endpoint_url,
        model=args.edge_model,
        token=token,
        max_new_tokens=args.max_new_tokens,
    )
    client.resolve_model()
    return client


@dataclass
class GraphMeta:
    record: dict[str, Any]
    nodes: list[dict[str, Any]]
    det_edges: list[dict[str, Any]]

    @property
    def graph_id(self) -> str:
        return graph_id_for(self.record)


def build_kg_stage(args: argparse.Namespace, store: ArtifactStore, admissions_path: Path) -> Path:
    output = store.local("kg_raw.jsonl")
    manifest_path = store.local("kg_raw_manifest.json")

    if not args.force and store.download("kg_raw.jsonl", output):
        LOG.info("[RESUME] build_kg found kg_raw.jsonl")
        store.download("kg_raw_manifest.json", manifest_path)
        return output

    graphs: list[dict[str, Any]] = []
    records = list(iter_jsonl(admissions_path))
    meta: list[GraphMeta] = []
    for record in records:
        nodes, index = build_nodes(record)
        meta.append(
            GraphMeta(
                record=record,
                nodes=nodes,
                det_edges=deterministic_edges(record, index),
            )
        )

    LOG.info("[SETUP] build_kg graphs=%s", len(meta))

    client = endpoint_client_from_args(args)
    llm_enabled = client is not None
    parsed: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    meta_by_gid = {item.graph_id: item for item in meta}
    graph_ids_with_llm = {
        item.graph_id for item in meta if llm_enabled and has_note(item.record)
    }

    deterministic_only = [item for item in meta if item.graph_id not in graph_ids_with_llm]
    for item in deterministic_only:
        graphs.append(
            assemble_graph(
                item.record,
                item.nodes,
                item.det_edges,
                {},
            )
        )

    if graph_ids_with_llm:
        assert client is not None
        tasks: list[tuple[str, str]] = []
        for gid in sorted(graph_ids_with_llm):
            for relation in SPECIALISTS:
                tasks.append((gid, relation))
        LOG.info(
            "[LLM] %s graphs x %s specialists = %s calls | workers=%s",
            len(graph_ids_with_llm),
            len(SPECIALISTS),
            len(tasks),
            args.workers,
        )
        completed_calls = 0
        with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    client.call,
                    make_prompt(
                        relation,
                        meta_by_gid[gid].nodes,
                        str(meta_by_gid[gid].record.get("note") or ""),
                    ),
                ): (gid, relation)
                for gid, relation in tasks
            }
            for future in cf.as_completed(futures):
                gid, relation = futures[future]
                parsed[gid][relation] = extract_json(future.result())
                completed_calls += 1
                if len(parsed[gid]) == len(SPECIALISTS):
                    item = meta_by_gid[gid]
                    graphs.append(
                        assemble_graph(
                            item.record,
                            item.nodes,
                            item.det_edges,
                            parsed.pop(gid),
                        )
                    )
                if completed_calls % 50 == 0:
                    LOG.info("[LLM] calls=%s/%s", completed_calls, len(tasks))

    write_jsonl(output, graphs)
    store.upload(output, "kg_raw.jsonl")
    relation_counts = Counter(edge["relation"] for graph in graphs for edge in graph.get("edges", []))
    evidence_counts = Counter(edge.get("evidence", "") for graph in graphs for edge in graph.get("edges", []))
    manifest = {
        "stage": "build_kg",
        "input": "admissions_normalized.jsonl",
        "graphs": len(graphs),
        "nodes": sum(len(graph.get("nodes", [])) for graph in graphs),
        "edges": sum(len(graph.get("edges", [])) for graph in graphs),
        "relations": dict(sorted(relation_counts.items())),
        "evidence": dict(sorted(evidence_counts.items())),
        "llm_mode": args.llm_mode,
        "llm_enabled": llm_enabled,
        "endpoint_url": args.endpoint_url if llm_enabled else "",
        "edge_model": client.model if client else "",
        "sha256": sha256_file(output),
    }
    write_json(manifest_path, manifest)
    store.upload(manifest_path, "kg_raw_manifest.json")
    LOG.info("[DONE] build_kg graphs=%s -> %s", len(graphs), output)
    return output


# --------------------------------------------------------------------------- #
# Optional note embedding stage
# --------------------------------------------------------------------------- #


def resolve_embed_device(preference: str) -> str:
    import torch

    if preference == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if preference == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--embed-device cuda was requested, but CUDA is not available")
    if (
        preference == "mps"
        and (not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available())
    ):
        raise RuntimeError("--embed-device mps was requested, but MPS is not available")
    return preference


def embed_notes_stage(args: argparse.Namespace, store: ArtifactStore, kg_path: Path) -> Path:
    output = store.local("kg_embedded.jsonl")
    manifest_path = store.local("kg_embedded_manifest.json")

    if not args.force and store.download("kg_embedded.jsonl", output):
        LOG.info("[RESUME] embed_notes found kg_embedded.jsonl")
        store.download("kg_embedded_manifest.json", manifest_path)
        return output

    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "note embedding requires torch and transformers. Run with uv so the "
            "script dependencies install, or install those packages manually."
        ) from exc

    records = list(iter_jsonl(kg_path))
    if not records:
        raise RuntimeError(f"0 KG records in {kg_path}")

    notes = [str(record.get("note") or "") for record in records]
    empty_notes = sum(1 for note in notes if not note.strip())
    device = resolve_embed_device(args.embed_device)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    LOG.info(
        "[EMBED] records=%s empty_notes=%s model=%s device=%s max_len=%s batch=%s",
        len(records),
        empty_notes,
        args.embed_model,
        device,
        args.embed_max_len,
        args.embed_batch_size,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.embed_model)
    model = AutoModel.from_pretrained(args.embed_model, torch_dtype=dtype).to(device).eval()
    dim = int(model.config.hidden_size)

    @torch.no_grad()
    def embed_batch(batch_texts: list[str]) -> list[list[float]]:
        tokens = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=args.embed_max_len,
            return_tensors="pt",
        ).to(device)
        hidden = model(**tokens).last_hidden_state
        mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        vectors = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        return vectors.float().cpu().tolist()

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = output.with_name(f"{output.name}.tmp")
    embedded = 0
    start_time = time.time()
    with tmp_output.open("w", encoding="utf-8") as handle:
        for start in range(0, len(records), args.embed_batch_size):
            end = min(start + args.embed_batch_size, len(records))
            vectors = embed_batch(notes[start:end])
            for record, vector in zip(records[start:end], vectors):
                item = dict(record)
                item["note_embedding"] = [round(float(value), 6) for value in vector]
                item["embed_model"] = args.embed_model
                item["embed_dim"] = dim
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            embedded = end
            if embedded == len(records) or (start // args.embed_batch_size) % 20 == 0:
                LOG.info("[EMBED] records=%s/%s", embedded, len(records))
    tmp_output.replace(output)

    manifest = {
        "stage": "embed_notes",
        "input": kg_path.name,
        "output": "kg_embedded.jsonl",
        "records": len(records),
        "notes": len(notes),
        "empty_notes": empty_notes,
        "embed_model": args.embed_model,
        "embed_dim": dim,
        "embed_device": device,
        "embed_max_len": args.embed_max_len,
        "embed_batch_size": args.embed_batch_size,
        "seconds": round(time.time() - start_time, 3),
        "sha256": sha256_file(output),
    }
    write_json(manifest_path, manifest)
    store.upload(output, "kg_embedded.jsonl")
    store.upload(manifest_path, "kg_embedded_manifest.json")
    LOG.info("[DONE] embed_notes records=%s dim=%s -> %s", len(records), dim, output)
    return output


# --------------------------------------------------------------------------- #
# FalkorDB upload stage
# --------------------------------------------------------------------------- #


IDENTIFIER_RE = re.compile(r"[^A-Za-z0-9_]+")
PROPERTY_RE = re.compile(r"[^A-Za-z0-9_]+")
TEXT_FIELD_KEYS = {"note", "input_block", "transcript"}


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

    def to_dict(self) -> dict[str, Any]:
        return {
            "records": self.records,
            "nodes": self.nodes,
            "edges": self.edges,
            "graph_links": self.graph_links,
            "skipped_edges": self.skipped_edges,
            "node_labels": dict(sorted(self.node_labels.items())),
            "edge_relations": dict(sorted(self.edge_relations.items())),
        }


@dataclass
class UploadRows:
    admissions: list[dict[str, Any]] = field(default_factory=list)
    nodes_by_label: dict[str, list[dict[str, Any]]] = field(default_factory=lambda: defaultdict(list))
    graph_links: list[dict[str, Any]] = field(default_factory=list)
    edges_by_relation: dict[str, list[dict[str, Any]]] = field(default_factory=lambda: defaultdict(list))
    stats: UploadStats = field(default_factory=UploadStats)


def sanitize_identifier(value: Any, default: str = "IDENTIFIER") -> str:
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
    raw = str(value or "").strip()
    if not raw:
        raw = default
    key = PROPERTY_RE.sub("_", raw).strip("_").lower()
    if not key:
        key = default
    if key[0].isdigit():
        key = f"_{key}"
    return key


def escape_multiline_string(value: str) -> str:
    return value.replace("\r\n", "\\n").replace("\r", "\\n").replace("\n", "\\n")


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
                return escape_multiline_string(json.dumps(value, ensure_ascii=False, default=str))
            if item_scalar is not None:
                items.append(item_scalar)
        return items
    return escape_multiline_string(json.dumps(value, ensure_ascii=False, default=str))


def dedup_key(existing: Mapping[str, Any], key: str) -> str:
    if key not in existing:
        return key
    suffix = 2
    while f"{key}_{suffix}" in existing:
        suffix += 1
    return f"{key}_{suffix}"


def flatten_properties(
    raw: Mapping[str, Any],
    *,
    prefix: str = "",
    max_depth: int = 1,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in raw.items():
        safe_key = sanitize_property_key(key)
        full_key = f"{prefix}_{safe_key}" if prefix else safe_key
        if isinstance(value, Mapping) and max_depth > 0:
            for nested_key, nested_value in flatten_properties(
                value,
                prefix=full_key,
                max_depth=max_depth - 1,
            ).items():
                if nested_value is not None:
                    out[dedup_key(out, nested_key)] = nested_value
            continue
        coerced = coerce_property_value(value)
        if coerced is not None:
            out[dedup_key(out, full_key)] = coerced
    return out


def node_local_id(node: Mapping[str, Any], idx: int) -> str:
    return str(node.get("id") or node.get("node_id") or f"N_{idx:05d}")


def node_text(node: Mapping[str, Any], local_id: str) -> str:
    return str(node.get("text") or node.get("name") or node.get("normalized_name") or local_id)


def edge_source(edge: Mapping[str, Any]) -> str:
    return str(edge.get("source") or edge.get("source_id") or "")


def edge_target(edge: Mapping[str, Any]) -> str:
    return str(edge.get("target") or edge.get("target_id") or "")


def edge_relation(edge: Mapping[str, Any]) -> str:
    return str(edge.get("relation") or edge.get("type") or "RELATED_TO")


def kg_node_id(graph_id: str, local_id: str) -> str:
    return f"{graph_id}:{local_id}"


def build_upload_rows(
    records: Iterable[Mapping[str, Any]],
    *,
    include_note_embedding: bool,
    include_text_fields: bool,
    link_graphs: bool,
    source_tag: str,
) -> UploadRows:
    rows = UploadRows()
    for record in records:
        graph_id = graph_id_for(record)
        rows.stats.records += 1

        graph_props_raw = {}
        for key, value in record.items():
            if key in {"nodes", "edges"}:
                continue
            if key == "note_embedding" and not include_note_embedding:
                continue
            if key in TEXT_FIELD_KEYS and not include_text_fields:
                continue
            graph_props_raw[key] = value
        graph_props = flatten_properties(graph_props_raw, max_depth=1)
        graph_props["graph_id"] = graph_id
        if source_tag:
            graph_props["source_tag"] = source_tag
        rows.admissions.append({"graph_id": graph_id, "properties": graph_props})

        known_nodes: set[str] = set()
        for idx, node in enumerate(record.get("nodes", []) or []):
            local_id = node_local_id(node, idx)
            known_nodes.add(local_id)
            node_type = sanitize_identifier(node.get("type"), "ENTITY")
            props = flatten_properties(node, max_depth=1)
            props.update(
                {
                    "kg_node_id": kg_node_id(graph_id, local_id),
                    "graph_id": graph_id,
                    "source_node_id": local_id,
                    "node_type": node_type,
                    "text": escape_multiline_string(node_text(node, local_id)),
                }
            )
            rows.nodes_by_label[node_type].append(
                {
                    "kg_node_id": kg_node_id(graph_id, local_id),
                    "properties": props,
                }
            )
            rows.stats.nodes += 1
            rows.stats.node_labels[node_type] += 1
            if link_graphs:
                rows.graph_links.append(
                    {
                        "graph_id": graph_id,
                        "kg_node_id": kg_node_id(graph_id, local_id),
                    }
                )
                rows.stats.graph_links += 1

        for idx, edge in enumerate(record.get("edges", []) or []):
            source = edge_source(edge)
            target = edge_target(edge)
            if not source or not target or source not in known_nodes or target not in known_nodes:
                rows.stats.skipped_edges += 1
                continue
            relation = sanitize_identifier(edge_relation(edge), "RELATED_TO")
            edge_id = str(edge.get("kg_edge_id") or f"{graph_id}:E_{idx:05d}")
            props = flatten_properties(edge, max_depth=1)
            props.update(
                {
                    "kg_edge_id": edge_id,
                    "graph_id": graph_id,
                    "source_node_id": source,
                    "target_node_id": target,
                    "relation": relation,
                }
            )
            rows.edges_by_relation[relation].append(
                {
                    "kg_edge_id": edge_id,
                    "source_id": kg_node_id(graph_id, source),
                    "target_id": kg_node_id(graph_id, target),
                    "properties": props,
                }
            )
            rows.stats.edges += 1
            rows.stats.edge_relations[relation] += 1
    return rows


class FalkorUploader:
    def __init__(self, graph: Any, timeout_ms: int | None = None):
        self.graph = graph
        self.timeout_ms = timeout_ms

    @classmethod
    def connect(cls, args: argparse.Namespace) -> "FalkorUploader":
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

    def query(self, query: str, params: Mapping[str, Any] | None = None) -> Any:
        return self.graph.query(query, params=dict(params or {}), timeout=self.timeout_ms)

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
            except Exception as exc:
                LOG.warning("[FALKOR] index %s%s failed: %s", label, properties, exc)

    def upload_rows(self, rows: UploadRows, batch_no: int) -> None:
        tag = f"batch={batch_no}"
        if rows.admissions:
            LOG.info("[FALKOR] %s upload KGGraph=%s", tag, len(rows.admissions))
            self.query(
                """
                UNWIND $items AS item
                MERGE (g:KGGraph {graph_id: item.graph_id})
                SET g += item.properties
                SET g:Admission
                """,
                {"items": rows.admissions},
            )
        for label, nodes in sorted(rows.nodes_by_label.items()):
            LOG.info("[FALKOR] %s upload nodes label=%s count=%s", tag, label, len(nodes))
            self.query(
                f"""
                UNWIND $items AS item
                MERGE (n:KGNode {{kg_node_id: item.kg_node_id}})
                SET n += item.properties
                SET n:{label}
                """,
                {"items": nodes},
            )
        if rows.graph_links:
            LOG.info("[FALKOR] %s upload graph_links=%s", tag, len(rows.graph_links))
            self.query(
                """
                UNWIND $items AS item
                MATCH (g:KGGraph {graph_id: item.graph_id})
                MATCH (n:KGNode {kg_node_id: item.kg_node_id})
                MERGE (g)-[:CONTAINS_NODE]->(n)
                """,
                {"items": rows.graph_links},
            )
        for relation, edges in sorted(rows.edges_by_relation.items()):
            LOG.info("[FALKOR] %s upload edges relation=%s count=%s", tag, relation, len(edges))
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


def batched_jsonl(path: Path, batch_size: int) -> Iterator[list[dict[str, Any]]]:
    batch: list[dict[str, Any]] = []
    for record in iter_jsonl(path):
        batch.append(record)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def falkor_upload_stage(args: argparse.Namespace, store: ArtifactStore, kg_path: Path) -> Path:
    manifest_path = store.local("falkor_upload_manifest.json")

    uploader = None
    if not args.dry_run_falkor:
        uploader = FalkorUploader.connect(args)
        if args.drop_existing:
            LOG.info("[FALKOR] drop existing graph %r", args.falkor_graph)
            uploader.drop_existing()
        if not args.skip_indexes:
            uploader.ensure_indexes()

    total = UploadStats()
    for batch_no, batch in enumerate(batched_jsonl(kg_path, args.falkor_batch_size), start=1):
        rows = build_upload_rows(
            batch,
            include_note_embedding=args.include_note_embedding,
            include_text_fields=not args.exclude_text_fields,
            link_graphs=not args.no_graph_links,
            source_tag=args.source_tag,
        )
        if uploader is not None:
            uploader.upload_rows(rows, batch_no)
        total.add(rows.stats)
        LOG.info(
            "[FALKOR] batch=%s total records=%s nodes=%s edges=%s skipped=%s",
            batch_no,
            total.records,
            total.nodes,
            total.edges,
            total.skipped_edges,
        )

    manifest = {
        "stage": "upload_falkor",
        "dry_run": args.dry_run_falkor,
        "falkor_graph": args.falkor_graph,
        "falkor_url_set": bool(args.falkor_url),
        "source": kg_path.name,
        "stats": total.to_dict(),
        "uploaded_at_utc": utc_now(),
    }
    write_json(manifest_path, manifest)
    store.upload(manifest_path, "falkor_upload_manifest.json")
    LOG.info("[DONE] upload_falkor stats=%s", total.to_dict())
    return manifest_path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def env_default(name: str, default: Any = None) -> Any:
    return os.getenv(name, default)


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--workdir", default=env_default("WORKDIR"))
    parser.add_argument("--output-repo", default=env_default("OUTPUT_REPO"))
    parser.add_argument("--public-output-repo", action="store_true")
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument("--force", action="store_true", help="Rerun stages even if output JSONLs exist")
    parser.add_argument(
        "--stop-after",
        choices=("normalize", "build-kg", "embed-notes", "upload-falkor"),
        default=env_default("STOP_AFTER"),
    )

    parser.add_argument("--raw-local-file", default=env_default("RAW_LOCAL_FILE"))
    parser.add_argument("--raw-repo", default=env_default("RAW_REPO", "on1onmangoes/fawkes-mimic-megaflat-admission-260615"))
    parser.add_argument("--raw-file", default=env_default("RAW_FILE", "fawkes_mimic_megaflat_admission_260615.jsonl"))
    parser.add_argument("--n-patients", type=int, default=int(env_default("N_PATIENTS", 1000)))
    parser.add_argument("--seed", type=int, default=int(env_default("SEED", 42)))
    parser.add_argument("--min-dx", type=int, default=int(env_default("MIN_DX", 3)))
    parser.add_argument("--max-dx", type=int, default=int(env_default("MAX_DX", 40)))
    parser.add_argument("--min-med", type=int, default=int(env_default("MIN_MED", 3)))
    parser.add_argument("--max-med", type=int, default=int(env_default("MAX_MED", 60)))
    parser.add_argument("--min-svc", type=int, default=int(env_default("MIN_SVC", 1)))
    parser.add_argument("--no-eligibility-filter", action="store_true")
    parser.add_argument("--include-noteless", action="store_true")
    parser.add_argument("--limit-admissions", type=int, default=None)

    parser.add_argument("--llm-mode", choices=("auto", "off", "required"), default=env_default("LLM_MODE", "auto"))
    parser.add_argument("--endpoint-url", default=env_default("ENDPOINT_URL", ""))
    parser.add_argument("--endpoint-token-env", default=env_default("ENDPOINT_TOKEN_ENV", "HF_TOKEN"))
    parser.add_argument("--edge-model", default=env_default("EDGE_MODEL", "google/medgemma-27b-text-it"))
    parser.add_argument("--workers", type=int, default=int(env_default("WORKERS", 16)))
    parser.add_argument("--max-new-tokens", type=int, default=int(env_default("MAX_NEW_TOKENS", 15000)))

    parser.add_argument(
        "--embed-notes",
        action="store_true",
        default=env_bool("EMBED_NOTES"),
        help="Add Clinical ModernBERT mean-pooled note embeddings to kg_embedded.jsonl.",
    )
    parser.add_argument("--embed-model", default=env_default("EMBED_MODEL", "Simonlee711/Clinical_ModernBERT"))
    parser.add_argument("--embed-max-len", type=int, default=int(env_default("EMBED_MAX_LEN", 8192)))
    parser.add_argument("--embed-batch-size", type=int, default=int(env_default("EMBED_BATCH_SIZE", 32)))
    parser.add_argument(
        "--embed-device",
        choices=("auto", "cuda", "cpu", "mps"),
        default=env_default("EMBED_DEVICE", "auto"),
    )

    parser.add_argument("--falkor-graph", default=env_default("FALKORDB_GRAPH", "fawkes_clinical_kg"))
    parser.add_argument("--falkor-url", default=env_default("FALKORDB_URL"))
    parser.add_argument("--falkor-host", default=env_default("FALKORDB_HOST", "localhost"))
    parser.add_argument("--falkor-port", type=int, default=int(env_default("FALKORDB_PORT", 6379)))
    parser.add_argument("--falkor-username", default=env_default("FALKORDB_USERNAME"))
    parser.add_argument("--falkor-password", default=env_default("FALKORDB_PASSWORD"))
    parser.add_argument("--falkor-ssl", action="store_true")
    parser.add_argument("--timeout-ms", type=int, default=None)
    parser.add_argument("--connect-timeout-sec", type=float, default=10.0)
    parser.add_argument("--socket-timeout-sec", type=float, default=30.0)
    parser.add_argument("--falkor-batch-size", type=int, default=int(env_default("FALKOR_BATCH_SIZE", 50)))
    parser.add_argument("--drop-existing", action="store_true")
    parser.add_argument("--skip-indexes", action="store_true")
    parser.add_argument("--dry-run-falkor", action="store_true")
    parser.add_argument(
        "--include-note-embedding",
        action="store_true",
        default=env_bool("INCLUDE_NOTE_EMBEDDING"),
        help="Store top-level note_embedding arrays on KGGraph during Falkor upload.",
    )
    parser.add_argument("--exclude-text-fields", action="store_true")
    parser.add_argument("--no-graph-links", action="store_true")
    parser.add_argument("--source-tag", default=env_default("SOURCE_TAG", "fawkes_falkor_e2e"))

    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        default=env_default("LOG_LEVEL", "INFO"),
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.n_patients < 0:
        raise SystemExit("--n-patients must be >= 0. Use 0 for all eligible patients.")
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    if args.embed_max_len <= 0:
        raise SystemExit("--embed-max-len must be positive")
    if args.embed_batch_size <= 0:
        raise SystemExit("--embed-batch-size must be positive")
    if args.stop_after == "embed-notes" and not args.embed_notes:
        raise SystemExit("--stop-after embed-notes requires --embed-notes")
    if args.falkor_batch_size <= 0:
        raise SystemExit("--falkor-batch-size must be positive")
    if not args.local_only and not args.output_repo:
        raise SystemExit("--output-repo is required unless --local-only is set")


def write_pipeline_manifest(args: argparse.Namespace, store: ArtifactStore) -> None:
    manifest = {
        "created_at_utc": utc_now(),
        "script": Path(__file__).name,
        "raw_repo": args.raw_repo,
        "raw_file": args.raw_file,
        "raw_local_file": args.raw_local_file,
        "output_repo": args.output_repo,
        "falkor_graph": args.falkor_graph,
        "llm_mode": args.llm_mode,
        "embed_notes": args.embed_notes,
        "embed_model": args.embed_model if args.embed_notes else "",
        "include_note_embedding": args.include_note_embedding,
    }
    path = store.local("pipeline_manifest.json")
    write_json(path, manifest)
    store.upload(path, "pipeline_manifest.json")


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    validate_args(args)

    workdir = Path(args.workdir) if args.workdir else Path("workdir")
    workdir.mkdir(parents=True, exist_ok=True)
    store = ArtifactStore(
        repo_id=args.output_repo,
        workdir=workdir,
        private=not args.public_output_repo,
        local_only=args.local_only,
    )
    store.ensure_repo()
    write_pipeline_manifest(args, store)

    LOG.info("[ENTRY] workdir=%s output_repo=%s", workdir, args.output_repo)
    admissions = normalize_stage(args, store)
    if args.stop_after == "normalize":
        return
    kg = build_kg_stage(args, store, admissions)
    if args.stop_after == "build-kg":
        return
    upload_source = embed_notes_stage(args, store, kg) if args.embed_notes else kg
    if args.stop_after == "embed-notes":
        return
    falkor_upload_stage(args, store, upload_source)


if __name__ == "__main__":
    main()
