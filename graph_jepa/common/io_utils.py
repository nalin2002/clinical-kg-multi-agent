"""Config loading, path resolution, patient discovery, and graph/JSON I/O.

Centralises every filesystem concern so the rest of the codebase never builds
paths by hand and never hardcodes a patient id.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import yaml

from .graph_schema import Graph

LOG = logging.getLogger("graph_jepa")


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def setup_logging(level: str = "INFO") -> logging.Logger:
    if not LOG.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
        LOG.addHandler(handler)
    LOG.setLevel(getattr(logging, level.upper(), logging.INFO))
    return LOG


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
class Config:
    """Thin wrapper over the parsed YAML with dotted-key access + path resolve."""

    def __init__(self, data: dict, config_path: Path):
        self.data = data
        self.config_path = config_path
        root = data.get("project_root")
        if root:
            self.project_root = Path(root).expanduser().resolve()
        else:
            # graph_jepa/config.yaml -> repo root is two levels up.
            self.project_root = config_path.resolve().parent.parent
        _load_dotenv(self.project_root / ".env")

    def get(self, dotted: str, default=None):
        node = self.data
        for key in dotted.split("."):
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    def path(self, dotted: str, default: Optional[str] = None) -> Path:
        """Resolve a config path value against the project root."""
        value = self.get(dotted, default)
        if value is None:
            raise KeyError(f"config path {dotted!r} is not set")
        p = Path(value)
        return p if p.is_absolute() else (self.project_root / p)

    def resolve(self, value: str) -> Path:
        p = Path(value)
        return p if p.is_absolute() else (self.project_root / p)


def _load_dotenv(env_path: Path) -> None:
    """Load KEY=VALUE pairs from a .env file into os.environ (existing env wins).

    Uses python-dotenv if available, otherwise a small built-in parser so no
    extra dependency is required. Keys already set in the environment are not
    overridden.
    """
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv(env_path, override=False)
        LOG.info("loaded environment from %s (python-dotenv)", env_path)
        return
    except ImportError:
        pass
    loaded = []
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    if loaded:
        LOG.info("loaded environment from %s: %s", env_path, ", ".join(loaded))


def load_config(config_path: str | Path) -> Config:
    path = Path(config_path)
    with open(path) as f:
        data = yaml.safe_load(f)
    return Config(data, path)


# --------------------------------------------------------------------------- #
# Patient discovery
# --------------------------------------------------------------------------- #
@dataclass
class Patient:
    patient_id: str
    transcript_path: Optional[Path]
    note_path: Optional[Path]
    meta_path: Optional[Path]

    @property
    def has_transcript(self) -> bool:
        return self.transcript_path is not None and self.transcript_path.exists()

    @property
    def has_note(self) -> bool:
        return self.note_path is not None and self.note_path.exists()

    def read_transcript(self) -> Optional[str]:
        return self.transcript_path.read_text(encoding="utf-8") if self.has_transcript else None

    def read_note(self) -> Optional[str]:
        return self.note_path.read_text(encoding="utf-8") if self.has_note else None


def discover_patients(cfg: Config) -> List[Patient]:
    """List patients under data.aci_bench_dir. Missing notes/transcripts are
    tolerated — the relevant field is simply ``None`` and callers skip."""
    root = cfg.path("data.aci_bench_dir")
    if not root.exists():
        LOG.warning("ACI-Bench dir not found: %s", root)
        return []
    tsx = cfg.get("data.transcript_suffix", ".txt")
    nsx = cfg.get("data.note_suffix", "_note.txt")
    msx = cfg.get("data.meta_suffix", "_meta.json")
    allow = cfg.get("data.patient_ids")

    patients: List[Patient] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        pid = child.name
        if allow and pid not in allow:
            continue
        tp = child / f"{pid}{tsx}"
        npth = child / f"{pid}{nsx}"
        mp = child / f"{pid}{msx}"
        patients.append(
            Patient(
                patient_id=pid,
                transcript_path=tp if tp.exists() else None,
                note_path=npth if npth.exists() else None,
                meta_path=mp if mp.exists() else None,
            )
        )
    return patients


def load_training_graphs(cfg: Config, monitor_frac: float = 0.15):
    """Load graphs for SELF-SUPERVISED world-model training (Setup B).

    Source dir = ``paths.<training.train_graph_source>`` (default ``llm_graphs`` —
    the EIR 13-agent transcript graphs). Returns ``(train, monitor, source_key)``
    where the split is GRAPH-wise and used ONLY for loss/accuracy monitoring.

    Silver graphs are NOT used here — they are reserved for evaluation — so there
    is no train/eval leakage and no patient-level split is required.
    """
    import random

    source = cfg.get("training.train_graph_source", "llm_graphs")
    graphs = load_graphs(cfg.path(f"paths.{source}"))
    ids = sorted(graphs.keys())
    random.Random(int(cfg.get("training.seed", 13))).shuffle(ids)
    n_mon = max(1, int(round(len(ids) * monitor_frac))) if len(ids) > 3 else 0
    mon = set(ids[:n_mon])
    train = [graphs[i] for i in ids if i not in mon]
    monitor = [graphs[i] for i in ids if i in mon]
    return train, monitor, source


def split_patients(cfg: Config, patient_ids: List[str]) -> Dict[str, List[str]]:
    """Deterministic patient-level train/val/test split (no leakage)."""
    import random

    seed = int(cfg.get("data.split.seed", 13))
    tr = float(cfg.get("data.split.train", 0.6))
    va = float(cfg.get("data.split.val", 0.2))
    ids = sorted(patient_ids)
    random.Random(seed).shuffle(ids)
    n = len(ids)
    n_tr = int(round(n * tr))
    n_va = int(round(n * va))
    return {
        "train": sorted(ids[:n_tr]),
        "val": sorted(ids[n_tr:n_tr + n_va]),
        "test": sorted(ids[n_tr + n_va:]),
    }


# --------------------------------------------------------------------------- #
# JSON / graph I/O
# --------------------------------------------------------------------------- #
def read_json(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, obj, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False)


def write_jsonl(path: str | Path, rows: List[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def graph_path(directory: str | Path, patient_id: str) -> Path:
    return Path(directory) / f"{patient_id}.json"


def save_graph(directory: str | Path, graph: Graph) -> Path:
    p = graph_path(directory, graph.patient_id)
    write_json(p, graph.to_dict())
    return p


def load_graph(path: str | Path) -> Graph:
    return Graph.from_dict(read_json(path))


def load_graphs(directory: str | Path) -> Dict[str, Graph]:
    """Load every ``*.json`` graph in a directory, keyed by patient_id."""
    directory = Path(directory)
    out: Dict[str, Graph] = {}
    if not directory.exists():
        return out
    for jf in sorted(directory.glob("*.json")):
        if jf.name.endswith("_meta.json") or jf.name.endswith("_stats.json"):
            continue
        try:
            g = load_graph(jf)
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            LOG.warning("skipping %s: %s", jf.name, exc)
            continue
        out[g.patient_id or jf.stem] = g
    return out


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
