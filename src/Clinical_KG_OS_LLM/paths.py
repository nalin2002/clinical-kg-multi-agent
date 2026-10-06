"""Repository-root paths (resolved from this file, not the process cwd)."""

from pathlib import Path


def project_root() -> Path:
    """Project root (parent of ``src/``)."""
    return Path(__file__).resolve().parent.parent.parent


def transcripts_dir() -> Path:
    """Evaluation bundle: per-patient dirs with ``.txt`` and ``*_standard_answer.json``."""
    return project_root() / "data" / "transcripts"


def curated_baseline_unified_path() -> Path:
    """Merged human-curated reference KG (build via ``stage_curated_subkgs`` + ``dump_graph``)."""
    return project_root() / "data" / "baseline_curated" / "unified_graph_curated.json"
