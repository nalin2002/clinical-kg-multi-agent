"""Default transcript directory resolution."""

from __future__ import annotations

from pathlib import Path


def default_transcripts_dir() -> Path:
    """Resolve the default transcript directory.

    Prefers the sibling ``Clinical_KG_OS_LLM`` package when it is importable
    (the original layout), otherwise falls back to this repo's bundled
    ``data/transcripts``. Override per-run with ``--transcripts-dir``.
    """
    try:
        from Clinical_KG_OS_LLM.paths import transcripts_dir  # type: ignore

        return transcripts_dir()
    except Exception:
        return Path(__file__).resolve().parent.parent.parent / "data" / "transcripts"


TRANSCRIPT_DIR = default_transcripts_dir()
