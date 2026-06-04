"""Transcript and note file I/O."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .paths import TRANSCRIPT_DIR

NOTE_HEADER = (
    "=== CLINICAL NOTE (post-visit clinician summary; use as supporting "
    "context for extraction) ==="
)


def read_transcript(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def read_note(txt_path: Path) -> str:
    """Load the sibling ``<res_id>_note.txt`` clinical note, if it exists."""
    note_path = txt_path.with_name(f"{txt_path.stem}_note.txt")
    return note_path.read_text(encoding="utf-8") if note_path.exists() else ""


def build_source_text(transcript: str, note: str = "") -> str:
    """Combine the dialogue transcript with an optional clinical note."""
    note = (note or "").strip()
    if not note:
        return transcript
    return f"{transcript}\n\n{NOTE_HEADER}\n{note}"


def get_transcript_files(
    res_ids: Optional[list[str]] = None, transcript_dir: Optional[Path] = None
) -> list[Path]:
    base = transcript_dir or TRANSCRIPT_DIR
    files = [
        d / f"{d.name}.txt"
        for d in sorted(base.glob("RES*"))
        if d.is_dir() and (d / f"{d.name}.txt").exists()
    ]
    if res_ids:
        wanted = set(res_ids)
        files = [f for f in files if f.parent.name in wanted]
    return files
