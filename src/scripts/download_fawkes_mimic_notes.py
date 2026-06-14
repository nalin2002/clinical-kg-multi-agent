#!/usr/bin/env python3
"""Download raw Fawkes MIMIC notes JSONL files from Hugging Face.

Usage:
    python download_fawkes_mimic_notes.py
    HF_TOKEN=... python download_fawkes_mimic_notes.py --out data/fawkes_mimic_notes

This downloads the repository's raw ``*.jsonl`` files. It does not use
``datasets.save_to_disk``, which writes Apache Arrow files.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download


DATASET_ID = "wmatbooth/fawkes-mimic-notes-1k-v3-rows4000-5000-260612"
DEFAULT_OUT = Path("data/fawkes-mimic-notes-1k-v3-rows4000-5000-260612")


def _file_summary(out_dir: Path) -> list[dict[str, object]]:
    return [
        {
            "path": str(path.relative_to(out_dir)),
            "bytes": path.stat().st_size,
        }
        for path in sorted(out_dir.rglob("*.jsonl"))
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUT),
        help=f"Directory to write the downloaded dataset (default: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional Hugging Face revision, branch, or commit SHA.",
    )
    parser.add_argument(
        "--token-env",
        default="HF_TOKEN",
        help="Environment variable containing a Hugging Face token, if needed.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    token = os.getenv(args.token_env)

    snapshot_download(
        repo_id=DATASET_ID,
        repo_type="dataset",
        revision=args.revision,
        token=token,
        local_dir=str(out_dir),
        allow_patterns=["*.jsonl", "**/*.jsonl"],
    )

    files = _file_summary(out_dir)
    if not files:
        raise SystemExit(
            "[fawkes-mimic-notes] no .jsonl files were downloaded. "
            "Check the dataset repo contents or revision."
        )

    manifest = {
        "hf_dataset": DATASET_ID,
        "revision": args.revision,
        "output_dir": str(out_dir),
        "files": files,
    }
    manifest_path = out_dir / "_download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"[fawkes-mimic-notes] downloaded raw JSONL files from {DATASET_ID}")
    print(f"[fawkes-mimic-notes] saved to {out_dir}")
    print(f"[fawkes-mimic-notes] manifest: {manifest_path}")
    for info in files:
        print(f"  {info['path']}: {info['bytes']} bytes")


if __name__ == "__main__":
    main()
