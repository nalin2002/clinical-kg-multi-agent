#!/usr/bin/env python3
"""Download raw Fawkes MIMIC graph files from Hugging Face.

Usage:
    python download_fawkes_mimic_graphs.py
    HF_TOKEN=... python download_fawkes_mimic_graphs.py --out data/fawkes_mimic_graphs

This mirrors ``download_fawkes_mimic_notes.py`` but targets the final graph
dataset. It downloads raw data files directly from the dataset repository
instead of using ``datasets.save_to_disk``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download


DATASET_ID = "wmatbooth/fawkes-mimic-graphs-complete-v8-rows4000-5000-260613"
DEFAULT_OUT = Path("data/fawkes-mimic-graphs-complete-v8-rows4000-5000-260613")
DATA_FILE_SUFFIXES = (
    ".json",
    ".jsonl",
    ".jsonl.gz",
    ".parquet",
    ".csv",
    ".csv.gz",
)
ALLOW_PATTERNS = [
    "*.json",
    "**/*.json",
    "*.jsonl",
    "**/*.jsonl",
    "*.jsonl.gz",
    "**/*.jsonl.gz",
    "*.parquet",
    "**/*.parquet",
    "*.csv",
    "**/*.csv",
    "*.csv.gz",
    "**/*.csv.gz",
]


def _is_data_file(path: Path) -> bool:
    name = path.name.lower()
    return any(name.endswith(suffix) for suffix in DATA_FILE_SUFFIXES)


def _file_summary(out_dir: Path) -> list[dict[str, object]]:
    return [
        {
            "path": str(path.relative_to(out_dir)),
            "bytes": path.stat().st_size,
        }
        for path in sorted(out_dir.rglob("*"))
        if path.is_file() and _is_data_file(path)
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
        allow_patterns=ALLOW_PATTERNS,
    )

    files = _file_summary(out_dir)
    if not files:
        raise SystemExit(
            "[fawkes-mimic-graphs] no graph data files were downloaded. "
            "Check the dataset repo contents, revision, or allow patterns."
        )

    manifest = {
        "hf_dataset": DATASET_ID,
        "revision": args.revision,
        "output_dir": str(out_dir),
        "allow_patterns": ALLOW_PATTERNS,
        "files": files,
    }
    manifest_path = out_dir / "_download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"[fawkes-mimic-graphs] downloaded raw graph files from {DATASET_ID}")
    print(f"[fawkes-mimic-graphs] saved to {out_dir}")
    print(f"[fawkes-mimic-graphs] manifest: {manifest_path}")
    for info in files:
        print(f"  {info['path']}: {info['bytes']} bytes")


if __name__ == "__main__":
    main()
