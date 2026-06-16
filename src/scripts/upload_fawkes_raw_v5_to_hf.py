#!/usr/bin/env python3
"""Upload the Fawkes raw Graph-JEPA v5 checkpoint bundle to Hugging Face.

Usage:
    python src/scripts/upload_fawkes_raw_v5_to_hf.py --repo-id USER/REPO --dry-run
    HF_TOKEN=... python src/scripts/upload_fawkes_raw_v5_to_hf.py --repo-id USER/REPO
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi


DEFAULT_MODEL_DIR = Path("ckpts/fawkes_raw_v5")
DEFAULT_EVAL_DIR = Path("outputs/fawkes_raw_global_v5")
MODEL_FILES = (
    "graph_jepa_v5.pt",
    "graph_jepa_v5_pretrain.pt",
    "config_v5.json",
    "config_v5_pretrain.json",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _required_model_files(model_dir: Path) -> list[tuple[Path, str]]:
    files = [(model_dir / name, name) for name in MODEL_FILES]
    missing = [str(path) for path, _name in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "missing required model artifact(s): " + ", ".join(missing)
        )
    return files


def _readme_operation(repo_id: str, model_dir: Path, eval_dir: Path | None) -> CommitOperationAdd:
    config = _read_json(model_dir / "config_v5.json")
    loo_eval = _maybe_read_eval(eval_dir, "loo_eval.json") if eval_dir else None
    llm_eval = _maybe_read_eval(eval_dir, "llm_vs_jepa_eval.json") if eval_dir else None

    lines = [
        "---",
        "library_name: pytorch",
        "tags:",
        "- graph-jepa",
        "- clinical-kg",
        "- pytorch",
        "---",
        "",
        "# Fawkes Raw Graph-JEPA v5",
        "",
        "Graph-JEPA v5 checkpoint trained on the Fawkes raw global graph split.",
        "",
        "## Files",
        "",
        "- `graph_jepa_v5.pt`: fine-tuned checkpoint",
        "- `graph_jepa_v5_pretrain.pt`: masked-pretraining checkpoint",
        "- `config_v5.json`: fine-tuned checkpoint config",
        "- `config_v5_pretrain.json`: masked-pretraining config",
        "- `README.md`: generated model card with summary metrics, when present",
        "",
        "## Configuration",
        "",
        f"- Encoder: `{config.get('encoder', 'unknown')}`",
        f"- Input dimension: `{config.get('model', {}).get('in_dim', 'unknown')}`",
        f"- Hidden dimension: `{config.get('model', {}).get('hidden_dim', 'unknown')}`",
        f"- Relations: `{config.get('model', {}).get('num_relations', 'unknown')}`",
        f"- Patches: `{config.get('model', {}).get('num_patches', 'unknown')}`",
    ]

    metrics = (loo_eval or {}).get("metrics", {})
    if metrics:
        lines.extend(
            [
                "",
                "## Leave-One-Out Evaluation",
                "",
                f"- Candidate mode: `{loo_eval.get('candidate_mode', 'unknown')}`",
                f"- Examples: `{metrics.get('n', 'unknown')}`",
                f"- MRR: `{metrics.get('mrr', 'unknown')}`",
                f"- Hits@1: `{metrics.get('hits1', 'unknown')}`",
                f"- Hits@3: `{metrics.get('hits3', 'unknown')}`",
                f"- Hits@10: `{metrics.get('hits10', 'unknown')}`",
            ]
        )

    jepa_metrics = (llm_eval or {}).get("jepa", {})
    llm_metrics = (llm_eval or {}).get("llm", {})
    if jepa_metrics and llm_metrics:
        lines.extend(
            [
                "",
                "## JEPA vs LLM Sample Evaluation",
                "",
                f"- JEPA MRR: `{jepa_metrics.get('mrr', 'unknown')}`",
                f"- JEPA Hits@1: `{jepa_metrics.get('hits1', 'unknown')}`",
                f"- LLM MRR: `{llm_metrics.get('mrr', 'unknown')}`",
                f"- LLM Hits@1: `{llm_metrics.get('hits1', 'unknown')}`",
            ]
        )

    lines.extend(
        [
            "",
            "## Loading",
            "",
            "```python",
            "import torch",
            "from graph_jepa_v5.training import load_model_checkpoint",
            "",
            "model, cfg = load_model_checkpoint(",
            '    "graph_jepa_v5.pt",',
            "    torch.device(\"cpu\"),",
            ")",
            "```",
            "",
            "This checkpoint uses the custom Graph-JEPA v5 code in this repository.",
            "Review dataset and license constraints before making this repo public.",
            "",
            f"Hub repo: `{repo_id}`",
        ]
    )
    content = "\n".join(lines).encode("utf-8") + b"\n"
    return CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=content)


def _maybe_read_eval(eval_dir: Path | None, name: str) -> dict | None:
    if eval_dir is None:
        return None
    path = eval_dir / name
    if not path.is_file():
        return None
    return _read_json(path)


def _build_operations(
    repo_id: str,
    model_dir: Path,
    eval_dir: Path | None,
) -> list[CommitOperationAdd]:
    operations = [
        CommitOperationAdd(path_in_repo=name, path_or_fileobj=path)
        for path, name in _required_model_files(model_dir)
    ]
    operations.append(_readme_operation(repo_id, model_dir, eval_dir))
    return operations


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Destination Hugging Face model repo, e.g. USER/fawkes-raw-v5.",
    )
    parser.add_argument(
        "--model-dir",
        default=str(DEFAULT_MODEL_DIR),
        help=f"Directory containing v5 checkpoint files (default: {DEFAULT_MODEL_DIR}).",
    )
    parser.add_argument(
        "--eval-dir",
        default=str(DEFAULT_EVAL_DIR),
        help=(
            "Local directory containing optional evaluation summaries for README metrics "
            f"(default: {DEFAULT_EVAL_DIR})."
        ),
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Do not read local evaluation summaries into the generated README.",
    )
    visibility = parser.add_mutually_exclusive_group()
    visibility.add_argument(
        "--private",
        dest="private",
        action="store_true",
        default=True,
        help="Create the repo as private when it does not exist (default).",
    )
    visibility.add_argument(
        "--public",
        dest="private",
        action="store_false",
        help="Create the repo as public when it does not exist.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional target branch or revision.",
    )
    parser.add_argument(
        "--commit-message",
        default="Upload Fawkes raw Graph-JEPA v5 checkpoint",
        help="Commit message for the Hub upload.",
    )
    parser.add_argument(
        "--token-env",
        default="HF_TOKEN",
        help="Environment variable containing a Hugging Face token.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print files that would be uploaded without contacting Hugging Face.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    root = _repo_root()
    model_dir = (root / args.model_dir).resolve()
    eval_dir = None if args.no_eval else (root / args.eval_dir).resolve()
    token = os.getenv(args.token_env)

    operations = _build_operations(args.repo_id, model_dir, eval_dir)
    if args.dry_run:
        print(f"[hf-upload] repo_id={args.repo_id}")
        print(f"[hf-upload] private={args.private}")
        print(f"[hf-upload] revision={args.revision or 'default'}")
        print("[hf-upload] files:")
        for operation in operations:
            print(f"  {operation.path_in_repo}")
        return

    api = HfApi(token=token)
    repo_url = api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=args.private,
        exist_ok=True,
    )
    commit = api.create_commit(
        repo_id=args.repo_id,
        repo_type="model",
        revision=args.revision,
        operations=operations,
        commit_message=args.commit_message,
    )
    print(f"[hf-upload] repo: {repo_url}")
    print(f"[hf-upload] commit: {commit.commit_url}")


if __name__ == "__main__":
    main()
