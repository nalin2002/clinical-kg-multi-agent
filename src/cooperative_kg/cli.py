"""CLI entry point for the cooperative multi-agent KG pipeline."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .clients import AnthropicClient, OpenRouterClient
from .constants import OUTPUT_SUFFIX, PROVIDER_MODELS
from .io import get_transcript_files
from .models import configure_models
from .paths import TRANSCRIPT_DIR
from .pipeline import process_one, run_all_batch


def load_client(provider: str):
    """Load API client based on provider. Reads api_keys.json for credentials."""
    with open("api_keys.json", encoding="utf-8") as f:
        api_keys = json.load(f)

    if provider == "anthropic":
        key = api_keys.get("anthropic")
        if not key:
            raise SystemExit('api_keys.json must contain a non-empty "anthropic" key')
        return AnthropicClient(key)

    key = api_keys.get("openrouter")
    if not key:
        raise SystemExit('api_keys.json must contain a non-empty "openrouter" key')
    return OpenRouterClient(key)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cooperative multi-agent clinical KG extractor")
    parser.add_argument("--output", required=True, help="Output directory for per-transcript KG JSON files")
    parser.add_argument("--res-ids", nargs="+", default=None, help="Optional RES IDs to process")
    parser.add_argument(
        "--transcripts-dir",
        default=None,
        help="Directory of RES*/RES*.txt transcripts (e.g. ACI-Bench). "
        f"Default: {TRANSCRIPT_DIR}",
    )
    parser.add_argument(
        "--provider",
        choices=list(PROVIDER_MODELS),
        default=None,
        help='API provider: "openrouter" or "anthropic". '
        'Overrides the "provider" field in api_keys.json.',
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Use Anthropic Message Batches API for 50%% cost reduction "
        "(requires --provider anthropic). Processes all transcripts in "
        "staged batches instead of one-at-a-time.",
    )
    args = parser.parse_args()

    if args.provider:
        provider = args.provider
    else:
        try:
            with open("api_keys.json", encoding="utf-8") as f:
                provider = json.load(f).get("provider", "openrouter")
        except FileNotFoundError:
            provider = "openrouter"

    if args.batch and provider != "anthropic":
        raise SystemExit("--batch requires --provider anthropic")

    configure_models(provider)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    transcript_dir = Path(args.transcripts_dir) if args.transcripts_dir else TRANSCRIPT_DIR
    client = load_client(provider)
    transcript_files = get_transcript_files(args.res_ids, transcript_dir)

    print("Cooperative Multi-Agent KG Pipeline")
    print(f"Provider: {provider}" + (" (batch mode)" if args.batch else ""))
    print(f"Models: {PROVIDER_MODELS[provider]}")
    print(f"Output: {output_dir}")
    print(f"Transcripts dir: {transcript_dir}")
    print(f"Transcripts: {len(transcript_files)}")
    print("=" * 60)

    if args.batch:
        success, failed, total_usage, details = run_all_batch(
            transcript_files, client, output_dir, PROVIDER_MODELS[provider]
        )
        total_tokens = {
            "prompt": total_usage.get("prompt_tokens", 0),
            "completion": total_usage.get("completion_tokens", 0),
        }
    else:
        success = failed = 0
        total_tokens = {"prompt": 0, "completion": 0}
        details: list[dict] = []

        for txt_path in transcript_files:
            try:
                res_id, status, nodes, edges, usage = process_one(txt_path, client, output_dir)
            except Exception as exc:
                res_id = txt_path.stem
                status, nodes, edges, usage = f"ERROR: {exc}", 0, 0, {}
                print(f"  {res_id}: {status}", flush=True)

            if status in {"OK", "SKIP"}:
                success += 1
            else:
                failed += 1
            total_tokens["prompt"] += usage.get("prompt_tokens", 0)
            total_tokens["completion"] += usage.get("completion_tokens", 0)
            details.append({"res_id": res_id, "status": status, "nodes": nodes, "edges": edges, **usage})
            time.sleep(0.2)

    stats = {
        "method": OUTPUT_SUFFIX,
        "provider": provider,
        "batch_mode": args.batch,
        "models": PROVIDER_MODELS[provider],
        "total_tokens": total_tokens,
        "success": success,
        "failed": failed,
        "details": details,
    }
    (output_dir / "_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print("=" * 60)
    print(f"Done. success={success} failed={failed}")
    print(f"Total tokens: {total_tokens['prompt'] + total_tokens['completion']}")
    print(f"Output: {output_dir}")
