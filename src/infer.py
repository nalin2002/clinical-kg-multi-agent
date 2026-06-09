"""End-to-end inference: transcripts -> sub-KGs -> unified KG -> JEPA-refined KG.

Usage::

    PYTHONPATH=src python src/infer.py \\
        --transcripts-dir data/transcripts \\
        --output outputs/my_run \\
        --checkpoint checkpoints/graph_jepa.pt

Steps:
  1. Run the cooperative multi-agent extractor on each transcript.
  2. Merge sub-KGs into a unified graph via embedding entity resolution.
  3. Score the unified graph with a Graph-JEPA checkpoint to produce the
     final refined KG (annotate-only by default; pass --prune-threshold to drop
     low-scoring edges).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from cooperative_kg.cli import load_client
from cooperative_kg.constants import OUTPUT_SUFFIX, PROVIDER_MODELS
from cooperative_kg.credentials import resolve_provider
from cooperative_kg.dump_graph import (
    ENCODER_LABELS,
    ER_SIMILARITY_THRESHOLD,
    build_unified_graph,
    entity_resolution,
    load_encoder,
    load_sub_kgs,
)
from cooperative_kg.io import get_transcript_files
from cooperative_kg.models import configure_models
from cooperative_kg.pipeline import process_one, run_all_batch
from graph_jepa.schema import PatientGraph


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _default_checkpoint(jepa_module: str) -> Path:
    name = "graph_jepa_v2.pt" if jepa_module == "graph_jepa_v2" else "graph_jepa.pt"
    return _repo_root() / "checkpoints" / name


def _load_jepa(jepa_module: str):
    if jepa_module == "graph_jepa_v2":
        from graph_jepa_v2.score import _load, _prune, score_graph

        return _load, score_graph, _prune, ".cache/graph_jepa_v2/encoder"
    from graph_jepa.score import _load, _prune, score_graph

    return _load, score_graph, _prune, ".cache/graph_jepa/bge"


def run_extraction(
    transcript_dir: Path,
    sub_kg_dir: Path,
    res_ids: list[str] | None,
    provider: str | None,
    batch: bool,
) -> dict:
    if batch and provider is None:
        provider = "anthropic"
    else:
        provider = resolve_provider(provider)
    if batch and provider != "anthropic":
        raise SystemExit("Batch mode requires --provider anthropic (or pass --no-batch for sequential extraction)")

    configure_models(provider)
    sub_kg_dir.mkdir(parents=True, exist_ok=True)

    client = load_client(provider)
    transcript_files = get_transcript_files(res_ids, transcript_dir)

    print("Step 1: Multi-agent KG extraction")
    print(f"  Provider   : {provider}" + (" (batch mode)" if batch else ""))
    print(f"  Transcripts: {len(transcript_files)} from {transcript_dir}")
    print(f"  Output     : {sub_kg_dir}")
    print("=" * 60)

    if batch:
        success, failed, total_usage, details = run_all_batch(
            transcript_files, client, sub_kg_dir, PROVIDER_MODELS[provider]
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
                res_id, status, nodes, edges, usage = process_one(txt_path, client, sub_kg_dir)
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
        "step": "extraction",
        "method": OUTPUT_SUFFIX,
        "provider": provider,
        "batch_mode": batch,
        "success": success,
        "failed": failed,
        "total_tokens": total_tokens,
        "details": details,
    }
    print(f"  Done: success={success} failed={failed}")
    return stats


def run_unification(
    sub_kg_dir: Path,
    output_dir: Path,
    name: str,
    encoder: str,
    threshold: float,
    cache_dir: str | None,
) -> tuple[Path, Path, dict]:
    print("\nStep 2: Unified KG (entity resolution merge)")
    print(f"  Input      : {sub_kg_dir}")
    print(f"  Encoder    : {encoder} ({ENCODER_LABELS[encoder]})")
    print(f"  Threshold  : {threshold}")
    print("=" * 60)

    embed_model = load_encoder(encoder, cache_dir)
    all_raw_entities, all_raw_edges, pass1_total, pass2_total, loaded = load_sub_kgs(sub_kg_dir)
    if loaded == 0:
        raise SystemExit(f"No sub-KG files found in {sub_kg_dir}")

    print(f"  Loaded {loaded} sub-KGs: {len(all_raw_entities)} entities, {len(all_raw_edges)} edges")

    unique_before = len({e["text"].strip().lower() for e in all_raw_entities})
    canonical_map, merge_decisions = entity_resolution(all_raw_entities, embed_model, threshold)
    unique_after = len(set(canonical_map.values()))
    print(f"  Entity resolution: {unique_before} -> {unique_after} canonical entities")

    nodes, edges = build_unified_graph(all_raw_entities, all_raw_edges, canonical_map)

    graph_path = output_dir / f"unified_graph_{name}.json"
    er_path = output_dir / f"er_merge_decisions_{name}.json"

    er_path.write_text(
        json.dumps(
            {
                "method": "embedding_similarity",
                "embedding_model": ENCODER_LABELS[encoder],
                "similarity_threshold": threshold,
                "total_raw_entities": len(all_raw_entities),
                "unique_before": unique_before,
                "unique_after": unique_after,
                "pass1_totals": pass1_total,
                "pass2_totals": pass2_total,
                "merge_decisions": merge_decisions,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    graph_path.write_text(
        json.dumps({"nodes": nodes, "edges": edges}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"  Saved unified graph: {graph_path} ({len(nodes)} nodes, {len(edges)} edges)")
    print(f"  Saved ER decisions  : {er_path}")

    stats = {
        "step": "unification",
        "sub_kgs_loaded": loaded,
        "nodes": len(nodes),
        "edges": len(edges),
        "unique_before": unique_before,
        "unique_after": unique_after,
        "merge_clusters": len(merge_decisions),
        "unified_graph": str(graph_path),
        "er_decisions": str(er_path),
    }
    return graph_path, er_path, stats


def run_jepa_refinement(
    unified_graph_path: Path,
    refined_graph_path: Path,
    checkpoint: Path,
    jepa_module: str,
    device: str,
    encoder_cache: str | None,
    prune_threshold: float | None,
) -> dict:
    import torch

    if not checkpoint.is_file():
        raise SystemExit(f"JEPA checkpoint not found: {checkpoint}")

    load_ckpt, score_graph, prune_fn, default_cache = _load_jepa(jepa_module)
    cache = encoder_cache or default_cache

    print("\nStep 3: Graph-JEPA refinement")
    print(f"  Input      : {unified_graph_path}")
    print(f"  Checkpoint : {checkpoint}")
    print(f"  Module     : {jepa_module}")
    print(f"  Device     : {device}")
    if prune_threshold is not None:
        print(f"  Prune      : drop edges with jepa_score < {prune_threshold}")
    print("=" * 60)

    torch_device = torch.device(device)
    model, encoder, cfg = load_ckpt(str(checkpoint), torch_device, cache)
    if prune_threshold is not None:
        cfg.score.prune_threshold = prune_threshold

    graph = PatientGraph.load(unified_graph_path)
    scores, flags = score_graph(graph, model, encoder, cfg, torch_device)
    graph.annotate_edges(scores, flags)

    pruned = 0
    if cfg.score.prune_threshold is not None:
        pruned = prune_fn(graph, cfg.score.prune_threshold)

    refined_graph_path.parent.mkdir(parents=True, exist_ok=True)
    graph.save(refined_graph_path)

    flagged = sum(1 for f in flags if f != "ok")
    print(
        f"  Saved refined graph: {refined_graph_path} "
        f"({len(graph.nodes)} nodes, {len(graph.edges)} edges, "
        f"{flagged} flagged" + (f", {pruned} pruned" if pruned else "") + ")"
    )

    return {
        "step": "jepa_refinement",
        "module": jepa_module,
        "checkpoint": str(checkpoint),
        "nodes": len(graph.nodes),
        "edges": len(graph.edges),
        "edges_scored": len(scores),
        "edges_flagged": flagged,
        "edges_pruned": pruned,
        "refined_graph": str(refined_graph_path),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="End-to-end inference: transcripts -> sub-KGs -> unified KG -> JEPA-refined KG"
    )
    p.add_argument(
        "--transcripts-dir",
        default=None,
        help="Directory of RES*/RES*.txt transcript folders (required unless --skip-extract)",
    )
    p.add_argument(
        "--output",
        required=True,
        help="Output root directory (writes sub_kgs/, unified graph, refined graph)",
    )
    p.add_argument(
        "--name",
        default=None,
        help="Suffix for unified/refined graph filenames (default: output dir name)",
    )
    p.add_argument("--res-ids", nargs="+", default=None, help="Optional RES IDs to process")
    p.add_argument(
        "--provider",
        choices=list(PROVIDER_MODELS),
        default=None,
        help='LLM provider: "openrouter" or "anthropic" (defaults to anthropic for batch extraction)',
    )
    p.add_argument(
        "--no-batch",
        action="store_true",
        help="Disable Anthropic Message Batches API and extract transcripts one at a time",
    )

    p.add_argument(
        "--skip-extract",
        action="store_true",
        help="Skip extraction and reuse existing sub-KGs in <output>/sub_kgs/",
    )
    p.add_argument(
        "--skip-unify",
        action="store_true",
        help="Skip unification and reuse existing unified graph in <output>/",
    )
    p.add_argument(
        "--skip-jepa",
        action="store_true",
        help="Skip JEPA refinement",
    )

    p.add_argument(
        "--encoder",
        choices=sorted(ENCODER_LABELS),
        default="bge",
        help="Embedding encoder for entity resolution (default: bge)",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=ER_SIMILARITY_THRESHOLD,
        help=f"ER similarity threshold (default: {ER_SIMILARITY_THRESHOLD})",
    )
    p.add_argument(
        "--cache-dir",
        default=None,
        help="Embedding cache directory for entity resolution",
    )

    p.add_argument(
        "--checkpoint",
        default=None,
        help="Graph-JEPA checkpoint .pt (default: checkpoints/graph_jepa.pt or v2 variant)",
    )
    p.add_argument(
        "--jepa-module",
        choices=["graph_jepa", "graph_jepa_v2"],
        default="graph_jepa",
        help="Graph-JEPA implementation to use for scoring (default: graph_jepa)",
    )
    p.add_argument("--device", default="cpu", help="Torch device for JEPA scoring")
    p.add_argument(
        "--encoder-cache",
        default=None,
        help="Encoder cache directory for JEPA scoring (module-specific default if omitted)",
    )
    p.add_argument(
        "--prune-threshold",
        type=float,
        default=None,
        help="Drop unified-graph edges with jepa_score below this value",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)

    if not args.skip_extract and not args.transcripts_dir:
        raise SystemExit("--transcripts-dir is required unless --skip-extract is set")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    transcript_dir = Path(args.transcripts_dir) if args.transcripts_dir else Path(".")
    sub_kg_dir = output_dir / "sub_kgs"
    name = args.name or output_dir.name

    unified_graph_path = output_dir / f"unified_graph_{name}.json"
    refined_graph_path = output_dir / f"unified_graph_{name}_refined.json"
    checkpoint = Path(args.checkpoint) if args.checkpoint else _default_checkpoint(args.jepa_module)

    print("Clinical KG Inference Pipeline")
    print(f"Transcripts : {transcript_dir}")
    print(f"Output root : {output_dir}")
    print("=" * 60)

    manifest: dict = {
        "transcripts_dir": str(transcript_dir),
        "output_dir": str(output_dir),
        "name": name,
        "steps": {},
    }

    if not args.skip_extract:
        manifest["steps"]["extraction"] = run_extraction(
            transcript_dir, sub_kg_dir, args.res_ids, args.provider, batch=not args.no_batch
        )
    else:
        print("Step 1: Skipped (--skip-extract)")
        if not sub_kg_dir.is_dir():
            raise SystemExit(f"--skip-extract set but sub-KG dir missing: {sub_kg_dir}")

    if not args.skip_unify:
        _, _, unify_stats = run_unification(
            sub_kg_dir, output_dir, name, args.encoder, args.threshold, args.cache_dir
        )
        manifest["steps"]["unification"] = unify_stats
    else:
        print("\nStep 2: Skipped (--skip-unify)")
        if not unified_graph_path.is_file():
            raise SystemExit(f"--skip-unify set but unified graph missing: {unified_graph_path}")
        manifest["steps"]["unification"] = {"skipped": True, "unified_graph": str(unified_graph_path)}

    if not args.skip_jepa:
        jepa_stats = run_jepa_refinement(
            unified_graph_path,
            refined_graph_path,
            checkpoint,
            args.jepa_module,
            args.device,
            args.encoder_cache,
            args.prune_threshold,
        )
        manifest["steps"]["jepa_refinement"] = jepa_stats
    else:
        print("\nStep 3: Skipped (--skip-jepa)")

    manifest_path = output_dir / "_infer_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print("Inference complete.")
    print(f"  Sub-KGs       : {sub_kg_dir}")
    print(f"  Unified KG    : {unified_graph_path}")
    if not args.skip_jepa:
        print(f"  Refined KG    : {refined_graph_path}")
    print(f"  Manifest      : {manifest_path}")


if __name__ == "__main__":
    main()
