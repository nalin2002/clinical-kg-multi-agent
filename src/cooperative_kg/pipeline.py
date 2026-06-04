"""End-to-end cooperative KG extraction pipeline."""

from __future__ import annotations

import json
from pathlib import Path

from .agents import (
    agent1_high_recall_entities,
    agent2_precision_filter,
    agent3_negations,
    agent4_relations,
    agent5_canonicalize,
    apply_canonicalization,
    merge_nodes,
)
from .clients import AnthropicClient
from .constants import OUTPUT_SUFFIX
from .io import build_source_text, read_note, read_transcript
from .prompts import (
    build_agent1_prompt,
    build_agent2_prompt,
    build_agent3_prompt,
    build_agent4_prompt,
    build_agent5_prompt,
)
from .text_utils import extract_json
from .validation import deterministic_human_style_enrichment, deterministic_validator


def run_pipeline(transcript: str, client, note: str = "") -> tuple[dict, dict]:
    usage_total = {"prompt_tokens": 0, "completion_tokens": 0}
    source = build_source_text(transcript, note)

    def add_usage(usage: dict) -> None:
        usage_total["prompt_tokens"] += usage.get("prompt_tokens", 0)
        usage_total["completion_tokens"] += usage.get("completion_tokens", 0)

    print("    [1/6] high-recall entities...", end=" ", flush=True)
    candidates, usage = agent1_high_recall_entities(source, client)
    add_usage(usage)
    print(f"{len(candidates)} candidates", flush=True)

    print("    [2/6] precision filter...", end=" ", flush=True)
    filtered, usage = agent2_precision_filter(candidates, source, client)
    add_usage(usage)
    print(f"{len(filtered)} kept", flush=True)

    print("    [3/6] negations...", end=" ", flush=True)
    negations, usage = agent3_negations(source, client)
    add_usage(usage)
    nodes = merge_nodes(filtered + negations)
    print(f"{len(negations)} negations, {len(nodes)} merged nodes", flush=True)

    print("    [4/6] relations...", end=" ", flush=True)
    edges, usage = agent4_relations(nodes, source, client)
    add_usage(usage)
    kg = {"nodes": nodes, "edges": edges}
    print(f"{len(edges)} candidate edges", flush=True)

    print("    [5/6] canonicalization...", end=" ", flush=True)
    kg, usage = agent5_canonicalize(kg, client)
    add_usage(usage)
    print(f"{len(kg.get('nodes', []))} nodes", flush=True)

    print("    [6/6] deterministic validator/enrichment...", end=" ", flush=True)
    kg = deterministic_validator(kg, source)
    kg = deterministic_human_style_enrichment(kg, source)
    print(f"{len(kg['nodes'])}n/{len(kg['edges'])}e", flush=True)
    return kg, usage_total


def process_one(txt_path: Path, client, output_dir: Path) -> tuple[str, str, int, int, dict]:
    res_id = txt_path.stem
    output_file = output_dir / f"{res_id}_{OUTPUT_SUFFIX}.json"
    if output_file.exists():
        return res_id, "SKIP", 0, 0, {}

    transcript = read_transcript(txt_path)
    note = read_note(txt_path)
    print(f"\n  {res_id}:{' (+note)' if note.strip() else ''}")
    kg, usage = run_pipeline(transcript, client, note)
    kg["_usage"] = usage
    kg["_method"] = OUTPUT_SUFFIX
    kg["_used_note"] = bool(note.strip())
    output_file.write_text(json.dumps(kg, indent=2, ensure_ascii=False), encoding="utf-8")
    return res_id, "OK", len(kg["nodes"]), len(kg["edges"]), usage


def run_all_batch(
    transcript_files: list[Path],
    client: AnthropicClient,
    output_dir: Path,
    models: dict[str, str],
) -> tuple[int, int, dict, list[dict]]:
    """Process all transcripts via Anthropic Batch API in staged batches."""
    work: dict[str, dict] = {}
    skipped = 0
    for f in transcript_files:
        res_id = f.stem
        if (output_dir / f"{res_id}_{OUTPUT_SUFFIX}.json").exists():
            print(f"  {res_id}: SKIP (exists)")
            skipped += 1
            continue
        note = read_note(f)
        work[res_id] = {
            "path": f,
            "source": build_source_text(read_transcript(f), note),
            "has_note": bool(note.strip()),
        }

    if not work:
        print("  Nothing to process.")
        return skipped, 0, {"prompt_tokens": 0, "completion_tokens": 0}, []

    res_ids = sorted(work)
    usage_total = {"prompt_tokens": 0, "completion_tokens": 0}

    def add_usage(u: dict) -> None:
        usage_total["prompt_tokens"] += u.get("prompt_tokens", 0)
        usage_total["completion_tokens"] += u.get("completion_tokens", 0)

    print(f"\n  [Batch 1/4] Agent 1 + Agent 3 ({len(res_ids) * 2} requests)...")
    reqs = []
    for rid in res_ids:
        src = work[rid]["source"]
        reqs.append({"custom_id": f"a1_{rid}", "model": models["recall"], "prompt": build_agent1_prompt(src)})
        reqs.append({"custom_id": f"a3_{rid}", "model": models["negation"], "prompt": build_agent3_prompt(src)})

    b1 = client.batch_generate(reqs)

    a1_results: dict[str, list[dict]] = {}
    a3_results: dict[str, list[dict]] = {}
    for rid in res_ids:
        content, usage = b1.get(f"a1_{rid}", ("", {}))
        add_usage(usage)
        nodes = extract_json(content)
        a1_results[rid] = nodes if isinstance(nodes, list) else []

        content, usage = b1.get(f"a3_{rid}", ("", {}))
        add_usage(usage)
        nodes = extract_json(content)
        a3_results[rid] = nodes if isinstance(nodes, list) else []

    print(
        f"    Agent 1: {sum(len(v) for v in a1_results.values())} candidates | "
        f"Agent 3: {sum(len(v) for v in a3_results.values())} negations"
    )

    reqs = []
    for rid in res_ids:
        if a1_results[rid]:
            reqs.append({
                "custom_id": f"a2_{rid}",
                "model": models["filter"],
                "prompt": build_agent2_prompt(a1_results[rid], work[rid]["source"]),
            })
    print(f"\n  [Batch 2/4] Agent 2 ({len(reqs)} requests)...")
    b2 = client.batch_generate(reqs) if reqs else {}

    a2_results: dict[str, list[dict]] = {}
    for rid in res_ids:
        if not a1_results[rid]:
            a2_results[rid] = []
            continue
        content, usage = b2.get(f"a2_{rid}", ("", {}))
        add_usage(usage)
        decision = extract_json(content)
        if isinstance(decision, dict):
            keep_ids = set(decision.get("keep_ids") or [])
            a2_results[rid] = [n for n in a1_results[rid] if n.get("id") in keep_ids] if keep_ids else a1_results[rid]
        else:
            a2_results[rid] = a1_results[rid]

    merged: dict[str, list[dict]] = {}
    for rid in res_ids:
        merged[rid] = merge_nodes(a2_results[rid] + a3_results[rid])
    print(f"    Merged: {sum(len(v) for v in merged.values())} total nodes")

    reqs = []
    for rid in res_ids:
        if merged[rid]:
            reqs.append({
                "custom_id": f"a4_{rid}",
                "model": models["relation"],
                "prompt": build_agent4_prompt(merged[rid], work[rid]["source"]),
            })
    print(f"\n  [Batch 3/4] Agent 4 ({len(reqs)} requests)...")
    b3 = client.batch_generate(reqs) if reqs else {}

    kgs: dict[str, dict] = {}
    for rid in res_ids:
        content, usage = b3.get(f"a4_{rid}", ("", {}))
        add_usage(usage)
        edges = extract_json(content)
        kgs[rid] = {"nodes": merged[rid], "edges": edges if isinstance(edges, list) else []}

    print(f"    Edges: {sum(len(v['edges']) for v in kgs.values())} total")

    reqs = []
    for rid in res_ids:
        if kgs[rid]["nodes"]:
            reqs.append({
                "custom_id": f"a5_{rid}",
                "model": models["canonicalize"],
                "prompt": build_agent5_prompt(kgs[rid]),
            })
    print(f"\n  [Batch 4/4] Agent 5 ({len(reqs)} requests)...")
    b4 = client.batch_generate(reqs) if reqs else {}

    for rid in res_ids:
        content, usage = b4.get(f"a5_{rid}", ("", {}))
        add_usage(usage)
        rewritten = extract_json(content)
        if isinstance(rewritten, list):
            kgs[rid] = apply_canonicalization(kgs[rid], rewritten)

    print("\n  [Local] Deterministic validator + enrichment...")
    details: list[dict] = []
    for rid in res_ids:
        source = work[rid]["source"]
        kgs[rid] = deterministic_validator(kgs[rid], source)
        kgs[rid] = deterministic_human_style_enrichment(kgs[rid], source)

        kg = kgs[rid]
        kg["_usage"] = usage_total
        kg["_method"] = OUTPUT_SUFFIX
        kg["_used_note"] = work[rid]["has_note"]
        out = output_dir / f"{rid}_{OUTPUT_SUFFIX}.json"
        out.write_text(json.dumps(kg, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"    {rid}: {len(kg['nodes'])}n/{len(kg['edges'])}e")
        details.append({
            "res_id": rid,
            "status": "OK",
            "nodes": len(kg["nodes"]),
            "edges": len(kg["edges"]),
        })

    return skipped + len(res_ids), 0, usage_total, details
