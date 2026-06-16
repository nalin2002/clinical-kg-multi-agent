"""Compare Graph-JEPA v5 against an LLM on sampled LOO edge recovery."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch

from graph_jepa.schema import IDX_TO_EDGE_TYPE

from .data import PatientGraphDataset
from .evaluate import (
    _chance_mrr,
    _positive_triples,
    _target_candidates,
    _trusted_edge_masks,
)
from .training import (
    add_data_args,
    build_checkpoint_encoder,
    build_graphs,
    load_model_checkpoint,
)


@dataclass(frozen=True)
class RecoveryQuery:
    graph_index: int
    edge_index: int
    source: int
    target: int
    relation: int
    candidates: tuple[int, ...]


@dataclass
class QueryResult:
    graph_index: int
    edge_index: int
    relation: str
    source: str
    target: str
    candidates: list[str]
    jepa_rank: int
    llm_rank: int
    llm_parse_ok: bool
    llm_parse_complete: bool
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    total_tokens: int
    finish_reason: str
    llm_response: str


def _load_dotenv_files() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    repo_root = Path(__file__).resolve().parents[2]
    load_dotenv(repo_root / ".env")
    load_dotenv()


def _api_base(provider: str) -> str:
    if provider == "openrouter":
        return "https://openrouter.ai/api/v1"
    if provider == "cerebras":
        return "https://api.cerebras.ai/v1"
    raise ValueError(f"unknown provider: {provider!r}")


def _api_key(provider: str) -> str:
    _load_dotenv_files()
    env_name = "OPENROUTER_API_KEY" if provider == "openrouter" else "CEREBRAS_API_KEY"
    key = os.environ.get(env_name, "").strip()
    if key:
        return key

    for path in (
        Path("api_keys.json"),
        Path(__file__).resolve().parents[2] / "api_keys.json",
    ):
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        key = str(data.get(provider) or "").strip()
        if key:
            return key

    raise SystemExit(f"Set {env_name} in .env or add {provider!r} to api_keys.json.")


def _node_text(node: dict) -> str:
    return str(
        node.get("text")
        or node.get("normalized_name")
        or node.get("name")
        or node.get("id")
        or ""
    ).strip()


def _node_label(graph, node_idx: int) -> str:
    node = graph.nodes[node_idx]
    return f"{node.get('type', '')}: {_node_text(node)}"


def _edge_label(graph, source: int, relation: int, target: int) -> str:
    return (
        f"{_node_label(graph, source)} --"
        f"{IDX_TO_EDGE_TYPE.get(relation, f'rel{relation}')}--> "
        f"{_node_label(graph, target)}"
    )


def _stable_query_seed(seed: int, graph_index: int, edge_index: int) -> int:
    return (int(seed) * 1_000_003 + graph_index * 9176 + edge_index * 101) & 0xFFFFFFFF


def _finalize_candidates(
    candidates: Iterable[int],
    *,
    target: int,
    max_candidates: int | None,
    rng: random.Random,
) -> tuple[int, ...]:
    candidates = list(candidates)
    if target not in candidates:
        return tuple()
    distractors = [candidate for candidate in candidates if candidate != target]
    if max_candidates is not None and len(candidates) > max_candidates:
        if max_candidates < 2:
            raise ValueError("--max-candidates must be at least 2")
        rng.shuffle(distractors)
        distractors = distractors[: max_candidates - 1]
    final = [target] + distractors
    rng.shuffle(final)
    return tuple(final)


def collect_recovery_queries(
    data_list: list,
    cfg,
    *,
    samples_per_relation: int,
    candidate_mode: str,
    max_candidates: int | None,
    seed: int,
    allow_duplicate_triples: bool = False,
) -> list[RecoveryQuery]:
    if samples_per_relation <= 0:
        raise ValueError("--samples-per-relation must be positive")

    grouped: dict[int, list[RecoveryQuery]] = defaultdict(list)
    for graph_index, data in enumerate(data_list):
        positive_mask, base_message_mask = _trusted_edge_masks(data, cfg)
        positive_indices = positive_mask.nonzero(as_tuple=False).flatten()
        if positive_indices.numel() == 0:
            continue
        true_triples = _positive_triples(data, positive_mask)
        message_triples = Counter(
            (
                int(source),
                int(target),
                int(relation),
            )
            for source, target, relation in zip(
                data.edge_index[0, base_message_mask].tolist(),
                data.edge_index[1, base_message_mask].tolist(),
                data.edge_type[base_message_mask].tolist(),
            )
        )

        for edge_index in positive_indices.detach().cpu().tolist():
            edge_index = int(edge_index)
            source = int(data.edge_index[0, edge_index].item())
            target = int(data.edge_index[1, edge_index].item())
            relation = int(data.edge_type[edge_index].item())
            if source == target:
                continue
            if (
                not allow_duplicate_triples
                and message_triples[(source, target, relation)] > 1
            ):
                continue

            candidates = _target_candidates(
                data,
                source=source,
                target=target,
                relation=relation,
                true_triples=true_triples,
                candidate_mode=candidate_mode,
            )
            if target not in candidates or len(candidates) < 2:
                continue

            query_rng = random.Random(_stable_query_seed(seed, graph_index, edge_index))
            final_candidates = _finalize_candidates(
                candidates,
                target=target,
                max_candidates=max_candidates,
                rng=query_rng,
            )
            if target not in final_candidates or len(final_candidates) < 2:
                continue
            grouped[relation].append(
                RecoveryQuery(
                    graph_index=graph_index,
                    edge_index=edge_index,
                    source=source,
                    target=target,
                    relation=relation,
                    candidates=final_candidates,
                )
            )

    rng = random.Random(seed)
    selected: list[RecoveryQuery] = []
    for relation in sorted(grouped):
        queries = grouped[relation]
        rng.shuffle(queries)
        selected.extend(queries[:samples_per_relation])
    selected.sort(key=lambda q: (q.relation, q.graph_index, q.edge_index))
    return selected


def _context_edge_indices(
    data,
    query: RecoveryQuery,
    base_message_mask: torch.Tensor,
    *,
    mode: str,
    limit: int,
) -> list[int]:
    if mode == "none" or limit <= 0:
        return []
    if mode not in {"source", "full"}:
        raise ValueError(f"unknown context mode: {mode!r}")

    candidate_nodes = set(query.candidates)
    rows = []
    for edge_index in base_message_mask.nonzero(as_tuple=False).flatten().tolist():
        edge_index = int(edge_index)
        if edge_index == query.edge_index:
            continue
        source = int(data.edge_index[0, edge_index].item())
        target = int(data.edge_index[1, edge_index].item())
        if mode == "source" and query.source not in (source, target):
            continue
        if query.source in (source, target):
            priority = 0
        elif source in candidate_nodes or target in candidate_nodes:
            priority = 1
        else:
            priority = 2
        rows.append((priority, edge_index))
    rows.sort()
    return [edge_index for _priority, edge_index in rows[:limit]]


def build_prompt(
    graph,
    data,
    cfg,
    query: RecoveryQuery,
    *,
    context_mode: str,
    max_context_edges: int,
) -> str:
    _positive_mask, base_message_mask = _trusted_edge_masks(data, cfg)
    context_edges = _context_edge_indices(
        data,
        query,
        base_message_mask,
        mode=context_mode,
        limit=max_context_edges,
    )
    relation = IDX_TO_EDGE_TYPE.get(query.relation, f"rel{query.relation}")
    lines = [
        "You are ranking candidate target nodes for a masked clinical KG edge.",
        "The original edge has been removed from the graph context.",
        "",
        "Masked query:",
        f"{_node_label(graph, query.source)} --{relation}--> ?",
        "",
        "Candidate target nodes:",
    ]
    for idx, candidate in enumerate(query.candidates, start=1):
        lines.append(f"{idx}. {_node_label(graph, candidate)}")

    if context_edges:
        lines.extend(["", "Graph context edges:"])
        for edge_index in context_edges:
            source = int(data.edge_index[0, edge_index].item())
            target = int(data.edge_index[1, edge_index].item())
            rel = int(data.edge_type[edge_index].item())
            lines.append(f"- {_edge_label(graph, source, rel, target)}")
    else:
        lines.extend(["", "Graph context edges: none"])

    lines.extend(
        [
            "",
            "Rank all candidate numbers from most likely to least likely.",
            'Return JSON only, exactly like: {"ranking":[1,2,3]}',
        ]
    )
    return "\n".join(lines)


def parse_ranking(text: str, candidate_count: int) -> tuple[list[int], bool, bool]:
    """Return zero-based candidate indices plus parse status flags."""

    raw_items: list = []
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            payload = json.loads(match.group(0))
            value = payload.get("ranking") or payload.get("ranked_candidates")
            if isinstance(value, list):
                raw_items = value
        except json.JSONDecodeError:
            raw_items = []

    if not raw_items:
        raw_items = re.findall(r"\b\d+\b", text)

    order: list[int] = []
    seen: set[int] = set()
    for item in raw_items:
        numbers = re.findall(r"\d+", str(item))
        if not numbers:
            continue
        idx = int(numbers[0]) - 1
        if 0 <= idx < candidate_count and idx not in seen:
            seen.add(idx)
            order.append(idx)

    parse_ok = bool(order)
    parse_complete = len(order) == candidate_count
    for idx in range(candidate_count):
        if idx not in seen:
            order.append(idx)
    return order, parse_ok, parse_complete


def _rank_from_order(order: list[int], true_position: int) -> int:
    try:
        return order.index(true_position) + 1
    except ValueError:
        return len(order)


@torch.no_grad()
def jepa_rank_query(
    model,
    data,
    query: RecoveryQuery,
    cfg,
    device: torch.device,
) -> int:
    data = data.clone().to(device)
    _positive_mask, base_message_mask = _trusted_edge_masks(data, cfg)
    message_mask = base_message_mask.clone()
    message_mask[query.edge_index] = False
    z_nodes = model.context_node_encoder(
        data.x,
        data.edge_index[:, message_mask],
        data.edge_type[message_mask],
    )
    candidates = torch.tensor(query.candidates, dtype=torch.long, device=device)
    sources = torch.full(
        (len(query.candidates),),
        query.source,
        dtype=torch.long,
        device=device,
    )
    relations = torch.full(
        (len(query.candidates),),
        query.relation,
        dtype=torch.long,
        device=device,
    )
    logits = model.edge_head(z_nodes[sources], z_nodes[candidates], relations)
    target_position = int(
        (candidates == query.target).nonzero(as_tuple=False)[0, 0].item()
    )
    return int((logits > logits[target_position]).sum().item()) + 1


class ChatRanker:
    def __init__(
        self,
        *,
        provider: str,
        model: str,
        api_key: str,
        base_url: str,
        temperature: float,
        max_tokens: int,
        reasoning_effort: str | None,
        reasoning_format: str | None,
        retries: int,
        sleep: float,
    ):
        from openai import OpenAI

        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.reasoning_format = reasoning_format
        self.retries = retries
        self.sleep = sleep
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=120)

    def rank(self, prompt: str) -> tuple[str, dict[str, int | str]]:
        last_error = None
        for attempt in range(self.retries):
            try:
                extra_body = {}
                if self.reasoning_format:
                    extra_body["reasoning_format"] = self.reasoning_format
                kwargs = {}
                if self.reasoning_effort:
                    kwargs["reasoning_effort"] = self.reasoning_effort
                if extra_body:
                    kwargs["extra_body"] = extra_body
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    **kwargs,
                )
                usage = {}
                if completion.usage:
                    prompt_tokens = completion.usage.prompt_tokens or 0
                    completion_tokens = completion.usage.completion_tokens or 0
                    total_tokens = getattr(completion.usage, "total_tokens", None)
                    completion_details = getattr(
                        completion.usage,
                        "completion_tokens_details",
                        None,
                    )
                    reasoning_tokens = (
                        getattr(completion_details, "reasoning_tokens", 0)
                        if completion_details
                        else 0
                    )
                    usage = {
                        "prompt_tokens": int(prompt_tokens),
                        "completion_tokens": int(completion_tokens),
                        "reasoning_tokens": int(reasoning_tokens or 0),
                        "total_tokens": int(
                            total_tokens
                            if total_tokens is not None
                            else prompt_tokens + completion_tokens
                        ),
                        "finish_reason": str(completion.choices[0].finish_reason or ""),
                    }
                return completion.choices[0].message.content or "", usage
            except Exception as exc:
                last_error = exc
                print(
                    f"[LLM] retry {attempt + 1}/{self.retries} failed: {exc}",
                    flush=True,
                )
                time.sleep(self.sleep * (2**attempt))
        print(
            f"[LLM] giving up after {self.retries} attempts: {last_error}",
            flush=True,
        )
        return "", {}


def _summarize(results: list[QueryResult], rank_field: str) -> dict:
    if not results:
        return {
            "mrr": float("nan"),
            "hits1": 0.0,
            "hits3": 0.0,
            "hits10": 0.0,
            "n": 0,
            "per_rel": [],
        }

    total_rr = []
    total_hits = {1: 0, 3: 0, 10: 0}
    rel_rr: dict[str, list[float]] = defaultdict(list)
    rel_hits: dict[str, dict[int, int]] = defaultdict(lambda: {1: 0, 3: 0, 10: 0})
    rel_candidates: dict[str, list[int]] = defaultdict(list)
    for result in results:
        rank = int(getattr(result, rank_field))
        rr = 1.0 / rank
        total_rr.append(rr)
        rel_rr[result.relation].append(rr)
        rel_candidates[result.relation].append(len(result.candidates))
        for k in total_hits:
            if rank <= k:
                total_hits[k] += 1
                rel_hits[result.relation][k] += 1

    per_rel = []
    for relation in sorted(rel_rr, key=lambda rel: -len(rel_rr[rel])):
        n = len(rel_rr[relation])
        mean_candidates = sum(rel_candidates[relation]) / max(n, 1)
        per_rel.append(
            {
                "rel": relation,
                "n": n,
                "mrr": sum(rel_rr[relation]) / n,
                "h1": rel_hits[relation][1] / n,
                "h3": rel_hits[relation][3] / n,
                "h10": rel_hits[relation][10] / n,
                "C": mean_candidates,
                "chance_mrr": _chance_mrr(mean_candidates),
                "chance_h1": 1.0 / mean_candidates if mean_candidates >= 1 else 1.0,
            }
        )
    n = len(results)
    return {
        "mrr": sum(total_rr) / n,
        "hits1": total_hits[1] / n,
        "hits3": total_hits[3] / n,
        "hits10": total_hits[10] / n,
        "n": n,
        "per_rel": per_rel,
    }


def _print_summary(label: str, metrics: dict) -> None:
    print(
        f"[{label}] MRR={metrics['mrr']:.3f} H@1={metrics['hits1']:.3f} "
        f"H@3={metrics['hits3']:.3f} H@10={metrics['hits10']:.3f} "
        f"over {metrics['n']} queries"
    )


def _print_side_by_side(jepa: dict, llm: dict) -> None:
    jepa_by_rel = {row["rel"]: row for row in jepa["per_rel"]}
    llm_by_rel = {row["rel"]: row for row in llm["per_rel"]}
    print(
        "[PER-REL] relation                 n     C    chance  "
        "JEPA_MRR JEPA_H@1  LLM_MRR LLM_H@1"
    )
    for relation in sorted(jepa_by_rel, key=lambda rel: -jepa_by_rel[rel]["n"]):
        jr = jepa_by_rel[relation]
        lr = llm_by_rel.get(relation)
        if lr is None:
            continue
        print(
            f"[PER-REL] {relation:<22} {jr['n']:<5} {jr['C']:<4.0f} "
            f"{jr['chance_mrr']:<7.3f} {jr['mrr']:<8.3f} {jr['h1']:<8.3f} "
            f"{lr['mrr']:<7.3f} {lr['h1']:<7.3f}"
        )


def run(args) -> dict:
    device = torch.device(args.device)
    model, cfg = load_model_checkpoint(args.checkpoint, device)
    model.eval()
    encoder = build_checkpoint_encoder(cfg, args.encoder_cache)
    graphs = build_graphs(args, cfg)
    dataset = PatientGraphDataset(graphs, encoder)
    data_list = [dataset[idx] for idx in range(len(dataset))]

    queries = collect_recovery_queries(
        data_list,
        cfg,
        samples_per_relation=args.samples_per_relation,
        candidate_mode=args.candidate_mode,
        max_candidates=args.max_candidates,
        seed=args.seed,
        allow_duplicate_triples=args.allow_duplicate_triples,
    )
    if args.max_queries is not None:
        queries = queries[: args.max_queries]
    if not queries:
        raise ValueError("no eligible LOO queries found")

    counts = Counter(
        IDX_TO_EDGE_TYPE.get(q.relation, f"rel{q.relation}") for q in queries
    )
    print(
        f"[SAMPLE] queries={len(queries)} candidate_mode={args.candidate_mode} "
        f"samples_per_relation={args.samples_per_relation} relations={dict(counts)}"
    )

    llm_model = args.llm_model or os.environ.get(f"{args.provider.upper()}_MODEL", "")
    if not llm_model:
        raise SystemExit(
            f"Pass --llm-model or set {args.provider.upper()}_MODEL in the environment."
        )
    ranker = ChatRanker(
        provider=args.provider,
        model=llm_model,
        api_key=_api_key(args.provider),
        base_url=args.api_base or _api_base(args.provider),
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        reasoning_effort=args.reasoning_effort,
        reasoning_format=args.reasoning_format,
        retries=args.retries,
        sleep=args.retry_sleep,
    )

    results: list[QueryResult] = []
    records_path = Path(args.records_output) if args.records_output else None
    if records_path:
        records_path.parent.mkdir(parents=True, exist_ok=True)
        records_path.write_text("", encoding="utf-8")

    for idx, query in enumerate(queries, start=1):
        graph = graphs[query.graph_index]
        data = data_list[query.graph_index]
        relation = IDX_TO_EDGE_TYPE.get(query.relation, f"rel{query.relation}")
        jepa_rank = jepa_rank_query(model, data, query, cfg, device)
        prompt = build_prompt(
            graph,
            data,
            cfg,
            query,
            context_mode=args.context,
            max_context_edges=args.max_context_edges,
        )
        response, usage = ranker.rank(prompt)
        order, parse_ok, parse_complete = parse_ranking(response, len(query.candidates))
        true_position = query.candidates.index(query.target)
        llm_rank = (
            _rank_from_order(order, true_position)
            if parse_ok
            else len(query.candidates)
        )
        result = QueryResult(
            graph_index=query.graph_index,
            edge_index=query.edge_index,
            relation=relation,
            source=_node_label(graph, query.source),
            target=_node_label(graph, query.target),
            candidates=[
                _node_label(graph, candidate)
                for candidate in query.candidates
            ],
            jepa_rank=jepa_rank,
            llm_rank=llm_rank,
            llm_parse_ok=parse_ok,
            llm_parse_complete=parse_complete,
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            reasoning_tokens=int(usage.get("reasoning_tokens", 0)),
            total_tokens=int(usage.get("total_tokens", 0)),
            finish_reason=str(usage.get("finish_reason", "")),
            llm_response=response,
        )
        results.append(result)
        if records_path:
            with records_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(result)) + "\n")
        if idx % args.progress_every == 0 or idx == len(queries):
            print(f"[PROGRESS] {idx}/{len(queries)} queries complete", flush=True)
        if args.request_sleep > 0:
            time.sleep(args.request_sleep)

    jepa_metrics = _summarize(results, "jepa_rank")
    llm_metrics = _summarize(results, "llm_rank")
    parse_failures = sum(1 for result in results if not result.llm_parse_ok)
    incomplete = sum(1 for result in results if not result.llm_parse_complete)
    token_usage = {
        "prompt_tokens": sum(result.prompt_tokens for result in results),
        "completion_tokens": sum(result.completion_tokens for result in results),
        "reasoning_tokens": sum(result.reasoning_tokens for result in results),
        "total_tokens": sum(result.total_tokens for result in results),
    }
    finish_reasons = Counter(result.finish_reason for result in results)

    print("[RESULTS]")
    _print_summary("JEPA", jepa_metrics)
    _print_summary("LLM", llm_metrics)
    print(
        f"[LLM] parse_failures={parse_failures}/{len(results)} "
        f"incomplete_rankings={incomplete}/{len(results)}"
    )
    print(
        "[LLM] tokens "
        f"prompt={token_usage['prompt_tokens']} "
        f"completion={token_usage['completion_tokens']} "
        f"reasoning={token_usage['reasoning_tokens']} "
        f"total={token_usage['total_tokens']}"
    )
    print(f"[LLM] finish_reasons={dict(finish_reasons)}")
    _print_side_by_side(jepa_metrics, llm_metrics)

    payload = {
        "config": {
            "checkpoint": args.checkpoint,
            "provider": args.provider,
            "llm_model": llm_model,
            "candidate_mode": args.candidate_mode,
            "context": args.context,
            "samples_per_relation": args.samples_per_relation,
            "max_candidates": args.max_candidates,
            "max_context_edges": args.max_context_edges,
            "seed": args.seed,
            "allow_duplicate_triples": args.allow_duplicate_triples,
            "reasoning_effort": args.reasoning_effort,
            "reasoning_format": args.reasoning_format,
        },
        "sample_counts": dict(counts),
        "jepa": jepa_metrics,
        "llm": llm_metrics,
        "llm_parse_failures": parse_failures,
        "llm_incomplete_rankings": incomplete,
        "llm_token_usage": token_usage,
        "llm_finish_reasons": dict(finish_reasons),
        "records": [asdict(result) for result in results],
    }
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sample LOO edge-recovery queries and compare JEPA to an LLM"
    )
    add_data_args(parser)
    parser.add_argument("--checkpoint", default="checkpoints/graph_jepa_v5.pt")
    parser.add_argument("--encoder-cache", default=".cache/graph_jepa_v5/encoder")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--candidate-mode",
        choices=["schema", "same-type"],
        default="same-type",
    )
    parser.add_argument("--samples-per-relation", type=int, default=100)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--max-candidates", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--allow-duplicate-triples",
        action="store_true",
        help="Allow exact duplicate triples that can leak the hidden edge",
    )
    parser.add_argument(
        "--provider",
        choices=["openrouter", "cerebras"],
        default="openrouter",
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help="Provider model id. Or set OPENROUTER_MODEL/CEREBRAS_MODEL.",
    )
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument(
        "--reasoning-effort",
        choices=["none", "low", "medium", "high"],
        default="low",
        help=(
            "Reasoning control for Cerebras GPT-OSS/GLM. Use low for ranking; "
            "none is mainly for GLM."
        ),
    )
    parser.add_argument(
        "--reasoning-format",
        choices=["none", "parsed", "raw", "hidden"],
        default="hidden",
        help="Cerebras reasoning output format; hidden keeps records clean.",
    )
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--request-sleep", type=float, default=0.0)
    parser.add_argument("--context", choices=["none", "source", "full"], default="full")
    parser.add_argument("--max-context-edges", type=int, default=120)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--output", default=None, help="Final JSON summary path")
    parser.add_argument(
        "--records-output",
        default=None,
        help="Optional JSONL path written incrementally after each LLM call",
    )
    return parser


def main(argv=None) -> None:
    run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()
