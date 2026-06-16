"""Compare the HF LOO JEPA baseline, Graph-JEPA v5, and an LLM on sampled LOO queries."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import TransformerConv

from graph_jepa.schema import IDX_TO_EDGE_TYPE

from .data import PatientGraphDataset
from .evaluate import _chance_mrr
from .evaluate_llm import (
    ChatRanker,
    RecoveryQuery,
    _api_base,
    _api_key,
    _node_label,
    _rank_from_order,
    build_prompt,
    collect_recovery_queries,
    jepa_rank_query,
    parse_ranking,
)
from .training import (
    add_data_args,
    build_checkpoint_encoder,
    build_graphs,
    load_model_checkpoint,
)


LOO_NODE_TYPES = {
    "PATIENT": 0,
    "DIAGNOSIS": 1,
    "MEDICATION": 2,
    "MICROBIOLOGY": 3,
    "PROCEDURE": 4,
    "SERVICE": 5,
    "LAB_TEST": 6,
    "PROCUREMENT": 7,
}
LOO_RELATION_CANONICAL = {
    "HAS_DIAGNOSIS": 0,
    "TAKES_MEDICATION": 1,
    "TREATED_BY": 2,
    "HAD_MICROBIOLOGY": 3,
    "CO_OCCURS_WITH": 4,
    "UNDERWENT_PROCEDURE": 5,
    "MANAGED_BY_SERVICE": 6,
    "MANAGED_FOR": 7,
    "PERFORMED_FOR": 8,
    "CONFIRMS": 9,
    "HAD_LAB_TEST": 10,
    "DIAGNOSED_BY": 11,
    "TARGETS_ORGANISM": 12,
    "MONITORED_BY": 13,
    "ADMINISTERED_DURING": 14,
    "INDICATES": 15,
    "INVESTIGATED_BY": 16,
    "ASSOCIATED_WITH": 17,
    "COMPLICATED_BY": 18,
    "PART_OF_REGIMEN": 19,
}
LOO_RELATION_ALIASES = {
    "DIAGNORED_BY": "DIAGNOSED_BY",
    "TARGET_ORGANISM": "TARGETS_ORGANISM",
    "HAS_MICROBIOLOGY": "HAD_MICROBIOLOGY",
    "HAD_PROCEDURE": "UNDERWENT_PROCEDURE",
    "HAS_MEDICATION": "TAKES_MEDICATION",
    "MANAGES_FOR": "MANAGED_FOR",
}
LOO_NUMERIC_DIM = 6
LOO_SCORE_FEATS = [
    "model",
    "drug_link_cos",
    "dx_disease_cos",
    "het_treats_ctd",
    "het_treats_cpd",
    "het_drug_cos",
    "het_dx_cos",
    "het_resembles_drd",
    "het_presents_dps",
    "omop_src_cos",
    "omop_dst_cos",
    "omop_lca_dist",
    "prov_in_note",
    "prov_ratio",
]
DEFAULT_RAW_TEST_GRAPHS = "outputs/fawkes_raw_global_v5/test"
DEFAULT_V5_CHECKPOINT = "ckpts/fawkes_raw_v5/graph_jepa_v5.pt"
DEFAULT_LOO_FILENAME = "fawkes_jepa_loo_eval_v12_260615.pt"


@dataclass
class ComparisonResult:
    graph_index: int
    edge_index: int
    relation: str
    source: str
    target: str
    candidates: list[str]
    loo_jepa_rank: int
    v5_jepa_rank: int
    llm_rank: int
    llm_parse_ok: bool
    llm_parse_complete: bool
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    total_tokens: int
    finish_reason: str
    llm_response: str


@dataclass
class LooModelBundle:
    encoder: "LooEncoder"
    scorer: nn.Module
    config: dict
    entity_vocab: int
    num_base_relations: int
    use_scores: bool


def _loo_relation_id(relation: str | int) -> int | None:
    if isinstance(relation, int):
        relation = IDX_TO_EDGE_TYPE.get(relation, f"rel{relation}")
    normalized = str(relation or "").upper()
    canonical = LOO_RELATION_ALIASES.get(normalized, normalized)
    return LOO_RELATION_CANONICAL.get(canonical)


def _loo_entity_hash(name: str, entity_vocab: int) -> int:
    return int(hashlib.md5(name.encode("utf-8")).hexdigest(), 16) % entity_vocab


def _loo_score_vec(edge: dict) -> list[float]:
    labels = edge.get("labels")
    values = labels if isinstance(labels, dict) else edge
    out = []
    for key in LOO_SCORE_FEATS:
        value = values.get(key)
        if not isinstance(value, (int, float)):
            value = 0.0
        if key == "omop_lca_dist":
            value = math.exp(-float(value) / 5.0) if value else 0.0
        out.append(float(value))
    return out


def _loo_numeric_features(graph) -> list[float]:
    patient = graph.extra.get("_patient") or {}
    age = patient.get("anchor_age", patient.get("age", 0.0))
    try:
        age_value = float(age) / 100.0
    except (TypeError, ValueError):
        age_value = 0.0
    gender = str(patient.get("gender") or "").upper()
    return [
        age_value,
        1.0 if gender == "M" else 0.0,
        1.0 if gender == "F" else 0.0,
        0.0,
        0.0,
        0.0,
    ]


def _node_entity_name(node: dict) -> str:
    return str(node.get("normalized_name") or node.get("name") or "").strip()


def to_loo_graph_data(graph, *, entity_vocab: int, use_scores: bool) -> Data:
    """Convert a repo PatientGraph into the HF LOO baseline tensor contract."""

    id_to_idx: dict[str, int] = {}
    node_types = []
    entity_ids = []
    numeric_features = []
    numfeat = _loo_numeric_features(graph)
    for node in graph.nodes:
        node_id = str(node.get("id") or "")
        node_type = str(node.get("type") or "").upper()
        if not node_id:
            continue
        if node_type not in LOO_NODE_TYPES:
            raise KeyError(f"LOO baseline does not support node type {node_type!r}")
        id_to_idx[node_id] = len(node_types)
        node_types.append(LOO_NODE_TYPES[node_type])
        entity_ids.append(_loo_entity_hash(_node_entity_name(node), entity_vocab))
        numeric_features.append(numfeat)

    src, dst, rel, edge_feat = [], [], [], []
    for edge in graph.edges:
        source_id = str(edge.get("source_id") or edge.get("source") or "")
        target_id = str(edge.get("target_id") or edge.get("target") or "")
        relation = str(edge.get("type") or edge.get("relation") or "").upper()
        rel_id = _loo_relation_id(relation)
        if rel_id is None:
            continue
        if source_id not in id_to_idx or target_id not in id_to_idx:
            continue
        src.append(id_to_idx[source_id])
        dst.append(id_to_idx[target_id])
        rel.append(rel_id)
        edge_feat.append(_loo_score_vec(edge))

    data = Data()
    data.num_nodes = len(node_types)
    data.node_type = torch.tensor(node_types, dtype=torch.long)
    data.entity_id = torch.tensor(entity_ids, dtype=torch.long)
    data.numfeat = torch.tensor(numeric_features, dtype=torch.float)
    data.sem_id = torch.zeros(len(node_types), dtype=torch.long)
    if src:
        data.edge_index = torch.tensor([src, dst], dtype=torch.long)
        data.edge_type = torch.tensor(rel, dtype=torch.long)
        data.edge_feat = torch.tensor(edge_feat, dtype=torch.float)
    else:
        data.edge_index = torch.zeros((2, 0), dtype=torch.long)
        data.edge_type = torch.zeros((0,), dtype=torch.long)
        data.edge_feat = torch.zeros((0, len(LOO_SCORE_FEATS)), dtype=torch.float)
    if not use_scores:
        data.edge_feat = torch.zeros(
            (data.edge_type.numel(), len(LOO_SCORE_FEATS)),
            dtype=torch.float,
        )
    return data


def _loo_add_inverses(
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    edge_feat: torch.Tensor | None,
    *,
    num_base_relations: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if edge_index.size(1) == 0:
        return edge_index, edge_type, edge_feat
    inv_index = edge_index.flip(0)
    inv_type = edge_type + num_base_relations
    full_index = torch.cat([edge_index, inv_index], dim=1)
    full_type = torch.cat([edge_type, inv_type], dim=0)
    if edge_feat is None:
        return full_index, full_type, None
    return full_index, full_type, torch.cat([edge_feat, edge_feat], dim=0)


class LooEncoder(nn.Module):
    def __init__(
        self,
        *,
        hid: int,
        layers: int,
        heads: int,
        edge_emb: int,
        entity_vocab: int,
        num_node_types: int,
        num_relations: int,
        use_scores: bool,
        use_entity_emb: bool = True,
    ):
        super().__init__()
        if hid % heads:
            raise ValueError(f"LOO hidden size {hid} is not divisible by heads {heads}")
        self.use_scores = use_scores
        self.use_entity_emb = use_entity_emb
        self.type_emb = nn.Embedding(num_node_types, hid)
        self.entity_emb = nn.Embedding(entity_vocab, hid)
        self.num_proj = nn.Linear(LOO_NUMERIC_DIM, hid)
        self.rel_emb = nn.Embedding(num_relations, edge_emb)
        edge_dim = edge_emb + (len(LOO_SCORE_FEATS) if use_scores else 0)
        self.convs = nn.ModuleList(
            TransformerConv(
                hid,
                hid // heads,
                heads=heads,
                concat=True,
                edge_dim=edge_dim,
                dropout=0.0,
            )
            for _ in range(layers)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(hid) for _ in range(layers))

    def forward(
        self,
        node_type: torch.Tensor,
        entity_id: torch.Tensor,
        numfeat: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        edge_feat: torch.Tensor | None = None,
        sem_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del sem_id
        ident = self.entity_emb(entity_id) if self.use_entity_emb else 0
        h = self.type_emb(node_type) + ident + self.num_proj(numfeat)
        edge_attr = self.rel_emb(edge_type)
        if self.use_scores:
            if edge_feat is None:
                raise ValueError("LOO checkpoint expects edge score features")
            edge_attr = torch.cat([edge_attr, edge_feat], dim=-1)
        for conv, norm in zip(self.convs, self.norms):
            h = F.relu(norm(conv(h, edge_index, edge_attr)))
        return h


class LooDistMult(nn.Module):
    def __init__(self, *, hid: int, num_relations: int):
        super().__init__()
        self.rel = nn.Embedding(num_relations, hid)

    def forward(
        self,
        h: torch.Tensor,
        source: torch.Tensor,
        target: torch.Tensor,
        relation: torch.Tensor,
    ) -> torch.Tensor:
        return (h[source] * self.rel(relation) * h[target]).sum(dim=-1)


class LooMLPScorer(nn.Module):
    def __init__(self, *, hid: int, num_relations: int):
        super().__init__()
        self.rel = nn.Embedding(num_relations, hid)
        self.mlp = nn.Sequential(
            nn.Linear(3 * hid, hid),
            nn.ReLU(),
            nn.Linear(hid, 1),
        )

    def forward(
        self,
        h: torch.Tensor,
        source: torch.Tensor,
        target: torch.Tensor,
        relation: torch.Tensor,
    ) -> torch.Tensor:
        return self.mlp(torch.cat([h[source], h[target], self.rel(relation)], dim=-1)).squeeze(-1)


def _infer_layers(encoder_state: dict[str, torch.Tensor]) -> int:
    layer_ids = {
        int(key.split(".")[1])
        for key in encoder_state
        if key.startswith("convs.") and key.split(".")[1].isdigit()
    }
    return max(layer_ids) + 1 if layer_ids else 2


def load_loo_checkpoint(path: str | Path, device: torch.device) -> LooModelBundle:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    encoder_state = ckpt["encoder"]
    scorer_state = ckpt["scorer"]
    config = dict(ckpt.get("config", {}))

    hid = int(encoder_state["type_emb.weight"].shape[1])
    num_node_types = int(encoder_state["type_emb.weight"].shape[0])
    entity_vocab = int(encoder_state["entity_emb.weight"].shape[0])
    num_relations = int(encoder_state["rel_emb.weight"].shape[0])
    edge_emb = int(encoder_state["rel_emb.weight"].shape[1])
    layers = int(config.get("layers") or _infer_layers(encoder_state))
    heads = int(config.get("heads") or 4)
    use_scores = bool(config.get("use_scores", False))

    encoder = LooEncoder(
        hid=hid,
        layers=layers,
        heads=heads,
        edge_emb=edge_emb,
        entity_vocab=entity_vocab,
        num_node_types=num_node_types,
        num_relations=num_relations,
        use_scores=use_scores,
    ).to(device)
    encoder.load_state_dict(encoder_state)

    scorer_cls = (
        LooMLPScorer
        if any(k.startswith("mlp.") for k in scorer_state)
        else LooDistMult
    )
    scorer = scorer_cls(hid=hid, num_relations=num_relations).to(device)
    scorer.load_state_dict(scorer_state)
    encoder.eval()
    scorer.eval()
    return LooModelBundle(
        encoder=encoder,
        scorer=scorer,
        config=config,
        entity_vocab=entity_vocab,
        num_base_relations=num_relations // 2,
        use_scores=use_scores,
    )


def resolve_loo_checkpoint(args) -> Path:
    if args.loo_checkpoint:
        path = Path(args.loo_checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f"LOO checkpoint not found: {path}")
        return path
    if not args.loo_repo_id:
        raise SystemExit("Pass --loo-repo-id, or --loo-checkpoint for a local .pt file.")
    from huggingface_hub import hf_hub_download

    token = os.environ.get(args.hf_token_env)
    return Path(
        hf_hub_download(
            repo_id=args.loo_repo_id,
            filename=args.loo_filename,
            repo_type="model",
            revision=args.loo_revision,
            token=token,
        )
    )


def _loo_exact_edge_mask(data: Data, query: RecoveryQuery) -> torch.Tensor:
    rel_id = _loo_relation_id(query.relation)
    if rel_id is None or data.edge_type.numel() == 0:
        return torch.zeros(data.edge_type.size(0), dtype=torch.bool)
    return (
        (data.edge_index[0] == query.source)
        & (data.edge_index[1] == query.target)
        & (data.edge_type == rel_id)
    )


def filter_loo_compatible_queries(
    queries: list[RecoveryQuery],
    loo_data: list[Data],
) -> tuple[list[RecoveryQuery], Counter[str]]:
    kept = []
    skipped: Counter[str] = Counter()
    for query in queries:
        relation = IDX_TO_EDGE_TYPE.get(query.relation, f"rel{query.relation}")
        if _loo_relation_id(relation) is None:
            skipped[f"unsupported_relation:{relation}"] += 1
            continue
        data = loo_data[query.graph_index]
        if query.source >= data.num_nodes or query.target >= data.num_nodes:
            skipped["node_index_out_of_range"] += 1
            continue
        if not bool(_loo_exact_edge_mask(data, query).any()):
            skipped[f"missing_exact_edge:{relation}"] += 1
            continue
        kept.append(query)
    return kept, skipped


@torch.no_grad()
def loo_rank_query(
    bundle: LooModelBundle,
    data: Data,
    query: RecoveryQuery,
    device: torch.device,
) -> int:
    rel_id = _loo_relation_id(query.relation)
    if rel_id is None:
        raise ValueError(f"LOO baseline does not support relation id {query.relation}")
    data = data.clone().to(device)
    keep = ~_loo_exact_edge_mask(data, query).to(device)
    edge_feat = data.edge_feat[keep] if bundle.use_scores else None
    edge_index, edge_type, edge_feat = _loo_add_inverses(
        data.edge_index[:, keep],
        data.edge_type[keep],
        edge_feat,
        num_base_relations=bundle.num_base_relations,
    )
    h = bundle.encoder(
        data.node_type,
        data.entity_id,
        data.numfeat,
        edge_index,
        edge_type,
        edge_feat,
        data.sem_id,
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
        rel_id,
        dtype=torch.long,
        device=device,
    )
    scores = bundle.scorer(h, sources, candidates, relations)
    target_position = int(
        (candidates == query.target).nonzero(as_tuple=False)[0, 0].item()
    )
    return int((scores > scores[target_position]).sum().item()) + 1


def _summarize(results: list[ComparisonResult], rank_field: str) -> dict:
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
        reciprocal_rank = 1.0 / rank
        total_rr.append(reciprocal_rank)
        rel_rr[result.relation].append(reciprocal_rank)
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


def _print_three_way(loo: dict, v5: dict, llm: dict) -> None:
    loo_by_rel = {row["rel"]: row for row in loo["per_rel"]}
    v5_by_rel = {row["rel"]: row for row in v5["per_rel"]}
    llm_by_rel = {row["rel"]: row for row in llm["per_rel"]}
    print(
        "[PER-REL] relation                 n     C    chance  "
        "LOO_MRR LOO_H@1  V5_MRR V5_H@1  LLM_MRR LLM_H@1"
    )
    relations = sorted(v5_by_rel, key=lambda rel: -v5_by_rel[rel]["n"])
    for relation in relations:
        lr = loo_by_rel.get(relation)
        vr = v5_by_rel[relation]
        mr = llm_by_rel.get(relation)
        if lr is None or mr is None:
            continue
        print(
            f"[PER-REL] {relation:<22} {vr['n']:<5} {vr['C']:<4.0f} "
            f"{vr['chance_mrr']:<7.3f} {lr['mrr']:<7.3f} {lr['h1']:<8.3f} "
            f"{vr['mrr']:<6.3f} {vr['h1']:<7.3f} "
            f"{mr['mrr']:<7.3f} {mr['h1']:<7.3f}"
        )


def run(args) -> dict:
    device = torch.device(args.device)
    v5_model, v5_cfg = load_model_checkpoint(args.checkpoint, device)
    v5_model.eval()
    encoder = build_checkpoint_encoder(v5_cfg, args.encoder_cache)
    graphs = build_graphs(args, v5_cfg)
    dataset = PatientGraphDataset(graphs, encoder)
    data_list = [dataset[idx] for idx in range(len(dataset))]

    loo_checkpoint = resolve_loo_checkpoint(args)
    loo_bundle = load_loo_checkpoint(loo_checkpoint, device)
    loo_data = [
        to_loo_graph_data(
            graph,
            entity_vocab=loo_bundle.entity_vocab,
            use_scores=loo_bundle.use_scores,
        )
        for graph in graphs
    ]

    queries = collect_recovery_queries(
        data_list,
        v5_cfg,
        samples_per_relation=args.samples_per_relation,
        candidate_mode=args.candidate_mode,
        max_candidates=args.max_candidates,
        seed=args.seed,
        allow_duplicate_triples=args.allow_duplicate_triples,
    )
    queries, skipped = filter_loo_compatible_queries(queries, loo_data)
    if args.max_queries is not None:
        queries = queries[: args.max_queries]
    if not queries:
        raise ValueError("no eligible LOO-compatible recovery queries found")

    counts = Counter(
        IDX_TO_EDGE_TYPE.get(q.relation, f"rel{q.relation}") for q in queries
    )
    print(
        f"[SAMPLE] queries={len(queries)} candidate_mode={args.candidate_mode} "
        f"samples_per_relation={args.samples_per_relation} relations={dict(counts)}"
    )
    if skipped:
        print(f"[SAMPLE] skipped_for_loo={dict(skipped)}")

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
        reasoning_effort=(
            None if args.reasoning_effort == "none" else args.reasoning_effort
        ),
        reasoning_format=(
            None if args.reasoning_format == "none" else args.reasoning_format
        ),
        retries=args.retries,
        sleep=args.retry_sleep,
    )

    results: list[ComparisonResult] = []
    records_path = Path(args.records_output) if args.records_output else None
    if records_path:
        records_path.parent.mkdir(parents=True, exist_ok=True)
        records_path.write_text("", encoding="utf-8")

    for idx, query in enumerate(queries, start=1):
        graph = graphs[query.graph_index]
        data = data_list[query.graph_index]
        relation = IDX_TO_EDGE_TYPE.get(query.relation, f"rel{query.relation}")
        loo_rank = loo_rank_query(
            loo_bundle,
            loo_data[query.graph_index],
            query,
            device,
        )
        v5_rank = jepa_rank_query(v5_model, data, query, v5_cfg, device)
        prompt = build_prompt(
            graph,
            data,
            v5_cfg,
            query,
            context_mode=args.context,
            max_context_edges=args.max_context_edges,
        )
        response, usage = ranker.rank(prompt)
        order, parse_ok, parse_complete = parse_ranking(
            response,
            len(query.candidates),
        )
        true_position = query.candidates.index(query.target)
        llm_rank = (
            _rank_from_order(order, true_position)
            if parse_ok
            else len(query.candidates)
        )
        result = ComparisonResult(
            graph_index=query.graph_index,
            edge_index=query.edge_index,
            relation=relation,
            source=_node_label(graph, query.source),
            target=_node_label(graph, query.target),
            candidates=[
                _node_label(graph, candidate)
                for candidate in query.candidates
            ],
            loo_jepa_rank=loo_rank,
            v5_jepa_rank=v5_rank,
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

    loo_metrics = _summarize(results, "loo_jepa_rank")
    v5_metrics = _summarize(results, "v5_jepa_rank")
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
    _print_summary("LOO-JEPA", loo_metrics)
    _print_summary("V5-JEPA", v5_metrics)
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
    _print_three_way(loo_metrics, v5_metrics, llm_metrics)

    payload = {
        "config": {
            "checkpoint": args.checkpoint,
            "loo_checkpoint": str(loo_checkpoint),
            "loo_repo_id": args.loo_repo_id,
            "loo_filename": args.loo_filename,
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
        "skipped_for_loo": dict(skipped),
        "loo_jepa": loo_metrics,
        "v5_jepa": v5_metrics,
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
        description=(
            "Sample raw global Fawkes LOO edge-recovery queries and compare "
            "the HF LOO JEPA baseline, Graph-JEPA v5, and an LLM."
        )
    )
    add_data_args(parser)
    parser.set_defaults(
        data="mimic-subkgs",
        mimic_subkg_path=DEFAULT_RAW_TEST_GRAPHS,
    )
    parser.add_argument("--checkpoint", default=DEFAULT_V5_CHECKPOINT)
    parser.add_argument("--encoder-cache", default=".cache/graph_jepa_v5/encoder")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--loo-repo-id",
        default=None,
        help="Hugging Face model repo containing the LOO checkpoint.",
    )
    parser.add_argument(
        "--loo-filename",
        default=DEFAULT_LOO_FILENAME,
        help=f"Checkpoint filename in --loo-repo-id (default: {DEFAULT_LOO_FILENAME}).",
    )
    parser.add_argument(
        "--loo-revision",
        default=None,
        help="Optional Hugging Face revision, branch, or commit SHA.",
    )
    parser.add_argument(
        "--loo-checkpoint",
        default=None,
        help="Local LOO checkpoint path. If set, skips Hugging Face download.",
    )
    parser.add_argument(
        "--hf-token-env",
        default="HF_TOKEN",
        help="Environment variable containing a Hugging Face token.",
    )
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
    )
    parser.add_argument(
        "--reasoning-format",
        choices=["none", "parsed", "raw", "hidden"],
        default="hidden",
    )
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--request-sleep", type=float, default=0.0)
    parser.add_argument(
        "--context",
        choices=["none", "source", "full"],
        default="full",
    )
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
