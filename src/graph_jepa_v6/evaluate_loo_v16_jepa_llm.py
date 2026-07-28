"""Compare the HF v16 entity-note JEPA baseline, Graph-JEPA v6, and an LLM on sampled LOO queries.

The v16 trainer is not the v12 trainer with a new name; it changes the encoder input
contract in ways that this evaluator has to reproduce exactly:

* node features are ``6`` demographics plus a ``768``-dim Clinical-ModernBERT note vector
  placed on the entities the note grounds (``ground_by=prov``), zeros elsewhere,
* ``NOTE`` is a ninth node type and ``HAS_NOTE`` a twenty-first base relation, so the
  inverse-relation offset is 21 rather than 20,
* the v8 score vector is a sigmoid gate on the relation embedding instead of a concat,
* LLM edges with no KB hit and no note provenance are pruned before encoding.
"""

from __future__ import annotations

import argparse
import json
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

from .data import PatientGraphDataset, _grounded_note_nodes, _note_embedding
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
from .evaluate_loo_v12_jepa_llm import (
    DEFAULT_RAW_TEST_GRAPHS,
    DEFAULT_V6_CHECKPOINT,
    LOO_RELATION_ALIASES,
    LOO_SCORE_FEATS,
    LooDistMult,
    LooMLPScorer,
    _infer_layers,
    _loo_add_inverses,
    _loo_entity_hash,
    _loo_numeric_features,
    _loo_score_vec,
    _node_entity_name,
    _print_summary,
    _summarize,
)
from .training import (
    add_data_args,
    build_checkpoint_encoder,
    build_graphs,
    load_model_checkpoint,
)


V16_NODE_TYPES = {
    "PATIENT": 0,
    "DIAGNOSIS": 1,
    "MEDICATION": 2,
    "MICROBIOLOGY": 3,
    "PROCEDURE": 4,
    "SERVICE": 5,
    "LAB_TEST": 6,
    "PROCUREMENT": 7,
    "NOTE": 8,
}
V16_RELATION_CANONICAL = {
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
    "HAS_NOTE": 20,
}
V16_BASE_NUMERIC = 6
V16_EVIDENCE_FEATS = [
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
]
DEFAULT_V16_REPO_ID = "wmatbooth/fawkes-graph-jepa-v16-260615"
DEFAULT_V16_FILENAME = "fawkes_trainer_jepa_entity_note_v16_260615.pt"


@dataclass
class ComparisonResult:
    graph_index: int
    edge_index: int
    relation: str
    source: str
    target: str
    candidates: list[str]
    v16_jepa_rank: int
    v6_jepa_rank: int
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
class V16ModelBundle:
    encoder: "V16Encoder"
    scorer: nn.Module
    config: dict
    entity_vocab: int
    num_base_relations: int
    use_scores: bool
    numeric_dim: int
    note_embedding_dim: int
    note_ground_by: str
    prune_no_evidence: bool


def _v16_relation_id(relation: str | int) -> int | None:
    if isinstance(relation, int):
        relation = IDX_TO_EDGE_TYPE.get(relation, f"rel{relation}")
    normalized = str(relation or "").upper()
    canonical = LOO_RELATION_ALIASES.get(normalized, normalized)
    return V16_RELATION_CANONICAL.get(canonical)


def _v16_edge_values(edge: dict) -> dict:
    """v16 reads scores from ``labels``; the repo graphs carry them on the edge itself."""

    labels = edge.get("labels")
    return labels if isinstance(labels, dict) else edge


def _v16_has_evidence(edge: dict) -> bool:
    values = _v16_edge_values(edge)
    if values.get("prov_in_note"):
        return True
    for key in V16_EVIDENCE_FEATS:
        value = values.get(key)
        if isinstance(value, (int, float)) and value > 0:
            return True
    return False


def _v16_is_no_evidence_llm(edge: dict) -> bool:
    evidence = str(edge.get("evidence") or "").strip().lower()
    return evidence == "llm" and not _v16_has_evidence(edge)


def to_v16_graph_data(
    graph,
    *,
    entity_vocab: int,
    use_scores: bool,
    numeric_dim: int,
    note_embedding_dim: int,
    note_ground_by: str,
) -> Data:
    """Convert a repo PatientGraph into the HF v16 baseline tensor contract."""

    use_note = numeric_dim > V16_BASE_NUMERIC
    if use_note and numeric_dim != V16_BASE_NUMERIC + note_embedding_dim:
        raise ValueError(
            f"v16 numeric dim {numeric_dim} != {V16_BASE_NUMERIC} + "
            f"note embedding dim {note_embedding_dim}"
        )

    demographics = _loo_numeric_features(graph)
    if use_note:
        grounded = _grounded_note_nodes(graph, note_ground_by)
        note = _note_embedding(graph, note_embedding_dim)
        note_vector = note if note is not None else [0.0] * note_embedding_dim
        zero_note = [0.0] * note_embedding_dim
    else:
        grounded = set()
        note = None
        note_vector = []
        zero_note = []

    id_to_idx: dict[str, int] = {}
    node_types = []
    entity_ids = []
    numeric_features = []
    for idx, node in enumerate(graph.nodes):
        node_type = str(node.get("type") or "").upper()
        if node_type not in V16_NODE_TYPES:
            raise KeyError(f"v16 baseline does not support node type {node_type!r}")
        id_to_idx[str(node.get("id") or "")] = idx
        node_types.append(V16_NODE_TYPES[node_type])
        entity_ids.append(_loo_entity_hash(_node_entity_name(node), entity_vocab))
        if use_note:
            numeric_features.append(
                demographics + (note_vector if idx in grounded else zero_note)
            )
        else:
            numeric_features.append(list(demographics))

    src, dst, rel, edge_feat, no_evidence = [], [], [], [], []
    for edge in graph.edges:
        source_id = str(edge.get("source_id") or edge.get("source") or "")
        target_id = str(edge.get("target_id") or edge.get("target") or "")
        relation = str(edge.get("type") or edge.get("relation") or "").upper()
        rel_id = _v16_relation_id(relation)
        if rel_id is None:
            continue
        if source_id not in id_to_idx or target_id not in id_to_idx:
            continue
        src.append(id_to_idx[source_id])
        dst.append(id_to_idx[target_id])
        rel.append(rel_id)
        edge_feat.append(_loo_score_vec(edge))
        no_evidence.append(_v16_is_no_evidence_llm(edge))

    data = Data()
    data.num_nodes = len(node_types)
    data.node_type = torch.tensor(node_types, dtype=torch.long)
    data.entity_id = torch.tensor(entity_ids, dtype=torch.long)
    data.numfeat = torch.tensor(numeric_features, dtype=torch.float)
    data.sem_id = torch.zeros(len(node_types), dtype=torch.long)
    data.note_grounded = torch.tensor([len(grounded)], dtype=torch.long)
    data.note_present = torch.tensor([note is not None], dtype=torch.bool)
    if src:
        data.edge_index = torch.tensor([src, dst], dtype=torch.long)
        data.edge_type = torch.tensor(rel, dtype=torch.long)
        data.edge_feat = torch.tensor(edge_feat, dtype=torch.float)
        data.edge_no_evidence = torch.tensor(no_evidence, dtype=torch.bool)
    else:
        data.edge_index = torch.zeros((2, 0), dtype=torch.long)
        data.edge_type = torch.zeros((0,), dtype=torch.long)
        data.edge_feat = torch.zeros((0, len(LOO_SCORE_FEATS)), dtype=torch.float)
        data.edge_no_evidence = torch.zeros((0,), dtype=torch.bool)
    if not use_scores:
        data.edge_feat = torch.zeros(
            (data.edge_type.numel(), len(LOO_SCORE_FEATS)),
            dtype=torch.float,
        )
    return data


class V16Encoder(nn.Module):
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
        numeric_dim: int,
        use_scores: bool,
        use_entity_emb: bool = True,
    ):
        super().__init__()
        if hid % heads:
            raise ValueError(f"v16 hidden size {hid} is not divisible by heads {heads}")
        self.use_scores = use_scores
        self.use_entity_emb = use_entity_emb
        self.type_emb = nn.Embedding(num_node_types, hid)
        self.entity_emb = nn.Embedding(entity_vocab, hid)
        self.num_proj = nn.Linear(numeric_dim, hid)
        self.rel_emb = nn.Embedding(num_relations, edge_emb)
        # v16 gates the relation embedding with the v8 score vector, so edge_dim
        # stays at edge_emb instead of the v12 edge_emb + score concat.
        self.score_gate = nn.Sequential(
            nn.Linear(len(LOO_SCORE_FEATS), edge_emb),
            nn.ReLU(),
            nn.Linear(edge_emb, 1),
        )
        self.convs = nn.ModuleList(
            TransformerConv(
                hid,
                hid // heads,
                heads=heads,
                concat=True,
                edge_dim=edge_emb,
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
                raise ValueError("v16 checkpoint expects edge score features")
            edge_attr = edge_attr * torch.sigmoid(self.score_gate(edge_feat))
        for conv, norm in zip(self.convs, self.norms):
            h = F.relu(norm(conv(h, edge_index, edge_attr)))
        return h


def load_v16_checkpoint(path: str | Path, device: torch.device) -> V16ModelBundle:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    encoder_state = ckpt["encoder"]
    scorer_state = ckpt["scorer"]
    config = dict(ckpt.get("config", {}))

    hid = int(encoder_state["type_emb.weight"].shape[1])
    num_node_types = int(encoder_state["type_emb.weight"].shape[0])
    entity_vocab = int(encoder_state["entity_emb.weight"].shape[0])
    num_relations = int(encoder_state["rel_emb.weight"].shape[0])
    edge_emb = int(encoder_state["rel_emb.weight"].shape[1])
    numeric_dim = int(encoder_state["num_proj.weight"].shape[1])
    layers = int(config.get("layers") or _infer_layers(encoder_state))
    heads = int(config.get("heads") or 4)
    use_scores = bool(config.get("use_scores", False))
    note_embedding_dim = int(
        config.get("embed_dim") or max(numeric_dim - V16_BASE_NUMERIC, 0)
    )
    note_ground_by = str(config.get("ground_by") or "prov").lower()
    prune_no_evidence = bool(config.get("prune_no_evidence", True))

    encoder = V16Encoder(
        hid=hid,
        layers=layers,
        heads=heads,
        edge_emb=edge_emb,
        entity_vocab=entity_vocab,
        num_node_types=num_node_types,
        num_relations=num_relations,
        numeric_dim=numeric_dim,
        use_scores=use_scores,
    ).to(device)
    encoder.load_state_dict(encoder_state)

    scorer_cls = (
        LooMLPScorer
        if any(key.startswith("mlp.") for key in scorer_state)
        else LooDistMult
    )
    scorer = scorer_cls(hid=hid, num_relations=num_relations).to(device)
    scorer.load_state_dict(scorer_state)
    encoder.eval()
    scorer.eval()
    return V16ModelBundle(
        encoder=encoder,
        scorer=scorer,
        config=config,
        entity_vocab=entity_vocab,
        num_base_relations=num_relations // 2,
        use_scores=use_scores,
        numeric_dim=numeric_dim,
        note_embedding_dim=note_embedding_dim,
        note_ground_by=note_ground_by,
        prune_no_evidence=prune_no_evidence,
    )


def resolve_v16_checkpoint(args) -> Path:
    if args.v16_checkpoint:
        path = Path(args.v16_checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f"v16 checkpoint not found: {path}")
        return path
    if not args.v16_repo_id:
        raise SystemExit("Pass --v16-repo-id, or --v16-checkpoint for a local .pt file.")
    from huggingface_hub import hf_hub_download

    token = os.environ.get(args.hf_token_env)
    return Path(
        hf_hub_download(
            repo_id=args.v16_repo_id,
            filename=args.v16_filename,
            repo_type="model",
            revision=args.v16_revision,
            token=token,
        )
    )


def _v16_exact_edge_mask(data: Data, query: RecoveryQuery) -> torch.Tensor:
    rel_id = _v16_relation_id(query.relation)
    if rel_id is None or data.edge_type.numel() == 0:
        return torch.zeros(data.edge_type.size(0), dtype=torch.bool)
    return (
        (data.edge_index[0] == query.source)
        & (data.edge_index[1] == query.target)
        & (data.edge_type == rel_id)
    )


def filter_v16_compatible_queries(
    queries: list[RecoveryQuery],
    v16_data: list[Data],
) -> tuple[list[RecoveryQuery], Counter[str]]:
    kept = []
    skipped: Counter[str] = Counter()
    for query in queries:
        relation = IDX_TO_EDGE_TYPE.get(query.relation, f"rel{query.relation}")
        if _v16_relation_id(relation) is None:
            skipped[f"unsupported_relation:{relation}"] += 1
            continue
        data = v16_data[query.graph_index]
        if query.source >= data.num_nodes or query.target >= data.num_nodes:
            skipped["node_index_out_of_range"] += 1
            continue
        if not bool(_v16_exact_edge_mask(data, query).any()):
            skipped[f"missing_exact_edge:{relation}"] += 1
            continue
        kept.append(query)
    return kept, skipped


@torch.no_grad()
def v16_rank_query(
    bundle: V16ModelBundle,
    data: Data,
    query: RecoveryQuery,
    device: torch.device,
    *,
    prune_no_evidence: bool,
) -> int:
    rel_id = _v16_relation_id(query.relation)
    if rel_id is None:
        raise ValueError(f"v16 baseline does not support relation id {query.relation}")
    data = data.clone().to(device)
    dropped = _v16_exact_edge_mask(data, query).to(device)
    if prune_no_evidence:
        dropped = dropped | data.edge_no_evidence.to(device)
    keep = ~dropped
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


def _print_three_way(v16: dict, v6: dict, llm: dict) -> None:
    v16_by_rel = {row["rel"]: row for row in v16["per_rel"]}
    v6_by_rel = {row["rel"]: row for row in v6["per_rel"]}
    llm_by_rel = {row["rel"]: row for row in llm["per_rel"]}
    print(
        "[PER-REL] relation                 n     C    chance  "
        "V16_MRR V16_H@1  V6_MRR V6_H@1  LLM_MRR LLM_H@1"
    )
    relations = sorted(v6_by_rel, key=lambda rel: -v6_by_rel[rel]["n"])
    for relation in relations:
        br = v16_by_rel.get(relation)
        vr = v6_by_rel[relation]
        mr = llm_by_rel.get(relation)
        if br is None or mr is None:
            continue
        print(
            f"[PER-REL] {relation:<22} {vr['n']:<5} {vr['C']:<4.0f} "
            f"{vr['chance_mrr']:<7.3f} {br['mrr']:<7.3f} {br['h1']:<8.3f} "
            f"{vr['mrr']:<6.3f} {vr['h1']:<7.3f} "
            f"{mr['mrr']:<7.3f} {mr['h1']:<7.3f}"
        )


def _print_two_way(v16: dict, v6: dict) -> None:
    v16_by_rel = {row["rel"]: row for row in v16["per_rel"]}
    v6_by_rel = {row["rel"]: row for row in v6["per_rel"]}
    print(
        "[PER-REL] relation                 n     C    chance  "
        "V16_MRR V16_H@1  V6_MRR V6_H@1"
    )
    relations = sorted(v6_by_rel, key=lambda rel: -v6_by_rel[rel]["n"])
    for relation in relations:
        br = v16_by_rel.get(relation)
        vr = v6_by_rel[relation]
        if br is None:
            continue
        print(
            f"[PER-REL] {relation:<22} {vr['n']:<5} {vr['C']:<4.0f} "
            f"{vr['chance_mrr']:<7.3f} {br['mrr']:<7.3f} {br['h1']:<8.3f} "
            f"{vr['mrr']:<6.3f} {vr['h1']:<7.3f}"
        )


def run(args) -> dict:
    device = torch.device(args.device)
    v6_model, v6_cfg = load_model_checkpoint(args.checkpoint, device)
    v6_model.eval()
    encoder = build_checkpoint_encoder(v6_cfg, args.encoder_cache)
    graphs = build_graphs(args, v6_cfg)
    dataset = PatientGraphDataset(
        graphs,
        encoder,
        use_note_embeddings=v6_cfg.model.use_note_embeddings,
        note_embedding_dim=v6_cfg.model.note_embedding_dim,
        note_ground_by=v6_cfg.model.note_ground_by,
    )
    data_list = [dataset[idx] for idx in range(len(dataset))]

    v16_checkpoint = resolve_v16_checkpoint(args)
    v16_bundle = load_v16_checkpoint(v16_checkpoint, device)
    if args.prune_no_evidence == "auto":
        prune_no_evidence = v16_bundle.prune_no_evidence
    else:
        prune_no_evidence = args.prune_no_evidence == "on"
    print(
        f"[V16] numeric_dim={v16_bundle.numeric_dim} "
        f"note_embedding_dim={v16_bundle.note_embedding_dim} "
        f"ground_by={v16_bundle.note_ground_by} "
        f"use_scores={v16_bundle.use_scores} "
        f"num_base_relations={v16_bundle.num_base_relations} "
        f"prune_no_evidence={prune_no_evidence} "
        f"(checkpoint default {v16_bundle.prune_no_evidence})"
    )

    v16_data = [
        to_v16_graph_data(
            graph,
            entity_vocab=v16_bundle.entity_vocab,
            use_scores=v16_bundle.use_scores,
            numeric_dim=v16_bundle.numeric_dim,
            note_embedding_dim=v16_bundle.note_embedding_dim,
            note_ground_by=v16_bundle.note_ground_by,
        )
        for graph in graphs
    ]

    notes_present = sum(int(data.note_present[0]) for data in v16_data)
    grounded_nodes = sum(int(data.note_grounded[0]) for data in v16_data)
    total_nodes = sum(int(data.num_nodes) for data in v16_data)
    pruned_edges = sum(int(data.edge_no_evidence.sum()) for data in v16_data)
    total_edges = sum(int(data.edge_type.numel()) for data in v16_data)
    print(
        f"[V16] note_embeddings={notes_present}/{len(v16_data)} graphs | "
        f"grounded_nodes={grounded_nodes}/{total_nodes} | "
        f"no_evidence_llm_edges={pruned_edges}/{total_edges}"
    )
    if v16_bundle.numeric_dim > V16_BASE_NUMERIC and notes_present == 0:
        print(
            "[V16][WARN] the checkpoint expects note vectors but no graph carries "
            "note_embedding; v16 runs with an all-zero note block, which is off its "
            "training distribution. v6 is note-less on this data too."
        )

    queries = collect_recovery_queries(
        data_list,
        v6_cfg,
        samples_per_relation=args.samples_per_relation,
        candidate_mode=args.candidate_mode,
        max_candidates=args.max_candidates,
        seed=args.seed,
        allow_duplicate_triples=args.allow_duplicate_triples,
    )
    queries, skipped = filter_v16_compatible_queries(queries, v16_data)
    if args.max_queries is not None:
        queries = queries[: args.max_queries]
    if not queries:
        raise ValueError("no eligible v16-compatible recovery queries found")

    counts = Counter(
        IDX_TO_EDGE_TYPE.get(q.relation, f"rel{q.relation}") for q in queries
    )
    print(
        f"[SAMPLE] queries={len(queries)} candidate_mode={args.candidate_mode} "
        f"samples_per_relation={args.samples_per_relation} relations={dict(counts)}"
    )
    if skipped:
        print(f"[SAMPLE] skipped_for_v16={dict(skipped)}")

    if args.skip_llm:
        llm_model = ""
        ranker = None
    else:
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
        v16_rank = v16_rank_query(
            v16_bundle,
            v16_data[query.graph_index],
            query,
            device,
            prune_no_evidence=prune_no_evidence,
        )
        v6_rank = jepa_rank_query(v6_model, data, query, v6_cfg, device)
        if args.skip_llm:
            response = ""
            usage = {}
            parse_ok = False
            parse_complete = False
            llm_rank = 0
        else:
            prompt = build_prompt(
                graph,
                data,
                v6_cfg,
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
            v16_jepa_rank=v16_rank,
            v6_jepa_rank=v6_rank,
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

    v16_metrics = _summarize(results, "v16_jepa_rank")
    v6_metrics = _summarize(results, "v6_jepa_rank")

    print("[RESULTS]")
    _print_summary("V16-JEPA", v16_metrics)
    _print_summary("V6-JEPA", v6_metrics)

    if args.skip_llm:
        llm_metrics = None
        _print_two_way(v16_metrics, v6_metrics)
    else:
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
        _print_three_way(v16_metrics, v6_metrics, llm_metrics)

    payload = {
        "config": {
            "checkpoint": args.checkpoint,
            "v16_checkpoint": str(v16_checkpoint),
            "v16_repo_id": args.v16_repo_id,
            "v16_filename": args.v16_filename,
            "v16_config": v16_bundle.config,
            "prune_no_evidence": prune_no_evidence,
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
            "skip_llm": args.skip_llm,
        },
        "v16_inputs": {
            "graphs_with_note_embedding": notes_present,
            "graphs": len(v16_data),
            "grounded_nodes": grounded_nodes,
            "nodes": total_nodes,
            "no_evidence_llm_edges": pruned_edges,
            "edges": total_edges,
        },
        "sample_counts": dict(counts),
        "skipped_for_v16": dict(skipped),
        "v16_jepa": v16_metrics,
        "v6_jepa": v6_metrics,
        "records": [asdict(result) for result in results],
    }
    if not args.skip_llm:
        payload["llm"] = llm_metrics
        payload["llm_parse_failures"] = parse_failures
        payload["llm_incomplete_rankings"] = incomplete
        payload["llm_token_usage"] = token_usage
        payload["llm_finish_reasons"] = dict(finish_reasons)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sample raw global Fawkes LOO edge-recovery queries and compare "
            "the HF v16 entity-note JEPA baseline, Graph-JEPA v6, and an LLM."
        )
    )
    add_data_args(parser)
    parser.set_defaults(
        data="mimic-subkgs",
        mimic_subkg_path=DEFAULT_RAW_TEST_GRAPHS,
    )
    parser.add_argument("--checkpoint", default=DEFAULT_V6_CHECKPOINT)
    parser.add_argument("--encoder-cache", default=".cache/graph_jepa_v6/encoder")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--v16-repo-id",
        default=DEFAULT_V16_REPO_ID,
        help="Hugging Face model repo containing the v16 checkpoint.",
    )
    parser.add_argument(
        "--v16-filename",
        default=DEFAULT_V16_FILENAME,
        help=f"Checkpoint filename in --v16-repo-id (default: {DEFAULT_V16_FILENAME}).",
    )
    parser.add_argument(
        "--v16-revision",
        default=None,
        help="Optional Hugging Face revision, branch, or commit SHA.",
    )
    parser.add_argument(
        "--v16-checkpoint",
        default=None,
        help="Local v16 checkpoint path. If set, skips Hugging Face download.",
    )
    parser.add_argument(
        "--prune-no-evidence",
        choices=["auto", "on", "off"],
        default="auto",
        help=(
            "Drop LLM edges with no KB hit and no note provenance from the v16 "
            "context, as the v16 trainer does. 'auto' follows the checkpoint config."
        ),
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
        "--skip-llm",
        action="store_true",
        help="Only compare the two JEPA methods; skip all LLM calls.",
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
