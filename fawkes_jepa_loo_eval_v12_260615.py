# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch==2.4.1", "torch-geometric==2.6.1", "huggingface_hub>=0.25",
#   "scikit-learn>=1.3", "numpy",
# ]
# ///
# fawkes_jepa_loo_eval_v12_260615.py
# ============================================================================
# VERSION v12 (260615) — LEAVE-ONE-OUT edge recovery (the fair test). Trains the v9 baseline
#   (USE_SCORES=0 world model + frozen readout) UNCHANGED, then GRADES THE WHOLE GRAPH one edge at a time:
#     for every edge -> remove ONLY that edge, keep the entire rest of the graph, ask the model to recover it
#     (rank the true target vs same-type candidates, FILTERED for other true tails), put it back, next edge.
#   WHY: masking many edges at once (v8/v11 readout = 30% held out together) strips the mutual context the
#     inferred cross-links lean on (a drug manages a dx, that dx complicates another...), so it UNDER-states
#     recovery. Leave-one-out always gives full surrounding context = how a refiner actually works. No leak:
#     the masked edge AND its inverse are absent before encoding; no randomness -> exactly reproducible.
#   Reports [LOO] per-relation MRR/H@1/H@3/H@10 vs chance + [CONTRAST] 30%-mask -> leave-one-out for the 4
#     inferred LLM edges. The v11 batch-holdout EIR uplift is retained behind RUN_EIR=1 (OFF by default:
#     its full-gold edge-F1 is hub-dominated; the LOO per-relation number is the honest measure).
# ============================================================================
# VERSION v9 (260615) — TRAIN ON THE v8-SCORED CORPUS + INGEST THE v8 EDGE-SCORE VECTOR.
#   Built on v8 (the world-model proof). Two changes only; everything else (JEPA pretrain, EMA target,
#   slot-conditioned query, frozen DistMult readout, frozen per-patient eval quiz, determinism,
#   per-relation recovery) is carried over verbatim.
#   CHANGE 1 (loader): reads the per-range SCORED datasets (SCORED_REPOS, comma-sep jsonl of
#     graphs-stage2-v8-{R}); demographics from COMPLETE5K_REPO by subject_id. Fresh seeded
#     train/val/test split (no old fixed 965 test set).
#   CHANGE 2 (edge features): the encoder edge feature is the 14-dim v8 SCORE VECTOR per edge
#     (model, drug_link_cos, dx_disease_cos, het_*, omop_*_cos, omop_lca_dist, prov_in_note, prov_ratio),
#     observed-edges-only (no leak), instead of the single GPT-OSS confidence scalar. None -> 0.0.
#   v10 will add the edit head (keep/delete + relabel) + silver-edit supervision; v11 the EIR uplift evals.
# ============================================================================
# REAL Graph-JEPA world model for clinical KG refinement (WM@Booth). PHASE 1 = masked-node latent
# prediction (BYOL, EMA target) on ONE shared encoder = the world model; PHASE 2 = frozen-encoder edge
# recovery readout. Inverse-edge leakage + test-set-selection bugs fixed. CLAUDE.md: deterministic,
# [TAG] logging, NO fallback (unknown type/relation/missing data fails loud). Added 260615.

import os, json, copy, time, math, hashlib, logging
import numpy as np

if os.environ.get("DETERMINISTIC", "1").lower() in ("1", "true", "yes"):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # MUST precede import torch
import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import TransformerConv, global_mean_pool
from torch_geometric.utils import subgraph
from sklearn.metrics import roc_auc_score, average_precision_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("fawkes_jepa")

NODE_TYPES = {
    "PATIENT": 0,
    "DIAGNOSIS": 1,
    "MEDICATION": 2,
    "MICROBIOLOGY": 3,
    "PROCEDURE": 4,
    "SERVICE": 5,
    "LAB_TEST": 6,
    "PROCUREMENT": 7,
}
RELATION_CANONICAL = {
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
RELATION_ALIASES = {
    "DIAGNORED_BY": "DIAGNOSED_BY",
    "TARGET_ORGANISM": "TARGETS_ORGANISM",
    "HAS_MICROBIOLOGY": "HAD_MICROBIOLOGY",
    "HAD_PROCEDURE": "UNDERWENT_PROCEDURE",
    "HAS_MEDICATION": "TAKES_MEDICATION",
    "MANAGES_FOR": "MANAGED_FOR",
}
NUM_NODE_TYPES = len(NODE_TYPES)
NUM_BASE = len(RELATION_CANONICAL)
NUM_RELATIONS = 2 * NUM_BASE
NUMERIC_DIM = 6
# (v9) the v8 edge-score vector fed to the encoder. Numeric signals only; None -> 0.0. lca_dist scaled.
SCORE_FEATS = [
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
SCORE_DIM = len(SCORE_FEATS)


def score_vec(labels):
    labels = labels or {}
    out = []
    for k in SCORE_FEATS:
        v = labels.get(k)
        if not isinstance(v, (int, float)):
            v = 0.0
        if k == "omop_lca_dist":
            v = math.exp(-float(v) / 5.0) if v else 0.0  # distance -> closeness in (0,1]
        out.append(float(v))
    return out


def env(k, d=None, required=False):
    v = os.environ.get(k, d)
    if required and not v:
        raise RuntimeError(f"[FAILURE] env {k} unset; no fallback.")
    return v


SCORED_REPOS = [
    r.strip() for r in env("SCORED_REPOS", required=True).split(",") if r.strip()
]  # (v9) comma-sep scored jsonl datasets
COMPLETE5K_REPO = env("COMPLETE5K_REPO", "on1onmangoes/fawkes-mimic-complete5k-260611")
COMPLETE5K_FILE = env("COMPLETE5K_FILE", "fawkes_mimic_complete5k_260611.jsonl")
PUSH = str(os.environ.get("PUSH", "1")).lower() in ("1", "true", "yes")
OUTPUT_REPO = env("OUTPUT_REPO", required=PUSH)
JEPA_EPOCHS = int(os.environ.get("JEPA_EPOCHS", 60))
READOUT_EPOCHS = int(os.environ.get("READOUT_EPOCHS", 40))
BATCH = int(os.environ.get("BATCH", 16))
HID = int(os.environ.get("HID", 128))
LAYERS = int(os.environ.get("LAYERS", 2))
HEADS = int(os.environ.get("HEADS", 4))
EDGE_EMB = int(os.environ.get("EDGE_EMB", 32))
ENTITY_VOCAB = int(os.environ.get("ENTITY_VOCAB", 8192))
LR = float(os.environ.get("LR", 1e-3))
NODE_MASK = float(os.environ.get("NODE_MASK", 0.4))
EDGE_MASK = float(os.environ.get("EDGE_MASK", 0.3))
EMA_BASE = float(os.environ.get("EMA_BASE", 0.996))
EMA_FINAL = float(os.environ.get("EMA_FINAL", 0.9999))
VAL_FRAC = float(os.environ.get("VAL_FRAC", 0.1))
TEST_FRAC = float(os.environ.get("TEST_FRAC", 0.1))
FREEZE = str(os.environ.get("FREEZE_ENCODER", "1")).lower() in ("1", "true", "yes")
MRR_CAP = int(os.environ.get("MRR_CAP", 3000))
SEED = int(os.environ.get("SEED", 42))
DETERMINISTIC = os.environ.get("DETERMINISTIC", "1").lower() in ("1", "true", "yes")
QUERY_ENTITY = str(os.environ.get("QUERY_ENTITY", "0")).lower() in ("1", "true", "yes")
USE_ENTITY_EMB = str(os.environ.get("ENTITY_EMB", "1")).lower() in ("1", "true", "yes")
NEG_K = int(os.environ.get("NEG_K", 8))
TEMP = float(os.environ.get("TEMP", 1.0))
USE_SCORES = str(os.environ.get("USE_SCORES", "0")).lower() in (
    "1",
    "true",
    "yes",
)  # (v11) DEFAULT OFF — v9 ablation: scores-as-encoder-input hurt non-obvious; the EIR eval encodes structure-only
DECODER = os.environ.get("DECODER", "distmult").lower()
FREEZE_EVAL = str(os.environ.get("FREEZE_EVAL", "1")).lower() in ("1", "true", "yes")
LOSS = os.environ.get("LOSS", "infonce").lower()
SEMANTIC_ENT = str(os.environ.get("SEMANTIC_ENT", "0")).lower() in ("1", "true", "yes")
if HID % HEADS:
    raise ValueError(f"[FAILURE] HID {HID} not divisible by HEADS {HEADS}")


def set_seed(s):
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    np.random.seed(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if DETERMINISTIC:
        torch.use_deterministic_algorithms(True, warn_only=True)


def ehash(n):
    return int(hashlib.md5(n.encode("utf-8")).hexdigest(), 16) % ENTITY_VOCAB


def resolve_rel(r):
    if r in RELATION_CANONICAL:
        return RELATION_CANONICAL[r]
    if r in RELATION_ALIASES:
        return RELATION_CANONICAL[RELATION_ALIASES[r]]
    raise KeyError(f"[FAILURE] unknown relation '{r}'; no fallback.")


def add_inverses(ei, et, ef=None):  # (v9) ef is the (E,SCORE_DIM) feature matrix (or None)
    if ei.size(1) == 0:
        return (ei, et) if ef is None else (ei, et, ef)
    ei2 = torch.cat([ei, ei.flip(0)], 1)
    et2 = torch.cat([et, et + NUM_BASE])
    if ef is None:
        return ei2, et2
    return ei2, et2, torch.cat([ef, ef], 0)  # inverse edge inherits its forward edge's v8 scores


def load_demographics():
    from huggingface_hub import hf_hub_download

    p = hf_hub_download(COMPLETE5K_REPO, COMPLETE5K_FILE, repo_type="dataset")
    demo = {}
    for l in open(p):
        r = json.loads(l)
        demo[str(r.get("subject_id"))] = {"gender": r.get("gender"), "age": r.get("anchor_age")}
    logger.info(f"[DEMO] {len(demo)} patients from {COMPLETE5K_REPO}")
    return demo


def load_scored_graphs():
    from huggingface_hub import hf_hub_download, list_repo_files

    graphs = []
    for repo in SCORED_REPOS:
        fs = [f for f in list_repo_files(repo, repo_type="dataset") if f.endswith(".jsonl")]
        if not fs:
            raise FileNotFoundError(f"[FAILURE] no .jsonl in {repo}; no fallback.")
        n0 = len(graphs)
        for f in fs:
            for l in open(hf_hub_download(repo, f, repo_type="dataset")):
                l = l.strip()
                if l:
                    graphs.append(json.loads(l))
        logger.info(f"[SCORED] {repo}: +{len(graphs)-n0} graphs")
    if not graphs:
        raise RuntimeError("[FAILURE] 0 scored graphs loaded; no fallback.")
    return graphs


def numeric(demo_rec):
    if demo_rec is None:
        return [0.0] * NUMERIC_DIM
    age = demo_rec.get("age")
    age = float(age) / 100.0 if str(age).strip() not in ("", "None") else 0.0
    g = demo_rec.get("gender")
    return [
        age,
        1.0 if g == "M" else 0.0,
        1.0 if g == "F" else 0.0,
        0.0,
        0.0,
        0.0,
    ]  # (v9) age/sex; structure carries the rest (type-only ≥ full)


TARGET_RELS = {"MANAGED_FOR", "CONFIRMS", "COMPLICATED_BY", "INDICATES"}  # (v11) the 4 inferred LLM edges
SUPPORT_FEATS = [
    "dx_disease_cos",
    "het_treats_ctd",
    "het_treats_cpd",
    "het_resembles_drd",
    "het_presents_dps",
    "prov_ratio",
]
EIR_HOLDOUT = float(os.environ.get("EIR_HOLDOUT", 0.30))  # frac of gold LLM edges held out for the ADD test
EIR_FUZZY = float(os.environ.get("EIR_FUZZY", 0.80))  # EIR fuzzy-node match threshold (kg_similarity_scorer)
LOO_CAP = int(os.environ.get("LOO_CAP", 40000))  # (v12) max leave-one-out queries (each edge re-encodes the graph)
RUN_EIR = str(os.environ.get("RUN_EIR", "0")).lower() in (
    "1",
    "true",
    "yes",
)  # (v12) v11 batch-holdout EIR eval OFF by default (full-gold F1 is hub-dominated)


def support_graded(labels):  # (v11) graded evidence-support in [0,1], non-saturating
    labels = labels or {}
    return max(
        [float(labels.get(k) or 0.0) for k in SUPPORT_FEATS] + [0.0]
    )  # max KB endorsement / provenance ratio (no binary prov->1.0)


def to_data(g, demo):
    sid = str(g.get("subject_id"))
    drec = demo.get(sid)
    nodes, edges = g.get("nodes", []), g.get("edges", [])
    id2i, types, ents, numf = {}, [], [], []
    nf = numeric(drec)
    for i, n in enumerate(nodes):
        id2i[n["id"]] = i
        t = n.get("type")
        if t not in NODE_TYPES:
            raise KeyError(f"[FAILURE] unknown node type '{t}' subj {sid}")
        name = n.get("normalized_name") or n.get("name") or ""
        types.append(NODE_TYPES[t])
        ents.append(ehash(name))
        numf.append(nf)
    src, dst, et, feats = [], [], [], []
    for e in edges:
        s, d = e["source"], e["target"]
        if s not in id2i or d not in id2i:
            raise ValueError(f"[FAILURE] dangling edge subj {sid}")
        src.append(id2i[s])
        dst.append(id2i[d])
        et.append(resolve_rel(e.get("relation")))
        feats.append(score_vec(e.get("labels")))
    data = Data()
    data.num_nodes = len(nodes)
    data.n_edges_real = len(src)
    data.node_type = torch.tensor(types, dtype=torch.long)
    data.entity_id = torch.tensor(ents, dtype=torch.long)
    data.numfeat = torch.tensor(numf, dtype=torch.float)
    data.edge_index = (
        torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros((2, 0), dtype=torch.long)
    )  # FORWARD only
    data.edge_type = torch.tensor(et, dtype=torch.long) if et else torch.zeros((0,), dtype=torch.long)
    data.edge_feat = (
        torch.tensor(feats, dtype=torch.float) if feats else torch.zeros((0, SCORE_DIM), dtype=torch.float)
    )  # (v9) per-edge v8 score vector
    data.sem_id = torch.zeros(len(nodes), dtype=torch.long)
    data.gid = torch.tensor(
        [int(sid) if str(sid).isdigit() else int(hashlib.md5(str(sid).encode()).hexdigest()[:12], 16)], dtype=torch.long
    )
    return data


# ---- shared encoder (the world model) ----
class Encoder(nn.Module):
    def __init__(s):
        super().__init__()
        s.type_emb = nn.Embedding(NUM_NODE_TYPES, HID)
        s.entity_emb = nn.Embedding(ENTITY_VOCAB, HID)
        s.num_proj = nn.Linear(NUMERIC_DIM, HID)
        s.rel_emb = nn.Embedding(NUM_RELATIONS, EDGE_EMB)
        s.edim = EDGE_EMB + (SCORE_DIM if USE_SCORES else 0)  # (v9) v8 score vector appended to the relation embedding
        s.convs = nn.ModuleList(
            TransformerConv(HID, HID // HEADS, heads=HEADS, concat=True, edge_dim=s.edim, dropout=0.0)
            for _ in range(LAYERS)
        )
        s.norms = nn.ModuleList(nn.LayerNorm(HID) for _ in range(LAYERS))

    def forward(s, nt, eid, numf, ei, et, efeat=None, sem_id=None):
        ident = s.entity_emb(eid) if USE_ENTITY_EMB else 0
        h = s.type_emb(nt) + ident + s.num_proj(numf)
        ea = s.rel_emb(et)
        if USE_SCORES:
            if efeat is None:
                raise ValueError("[FAILURE] USE_SCORES on but efeat not passed to encoder; no fallback.")
            ea = torch.cat([ea, efeat], -1)
        for c, n in zip(s.convs, s.norms):
            h = F.relu(n(c(h, ei, ea)))
        return h


class JEPA(nn.Module):
    def __init__(s):
        super().__init__()
        s.ctx = Encoder()
        s.tgt = copy.deepcopy(s.ctx)
        for p in s.tgt.parameters():
            p.requires_grad_(False)
        s.pred = nn.Sequential(nn.Linear(2 * HID, HID), nn.ReLU(), nn.Linear(HID, HID))
        s.slot_rel = nn.Embedding(NUM_RELATIONS, HID)
        s.ema = EMA_BASE

    @torch.no_grad()
    def update(s):
        for pt, pc in zip(s.tgt.parameters(), s.ctx.parameters()):
            pt.mul_(s.ema).add_(pc, alpha=1 - s.ema)


def valid_mask(b, ng, ratio, device, tries=10):
    n = b.size(0)
    for _ in range(tries):
        m = torch.rand(n, device=device) < ratio
        tg = torch.zeros(ng, device=device).scatter_add_(0, b, m.float())
        ct = torch.zeros(ng, device=device).scatter_add_(0, b, (~m).float())
        if bool((tg > 0).all()) and bool((ct > 0).all()):
            return m
    raise RuntimeError(f"[FAILURE] no valid node mask after {tries} tries (ratio={ratio}); graphs too small.")


def jepa_step(model, b, device):
    b = b.to(device)
    nt, eid, numf, ei, et, bt = b.node_type, b.entity_id, b.numfeat, b.edge_index, b.edge_type, b.batch
    N = nt.size(0)
    ng = int(bt.max().item()) + 1
    tmask = valid_mask(bt, ng, NODE_MASK, device)
    cmask = ~tmask
    cn = cmask.nonzero(as_tuple=False).view(-1)
    if USE_SCORES:
        cei, cet, emask = subgraph(cn, ei, edge_attr=et, relabel_nodes=True, num_nodes=N, return_edge_mask=True)
        cef = b.edge_feat[emask]
        cei, cet, cef = add_inverses(cei, cet, cef)
        ch = model.ctx(nt[cn], eid[cn], numf[cn], cei, cet, cef, b.sem_id[cn])
    else:
        cei, cet = subgraph(cn, ei, edge_attr=et, relabel_nodes=True, num_nodes=N)
        cei, cet = add_inverses(cei, cet)
        ch = model.ctx(nt[cn], eid[cn], numf[cn], cei, cet, None, b.sem_id[cn])
    csum = global_mean_pool(ch, bt[cn], size=ng)
    with torch.no_grad():
        if USE_SCORES:
            fei, fet, fef = add_inverses(ei, et, b.edge_feat)
            th = model.tgt(nt, eid, numf, fei, fet, fef, b.sem_id)
        else:
            fei, fet = add_inverses(ei, et)
            th = model.tgt(nt, eid, numf, fei, fet, None, b.sem_id)
        emb_std = th.std(0).mean()
    tn = tmask.nonzero(as_tuple=False).view(-1)
    tgt = F.normalize(th[tn], dim=-1)
    g2c = torch.full((N,), -1, dtype=torch.long, device=device)
    g2c[cn] = torch.arange(cn.numel(), device=device)
    s_, d_ = ei[0], ei[1]
    m1 = tmask[s_] & cmask[d_]
    m2 = tmask[d_] & cmask[s_]
    tgt_ep = torch.cat([s_[m1], d_[m2]])
    nbr = torch.cat([d_[m1], s_[m2]])
    rels = torch.cat([et[m1], et[m2] + NUM_BASE])
    msg = ch[g2c[nbr]] + model.slot_rel(rels)
    slot = torch.zeros(N, HID, device=device).index_add_(0, tgt_ep, msg)
    cnt = torch.zeros(N, device=device).index_add_(0, tgt_ep, torch.ones(tgt_ep.numel(), device=device))
    slot = slot / cnt.clamp(min=1).unsqueeze(-1)
    q = slot[tn] + model.ctx.type_emb(nt[tn])
    if QUERY_ENTITY:
        q = q + model.ctx.entity_emb(eid[tn])
    pred = F.normalize(model.pred(torch.cat([csum[bt[tn]], q], -1)), dim=-1)
    return (2 - 2 * (pred * tgt).sum(-1)).mean(), emb_std


# ---- downstream edge-recovery readout (graph completion off frozen latents) ----
class Scorer(nn.Module):
    def __init__(s):
        super().__init__()
        s.rel = nn.Embedding(NUM_RELATIONS, HID)
        s.mlp = nn.Sequential(nn.Linear(3 * HID, HID), nn.ReLU(), nn.Linear(HID, 1))

    def forward(s, h, u, v, r):
        return s.mlp(torch.cat([h[u], h[v], s.rel(r)], -1)).squeeze(-1)


class DistMult(nn.Module):
    def __init__(s):
        super().__init__()
        s.rel = nn.Embedding(NUM_RELATIONS, HID)

    def forward(s, h, u, v, r):
        return (h[u] * s.rel(r) * h[v]).sum(-1)


def buckets(nt, bt):
    bk = bt * NUM_NODE_TYPES + nt
    o = torch.argsort(bk)
    return o, bk[o]


def same_type_k(targets, nt, bt, o, sb, K, gen=None):
    tb = bt[targets] * NUM_NODE_TYPES + nt[targets]
    lo = torch.searchsorted(sb, tb, right=False)
    hi = torch.searchsorted(sb, tb, right=True)
    span = (hi - lo).clamp(min=1).float()
    r = torch.rand(targets.numel(), K, device=targets.device, generator=gen)
    pick = (r * span.unsqueeze(1)).long() + lo.unsqueeze(1)
    return o[pick.clamp(max=o.numel() - 1)]


def readout_step(enc, scorer, b, device, train, gen=None):
    b = b.to(device)
    nt, eid, numf, ei, et, bt = b.node_type, b.entity_id, b.numfeat, b.edge_index, b.edge_type, b.batch
    E = ei.size(1)
    if E < 2:
        return None
    perm = torch.randperm(E, device=device, generator=gen)
    k = max(1, int(EDGE_MASK * E))
    hold = perm[:k]
    obs = perm[k:]
    if USE_SCORES:
        oei, oet, oef = add_inverses(
            ei[:, obs], et[obs], b.edge_feat[obs]
        )  # observed-only -> held-out scores never seen (no leak)
    else:
        oei, oet = add_inverses(ei[:, obs], et[obs])
        oef = None
    ctxmgr = torch.enable_grad() if (train and not FREEZE) else torch.no_grad()
    with ctxmgr:
        h = enc(nt, eid, numf, oei, oet, oef, b.sem_id)
    pu, pv, pr = ei[0, hold], ei[1, hold], et[hold]
    o, sb = buckets(nt, bt)
    nvk = same_type_k(pv, nt, bt, o, sb, NEG_K, gen=gen)
    Pn = pu.numel()
    u_rep = pu.unsqueeze(1).expand(Pn, NEG_K).reshape(-1)
    r_rep = pr.unsqueeze(1).expand(Pn, NEG_K).reshape(-1)
    pos = scorer(h, pu, pv, pr).view(Pn, 1)
    neg = scorer(h, u_rep, nvk.reshape(-1), r_rep).view(Pn, NEG_K)
    logits = torch.cat([pos, neg], 1) / TEMP
    logits[:, 1:][nvk == pv.unsqueeze(1)] = -1e9
    if LOSS == "bce":
        pos1 = pos.squeeze(1)
        neg1 = neg[:, 0]
        logits_bce = torch.cat([pos1, neg1])
        labels = torch.cat([torch.ones_like(pos1), torch.zeros_like(neg1)])
        loss = F.binary_cross_entropy_with_logits(logits_bce, labels)
    else:
        loss = F.cross_entropy(logits, torch.zeros(Pn, dtype=torch.long, device=device))
    qsig = (int(hold.sum().item()) * 1000003 + int(nvk.reshape(-1).sum().item()) + E * 7919) if gen is not None else 0
    return (
        loss,
        pos.squeeze(1).detach(),
        neg[:, 0].detach(),
        (nt[pu] != 0).detach(),
        (h.detach(), pu, pv, pr, nt, bt, qsig),
    )


@torch.no_grad()
def evaluate(enc, scorer, loader, device):
    enc.eval()
    scorer.eval()
    P, N, NP = [], [], []
    rr = []
    hits = {1: 0, 3: 0, 10: 0}
    nmrr = 0
    qsig_tot = 0
    from collections import defaultdict

    rel_rr = defaultdict(list)
    rel_hits = defaultdict(lambda: {1: 0, 3: 0, 10: 0})
    rel_n = defaultdict(int)
    rel_C = defaultdict(list)
    for b in loader:
        gen = torch.Generator(device=device).manual_seed(int(b.gid[0]) & 0x7FFFFFFFFFFFFFFF) if FREEZE_EVAL else None
        r = readout_step(enc, scorer, b, device, train=False, gen=gen)
        if r is None:
            continue
        _, pos, neg, nonpat, extra = r
        P.append(pos)
        N.append(neg)
        NP.append(nonpat)
        h, pu, pv, pr, nt, bt, qsig = extra
        qsig_tot = (qsig_tot + qsig) & 0xFFFFFFFFFFFF
        o, sb = buckets(nt, bt)
        for i in range(pu.numel()):
            if nmrr >= MRR_CAP:
                break
            u, v, r_ = int(pu[i]), int(pv[i]), int(pr[i])
            tb = int(bt[v]) * NUM_NODE_TYPES + int(nt[v])
            lo = int(torch.searchsorted(sb, torch.tensor(tb, device=device), right=False))
            hi = int(torch.searchsorted(sb, torch.tensor(tb, device=device), right=True))
            cand = o[lo:hi]
            if cand.numel() < 2:
                continue
            sc = scorer(
                h,
                torch.full((cand.numel(),), u, device=device, dtype=torch.long),
                cand,
                torch.full((cand.numel(),), r_, device=device, dtype=torch.long),
            )
            rank = int((sc > sc[(cand == v).nonzero(as_tuple=False)[0, 0]]).sum().item()) + 1
            rr.append(1.0 / rank)
            rel_rr[r_].append(1.0 / rank)
            rel_n[r_] += 1
            rel_C[r_].append(int(cand.numel()))
            for kk in hits:
                if rank <= kk:
                    hits[kk] += 1
                    rel_hits[r_][kk] += 1
            nmrr += 1
    pos = torch.cat(P).cpu().numpy()
    neg = torch.cat(N).cpu().numpy()
    npm = torch.cat(NP).cpu().numpy()
    y = np.concatenate([np.ones_like(pos), np.zeros_like(neg)])
    s = np.concatenate([pos, neg])
    auc, ap = roc_auc_score(y, s), average_precision_score(y, s)
    sp = pos[npm]
    if len(sp) > 5:
        y2 = np.concatenate([np.ones_like(sp), np.zeros_like(neg)])
        s2 = np.concatenate([sp, neg])
        auc2 = roc_auc_score(y2, s2)
    else:
        auc2 = float("nan")
    mrr = float(np.mean(rr)) if rr else float("nan")
    H = {k: hits[k] / max(nmrr, 1) for k in hits}
    ID2REL = {v: k for k, v in RELATION_CANONICAL.items()}
    per_rel = []
    for r_ in sorted(rel_n, key=lambda x: -rel_n[x]):
        n = rel_n[r_]
        C = float(np.mean(rel_C[r_]))
        per_rel.append(
            {
                "rel": ID2REL.get(r_, f"rel{r_}"),
                "n": n,
                "mrr": float(np.mean(rel_rr[r_])),
                "h1": rel_hits[r_][1] / n,
                "h10": rel_hits[r_][10] / n,
                "C": C,
                "chance_mrr": (math.log(C) + 0.5772) / C if C > 1 else 1.0,
                "chance_h1": 1.0 / C if C >= 1 else 1.0,
            }
        )
    return {
        "auc": auc,
        "ap": ap,
        "auc_nonobvious": auc2,
        "mrr": mrr,
        "hits1": H[1],
        "hits3": H[3],
        "hits10": H[10],
        "n_mrr": nmrr,
        "qsig": qsig_tot,
        "per_rel": per_rel,
    }


# ---- (v11) EIR scoring method (kg_similarity_scorer.py) + the refiner uplift eval ----
from difflib import SequenceMatcher


def _norm(t):
    return " ".join(str(t or "").lower().split())


def edge_prf(pred, gold):  # directional (src_text, REL, dst_text) exact-triple set P/R/F1
    if not gold:
        return (1.0, 1.0, 1.0)
    if not pred:
        return (0.0, 0.0, 0.0)
    tp = len(pred & gold)
    p = tp / len(pred)
    r = tp / len(gold)
    f = 2 * p * r / (p + r) if p + r > 0 else 0.0
    return p, r, f


@torch.no_grad()
def eir_uplift_eval(enc, scorer, pairs, tau, device):
    # refiner per TEST graph: GOLD = backbone + LLM edges with graded support>=tau; hold out EIR_HOLDOUT of gold LLM
    # edges; RAW = draft - held; REFINED = (keep supported, drop unsupported) + model-ADD (recover held-out top-1).
    enc.eval()
    scorer.eval()
    from collections import defaultdict

    A = defaultdict(list)
    add = defaultdict(lambda: [0, 0])  # add[REL]=[recovered, held]
    for d, g in pairs:
        nodes = g["nodes"]
        names = [(n.get("normalized_name") or n.get("name") or "") for n in nodes]
        idof = {n["id"]: i for i, n in enumerate(nodes)}
        E = []
        for e in g["edges"]:
            if e["source"] not in idof or e["target"] not in idof:
                continue
            try:
                rid = resolve_rel(e.get("relation"))
            except Exception:
                continue
            u = idof[e["source"]]
            v = idof[e["target"]]
            REL = str(e.get("relation", "")).upper()
            llm = e.get("evidence") == "llm"
            sup = support_graded(e.get("labels")) if llm else 1.0
            E.append(
                {
                    "u": u,
                    "v": v,
                    "REL": REL,
                    "rid": rid,
                    "llm": llm,
                    "sup": sup,
                    "tri": (_norm(names[u]), REL, _norm(names[v])),
                }
            )
        gold = set(e["tri"] for e in E if (not e["llm"]) or e["sup"] >= tau)
        gold_llm = [e for e in E if e["llm"] and e["sup"] >= tau]
        if not gold or int(d.num_nodes) < 3:
            continue
        gen = torch.Generator().manual_seed(int(d.gid[0]) & 0x7FFFFFFF)
        order = torch.randperm(len(gold_llm), generator=gen).tolist() if gold_llm else []
        held = [gold_llm[i] for i in order[: int(round(EIR_HOLDOUT * len(gold_llm)))]]
        held_tri = set(e["tri"] for e in held)
        raw_set = set(e["tri"] for e in E) - held_tri
        kept = [e for e in E if ((not e["llm"]) or e["sup"] >= tau) and e["tri"] not in held_tri]
        kept_set = set(e["tri"] for e in kept)
        nt = d.node_type.to(device)
        ntl = nt.tolist()
        if kept:
            oei = torch.tensor([[e["u"] for e in kept], [e["v"] for e in kept]], dtype=torch.long, device=device)
            oet = torch.tensor([e["rid"] for e in kept], dtype=torch.long, device=device)
        else:
            oei = torch.zeros((2, 0), dtype=torch.long, device=device)
            oet = torch.zeros((0,), dtype=torch.long, device=device)
        oei, oet = add_inverses(oei, oet)
        h = enc(nt, d.entity_id.to(device), d.numfeat.to(device), oei, oet, None, d.sem_id.to(device))
        added = set()
        for e in held:
            cand = [j for j in range(len(nodes)) if ntl[j] == ntl[e["v"]] and j != e["u"]]
            add[e["REL"]][1] += 1
            if not cand:
                continue
            ct = torch.tensor(cand, dtype=torch.long, device=device)
            sc = scorer(
                h,
                torch.full((len(cand),), e["u"], dtype=torch.long, device=device),
                ct,
                torch.full((len(cand),), e["rid"], dtype=torch.long, device=device),
            )
            vpred = cand[int(sc.argmax())]
            added.add((_norm(names[e["u"]]), e["REL"], _norm(names[vpred])))
            if vpred == e["v"]:
                add[e["REL"]][0] += 1
        refined = kept_set | added
        rp, rr, rf = edge_prf(raw_set, gold)
        fp, fr, ff = edge_prf(refined, gold)
        A["rawP"].append(rp)
        A["rawR"].append(rr)
        A["rawF"].append(rf)
        A["refP"].append(fp)
        A["refR"].append(fr)
        A["refF"].append(ff)
    mean = lambda x: float(np.mean(x)) if x else float("nan")
    out = {k: mean(A[k]) for k in ("rawP", "rawR", "rawF", "refP", "refR", "refF")}
    out["n_graphs"] = len(A["rawF"])
    out["add"] = {
        k: {"rec": v[0], "held": v[1], "recall": (v[0] / v[1] if v[1] else float("nan"))} for k, v in add.items()
    }
    return out


@torch.no_grad()
def loo_evaluate(enc, scorer, graphs, device, cap=LOO_CAP):
    # (v12) LEAVE-ONE-OUT: mask exactly ONE edge, keep the entire rest of the graph, recover it.
    # Filtered ranking (drop other true tails of (u,rel)); no randomness -> exact/reproducible. No leak:
    # the masked edge's forward AND inverse are absent (inverses are built only over the kept edges).
    enc.eval()
    scorer.eval()
    from collections import defaultdict

    rel_rr = defaultdict(list)
    rel_hits = defaultdict(lambda: {1: 0, 3: 0, 10: 0})
    rel_n = defaultdict(int)
    rel_C = defaultdict(list)
    nq = 0
    for d in graphs:
        if nq >= cap:
            break
        d = d.to(device)
        ei, et, nt = d.edge_index, d.edge_type, d.node_type
        E = ei.size(1)
        if E < 2:
            continue
        ei0, ei1 = ei[0], ei[1]
        for i in range(E):
            if nq >= cap:
                break
            u = int(ei0[i])
            v = int(ei1[i])
            r = int(et[i])
            if u == v:
                continue
            keep = torch.ones(E, dtype=torch.bool, device=device)
            keep[i] = False
            oei, oet = add_inverses(ei[:, keep], et[keep])  # full graph minus this ONE edge (its inverse is gone too)
            h = enc(nt, d.entity_id, d.numfeat, oei, oet, None, d.sem_id)
            cand = (nt == nt[v]).nonzero(as_tuple=False).view(-1)
            cand = cand[cand != u]
            others = ei1[(ei0 == u) & (et == r)]
            others = others[others != v]  # filtered: other true tails of (u,rel)
            if others.numel() > 0:
                cand = cand[~torch.isin(cand, others)]
            if cand.numel() < 2 or int((cand == v).sum()) == 0:
                continue
            sc = scorer(
                h,
                torch.full((cand.numel(),), u, dtype=torch.long, device=device),
                cand,
                torch.full((cand.numel(),), r, dtype=torch.long, device=device),
            )
            rank = int((sc > sc[(cand == v).nonzero(as_tuple=False)[0, 0]]).sum().item()) + 1
            rel_rr[r].append(1.0 / rank)
            rel_n[r] += 1
            rel_C[r].append(int(cand.numel()))
            for kk in (1, 3, 10):
                if rank <= kk:
                    rel_hits[r][kk] += 1
            nq += 1
    ID2REL = {v: k for k, v in RELATION_CANONICAL.items()}
    per_rel = []
    all_rr = []
    tot = {1: 0, 3: 0, 10: 0}
    tot_n = 0
    for r_ in sorted(rel_n, key=lambda x: -rel_n[x]):
        n = rel_n[r_]
        C = float(np.mean(rel_C[r_]))
        all_rr += rel_rr[r_]
        tot_n += n
        for kk in tot:
            tot[kk] += rel_hits[r_][kk]
        per_rel.append(
            {
                "rel": ID2REL.get(r_, f"rel{r_}"),
                "n": n,
                "mrr": float(np.mean(rel_rr[r_])),
                "h1": rel_hits[r_][1] / n,
                "h3": rel_hits[r_][3] / n,
                "h10": rel_hits[r_][10] / n,
                "C": C,
                "chance_mrr": (math.log(C) + 0.5772) / C if C > 1 else 1.0,
                "chance_h1": 1.0 / C if C >= 1 else 1.0,
            }
        )
    return {
        "mrr": float(np.mean(all_rr)) if all_rr else float("nan"),
        "hits1": tot[1] / max(tot_n, 1),
        "hits3": tot[3] / max(tot_n, 1),
        "hits10": tot[10] / max(tot_n, 1),
        "n": tot_n,
        "per_rel": per_rel,
    }


def main():
    logger.info(
        f"[ENTRY] LOO v12 | scored_repos={SCORED_REPOS} use_scores={USE_SCORES} score_dim={SCORE_DIM} jepa_ep={JEPA_EPOCHS} readout_ep={READOUT_EPOCHS} loo_cap={LOO_CAP} run_eir={RUN_EIR} "
        f"hid={HID} node_mask={NODE_MASK} edge_mask={EDGE_MASK} entity_emb={USE_ENTITY_EMB} decoder={DECODER} loss={LOSS} neg_k={NEG_K} freeze_eval={FREEZE_EVAL} deterministic={DETERMINISTIC} val={VAL_FRAC} test={TEST_FRAC} seed={SEED}"
    )
    set_seed(SEED)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"[GPU] {dev}" + (f" {torch.cuda.get_device_name(0)}" if dev.type == "cuda" else ""))
    demo = load_demographics()
    raw = load_scored_graphs()
    # vocab audit (fail loud)
    rels = set()
    types = set()
    for g in raw:
        for n in g.get("nodes", []):
            types.add(n.get("type"))
        for e in g.get("edges", []):
            rels.add(e.get("relation"))
    ur = sorted(r for r in rels if r and r not in (set(RELATION_CANONICAL) | set(RELATION_ALIASES)))
    ut = sorted(t for t in types if t and t not in NODE_TYPES)
    logger.info(f"[VOCAB] rel={len(rels)} type={len(types)} unknown_rel={len(ur)} unknown_type={len(ut)}")
    if ur or ut:
        raise RuntimeError(f"[FAILURE] unknown relations={ur} types={ut}; no fallback.")
    items = []
    for g in raw:
        d = to_data(g, demo)
        if d.num_nodes >= 3 and d.edge_index.size(1) >= 4:
            items.append((d, g))  # keep raw graph alongside for the EIR eval
    rng = np.random.RandomState(SEED)
    idx = rng.permutation(len(items))
    nte = int(TEST_FRAC * len(items))
    nval = int(VAL_FRAC * len(items))
    te_pairs = [items[i] for i in idx[:nte]]
    val_pairs = [items[i] for i in idx[nte : nte + nval]]
    tr_pairs = [items[i] for i in idx[nte + nval :]]
    te = [d for d, _ in te_pairs]
    val = [d for d, _ in val_pairs]
    tr = [d for d, _ in tr_pairs]
    logger.info(f"[DATA] graphs={len(items)} -> TRAIN={len(tr)} VAL={len(val)} TEST={len(te)} (seeded split)")

    # ---- PHASE 1: JEPA world-model pretraining ----
    model = JEPA().to(dev)
    opt = torch.optim.Adam(
        [p for p in model.ctx.parameters() if p.requires_grad]
        + list(model.pred.parameters())
        + list(model.slot_rel.parameters()),
        lr=LR,
    )
    logger.info(f"[MODEL] JEPA params={sum(p.numel() for p in model.parameters()):,} edge_dim={model.ctx.edim}")
    tl = DataLoader(tr, batch_size=BATCH, shuffle=True)
    for ep in range(1, JEPA_EPOCHS + 1):
        model.ctx.train()
        model.ema = EMA_FINAL - (EMA_FINAL - EMA_BASE) * (math.cos(math.pi * (ep - 1) / max(JEPA_EPOCHS, 1)) + 1) / 2
        t0 = time.perf_counter()
        tot = ts = nb = 0
        for b in tl:
            opt.zero_grad()
            loss, es = jepa_step(model, b, dev)
            loss.backward()
            opt.step()
            model.update()
            tot += loss.item()
            ts += es.item()
            nb += 1
        if ep % 10 == 0 or ep == JEPA_EPOCHS:
            logger.info(
                f"[LATENCY] JEPA epoch={ep}/{JEPA_EPOCHS} loss={tot/nb:.4f} emb_std={ts/nb:.4f} ema={model.ema:.4f} took={(time.perf_counter()-t0)*1000:.0f}ms"
            )

    # ---- PHASE 2: downstream edge-recovery readout on the (frozen) world-model encoder ----
    enc = model.ctx
    if FREEZE:
        for p in enc.parameters():
            p.requires_grad_(False)
        enc.eval()
    scorer = (DistMult() if DECODER == "distmult" else Scorer()).to(dev)
    params = list(scorer.parameters()) + ([] if FREEZE else list(enc.parameters()))
    ropt = torch.optim.Adam(params, lr=LR)
    eb = 1 if FREEZE_EVAL else BATCH
    rl = DataLoader(tr, batch_size=BATCH, shuffle=True)
    vl = DataLoader(val, batch_size=eb, shuffle=False)
    el = DataLoader(te, batch_size=eb, shuffle=False)
    best = {"auc": 0.0}
    best_state = None
    for ep in range(1, READOUT_EPOCHS + 1):
        scorer.train()
        if not FREEZE:
            enc.train()
        t0 = time.perf_counter()
        tot = nb = 0
        for b in rl:
            r = readout_step(enc, scorer, b, dev, train=True)
            if r is None:
                continue
            ropt.zero_grad()
            r[0].backward()
            ropt.step()
            tot += r[0].item()
            nb += 1
        if ep % 5 == 0 or ep == READOUT_EPOCHS:
            vm = evaluate(enc, scorer, vl, dev)
            logger.info(
                f"[LATENCY] READOUT epoch={ep}/{READOUT_EPOCHS} loss={tot/max(nb,1):.4f} took={(time.perf_counter()-t0)*1000:.0f}ms "
                f"| VAL auc={vm['auc']:.3f} nonobv={vm['auc_nonobvious']:.3f} MRR={vm['mrr']:.3f} H@1={vm['hits1']:.3f} H@10={vm['hits10']:.3f}"
            )
            if vm["auc"] > best["auc"]:
                best = {**vm, "epoch": ep}
                best_state = copy.deepcopy(scorer.state_dict())

    if best_state is not None:
        scorer.load_state_dict(best_state)
    tm = evaluate(enc, scorer, el, dev)
    logger.info(
        f"[RESULT] TEST (val-selected) | auc={tm['auc']:.3f} ap={tm['ap']:.3f} non-obvious_auc={tm['auc_nonobvious']:.3f} "
        f"MRR={tm['mrr']:.3f} Hits@1={tm['hits1']:.3f} Hits@3={tm['hits3']:.3f} Hits@10={tm['hits10']:.3f} (n_mrr={tm['n_mrr']}) quiz_sig={tm['qsig']} frozen={FREEZE_EVAL} scores={USE_SCORES}"
    )
    logger.info("[PER-REL] ONE shared latent, edge recovery by relation (sorted by edge count):")
    for rr_ in tm["per_rel"]:
        flag = (
            " <== ABOVE chance"
            if rr_["mrr"] > 1.5 * rr_["chance_mrr"]
            else (" <== ~chance" if rr_["mrr"] < 1.2 * rr_["chance_mrr"] else "")
        )
        logger.info(
            f"[PER-REL] {rr_['rel']:<22} n={rr_['n']:<5} C={rr_['C']:.0f}  MRR={rr_['mrr']:.3f} (chance {rr_['chance_mrr']:.3f})  H@1={rr_['h1']:.3f}  H@10={rr_['h10']:.3f}{flag}"
        )

    # ---- (v12) LEAVE-ONE-OUT edge recovery: mask ONE edge, keep the rest, recover it (full context, filtered) ----
    loo = loo_evaluate(enc, scorer, te, dev)
    logger.info(
        f"[LOO] leave-one-out (mask 1 edge, full context, filtered) | MRR={loo['mrr']:.3f} H@1={loo['hits1']:.3f} H@3={loo['hits3']:.3f} H@10={loo['hits10']:.3f} over {loo['n']} edges"
    )
    logger.info("[LOO] edge recovery by relation (one edge masked at a time, full surrounding context):")
    for rr_ in loo["per_rel"]:
        flag = (
            " <== ABOVE chance"
            if rr_["mrr"] > 1.5 * rr_["chance_mrr"]
            else (" <== ~chance" if rr_["mrr"] < 1.2 * rr_["chance_mrr"] else "")
        )
        logger.info(
            f"[LOO] {rr_['rel']:<22} n={rr_['n']:<5} C={rr_['C']:.0f}  MRR={rr_['mrr']:.3f} (chance {rr_['chance_mrr']:.3f})  H@1={rr_['h1']:.3f}  H@10={rr_['h10']:.3f}{flag}"
        )
    bm = {x["rel"]: x for x in tm["per_rel"]}
    lo = {x["rel"]: x for x in loo["per_rel"]}
    logger.info(
        "[CONTRAST] 4 inferred LLM edges — 30%-batch-mask MRR -> leave-one-out MRR (gain from keeping full context):"
    )
    for rel in sorted(TARGET_RELS):
        a = bm.get(rel)
        c = lo.get(rel)
        if a and c:
            logger.info(
                f"[CONTRAST] {rel:<16} {a['mrr']:.3f} -> {c['mrr']:.3f}  ({c['mrr']-a['mrr']:+.3f})  | H@1 {a['h1']:.3f} -> {c['h1']:.3f}"
            )

    # ---- (v11, optional) EIR batch-holdout uplift — OFF by default (full-gold F1 is hub-dominated; LOO above is the honest measure) ----
    tau = None
    eir = None
    if RUN_EIR:
        sup_v = [
            support_graded(e.get("labels")) for _, g in val_pairs for e in g["edges"] if e.get("evidence") == "llm"
        ]
        tau = float(np.quantile(sup_v, 0.5)) if sup_v else 0.5
        logger.info(
            f"[EIR] gold/keep tau (VAL median graded LLM support) = {tau:.3f} | holdout={EIR_HOLDOUT} fuzzy={EIR_FUZZY}"
        )
        eir = eir_uplift_eval(enc, scorer, te_pairs, tau, dev)
        logger.info(
            f"[EIR-UPLIFT] TEST edge-triple vs v8-gold ({eir['n_graphs']} graphs) | RAW P={eir['rawP']:.3f} R={eir['rawR']:.3f} F1={eir['rawF']:.3f}"
        )
        logger.info(
            f"[EIR-UPLIFT] TEST edge-triple vs v8-gold | REFINED P={eir['refP']:.3f} R={eir['refR']:.3f} F1={eir['refF']:.3f}  (F1 {eir['refF']-eir['rawF']:+.3f} | precision {eir['refP']-eir['rawP']:+.3f} = DISCONNECT, recall {eir['refR']-eir['rawR']:+.3f} = model ADD)"
        )
        logger.info("[EIR-ADD] held-out evidence-supported edges recovered (top-1 same-type), the 4 LLM targets:")
        for rel in sorted(TARGET_RELS):
            a = eir["add"].get(rel, {"rec": 0, "held": 0, "recall": float("nan")})
            logger.info(
                f"[EIR-ADD] {rel:<16} recovered = {a['rec']}/{a['held']} = {(a['recall'] if a['recall']==a['recall'] else 0.0):.3f}"
            )

    ckpt = "fawkes_jepa_loo_eval_v12_260615.pt"
    torch.save(
        {
            "encoder": enc.state_dict(),
            "scorer": scorer.state_dict(),
            "config": {
                "model": "graph_jepa_loo_eval_v12",
                "hid": HID,
                "layers": LAYERS,
                "heads": HEADS,
                "use_scores": USE_SCORES,
                "node_mask": NODE_MASK,
                "edge_mask": EDGE_MASK,
                "loo_cap": LOO_CAP,
                "run_eir": RUN_EIR,
                "eir_tau": tau,
                "scored_repos": SCORED_REPOS,
                "seed": SEED,
            },
            "recovery_test_batchmask": tm,
            "recovery_test_loo": loo,
            "eir": eir,
        },
        ckpt,
    )
    if not PUSH:
        logger.info(f"[DONE] PUSH=0 — saved local {ckpt}")
        return
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(OUTPUT_REPO, repo_type="model", exist_ok=True, private=True)
    api.upload_file(path_or_fileobj=ckpt, path_in_repo=ckpt, repo_id=OUTPUT_REPO, repo_type="model")
    logger.info(
        f"[DONE] v12 LOO -> https://huggingface.co/{OUTPUT_REPO}/blob/main/{ckpt} | leave-one-out MRR={loo['mrr']:.3f} H@1={loo['hits1']:.3f} (vs 30%-mask MRR={tm['mrr']:.3f}) over {loo['n']} edges"
    )


if __name__ == "__main__":
    main()
