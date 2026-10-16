# /// script
# requires-python = ">=3.10"
# dependencies = ["datasets>=2.19", "duckdb>=1.0", "torch", "transformers>=4.40",
#                 "huggingface_hub>=0.25", "hf_transfer", "pyarrow"]
#
# [[tool.uv.index]]
# name = "pytorch-cu121"
# url = "https://download.pytorch.org/whl/cu121"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch-cu121" }
# ///
# fawkes_graph_stage2_score_v8_260613.py
# ============================================================================
# STAGE 2 v8 — PRODUCTION edge scorer for the MIMIC patient knowledge graphs
# (Graph-JEPA training data; figures destined for a scientific paper).
#
# PRINCIPLES (CLAUDE.md): actuals only, NO fallback code, FAIL LOUDLY, deterministic,
# semantic similarity (no hard code-matching), reproducible. Every signal is computed
# from a real reference KB; nothing is stubbed. The SAME label schema is written on every
# edge (label everything, prune downstream); KB signals fire only on the LLM relations they
# can speak to, and deterministic/backbone edges keep model=1.0 (structured ground truth).
#
# ADDITIVE over the working v5/v7 "medicine" metric (kept verbatim):
#   model          : Stage-1 LLM confidence (1.0 on deterministic edges)
#   drug_link_cos  : clean_drug(med) -> nearest RxNorm INGREDIENT by SapBERT cosine  -> rxcui   [v5/v7]
#   dx_disease_cos : dx -> nearest disease in that rxcui's MED-RT may_treat list (RxNav)         [v5/v7]
#   matched_disease: the matched may_treat indication name                                       [v5/v7]
# NEW this iteration (only signals the KBs actually carry — verified by inspection):
#   het_treats_ctd / het_treats_cpd : type-constrained Hetionet Compound->Disease  (CtD / CpD)
#   het_resembles_drd               : type-constrained Hetionet Disease-resembles-Disease (DrD)
#   het_presents_dps                : type-constrained Hetionet Disease-presents-Symptom (DpS)
#   omop_src/omop_dst (+cos)        : SapBERT resolution to OMOP standard concept (RxNorm / SNOMED)
#   omop_lca_dist                   : dx<->dx hierarchy distance via CONCEPT_ANCESTOR lowest-common-ancestor
#   prov_in_note / prov_ratio       : cited quote verified against the source note (only if quotes present)
#
# G1 FIX: the MED-RT may_treat KB is fetched ONCE per unique rxcui, with transient errors RETRIED
#   then RAISED (never silently treated as empty — that was the run-to-run dxΦ flicker), and the
#   resulting {rxcui -> [diseases]} table is FROZEN to its own dataset for reproducibility.
#
# Launch (A100):
#   hf jobs uv run --flavor a100-large -s HF_TOKEN --timeout 90m -d fawkes_graph_stage2_score_v8_260613.py
# ============================================================================
import os, sys, re, json, time, logging, urllib.request, urllib.error
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
sys.stdout.reconfigure(line_buffering=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("stage2_v8")

# ── config ──
STAGE1_REPO  = os.environ.get("STAGE1_REPO",  "on1onmangoes/fawkes-mimic-graphs-stage1-rows4000-5000-260613")
NOTES_REPO   = os.environ.get("NOTES_REPO",   "on1onmangoes/fawkes-mimic-notes-1k-v3-rows4000-5000-260612")
RXNORM_REPO  = os.environ.get("RXNORM_REPO",  "on1onmangoes/fawkes-kg-rxnorm-ingredients-260613")
HET_REPO     = os.environ.get("HETIONET_REPO","on1onmangoes/fawkes-kg-hetionet-260613")
OMOP_CONCEPT = os.environ.get("OMOP_CONCEPT_REPO", "on1onmangoes/fawkes-omop-concept-embedded-260613")
OMOP_ANC     = os.environ.get("OMOP_ANC_REPO", "on1onmangoes/fawkes-omop-concept-ancestor-260613")
SEM_MODEL    = os.environ.get("SEM_MODEL",    "cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
OUT_REPO     = os.environ.get("OUTPUT_REPO",  "on1onmangoes/fawkes-mimic-graphs-stage2-v8-rows4000-5000-260613")
MAYTREAT_REPO= os.environ.get("MAYTREAT_REPO","on1onmangoes/fawkes-medrt-maytreat-cache-260613")
CHUNK        = int(os.environ.get("CHUNK", 50000))     # OMOP concept rows per GPU cosine chunk
RXNAV_WORKERS= int(os.environ.get("RXNAV_WORKERS", 8))

LLM_RELATIONS = {"MANAGED_FOR", "COMPLICATED_BY", "INDICATES", "CONFIRMS"}
RUN = {"datestamp": "260613", "sem_model": SEM_MODEL, "stage1": STAGE1_REPO,
       "kbs": {"rxnorm": RXNORM_REPO, "hetionet": HET_REPO, "omop_concept": OMOP_CONCEPT, "omop_ancestor": OMOP_ANC},
       "signals": [], "coverage": {}}

import torch
from datasets import load_dataset, Dataset
from huggingface_hub import hf_hub_download, list_repo_files, HfApi
DEV = "cuda" if torch.cuda.is_available() else "cpu"
if DEV != "cuda":                                            # FAIL LOUD — production run is GPU
    raise SystemExit("[FAIL] CUDA not available. This is a production A100 job; refusing to run on CPU.")
log.info(f"[ENTRY] stage2_v8 device={DEV}")

# ── SapBERT (deterministic embeddings) ──
from transformers import AutoTokenizer, AutoModel
log.info(f"[MODEL] loading {SEM_MODEL}")
_tk = AutoTokenizer.from_pretrained(SEM_MODEL); _md = AutoModel.from_pretrained(SEM_MODEL).eval().to(DEV)
@torch.no_grad()
def embed(texts, bs=512):
    out = []
    for i in range(0, len(texts), bs):
        b = _tk(texts[i:i+bs], padding=True, truncation=True, max_length=32, return_tensors="pt").to(DEV)
        out.append(torch.nn.functional.normalize(_md(**b).last_hidden_state[:, 0], dim=1))
    return torch.cat(out) if out else torch.empty(0, 768, device=DEV)

# ── dose/route/brand strip — the proven #1 quality lever (verbatim from semlink v5) ──
_DOSE = re.compile(r'\b\d+\.?\d*\s*(mg|mcg|ug|ml|gm|g|units?|unit|meq|mmol|iu)\b', re.I)
_FORM = re.compile(r'\b(neb|nebu|soln|solution|oral|liqu|liquid|tablet|tab|cap|capsule|inj|injection|'
                   r'extended|release|er|xl|xr|sr|cr|la|ec|dr|mini|bag|plus|drip|inhalation|inhal|'
                   r'spray|patch|flush|nf|gel|powder|vial|amp|puff|drops?|ointment|cream|supp|'
                   r'po|iv|im|sc|sq|ng|pr|sl|td)\b', re.I)
def clean_drug(s):
    s = re.sub(r'\([^)]*\)', ' ', s or '')
    s = re.sub(r'\*[^*]*\*', ' ', s)
    s = re.sub(r'\d+\.?\d*\s*%', ' ', s)
    s = _DOSE.sub(' ', s); s = _FORM.sub(' ', s)
    s = re.sub(r'[^a-zA-Z ]', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()

# ── MED-RT may_treat via RxNav, by rxcui. Transient error -> retry -> RAISE. 404/empty -> valid empty. ──
def fetch_may_treat(rc):
    url = f"https://rxnav.nlm.nih.gov/REST/rxclass/class/byRxcui.json?rxcui={rc}&relaSource=MEDRT&rela=may_treat"
    last = None
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=20) as r:
                d = json.loads(r.read())
            items = d.get("rxclassDrugInfoList", {}).get("rxclassDrugInfo", [])
            return rc, sorted({i["rxclassMinConceptItem"]["className"] for i in items})
        except urllib.error.HTTPError as he:
            if he.code == 404:
                return rc, []                                  # valid: drug has no MED-RT may_treat class
            last = he; time.sleep(1.5 * (attempt + 1))
        except Exception as ex:
            last = ex; time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"[FAILURE] RxNav may_treat failed for rxcui={rc} after 5 attempts: {last!r}")

# ── 1. frozen Stage-1 graphs ──
nf = next(f for f in list_repo_files(STAGE1_REPO, repo_type="dataset") if f.endswith(".jsonl"))
graphs = [json.loads(l) for l in open(hf_hub_download(STAGE1_REPO, nf, repo_type="dataset"))]
if not graphs:
    raise SystemExit(f"[FAIL] no graphs loaded from {STAGE1_REPO}")
n_edges = sum(len(g["edges"]) for g in graphs)
log.info(f"[STAGE1] {len(graphs)} graphs, {n_edges} edges")

# unique names by node type
def names_of(types):
    return sorted({nd["name"] for g in graphs for nd in g["nodes"] if nd["type"] in types})
DRUG_NAMES = names_of({"MEDICATION"})
DX_NAMES   = names_of({"DIAGNOSIS"})
log.info(f"[NAMES] {len(DRUG_NAMES)} drug, {len(DX_NAMES)} dx unique")
NVEC = {n: v for n, v in zip(DRUG_NAMES + DX_NAMES, embed(DRUG_NAMES + DX_NAMES))}  # raw-name embeddings (Het/OMOP)

# ── 2. drug -> RxNorm ingredient (drug_link_cos, rxcui) — semantic NN over the embedded RxNorm KB ──
rx = load_dataset(RXNORM_REPO, split="train")
if "rxcui" not in rx.column_names or "embedding" not in rx.column_names:
    raise SystemExit(f"[FAIL] {RXNORM_REPO} missing rxcui/embedding columns: {rx.column_names}")
RX_CUI = rx["rxcui"]; RX_E = torch.tensor(rx["embedding"], device=DEV)
clean_names = [clean_drug(n) for n in DRUG_NAMES]
CLEAN_E = embed(clean_names)
sims = CLEAN_E @ RX_E.T                                        # [n_drug, n_rxnorm]
mx, ix = sims.max(dim=1)
drug2rx = {DRUG_NAMES[i]: (RX_CUI[int(ix[i])], round(float(mx[i]), 3)) for i in range(len(DRUG_NAMES))}
log.info(f"[RXNORM] resolved {len(drug2rx)} drugs -> rxcui (ingredient index {len(RX_CUI)})")

# ── 3. MED-RT may_treat KB — fetch ONCE per unique rxcui (retry+raise), FREEZE to a dataset ──
uniq_rx = sorted({rc for rc, _ in drug2rx.values()})
log.info(f"[MAYTREAT] fetching MED-RT may_treat for {len(uniq_rx)} unique rxcui via RxNav ({RXNAV_WORKERS} workers)")
t0 = time.time(); RX2DIS = {}
with ThreadPoolExecutor(max_workers=RXNAV_WORKERS) as ex:
    for rc, dis in ex.map(fetch_may_treat, uniq_rx):          # any RuntimeError here propagates -> job fails loudly
        RX2DIS[rc] = dis
n_with = sum(1 for v in RX2DIS.values() if v)
log.info(f"[MAYTREAT] {n_with}/{len(uniq_rx)} rxcui have >=1 may_treat indication ({time.time()-t0:.0f}s)")
# freeze cache dataset (reproducibility)
api = HfApi()
cache_rows = [{"rxcui": rc, "may_treat_diseases": RX2DIS[rc]} for rc in uniq_rx]
Dataset.from_list(cache_rows).push_to_hub(MAYTREAT_REPO, private=True)
log.info(f"[MAYTREAT] frozen -> https://huggingface.co/datasets/{MAYTREAT_REPO}")
# embed every unique may_treat disease string once
all_dis = sorted({d for v in RX2DIS.values() for d in v})
DISVEC = {d: v for d, v in zip(all_dis, embed([d.lower() for d in all_dis]))} if all_dis else {}
def dx_disease(dx_name, rc):
    dis = RX2DIS.get(rc, [])
    if not dis or dx_name not in NVEC:
        return (None, None)
    M = torch.stack([DISVEC[d] for d in dis]); s = M @ NVEC[dx_name]
    j = int(s.argmax()); return (round(float(s[j]), 3), dis[j])

# ── 4. Hetionet — TYPE-CONSTRAINED nodes (by kind) + typed metaedge adjacency ──
hn = load_dataset(HET_REPO, "nodes", split="train")
he = load_dataset(HET_REPO, "edges", split="train")
def kind_index(kind):
    ids = [i for i, k in zip(hn["id"], hn["kind"]) if k == kind]
    if not ids: raise SystemExit(f"[FAIL] Hetionet has no '{kind}' nodes")
    emb = torch.stack([torch.tensor(e, device=DEV) for e, k in zip(hn["embedding"], hn["kind"]) if k == kind])
    return ids, torch.nn.functional.normalize(emb, dim=1)
COMP_ID, COMP_E = kind_index("Compound")
DIS_ID,  DIS_E  = kind_index("Disease")
SYM_ID,  SYM_E  = kind_index("Symptom")
META = defaultdict(set)
for s, m, t in zip(he["source"], he["metaedge"], he["target"]):
    if m in ("CtD", "CpD", "DpS", "DrD"):
        META[m].add((s, t))
        if m == "DrD": META[m].add((t, s))                    # resembles is symmetric
for m in ("CtD", "CpD", "DpS", "DrD"):
    if not META[m]: raise SystemExit(f"[FAIL] Hetionet metaedge {m} absent")
log.info(f"[HETIONET] kinds: {len(COMP_ID)} Compound, {len(DIS_ID)} Disease, {len(SYM_ID)} Symptom | "
         f"CtD={len(META['CtD'])} CpD={len(META['CpD'])} DpS={len(META['DpS'])} DrD={len(META['DrD'])//2}")
def nearest(name, ids, E):
    v = NVEC.get(name)
    if v is None: return (None, None)
    s = E @ v; i = int(s.argmax()); return (ids[i], round(float(s[i]), 3))

# ── 5. OMOP resolution — chunk the 2.1M concept embeddings on GPU (never hold all); drugs->Drug, dx->Condition ──
DRUG_RES_E = torch.stack([NVEC[n] for n in DRUG_NAMES]) if DRUG_NAMES else torch.empty(0, 768, device=DEV)
DX_RES_E   = torch.stack([NVEC[n] for n in DX_NAMES])   if DX_NAMES   else torch.empty(0, 768, device=DEV)
def best_init(n): return (torch.full((n,), -1.0, device=DEV), [None]*n)
d_cos, d_cid = best_init(len(DRUG_NAMES)); x_cos, x_cid = best_init(len(DX_NAMES))
def flush(cids, embs, MIM_E, bcos, bcid):
    if not cids or MIM_E.shape[0] == 0: return
    B = torch.tensor(embs, device=DEV); sims = MIM_E @ B.T; mx, ix = sims.max(dim=1)
    upd = mx > bcos; bcos[upd] = mx[upd]
    for i in torch.nonzero(upd).flatten().tolist(): bcid[i] = cids[int(ix[i])]
oc = load_dataset(OMOP_CONCEPT, split="train", streaming=True)
bd, bx, seen = ([], []), ([], []), 0
for r in oc:
    dom = r.get("domain_id")
    if dom == "Drug":      bd[0].append(r["concept_id"]); bd[1].append(r["embedding"])
    elif dom == "Condition": bx[0].append(r["concept_id"]); bx[1].append(r["embedding"])
    seen += 1
    if len(bd[0]) >= CHUNK: flush(bd[0], bd[1], DRUG_RES_E, d_cos, d_cid); bd = ([], [])
    if len(bx[0]) >= CHUNK: flush(bx[0], bx[1], DX_RES_E, x_cos, x_cid); bx = ([], [])
    if seen % 500000 == 0: log.info(f"[OMOP-CONCEPT] streamed {seen:,}")
flush(bd[0], bd[1], DRUG_RES_E, d_cos, d_cid); flush(bx[0], bx[1], DX_RES_E, x_cos, x_cid)
omop_drug = {DRUG_NAMES[i]: (d_cid[i], round(float(d_cos[i]), 3)) for i in range(len(DRUG_NAMES))}
omop_dx   = {DX_NAMES[i]:   (x_cid[i], round(float(x_cos[i]), 3)) for i in range(len(DX_NAMES))}
def omop_concept(name, ntype):
    return (omop_drug if ntype == "MEDICATION" else omop_dx).get(name, (None, None))
cond_ids = {c for c, _ in omop_dx.values() if c is not None}
log.info(f"[OMOP] resolved {len(omop_drug)} drugs, {len(omop_dx)} dx ({len(cond_ids)} unique condition concepts)")

# ── 6. OMOP CONCEPT_ANCESTOR via DuckDB, filtered to our condition concepts -> ancestors for LCA distance ──
import duckdb
con = duckdb.connect(); con.execute("CREATE TABLE dx(cid VARCHAR)")
con.executemany("INSERT INTO dx VALUES (?)", [(str(c),) for c in cond_ids])
anc_p = [hf_hub_download(OMOP_ANC, f, repo_type="dataset")
         for f in list_repo_files(OMOP_ANC, repo_type="dataset") if f.endswith(".parquet")]
rows = con.execute(f"""
  SELECT CAST(descendant_concept_id AS VARCHAR) d, CAST(ancestor_concept_id AS VARCHAR) a,
         MIN(min_levels_of_separation) lv
  FROM read_parquet({anc_p})
  WHERE CAST(descendant_concept_id AS VARCHAR) IN (SELECT cid FROM dx)
  GROUP BY 1,2""").fetchall()
ANC = defaultdict(dict)
for d, a, lv in rows: ANC[d][a] = int(lv)
log.info(f"[OMOP-ANC] ancestor map for {len(ANC)} condition concepts ({len(rows)} pairs)")
def lca_dist(ca, cb):
    if not ca or not cb or ca == cb: return None
    A, B = ANC.get(ca), ANC.get(cb)
    if not A or not B: return None
    common = set(A) & set(B)
    return min(A[c] + B[c] for c in common) if common else None

# ── 7. provenance — only if the input actually carries quotes (else loud WARNING, recorded null) ──
HAS_QUOTES = any(e.get("quote") for g in graphs for e in g["edges"])
NOTE = {}
if HAS_QUOTES:
    nd = load_dataset(NOTES_REPO, split="train")
    key = "hadm_id" if "hadm_id" in nd.column_names else "subject_id"
    tcol = next(c for c in nd.column_names if c in ("note", "text", "discharge_note", "synthetic_note"))
    NOTE = {str(k): (t or "") for k, t in zip(nd[key], nd[tcol])}
    log.info(f"[PROV] notes loaded ({len(NOTE)}) keyed by {key}")
else:
    log.warning("[THRESHOLD] input graphs carry NO quotes -> provenance is UNAVAILABLE this run "
                "(re-run Stage 1 with the provenance fix). prov_* will be null.")
RUN["coverage"]["has_quotes"] = HAS_QUOTES
_norm = lambda s: re.sub(r"\s+", " ", (s or "").lower()).strip()
def provenance(g, e):
    q = e.get("quote")
    if not (HAS_QUOTES and q): return (None, None)
    note = NOTE.get(str(g.get("hadm_id", g.get("subject_id", "")))) or ""
    nq, nn = _norm(q), _norm(note)
    if not nq or not nn: return (0, 0.0)
    if nq in nn: return (1, 1.0)
    from difflib import SequenceMatcher
    return (0, round(SequenceMatcher(None, nq, nn[:8000]).quick_ratio(), 3))

# ── 8. assemble UNIFORM label schema on every edge (null where a signal does not apply) ──
BLANK = {"model": None, "omop_src": None, "omop_src_cos": None, "omop_dst": None, "omop_dst_cos": None,
         "omop_lca_dist": None, "drug_link_cos": None, "rxcui": None, "dx_disease_cos": None,
         "matched_disease": None, "het_treats_ctd": None, "het_treats_cpd": None, "het_drug_cos": None,
         "het_dx_cos": None, "het_resembles_drd": None, "het_presents_dps": None,
         "prov_in_note": None, "prov_ratio": None}
RUN["signals"] = [k for k in BLANK if k != "model"]
cov = defaultdict(int); t0 = time.time()
for g in graphs:
    nm  = {nd["id"]: nd["name"] for nd in g["nodes"]}
    typ = {nd["id"]: nd["type"] for nd in g["nodes"]}
    for e in g["edges"]:
        lab = dict(BLANK); lab["model"] = e.get("confidence")
        rel = e["relation"]
        if rel in LLM_RELATIONS:
            s, d = nm.get(e["source"], ""), nm.get(e["target"], "")
            st, dt = typ.get(e["source"]), typ.get(e["target"])
            # universal OMOP resolution (where the endpoint is a drug or dx)
            if st in ("MEDICATION", "DIAGNOSIS"): lab["omop_src"], lab["omop_src_cos"] = omop_concept(s, st)
            if dt in ("MEDICATION", "DIAGNOSIS"): lab["omop_dst"], lab["omop_dst_cos"] = omop_concept(d, dt)
            if rel == "MANAGED_FOR" and st == "MEDICATION" and dt == "DIAGNOSIS":
                rc, dlc = drug2rx.get(s, (None, None)); lab["rxcui"], lab["drug_link_cos"] = rc, dlc
                if rc is not None: lab["dx_disease_cos"], lab["matched_disease"] = dx_disease(d, rc)
                c, ccos = nearest(s, COMP_ID, COMP_E); dd, dcos = nearest(d, DIS_ID, DIS_E)
                lab["het_drug_cos"], lab["het_dx_cos"] = ccos, dcos
                lab["het_treats_ctd"] = 1 if (c, dd) in META["CtD"] else 0
                lab["het_treats_cpd"] = 1 if (c, dd) in META["CpD"] else 0
            elif rel == "COMPLICATED_BY" and st == "DIAGNOSIS" and dt == "DIAGNOSIS":
                d1, c1 = nearest(s, DIS_ID, DIS_E); d2, c2 = nearest(d, DIS_ID, DIS_E)
                lab["het_dx_cos"] = min(x for x in (c1, c2) if x is not None) if (c1 and c2) else None
                lab["het_resembles_drd"] = 1 if (d1, d2) in META["DrD"] else 0
                lab["omop_lca_dist"] = lca_dist(lab["omop_src"], lab["omop_dst"])
            elif rel == "INDICATES" and st == "DIAGNOSIS" and dt == "DIAGNOSIS":
                da, _ = nearest(s, DIS_ID, DIS_E); sb, _ = nearest(d, SYM_ID, SYM_E)
                db, _ = nearest(d, DIS_ID, DIS_E); sa, _ = nearest(s, SYM_ID, SYM_E)
                lab["het_presents_dps"] = 1 if ((da, sb) in META["DpS"] or (db, sa) in META["DpS"]) else 0
                lab["omop_lca_dist"] = lca_dist(lab["omop_src"], lab["omop_dst"])
            lab["prov_in_note"], lab["prov_ratio"] = provenance(g, e)
            for k in ("dx_disease_cos", "het_treats_ctd", "het_resembles_drd", "het_presents_dps", "omop_lca_dist"):
                if lab[k] is not None: cov[k] += 1
        e["labels"] = lab
        cov[f"rel::{rel}"] += 1
log.info(f"[SCORE] {n_edges} edges scored in {time.time()-t0:.0f}s")
RUN["coverage"].update({k: cov[k] for k in cov})
for k in ("dx_disease_cos", "het_treats_ctd", "het_treats_cpd", "het_resembles_drd", "het_presents_dps", "omop_lca_dist"):
    log.info(f"[COVERAGE] {k}: {cov.get(k,0)}")

# ── 9. push scored graphs + reproducibility manifest ──
out = "fawkes_graphs_stage2_v8_rows4000-5000_260613.jsonl"
with open(out, "w") as f:
    for g in graphs: f.write(json.dumps(g, ensure_ascii=False) + "\n")
api.create_repo(OUT_REPO, repo_type="dataset", private=True, exist_ok=True)
api.upload_file(path_or_fileobj=out, path_in_repo=out, repo_id=OUT_REPO, repo_type="dataset")
with open("run_manifest_260613.json", "w") as f: json.dump(RUN, f, indent=2)
api.upload_file(path_or_fileobj="run_manifest_260613.json", path_in_repo="run_manifest_260613.json",
                repo_id=OUT_REPO, repo_type="dataset")
log.info(f"[DONE] {len(graphs)} graphs, {n_edges} edges -> https://huggingface.co/datasets/{OUT_REPO}")
