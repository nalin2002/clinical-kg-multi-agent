# /// script
# requires-python = ">=3.10"
# dependencies = ["huggingface_hub>=0.25", "hf_transfer"]
# ///
# fawkes_build_1k_patients_graphs_v2_260615.py
# ONE job, two stages, on the MEGA flat dataset (all MIMIC admissions + REAL notes):
#   STAGE A — select 1000 PATIENTS (eligible = >=1 admission meeting the complete criteria), keep ALL their
#             admissions, push the 1K cohort dataset.
#   STAGE B — build the per-admission GRAPH for each (deterministic backbone + 4 specialist LLM agents reading
#             the REAL discharge note via the medgemma-27b ENDPOINT), push the graphs dataset.
#   v2 (260615): STAGE B builds graphs ONLY for admissions that HAVE a discharge note (NOTED_ONLY, default 1).
#             Note-less admissions can't produce the 4 inferred LLM edges (the agents read the note) — they'd be
#             backbone-only and useless for training — so they are SKIPPED. Of the 2,969 cohort admissions, 1,912
#             (64.4%) are noted. Stage A cohort upload is unchanged (still all admissions).
# GRAPHS USE THE ENDPOINT (medgemma-27b-text-it on the H200 inference endpoint) — this is a CPU client job with
# WORKERS threads; the endpoint MUST be running. No local GPU. Same proven build logic as the Stage-1 builder.
import os, sys, re, json, time, random, logging, urllib.request, urllib.error
import concurrent.futures as cf
from collections import defaultdict
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
sys.stdout.reconfigure(line_buffering=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("p1k")

def env(k, d=None, required=False):
    v = os.environ.get(k, d)
    if required and not v: raise RuntimeError(f"[FAILURE] env {k} unset; no fallback.")
    return v
MEGA_REPO    = env("MEGA_REPO", "on1onmangoes/fawkes-mimic-megaflat-admission-260615")
MEGA_FILE    = env("MEGA_FILE", "fawkes_mimic_megaflat_admission_260615.jsonl")
PATIENTS_OUT = env("PATIENTS_OUT", "on1onmangoes/fawkes-1k-patients-alladm-260615")
GRAPHS_OUT   = env("GRAPHS_OUT", "on1onmangoes/fawkes-1k-patients-graphs-260615")
N_PATIENTS   = int(env("N_PATIENTS", 1000)); SEED = int(env("SEED", 42))
# complete criteria (same as complete5k) — patient is ELIGIBLE if any admission meets these; then keep ALL admissions
MIN_DX, MAX_DX = int(env("MIN_DX", 3)), int(env("MAX_DX", 40))
MIN_MED, MAX_MED = int(env("MIN_MED", 3)), int(env("MAX_MED", 60))
MIN_SVC = int(env("MIN_SVC", 1))
ENDPOINT = env("ENDPOINT_URL", "https://kyn9esljq4te1xgz.us-east-2.aws.endpoints.huggingface.cloud")
EDGE_MODEL = env("EDGE_MODEL", "google/medgemma-27b-text-it")
WORKERS = int(env("WORKERS", 16)); MAXTOK = int(env("MAX_NEW_TOKENS", 15000)); CKPT = int(env("CHECKPOINT", 20))
BUILD_GRAPHS = str(env("BUILD_GRAPHS", "1")).lower() in ("1", "true", "yes")
NOTED_ONLY = str(env("NOTED_ONLY", "1")).lower() in ("1", "true", "yes")   # (v2) Stage B: build graphs ONLY for admissions with a discharge note (skip note-less)

# ── STAGE B build logic (verbatim from fawkes_graph_stage1_build_v2) ──────────
SPEC_TMPL = """<role>
You are an EXPERT {persona} extracting {REL} edges from ONE hospital discharge note. You receive the full NODE list (the patient's structured items) and the numbered note. Emit edges in two buckets.
</role>

<nodes>
{nodes}
</nodes>

<note>
{note}
</note>

<task>
A {REL} edge means: {definition}
Valid src -> dst type patterns:
{patterns}
Only emit pairs matching one of these type patterns. Arrow direction is src -> dst. Never reverse.
{extra}
Err toward INCLUSION on MATCHED (both endpoints are <nodes> entries); precision is handled downstream.

Walk every plausible (src, dst) pair drawn from <nodes>:
  Step 1 - CANDIDATE WALK. For each pair decide ACCEPT / REJECT / UNCERTAIN for "{question}". Cite the note sentence number.
  Step 2 - RESOLVE every UNCERTAIN to ACCEPT or REJECT.
  Step 3 - VERIFY LABELS. MATCHED endpoints are character-for-character copies of a <nodes> name.
  Step 4 - ANTI-COLLAPSE. One src may link to several dst and vice-versa; include each.

Two buckets:
  MATCHED - both src and dst are <nodes> names (use the EXACT name string).
  OTHER   - at least one endpoint is named/implied in the note but NOT a <nodes> entry.

NEGATION BLOCKS EMISSION: if the note denies/excludes the link, do not emit it. DEDUPLICATE within each bucket.
</task>

<output>
Emit ONE JSON object, nothing else:
{{"rationale": "<=60 words", "matched": [{{"src":"...","dst":"...","sentence":"[n]","quote":"...","confidence":0.0}}], "other": [{{"src":"...","dst":"...","sentence":"[n]","quote":"...","confidence":0.0}}]}}
Set "confidence" to your ACTUAL certainty THIS specific link is correct — a graded value 0.30-0.99, and VARY it.
Any bucket may be []. Both keys must be present.
</output>"""
SPECIALISTS = {
    "MANAGED_FOR": ("clinical pharmacist",
        "a MEDICATION is given to treat / manage / control a DIAGNOSIS (active problem or chronic condition). The most valuable edge — a medication paired with the problem it addresses.",
        "  - MEDICATION -> DIAGNOSIS", "a medication is taken for a diagnosis",
        "Use standard clinical knowledge to link a listed MEDICATION to the listed DIAGNOSIS it treats (e.g. metoprolol -> hypertension; vancomycin -> a listed infection). Supportive fluids/flushes/vaccines have no MANAGED_FOR target — skip them. CRITICAL NO-TARGET RULE: if a medication has NO diagnosis in the node list that it plausibly treats, emit NOTHING for it — NEVER force a link to an unrelated diagnosis. CRITICAL MED-LIST RULE: when several medications appear together in ONE note sentence (a medication list), that shared sentence is CO-LOCATION, NOT shared purpose — judge EACH medication on its OWN pharmacology and match it to the SPECIFIC diagnosis it treats."),
    "CONFIRMS": ("clinical internist", "a PROCEDURE established / was diagnostic-of a DIAGNOSIS.",
        "  - PROCEDURE -> DIAGNOSIS", "a procedure confirmed a diagnosis",
        "Only emit when the note ties the procedure to the diagnosis it established."),
    "COMPLICATED_BY": ("clinical internist", "the acute course / primary DIAGNOSIS was COMPLICATED BY another DIAGNOSIS that arose this stay. NOT every comorbidity.",
        "  - DIAGNOSIS -> DIAGNOSIS", "the first was complicated by the second this admission",
        "Do NOT link chronic comorbidities as complications."),
    "INDICATES": ("clinical internist", "a presenting SYMPTOM/finding points to a DIAGNOSIS worked up this stay.",
        "  - DIAGNOSIS(PRESENTING) -> DIAGNOSIS", "the presenting feature points to the diagnosis",
        "STRICT: src MUST be a (PRESENTING) node. Never link two established diagnoses; never emit a reverse pair. No (PRESENTING) node -> emit nothing."),
}
def norm_name(s): return re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9 ]', ' ', (s or '').lower())).strip()
_JSON_DEC = json.JSONDecoder()
def extract_json(txt):
    s = re.sub(r"<think>.*?</think>", "", txt or "", flags=re.DOTALL).strip()
    if not s: return {}
    try: return json.loads(s)
    except json.JSONDecodeError: pass
    if s.startswith("```"):
        m = re.match(r"```(?:json)?\s*(.*?)\s*```", s, re.DOTALL)
        if m:
            try: return json.loads(m.group(1).strip())
            except json.JSONDecodeError: s = m.group(1).strip()
    best, best_len = None, -1
    for i, ch in enumerate(s):
        if ch != "{": continue
        try: obj, end = _JSON_DEC.raw_decode(s[i:])
        except json.JSONDecodeError: continue
        if isinstance(obj, dict) and end > best_len: best, best_len = obj, end
    if best is not None: return best
    try: return json.loads(re.sub(r",(\s*[}\]])", r"\1", s))
    except json.JSONDecodeError: return {}
def build_nodes(r):
    nodes, idx = [], {}
    def add(t, name, presenting=False):
        name = (name or "").strip()
        if not name: return
        k = (t, name.lower())
        if k in idx:
            if presenting: idx[k]["presenting"] = True
            return
        n = {"id": f"N_{len(nodes):03d}", "type": t, "name": name, "normalized_name": name.lower()}
        if presenting: n["presenting"] = True
        nodes.append(n); idx[k] = n
    add("PATIENT", f"patient_{r['subject_id']}")
    for sym in r.get("symptoms", []): add("DIAGNOSIS", sym, presenting=True)
    for dx in r.get("diagnoses", []): add("DIAGNOSIS", dx)
    for mh in r.get("medical_history", []): add("DIAGNOSIS", mh)
    for m in r.get("medications", []): add("MEDICATION", m["name"] if isinstance(m, dict) else m)
    for p in r.get("procedures", []): add("PROCEDURE", p)
    for mb in r.get("microbiology", []): add("MICROBIOLOGY", mb.get("organism") if isinstance(mb, dict) else mb)
    for sv in r.get("services", []): add("SERVICE", sv)
    return nodes, idx
def nid_of(idx, t, name): return (idx.get((t, (name or "").strip().lower())) or {}).get("id")
def deterministic_edges(r, idx):
    E, pid = [], nid_of(idx, "PATIENT", f"patient_{r['subject_id']}")
    def e(s, rel, d):
        if s and d and s != d: E.append({"source": s, "target": d, "relation": rel, "confidence": 1.0, "evidence": "structured"})
    for dx in r.get("symptoms", []) + r.get("diagnoses", []) + r.get("medical_history", []): e(pid, "HAS_DIAGNOSIS", nid_of(idx, "DIAGNOSIS", dx))
    for m in r.get("medications", []): e(pid, "TAKES_MEDICATION", nid_of(idx, "MEDICATION", m["name"] if isinstance(m, dict) else m))
    for p in r.get("procedures", []): e(pid, "UNDERWENT_PROCEDURE", nid_of(idx, "PROCEDURE", p))
    for sv in r.get("services", []): e(pid, "MANAGED_BY_SERVICE", nid_of(idx, "SERVICE", sv))
    for mb in r.get("microbiology", []):
        org = nid_of(idx, "MICROBIOLOGY", mb.get("organism") if isinstance(mb, dict) else mb)
        e(pid, "HAD_MICROBIOLOGY", org)
        for s in (mb.get("susceptibilities", []) if isinstance(mb, dict) else []):
            if s.get("interpretation") == "S":
                ab = nid_of(idx, "MEDICATION", s.get("antibiotic", ""))
                if ab: e(ab, "TARGETS_ORGANISM", org)
    return E
def make_prompt(rel, nodes, note):
    persona, definition, patterns, question, extra = SPECIALISTS[rel]
    nl = "\n".join(f'  {i}. "{n["name"]}" ({n["type"]}{", PRESENTING" if n.get("presenting") else ""})' for i, n in enumerate(nodes, 1))
    return SPEC_TMPL.format(persona=persona, REL=rel, definition=definition, patterns=patterns, question=question, extra=extra, nodes=nl, note=note or "")
def resolve(nm, nodes):
    q = norm_name(nm)
    for n in nodes:
        if norm_name(n["name"]) == q: return n["id"]
    for n in nodes:
        if q and (q in norm_name(n["name"]) or norm_name(n["name"]) in q): return n["id"]
    return None
_HEAD = {"Authorization": "Bearer " + os.environ["HF_TOKEN"], "Content-Type": "application/json"}
def resolve_model():
    global EDGE_MODEL
    try:
        req = urllib.request.Request(ENDPOINT.rstrip("/") + "/v1/models", headers=_HEAD)
        with urllib.request.urlopen(req, timeout=30) as r:
            served = json.loads(r.read())["data"][0]["id"]
        if served and served != EDGE_MODEL: log.info(f"[MODEL] {EDGE_MODEL!r} -> served {served!r}"); EDGE_MODEL = served
    except Exception as e:
        log.warning(f"[MODEL] /v1/models probe failed ({type(e).__name__}); using {EDGE_MODEL!r}")
def endpoint_call(prompt):
    body = json.dumps({"model": EDGE_MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": MAXTOK, "temperature": 0}).encode()
    url = ENDPOINT.rstrip("/") + "/v1/chat/completions"
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, data=body, headers=_HEAD)
            with urllib.request.urlopen(req, timeout=600) as r:
                return json.loads(r.read())["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            if e.code in (408, 429, 503) or 500 <= e.code < 600: time.sleep(2 * (2 ** attempt)); continue
            return ""
        except Exception:
            time.sleep(2 * (2 ** attempt))
    return ""

def main():
    from huggingface_hub import hf_hub_download, HfApi
    api = HfApi()
    # ── STAGE A: select 1000 patients (all admissions) from the mega dataset ──
    mega = hf_hub_download(MEGA_REPO, MEGA_FILE, repo_type="dataset")
    elig = set(); n_adm = 0
    for l in open(mega):
        l = l.strip()
        if not l: continue
        r = json.loads(l); n_adm += 1
        if MIN_DX <= r.get("n_diagnoses", 0) <= MAX_DX and MIN_MED <= r.get("n_medications", 0) <= MAX_MED and r.get("n_services", 0) >= MIN_SVC:
            elig.add(str(r["subject_id"]))
    pats = sorted(elig); random.Random(SEED).shuffle(pats); keep = set(pats[:N_PATIENTS])
    log.info(f"[SELECT] {n_adm:,} admissions -> {len(elig):,} eligible patients -> seed-{SEED} sample {len(keep)} patients (ALL their admissions)")
    recs = []
    for l in open(mega):
        l = l.strip()
        if not l: continue
        r = json.loads(l)
        if str(r["subject_id"]) in keep: recs.append(r)
    npat = len({r["subject_id"] for r in recs})
    log.info(f"[COHORT] {len(recs):,} admissions for {npat} patients (mean {len(recs)/max(npat,1):.1f} adm/patient)")
    local = "fawkes_1k_patients_alladm_260615.jsonl"
    with open(local, "w") as f:
        for r in recs: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    api.create_repo(PATIENTS_OUT, repo_type="dataset", private=True, exist_ok=True)
    api.upload_file(path_or_fileobj=local, path_in_repo=local, repo_id=PATIENTS_OUT, repo_type="dataset")
    log.info(f"[DONE A] 1K cohort -> https://huggingface.co/datasets/{PATIENTS_OUT} ({len(recs):,} admissions)")
    if not BUILD_GRAPHS:
        log.info("[SKIP B] BUILD_GRAPHS=0 — stopping after the cohort"); return

    # ── STAGE B: build the per-admission graph (endpoint) ──
    def has_note(r): return bool(r.get("note") and str(r.get("note")).strip())
    build_recs = [r for r in recs if has_note(r)] if NOTED_ONLY else recs   # (v2) skip note-less admissions (no inferred edges possible)
    if NOTED_ONLY:
        log.info(f"[NOTE-FILTER] NOTED_ONLY=1 — {len(build_recs):,}/{len(recs):,} admissions have a discharge note; {len(recs)-len(build_recs):,} note-less SKIPPED (backbone-only, no inferred edges)")
    recmeta = []
    for r in build_recs:
        nodes, idx = build_nodes(r); recmeta.append({"r": r, "nodes": nodes, "det": deterministic_edges(r, idx)})
    log.info(f"[SETUP B] {len(recmeta)} admissions x {len(SPECIALISTS)} agents = {len(recmeta)*len(SPECIALISTS):,} endpoint calls | workers={WORKERS} (endpoint must be running)")
    resolve_model()
    gfile = "fawkes_1k_patients_graphs_260615.jsonl"
    api.create_repo(GRAPHS_OUT, repo_type="dataset", private=True, exist_ok=True)
    def assemble(ri, parsed_by_rel):
        meta = recmeta[ri]; nodes = meta["nodes"]; llm_e, seen = [], set()
        for rel, parsed in parsed_by_rel.items():
            for m in (parsed.get("matched") or []):
                s, d = resolve(m.get("src"), nodes), resolve(m.get("dst"), nodes)
                if s and d and s != d:
                    cv = m.get("confidence"); conf = round(float(cv), 2) if isinstance(cv, (int, float)) and 0 < cv <= 1 else 0.5
                    llm_e.append({"source": s, "target": d, "relation": rel, "confidence": conf, "evidence": "llm",
                                  "sentence": m.get("sentence", ""), "quote": (m.get("quote") or "")[:300]})
        edges = []
        for e in meta["det"] + llm_e:
            k = (e["source"], e["relation"], e["target"])
            if k in seen: continue
            seen.add(k); edges.append(e)
        return {"subject_id": meta["r"]["subject_id"], "hadm_id": meta["r"].get("hadm_id"),
                "nodes": nodes, "edges": edges, "n_nodes": len(nodes), "n_edges": len(edges)}
    def push(gs):
        with open(gfile, "w") as f:
            for g in gs: f.write(json.dumps(g, ensure_ascii=False) + "\n")
        api.upload_file(path_or_fileobj=gfile, path_in_repo=gfile, repo_id=GRAPHS_OUT, repo_type="dataset")
    graphs, done = [], set()
    try:
        for l in open(hf_hub_download(GRAPHS_OUT, gfile, repo_type="dataset")):
            g = json.loads(l); graphs.append(g); done.add(g.get("hadm_id"))
        log.info(f"[RESUME] {len(graphs)} graphs already built — skipping")
    except Exception: pass
    todo = [ri for ri, m in enumerate(recmeta) if m["r"].get("hadm_id") not in done]
    if not todo: log.info(f"[DONE B] already complete: {len(graphs)} graphs"); return
    t0 = time.time(); parsed = defaultdict(dict)
    with cf.ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futs = {pool.submit(endpoint_call, make_prompt(rel, recmeta[ri]["nodes"], recmeta[ri]["r"].get("note"))): (ri, rel) for ri in todo for rel in SPECIALISTS}
        total = len(futs); completed = since = 0
        for fut in cf.as_completed(futs):
            ri, rel = futs[fut]; parsed[ri][rel] = extract_json(fut.result()); completed += 1
            if len(parsed[ri]) == len(SPECIALISTS):
                graphs.append(assemble(ri, parsed.pop(ri))); since += 1
                if since >= CKPT: push(graphs); since = 0; log.info(f"[CKPT] {len(graphs)} graphs ({completed}/{total} calls)")
            if completed % 50 == 0: log.info(f"  [GEN] {completed}/{total}")
    push(graphs)
    te = sum(g["n_edges"] for g in graphs)
    log.info(f"[DONE B] {len(graphs)} graphs | {te:,} edges | {time.time()-t0:.0f}s -> https://huggingface.co/datasets/{GRAPHS_OUT}")

if __name__ == "__main__":
    main()
