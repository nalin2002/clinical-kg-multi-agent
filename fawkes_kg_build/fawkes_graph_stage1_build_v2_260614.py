# /// script
# requires-python = ">=3.10"
# dependencies = ["datasets>=2.19", "huggingface_hub>=0.25", "hf_transfer"]
# ///
# fawkes_graph_stage1_build_v2_260614.py
# STAGE 1 (SHARDED) v2 — IDENTICAL generation to fawkes_graph_stage1_build_sharded_260613.py (proven v7
# prompts + robust extractor + deterministic backbone + quotes kept). ONLY change vs 260613: the shard
# slice AND the per-shard output filename are computed INSIDE the script from a single SHARD index, so
# the shell never builds a filename (kills the $S_260613 -> empty-var bug that collided all 10 shards
# into one file). Each job: SHARD in 0..N_SHARDS-1; slice = ALL[SHARD*chunk:(SHARD+1)*chunk];
# filename = data/stage1_{RANGE}_shard{SHARD:02d}_260614.jsonl. Run N jobs varying ONLY -e SHARD=$S.
import os, sys, re, json, time, argparse, logging
import urllib.request, urllib.error
import concurrent.futures as cf
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
sys.stdout.reconfigure(line_buffering=True)
from collections import defaultdict
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("stage1")

def cfg():
    ap = argparse.ArgumentParser()
    ap.add_argument("--notes-repo", default=os.environ.get("NOTES_REPO", "on1onmangoes/fawkes-mimic-notes-1k-v3-rows3000-4000-260612"))
    ap.add_argument("--notes-file", default=os.environ.get("NOTES_FILE", ""))
    ap.add_argument("--shard",      type=int, default=int(os.environ.get("SHARD", 0)))       # this shard index 0..N-1
    ap.add_argument("--n-shards",   type=int, default=int(os.environ.get("N_SHARDS", 10)))   # total shards
    ap.add_argument("--range",      default=os.environ.get("RANGE", "rows0-1000"))           # label for output filename
    ap.add_argument("--model",      default=os.environ.get("EDGE_MODEL", "google/medgemma-27b-text-it"))
    ap.add_argument("--endpoint",   default=os.environ.get("ENDPOINT_URL", "https://kyn9esljq4te1xgz.us-east-2.aws.endpoints.huggingface.cloud"))
    ap.add_argument("--workers",    type=int, default=int(os.environ.get("WORKERS", 16)))
    ap.add_argument("--maxtok",     type=int, default=int(os.environ.get("MAX_NEW_TOKENS", 15000)))
    ap.add_argument("--output",     default=os.environ.get("OUTPUT_REPO", "on1onmangoes/fawkes-mimic-graphs-stage1-rows3000-4000-260613"))
    return ap.parse_args()
A = cfg()

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
        "Use standard clinical knowledge to link a listed MEDICATION to the listed DIAGNOSIS it treats (e.g. metoprolol -> hypertension; vancomycin -> a listed infection). Supportive fluids/flushes/vaccines have no MANAGED_FOR target — skip them. CRITICAL NO-TARGET RULE: if a medication has NO diagnosis in the node list that it plausibly treats, emit NOTHING for it — NEVER force a link to an unrelated diagnosis (e.g. do not link an SSRI antidepressant to asthma when there is no mood-disorder node). CRITICAL MED-LIST RULE: when several medications appear together in ONE note sentence (a medication list), that shared sentence is CO-LOCATION, NOT shared purpose — do NOT link them all to the same (dominant) diagnosis. Judge EACH medication on its OWN pharmacology and match it to the SPECIFIC diagnosis it treats."),
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
    except json.JSONDecodeError: pass
    edges, i, n = [], 0, len(s)
    while i < n:
        if s[i] == "{":
            depth = 0; j = i; instr = esc = False; closed = False
            while j < n:
                c = s[j]
                if esc: esc = False
                elif c == "\\": esc = True
                elif c == '"': instr = not instr
                elif not instr:
                    if c == "{": depth += 1
                    elif c == "}":
                        depth -= 1
                        if depth == 0: closed = True; break
                j += 1
            if not closed: break
            try:
                d = json.loads(s[i:j+1])
                if isinstance(d, dict) and d.get("src") and d.get("dst"): edges.append(d)
            except json.JSONDecodeError: pass
            i = j + 1
        else:
            i += 1
    return {"matched": edges, "other": []} if edges else {}

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
    return SPEC_TMPL.format(persona=persona, REL=rel, definition=definition, patterns=patterns, question=question, extra=extra, nodes=nl, note=note)
def resolve(nm, nodes):
    q = norm_name(nm)
    for n in nodes:
        if norm_name(n["name"]) == q: return n["id"]
    for n in nodes:
        if q and (q in norm_name(n["name"]) or norm_name(n["name"]) in q): return n["id"]
    return None

_HEAD = {"Authorization": "Bearer " + os.environ["HF_TOKEN"], "Content-Type": "application/json"}
def resolve_model():
    try:
        req = urllib.request.Request(A.endpoint.rstrip("/") + "/v1/models", headers=_HEAD)
        with urllib.request.urlopen(req, timeout=30) as r:
            served = json.loads(r.read())["data"][0]["id"]
        if served and served != A.model:
            log.info(f"[MODEL] {A.model!r} -> served {served!r}"); A.model = served
    except Exception as e:
        log.warning(f"[MODEL] /v1/models probe failed ({type(e).__name__}); using {A.model!r}")
def endpoint_call(prompt):
    body = json.dumps({"model": A.model, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": A.maxtok, "temperature": 0}).encode()
    url = A.endpoint.rstrip("/") + "/v1/chat/completions"
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
    from huggingface_hub import hf_hub_download, HfApi, list_repo_files
    api = HfApi()
    nf = A.notes_file
    if not nf:
        cands = [f for f in list_repo_files(A.notes_repo, repo_type="dataset") if f.endswith(".jsonl")]
        if not cands: raise SystemExit(f"[FAIL] no .jsonl in {A.notes_repo}")
        nf = cands[0]; log.info(f"[NOTES] auto-found {nf}")
    p = hf_hub_download(A.notes_repo, nf, repo_type="dataset")
    ALL = [json.loads(l) for l in open(p)]
    chunk = -(-len(ALL) // A.n_shards)                    # ceil -> contiguous per-shard size; covers ALL rows
    start = A.shard * chunk
    R = ALL[start : start + chunk]                        # this shard's contiguous slice, computed in-script
    if not R: raise SystemExit(f"[FAIL] empty shard {A.shard}/{A.n_shards}: start={start} chunk={chunk} (file has {len(ALL)} rows)")
    log.info(f"[SHARD {A.shard}/{A.n_shards}] rows [{start}:{start+len(R)}] of {len(ALL)} from {A.notes_repo}")
    recmeta, tasks = [], []
    for ri, r in enumerate(R):
        nodes, idx = build_nodes(r); det = deterministic_edges(r, idx)
        recmeta.append({"r": r, "nodes": nodes, "det": det})
        for rel in SPECIALISTS: tasks.append((ri, rel))
    log.info(f"[SETUP] {len(R)} transcripts x {len(SPECIALISTS)} specialists = {len(tasks)} calls | workers={A.workers}")
    resolve_model()
    fn = f"data/stage1_{A.range}_shard{A.shard:02d}_260614.jsonl"   # filename owned by the script — no shell $S
    local = os.path.basename(fn)
    CKPT = int(os.environ.get("CHECKPOINT", 20))      # push every N completed patients (interval save)
    api.create_repo(A.output, repo_type="dataset", private=True, exist_ok=True)

    def assemble(ri, parsed_by_rel):
        meta = recmeta[ri]; nodes, det = meta["nodes"], meta["det"]; llm_e, seen = [], set()
        for rel, parsed in parsed_by_rel.items():
            for m in (parsed.get("matched") or []):
                s, d = resolve(m.get("src"), nodes), resolve(m.get("dst"), nodes)
                if s and d and s != d:
                    cv = m.get("confidence"); conf = round(float(cv), 2) if isinstance(cv, (int, float)) and 0 < cv <= 1 else 0.5
                    llm_e.append({"source": s, "target": d, "relation": rel, "confidence": conf, "evidence": "llm",
                                  "sentence": m.get("sentence", ""), "quote": (m.get("quote") or "")[:300]})
        edges = []
        for e in det + llm_e:
            k = (e["source"], e["relation"], e["target"])
            if k in seen: continue
            seen.add(k); edges.append(e)
        return {"subject_id": meta["r"]["subject_id"], "hadm_id": meta["r"].get("hadm_id"),
                "nodes": nodes, "edges": edges, "n_nodes": len(nodes), "n_edges": len(edges)}

    def push(gs):
        with open(local, "w") as f:
            for g in gs: f.write(json.dumps(g, ensure_ascii=False) + "\n")
        api.upload_file(path_or_fileobj=local, path_in_repo=fn, repo_id=A.output, repo_type="dataset")

    # RESUME: load any graphs already pushed for this shard, skip those patients
    graphs, done_hadm = [], set()
    try:
        for l in open(hf_hub_download(A.output, fn, repo_type="dataset")):
            g = json.loads(l); graphs.append(g); done_hadm.add(g.get("hadm_id"))
        log.info(f"[RESUME] {len(graphs)} graphs already pushed for {fn} — skipping them")
    except Exception:
        pass
    todo = [ri for ri, meta in enumerate(recmeta) if meta["r"].get("hadm_id") not in done_hadm]
    if not todo:
        log.info(f"[DONE] shard already complete: {len(graphs)} graphs"); return

    t0 = time.time(); parsed = defaultdict(dict)
    with cf.ThreadPoolExecutor(max_workers=A.workers) as pool:
        futs = {pool.submit(endpoint_call, make_prompt(rel, recmeta[ri]["nodes"], recmeta[ri]["r"]["note"])): (ri, rel)
                for ri in todo for rel in SPECIALISTS}
        total = len(futs); completed = since = 0
        for fut in cf.as_completed(futs):
            ri, rel = futs[fut]
            parsed[ri][rel] = extract_json(fut.result()); completed += 1
            if len(parsed[ri]) == len(SPECIALISTS):          # patient fully generated -> assemble
                graphs.append(assemble(ri, parsed.pop(ri))); since += 1
                if since >= CKPT:
                    push(graphs); since = 0
                    log.info(f"[CKPT] pushed {len(graphs)} graphs ({completed}/{total} calls)")
            if completed % 20 == 0: log.info(f"  [GEN] {completed}/{total}")
    push(graphs)
    te = sum(g["n_edges"] for g in graphs)
    log.info(f"[DONE] shard{A.shard:02d}: {len(graphs)} graphs | {te} edges | {time.time()-t0:.0f}s -> {A.output}/{fn}")

if __name__ == "__main__":
    main()
