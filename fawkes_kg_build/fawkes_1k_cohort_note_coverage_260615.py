# /// script
# requires-python = ">=3.10"
# dependencies = ["huggingface_hub>=0.25"]
# ///
# fawkes_1k_cohort_note_coverage_260615.py
# Count admissions WITH vs WITHOUT a real discharge note in the 1K-patient cohort (one row per admission),
# plus per-patient note coverage. Read-only, fail-loud, parameterized. Run locally or as a tiny CPU job.
import os, json, logging
from collections import defaultdict
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("notecov")

def env(k, d=None, required=False):
    v = os.environ.get(k, d)
    if required and not v: raise RuntimeError(f"[FAILURE] env {k} unset; no fallback.")
    return v
DATA_REPO = env("DATA_REPO", "on1onmangoes/fawkes-1k-patients-alladm-260615")
DATA_FILE = env("DATA_FILE", "fawkes_1k_patients_alladm_260615.jsonl")

def main():
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(DATA_REPO, DATA_FILE, repo_type="dataset")
    tot = wn = 0
    pt, pn = defaultdict(int), defaultdict(int)            # admissions / noted-admissions per patient
    for l in open(p):
        l = l.strip()
        if not l: continue
        r = json.loads(l); tot += 1; s = str(r.get("subject_id")); pt[s] += 1
        if r.get("note") and str(r.get("note")).strip(): wn += 1; pn[s] += 1
    if not tot: raise RuntimeError(f"[FAILURE] 0 records in {DATA_REPO}/{DATA_FILE}; no fallback.")
    log.info(f"[NOTES] {DATA_REPO}/{DATA_FILE}")
    log.info(f"  total admissions        : {tot:,}")
    log.info(f"  WITH discharge note     : {wn:,} ({100*wn/tot:.1f}%)")
    log.info(f"  WITHOUT discharge note  : {tot-wn:,} ({100*(tot-wn)/tot:.1f}%)")
    log.info(f"  patients                : {len(pt):,}")
    log.info(f"  patients 0 noted adm    : {sum(1 for s in pt if pn[s]==0):,}")
    log.info(f"  patients ALL adm noted  : {sum(1 for s in pt if pn[s]==pt[s]):,}")

if __name__ == "__main__":
    main()
