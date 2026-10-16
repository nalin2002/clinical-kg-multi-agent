# /// script
# requires-python = ">=3.10"
# dependencies = ["torch==2.4.1", "transformers>=4.48", "huggingface_hub>=0.25", "numpy"]   # torch pinned to CUDA-12 so it uses the GPU (unpinned -> CUDA-13 wheels -> driver too old -> CPU)
# ///
# fawkes_embed_notes_modernbert_260615.py
# Embed the per-admission `note` with CLINICAL-MODERNBERT (mean-pooled, one vector / note) for ALL records of
# the full graph dataset, and save a NEW dataset = the original records + `note_embedding` inline (for the NOTE
# node feature). GPU job (fastest). Deterministic (eval, no sampling). Parameterized, fail-loud (no fallback).
import os, json, logging
import torch, numpy as np
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("embed")

def env(k, d=None, required=False):
    v = os.environ.get(k, d)
    if required and not v: raise RuntimeError(f"[FAILURE] env {k} unset; no fallback.")
    return v
DATA_REPO   = env("DATA_REPO", "on1onmangoes/fawkes-training-graph-full-260615")
DATA_FILE   = env("DATA_FILE", "fawkes_training_graph_full_260615.jsonl")
EMBED_MODEL = env("EMBED_MODEL", "Simonlee711/Clinical_ModernBERT")   # Clinical-ModernBERT (override if the repo id differs)
OUTPUT_REPO = env("OUTPUT_REPO", "on1onmangoes/fawkes-training-graph-full-embedded-260615")
OUT_FILE    = env("OUT_FILE", "fawkes_training_graph_full_embedded_260615.jsonl")
MAX_LEN     = int(env("MAX_LEN", 8192)); BATCH = int(env("BATCH", 32))
PRIVATE     = str(env("PRIVATE", "1")).lower() in ("1", "true", "yes")

def main():
    from huggingface_hub import hf_hub_download, HfApi
    from transformers import AutoTokenizer, AutoModel
    p = hf_hub_download(DATA_REPO, DATA_FILE, repo_type="dataset")
    recs = [json.loads(l) for l in open(p) if l.strip()]
    log.info(f"[DATA] {len(recs)} records from {DATA_REPO}")
    notes = [(r.get("note") or "") for r in recs]
    n_empty = sum(1 for t in notes if not t.strip())
    log.info(f"[NOTES] {len(notes)} notes ({n_empty} empty)")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"[GPU] {dev}" + (f" {torch.cuda.get_device_name(0)}" if dev == "cuda" else ""))
    tok = AutoTokenizer.from_pretrained(EMBED_MODEL)
    model = AutoModel.from_pretrained(EMBED_MODEL, torch_dtype=torch.bfloat16 if dev == "cuda" else torch.float32).to(dev).eval()
    dim = int(model.config.hidden_size)
    log.info(f"[MODEL] {EMBED_MODEL} hidden_size={dim} max_len={MAX_LEN} batch={BATCH}")

    @torch.no_grad()
    def embed(batch_texts):
        t = tok(batch_texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(dev)
        h = model(**t).last_hidden_state                                  # (B, L, D)
        m = t["attention_mask"].unsqueeze(-1).to(h.dtype)                 # (B, L, 1)
        v = (h * m).sum(1) / m.sum(1).clamp(min=1)                        # masked mean -> (B, D)
        return v.float().cpu().numpy()

    embs = np.zeros((len(notes), dim), dtype=np.float32)
    import time; t0 = time.time()
    for i in range(0, len(notes), BATCH):
        embs[i:i+BATCH] = embed(notes[i:i+BATCH])
        if (i // BATCH) % 20 == 0: log.info(f"  [EMB] {min(i+BATCH,len(notes))}/{len(notes)}")
    log.info(f"[EMB] done {len(notes)} notes in {time.time()-t0:.0f}s")

    with open(OUT_FILE, "w") as f:
        for r, e in zip(recs, embs):
            r["note_embedding"] = [round(float(x), 6) for x in e]
            r["embed_model"] = EMBED_MODEL; r["embed_dim"] = dim
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.info(f"[WRITE] {OUT_FILE} ({os.path.getsize(OUT_FILE):,} bytes)")

    api = HfApi(); api.create_repo(OUTPUT_REPO, repo_type="dataset", private=PRIVATE, exist_ok=True)
    api.upload_file(path_or_fileobj=OUT_FILE, path_in_repo=OUT_FILE, repo_id=OUTPUT_REPO, repo_type="dataset")
    readme = (f"---\nlicense: other\ntags: [mimic-iv, clinical, fawkes, embeddings]\n---\n\n"
              f"# Fawkes training-graph-full + NOTE embeddings — PRIVATE\n\n"
              f"{len(recs)} per-admission graphs (note + MIMIC + nodes + edges + v8 scores) with the per-admission "
              f"**`note_embedding`** ({dim}-d, mean-pooled **{EMBED_MODEL}**) inline — the feature for the NOTE node. "
              f"MIMIC-IV credentialed (PhysioNet DUA) — keep private.\n")
    api.upload_file(path_or_fileobj=readme.encode(), path_in_repo="README.md", repo_id=OUTPUT_REPO, repo_type="dataset")
    log.info(f"[DONE] -> https://huggingface.co/datasets/{OUTPUT_REPO} ({len(recs)} records, note_embedding dim={dim})")

if __name__ == "__main__":
    main()
