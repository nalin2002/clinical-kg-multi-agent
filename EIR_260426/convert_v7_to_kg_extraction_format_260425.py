"""
v7 → kg_extraction.py format converter (260425)
================================================
Reads our v7 per-patient per-category outputs and emits per-patient unified
JSON files in the schema produced by Clinical_KG_OS_LLM/kg_extraction.py, so
that Clinical_KG_OS_LLM/dump_graph.py and kg_similarity_scorer.py can consume
them without modification.

Inputs (decoupled so we can mix runs, e.g. V14 INDICATES + V13 other edges):
  --nodes-dir : v7 run dir whose RES*/ subdirs contain the 7 entity-category JSONs
                (RES0XXX_SYMPTOM.json, RES0XXX_DIAGNOSIS.json, ...)
  --edges-dir : v7 run dir whose RES*/ subdirs contain the 6 edge-category JSONs
                (RES0XXX_INDICATES.json, RES0XXX_CAUSES.json, ...)
  --output    : output dir for per-patient {RES_ID}_v7.json files

Output schema (matches kg_extraction.py / dump_graph.py expectations):
  {
    "nodes": [{"id": "N_001", "text": "...", "type": "SYMPTOM",
               "evidence": "...", "turn_id": "P-1"}, ...],
    "edges": [{"source_id": "N_001", "target_id": "N_010", "type": "INDICATES",
               "evidence": "...", "turn_id": "D-39"}, ...]
  }

Fail-loudly behavior:
  - If --nodes-dir or --edges-dir is missing a patient that exists in the other,
    we raise.
  - Edges whose src/dst text cannot be resolved to a node id are SKIPPED and
    counted; the script prints a per-edge-category unresolved tally so the user
    can see how much information the converter dropped.
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
)
logger = logging.getLogger("convert_v7_260425")

ENTITY_CATEGORIES = [
    "SYMPTOM", "DIAGNOSIS", "TREATMENT", "PROCEDURE",
    "LOCATION", "MEDICAL_HISTORY", "LAB_RESULT",
]
EDGE_CATEGORIES = [
    "INDICATES", "CAUSES", "LOCATED_AT", "RULES_OUT", "TAKEN_FOR", "CONFIRMS",
]

# Preferred src/dst types per edge category (used for first-pass node lookup).
# These match our v7 prompt schema; we still fall back to any-type if no hit.
EDGE_TYPE_SCHEMA = {
    "INDICATES":  {"src": ["SYMPTOM", "LAB_RESULT", "PROCEDURE", "MEDICAL_HISTORY"],
                   "dst": ["DIAGNOSIS"]},
    "CAUSES":     {"src": ["MEDICAL_HISTORY", "DIAGNOSIS", "TREATMENT"],
                   "dst": ["DIAGNOSIS", "SYMPTOM"]},
    "LOCATED_AT": {"src": ["SYMPTOM"],
                   "dst": ["LOCATION"]},
    "RULES_OUT":  {"src": ["PROCEDURE", "LAB_RESULT", "SYMPTOM"],
                   "dst": ["DIAGNOSIS"]},
    "TAKEN_FOR":  {"src": ["TREATMENT"],
                   "dst": ["MEDICAL_HISTORY", "DIAGNOSIS"]},
    "CONFIRMS":   {"src": ["LAB_RESULT", "PROCEDURE"],
                   "dst": ["DIAGNOSIS"]},
}


def _norm(s: str) -> str:
    return (s or "").strip().lower()


def _entry_text(entry: dict) -> str:
    """`matched` entries use `label`; `other`/`fine_grained` use `text`."""
    return entry.get("label") or entry.get("text") or ""


def list_res_ids(d: Path) -> list:
    if not d.exists():
        raise SystemExit(f"[FATAL] directory does not exist: {d}")
    return sorted(p.name for p in d.iterdir() if p.is_dir() and p.name.startswith("RES"))


def load_entity_nodes(res_dir: Path, res_id: str):
    """Walk the 7 entity files; return (nodes_list, lookup_by_text_type, lookup_by_text)."""
    nodes = []
    by_text_type = {}                  # (text_lower, TYPE) -> node_id
    by_text = defaultdict(list)        # text_lower -> [(TYPE, node_id), ...]
    counter = 0

    for cat in ENTITY_CATEGORIES:
        f = res_dir / f"{res_id}_{cat}.json"
        if not f.exists():
            logger.warning(f"[NODES] {res_id}: missing {f.name}")
            continue
        with open(f) as fh:
            data = json.load(fh)

        for bucket in ("matched", "other", "fine_grained"):
            for entry in data.get(bucket, []) or []:
                text = _entry_text(entry).strip()
                if not text:
                    continue
                key = (_norm(text), cat)
                if key in by_text_type:
                    # Same text+type collapsed across buckets: keep first occurrence.
                    continue
                counter += 1
                nid = f"N_{counter:03d}"
                node = {
                    "id": nid,
                    "text": text,
                    "type": cat,
                    "evidence": entry.get("quote", ""),
                    "turn_id": entry.get("turn_id", ""),
                }
                nodes.append(node)
                by_text_type[key] = nid
                by_text[_norm(text)].append((cat, nid))

    return nodes, by_text_type, by_text


def resolve_endpoint(text: str, allowed_types: list, by_text_type: dict, by_text: dict):
    """Return node_id or None. Tries (text, t) for each allowed t, then any-type fallback."""
    t_norm = _norm(text)
    if not t_norm:
        return None
    for t in allowed_types:
        nid = by_text_type.get((t_norm, t))
        if nid:
            return nid
    candidates = by_text.get(t_norm, [])
    if candidates:
        # Prefer allowed types if any matched at all (we wouldn't be here), else first
        return candidates[0][1]
    return None


def load_edges(res_dir: Path, res_id: str, by_text_type: dict, by_text: dict):
    """Walk the 6 edge files; return (edges_list, unresolved_counts_by_category)."""
    edges = []
    unresolved = defaultdict(lambda: {"src": 0, "dst": 0, "both": 0, "total": 0})

    for cat in EDGE_CATEGORIES:
        f = res_dir / f"{res_id}_{cat}.json"
        if not f.exists():
            logger.warning(f"[EDGES] {res_id}: missing {f.name}")
            continue
        with open(f) as fh:
            data = json.load(fh)

        schema = EDGE_TYPE_SCHEMA[cat]
        for bucket in ("matched", "other", "fine_grained"):
            for entry in data.get(bucket, []) or []:
                src_text = (entry.get("src") or "").strip()
                dst_text = (entry.get("dst") or "").strip()
                if not src_text or not dst_text:
                    continue

                src_id = resolve_endpoint(src_text, schema["src"], by_text_type, by_text)
                dst_id = resolve_endpoint(dst_text, schema["dst"], by_text_type, by_text)

                if src_id and dst_id:
                    edges.append({
                        "source_id": src_id,
                        "target_id": dst_id,
                        "type": cat,
                        "evidence": entry.get("quote", ""),
                        "turn_id": entry.get("turn_id", ""),
                    })
                else:
                    u = unresolved[cat]
                    if src_id is None and dst_id is None:
                        u["both"] += 1
                    elif src_id is None:
                        u["src"] += 1
                    else:
                        u["dst"] += 1
                    u["total"] += 1

    return edges, dict(unresolved)


def convert_one(res_id: str, nodes_dir: Path, edges_dir: Path, out_dir: Path):
    nrd = nodes_dir / res_id
    erd = edges_dir / res_id
    if not nrd.exists():
        raise SystemExit(f"[FATAL] nodes-dir missing patient: {nrd}")
    if not erd.exists():
        raise SystemExit(f"[FATAL] edges-dir missing patient: {erd}")

    nodes, by_text_type, by_text = load_entity_nodes(nrd, res_id)
    edges, unresolved = load_edges(erd, res_id, by_text_type, by_text)

    out_file = out_dir / f"{res_id}_v7.json"
    with open(out_file, "w") as f:
        json.dump({"nodes": nodes, "edges": edges}, f, indent=2, ensure_ascii=False)

    return {
        "res_id": res_id,
        "n_nodes": len(nodes),
        "n_edges": len(edges),
        "unresolved_by_category": unresolved,
        "out_file": str(out_file),
    }


def main():
    ap = argparse.ArgumentParser(description="Convert v7 per-category outputs to kg_extraction.py per-patient format")
    ap.add_argument("--nodes-dir", required=True, help="v7 run dir containing RES*/ entity JSONs")
    ap.add_argument("--edges-dir", required=True, help="v7 run dir containing RES*/ edge JSONs")
    ap.add_argument("--output",    required=True, help="output dir for per-patient {RES}_v7.json files")
    ap.add_argument("--res-ids", nargs="+", default=None, help="optional subset")
    args = ap.parse_args()

    nodes_dir = Path(args.nodes_dir)
    edges_dir = Path(args.edges_dir)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_ids = list_res_ids(nodes_dir)
    e_ids = list_res_ids(edges_dir)

    common = sorted(set(n_ids) & set(e_ids))
    only_n = sorted(set(n_ids) - set(e_ids))
    only_e = sorted(set(e_ids) - set(n_ids))
    if only_n:
        logger.warning(f"[MISMATCH] in --nodes-dir but not --edges-dir: {only_n}")
    if only_e:
        logger.warning(f"[MISMATCH] in --edges-dir but not --nodes-dir: {only_e}")

    res_ids = common
    if args.res_ids:
        res_ids = [r for r in res_ids if r in args.res_ids]

    logger.info(f"[ENTRY] nodes-dir={nodes_dir}")
    logger.info(f"[ENTRY] edges-dir={edges_dir}")
    logger.info(f"[ENTRY] output={out_dir}")
    logger.info(f"[ENTRY] processing {len(res_ids)} patients")

    summary = []
    total_nodes = 0
    total_edges = 0
    total_unresolved = defaultdict(lambda: {"src": 0, "dst": 0, "both": 0, "total": 0})

    for res_id in res_ids:
        info = convert_one(res_id, nodes_dir, edges_dir, out_dir)
        summary.append(info)
        total_nodes += info["n_nodes"]
        total_edges += info["n_edges"]
        for cat, u in info["unresolved_by_category"].items():
            for k, v in u.items():
                total_unresolved[cat][k] += v
        logger.info(
            f"[OK] {res_id}: nodes={info['n_nodes']} edges={info['n_edges']} "
            f"unresolved={sum(u['total'] for u in info['unresolved_by_category'].values())}"
        )

    summary_path = out_dir / "_convert_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "nodes_dir": str(nodes_dir),
            "edges_dir": str(edges_dir),
            "n_patients": len(res_ids),
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "total_unresolved_by_edge_category": dict(total_unresolved),
            "per_patient": summary,
        }, f, indent=2, ensure_ascii=False)

    logger.info("=" * 60)
    logger.info(f"[SUMMARY] patients={len(res_ids)} total_nodes={total_nodes} total_edges={total_edges}")
    if total_unresolved:
        logger.info("[SUMMARY] unresolved edges (skipped) by category:")
        for cat in EDGE_CATEGORIES:
            u = total_unresolved.get(cat, {})
            if u and u.get("total", 0):
                logger.info(f"  {cat:12} total={u['total']} (src_missing={u['src']} dst_missing={u['dst']} both={u['both']})")
    logger.info(f"[SUMMARY] wrote {summary_path}")


if __name__ == "__main__":
    sys.exit(main())
