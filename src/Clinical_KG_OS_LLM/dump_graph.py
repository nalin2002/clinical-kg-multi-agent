"""
Unified KG Graph Merger
=======================
Merge sub-KGs into unified graph using entity resolution (BGE-M3 + cosine similarity).

Usage:
    python dump_graph.py --input baseline_naive/sub_kgs --output baseline_naive/
"""

import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# === Configuration ===
ER_SIMILARITY_THRESHOLD = 0.85
NEGATION_PREFIXES = ("absent ", "no ", "not ", "denied ", "negative ")


def normalize(name):
    return name.strip().lower()


def is_negated(name):
    return any(normalize(name).startswith(p) for p in NEGATION_PREFIXES)


def cosine_sim(a, b):
    d = np.dot(a, b)
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(d / n) if n > 0 else 0.0


def find_sub_kg_files(input_dir: Path) -> list:
    """Find all sub-KG JSON files in the input directory."""
    files = []
    for f in sorted(input_dir.glob("RES*.json")):
        if f.name.startswith("RES") and not f.name.startswith("_"):
            files.append(f)
    return files


def load_sub_kgs(input_dir: Path) -> tuple:
    """Load all sub-KGs from a directory."""
    all_raw_entities = []
    all_raw_edges = []
    pass1_total = {"nodes": 0, "edges": 0}
    pass2_total = {"nodes": 0, "edges": 0}
    loaded = 0

    for kg_path in find_sub_kg_files(input_dir):
        res_id = kg_path.stem.split("_")[0]  # Extract RES0198 from RES0198_naive.json

        with open(kg_path) as f:
            kg = json.load(f)

        # Track pass1 vs pass2 stats if available
        if "_meta" in kg:
            pass1_total["nodes"] += kg["_meta"].get("pass1", {}).get("nodes", 0)
            pass1_total["edges"] += kg["_meta"].get("pass1", {}).get("edges", 0)
            pass2_total["nodes"] += kg["_meta"].get("pass2", {}).get("nodes", 0)
            pass2_total["edges"] += kg["_meta"].get("pass2", {}).get("edges", 0)

        for node in kg.get("nodes", []):
            all_raw_entities.append({
                "res_id": res_id,
                "kg_id": node["id"],
                "text": node["text"],
                "type": node.get("type", "entity"),
                "evidence": node.get("evidence", ""),
                "turn_id": node.get("turn_id", ""),
            })

        for edge in kg.get("edges", []):
            all_raw_edges.append({
                "res_id": res_id,
                "source_id": edge["source_id"],
                "target_id": edge["target_id"],
                "type": edge["type"],
                "evidence": edge.get("evidence", ""),
                "turn_id": edge.get("turn_id", ""),
            })

        loaded += 1

    return all_raw_entities, all_raw_edges, pass1_total, pass2_total, loaded


def entity_resolution(all_raw_entities: list, embed_model, threshold: float) -> tuple:
    """Perform entity resolution using embedding similarity."""
    # Group by type
    type_groups = defaultdict(list)
    for ent in all_raw_entities:
        type_groups[ent["type"]].append(ent)

    merge_decisions = []
    canonical_map = {}

    for ent_type, entities in type_groups.items():
        unique_names = sorted(set(normalize(e["text"]) for e in entities))
        print(f"  {ent_type}: {len(unique_names)} unique names")

        if len(unique_names) <= 1:
            canon = unique_names[0] if unique_names else ""
            for e in entities:
                canonical_map[(e["res_id"], e["kg_id"])] = canon
            continue

        # Compute embeddings
        embeddings = {}
        emb_results = embed_model.get_text_embedding_batch(unique_names)
        for name, emb in zip(unique_names, emb_results):
            embeddings[name] = np.array(emb)

        # Cluster by similarity
        clusters = []
        assigned = set()
        for i, name_i in enumerate(unique_names):
            if name_i in assigned:
                continue
            cluster = [name_i]
            assigned.add(name_i)
            neg_i = is_negated(name_i)
            similarities = {}
            for j, name_j in enumerate(unique_names):
                if j <= i or name_j in assigned:
                    continue
                if is_negated(name_j) != neg_i:
                    continue
                sim = cosine_sim(embeddings[name_i], embeddings[name_j])
                if sim >= threshold:
                    cluster.append(name_j)
                    assigned.add(name_j)
                    similarities[name_j] = round(sim, 4)
            clusters.append((cluster, similarities))

        # Map to canonical names
        name_to_canonical = {}
        for cluster, sims in clusters:
            canonical = min(cluster, key=len)
            for name in cluster:
                name_to_canonical[name] = canonical

            if len(cluster) > 1:
                merge_decisions.append({
                    "entity_type": ent_type,
                    "cluster": cluster,
                    "canonical_name": canonical,
                    "similarities": sims,
                })

        for e in entities:
            canonical_map[(e["res_id"], e["kg_id"])] = name_to_canonical.get(
                normalize(e["text"]), normalize(e["text"])
            )

    return canonical_map, merge_decisions


def build_unified_graph(all_raw_entities: list, all_raw_edges: list, canonical_map: dict) -> tuple:
    """Build the unified graph from resolved entities."""
    entity_nodes_map = {}
    node_key_to_id = {}
    node_counter = 1

    for ent in all_raw_entities:
        canon = canonical_map[(ent["res_id"], ent["kg_id"])]
        node_key = f"{canon}|{ent['type']}"
        if node_key not in entity_nodes_map:
            node_id = f"E_{node_counter:04d}"
            node_counter += 1
            node_key_to_id[node_key] = node_id
            entity_nodes_map[node_key] = {
                "id": node_id,
                "text": canon,
                "type": ent["type"],
                "occurrences": [],
            }
        entity_nodes_map[node_key]["occurrences"].append({
            "res_id": ent["res_id"],
            "turn_id": ent["turn_id"],
        })

    # Map (res_id, kg_id) -> node_id
    id_to_node_id = {}
    for ent in all_raw_entities:
        canon = canonical_map[(ent["res_id"], ent["kg_id"])]
        node_key = f"{canon}|{ent['type']}"
        id_to_node_id[(ent["res_id"], ent["kg_id"])] = node_key_to_id[node_key]

    # Build edges
    edges = []
    for edge in all_raw_edges:
        src_id = id_to_node_id.get((edge["res_id"], edge["source_id"]))
        tgt_id = id_to_node_id.get((edge["res_id"], edge["target_id"]))
        if src_id and tgt_id:
            edges.append({
                "source_id": src_id,
                "target_id": tgt_id,
                "type": edge["type"],
                "res_id": edge["res_id"],
                "turn_id": edge["turn_id"],
            })

    return list(entity_nodes_map.values()), edges


def main():
    parser = argparse.ArgumentParser(description="Merge sub-KGs into unified graph")
    parser.add_argument("--input", type=str, required=True, help="Input directory containing sub-KG JSON files")
    parser.add_argument("--output", type=str, required=True, help="Output directory for unified graph")
    parser.add_argument("--name", type=str, default=None, help="Output filename suffix (default: derived from input dir)")
    parser.add_argument("--threshold", type=float, default=0.85, help="ER similarity threshold (default: 0.85)")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Derive output filename from input directory name or --name
    if args.name:
        suffix = args.name
    else:
        parent_name = input_dir.parent.name if input_dir.name == "sub_kgs" else input_dir.name
        suffix = parent_name.replace("baseline_", "")

    graph_filename = f"unified_graph_{suffix}.json"
    er_filename = f"er_merge_decisions_{suffix}.json"

    print(f"KG Graph Merger")
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir / graph_filename}")
    print("=" * 60)

    # Load embedding model
    print("Loading bge-m3...")
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")

    # Load sub-KGs
    all_raw_entities, all_raw_edges, pass1_total, pass2_total, loaded = load_sub_kgs(input_dir)
    print(f"Loaded {loaded} sub-KGs: {len(all_raw_entities)} entities, {len(all_raw_edges)} edges")

    if pass1_total["nodes"] > 0:
        print(f"  Pass1 totals: {pass1_total['nodes']} nodes, {pass1_total['edges']} edges")
        print(f"  Pass2 totals: {pass2_total['nodes']} nodes, {pass2_total['edges']} edges")

    # Entity resolution
    unique_before = len(set(normalize(e["text"]) for e in all_raw_entities))
    canonical_map, merge_decisions = entity_resolution(all_raw_entities, embed_model, args.threshold)
    unique_after = len(set(canonical_map.values()))
    print(f"\nEntity Resolution: {unique_before} -> {unique_after} canonical entities")

    # Build unified graph
    nodes, edges = build_unified_graph(all_raw_entities, all_raw_edges, canonical_map)

    # Save ER merge decisions
    er_path = output_dir / er_filename
    with open(er_path, "w") as f:
        json.dump({
            "method": "embedding_similarity",
            "embedding_model": "BAAI/bge-m3",
            "similarity_threshold": args.threshold,
            "total_raw_entities": len(all_raw_entities),
            "unique_before": unique_before,
            "unique_after": unique_after,
            "pass1_totals": pass1_total,
            "pass2_totals": pass2_total,
            "merge_decisions": merge_decisions,
        }, f, indent=2, ensure_ascii=False)
    print(f"Saved: {er_path}")

    # Save unified graph
    graph_path = output_dir / graph_filename
    with open(graph_path, "w") as f:
        json.dump({
            "nodes": nodes,
            "edges": edges,
        }, f, indent=2, ensure_ascii=False)
    print(f"Saved: {graph_path}")

    print(f"\nUnified graph: {len(nodes)} nodes, {len(edges)} edges")
    print("Done.")


if __name__ == "__main__":
    main()
