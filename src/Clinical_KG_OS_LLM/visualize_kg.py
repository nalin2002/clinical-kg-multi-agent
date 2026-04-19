"""
Knowledge Graph Visualizer
==========================
Render a unified KG (nodes + edges) as a graph figure.

Usage:
    python -m src.Clinical_KG_OS_LLM.visualize_kg --kg path/to/unified_graph.json --output graph.png
    python -m src.Clinical_KG_OS_LLM.visualize_kg --kg my_kg_naive/unified_graph_my_kg.json --output my_kg.png --res-id RES0198

Options:
    --kg       Path to unified KG JSON (required)
    --output   Output image path (default: kg_stem_graph.png)
    --res-id   Filter to one patient (optional; shows full graph if omitted)
    --max-nodes Limit nodes shown (default: 50; use 0 for no limit)
"""
import os

# Must set before any matplotlib import (overrides Jupyter's matplotlib_inline)
os.environ["MPLBACKEND"] = "Agg"

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx


def load_kg(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def filter_by_patient(kg: dict, res_id: str) -> tuple[list, list]:
    """Filter nodes and edges to those involving the given patient."""
    node_ids = set()
    for n in kg["nodes"]:
        for occ in n.get("occurrences", []):
            if occ.get("res_id") == res_id:
                node_ids.add(n["id"])
                break

    nodes = [n for n in kg["nodes"] if n["id"] in node_ids]
    edges = [
        e for e in kg["edges"]
        if e.get("res_id") == res_id
        and e["source_id"] in node_ids
        and e["target_id"] in node_ids
    ]
    return nodes, edges


def build_graph(nodes: list, edges: list, max_nodes: int = 50):
    """Build networkx graph from nodes and edges."""
    G = nx.DiGraph()
    node_by_id = {n["id"]: n for n in nodes}

    for n in nodes[:max_nodes] if max_nodes else nodes:
        G.add_node(n["id"], text=n["text"], type=n.get("type", "ENTITY"))

    for e in edges:
        if G.has_node(e["source_id"]) and G.has_node(e["target_id"]):
            G.add_edge(e["source_id"], e["target_id"], type=e.get("type", "RELATED_TO"))

    return G


def draw_graph(G: nx.DiGraph, output_path: Path):
    """Draw graph with matplotlib."""
    if G.number_of_nodes() == 0:
        print("No nodes to visualize.")
        return

    # Layout
    pos = nx.spring_layout(G, k=1.5, iterations=50, seed=42)

    fig, ax = plt.subplots(figsize=(14, 10))

    # Node colors by type
    type_colors = {
        "SYMPTOM": "#E74C3C",
        "DIAGNOSIS": "#3498DB",
        "TREATMENT": "#2ECC71",
        "PROCEDURE": "#9B59B6",
        "LOCATION": "#F39C12",
        "MEDICAL_HISTORY": "#1ABC9C",
        "LAB_RESULT": "#34495E",
    }
    default_color = "#95A5A6"

    node_colors = [
        type_colors.get(n[1].get("type", "").upper(), default_color)
        for n in G.nodes(data=True)
    ]

    # Draw
    nx.draw_networkx_nodes(
        G, pos, node_color=node_colors, node_size=800, alpha=0.9, ax=ax
    )
    nx.draw_networkx_edges(
        G, pos, edge_color="#BDC3C7", arrows=True, arrowsize=15, ax=ax
    )

    # Labels: use text, truncate if long
    labels = {n: (d["text"][:20] + "…" if len(d["text"]) > 20 else d["text"]) for n, d in G.nodes(data=True)}
    nx.draw_networkx_labels(
        G, pos, labels, font_size=7, font_weight="bold", ax=ax
    )

    # Edge labels (type)
    edge_labels = {(u, v): d.get("type", "")[:8] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=6, ax=ax)

    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize knowledge graph")
    parser.add_argument("--kg", type=str, required=True, help="Path to unified KG JSON")
    parser.add_argument("--output", type=str, default=None, help="Output image path")
    parser.add_argument("--res-id", type=str, default=None, help="Filter to one patient")
    parser.add_argument("--max-nodes", type=int, default=50, help="Max nodes to show (0 = no limit)")
    args = parser.parse_args()

    kg_path = Path(args.kg)
    kg = load_kg(str(kg_path))

    if args.res_id:
        nodes, edges = filter_by_patient(kg, args.res_id)
        print(f"Filtered to {args.res_id}: {len(nodes)} nodes, {len(edges)} edges")
    else:
        nodes = kg.get("nodes", [])
        edges = kg.get("edges", [])

    G = build_graph(nodes, edges, args.max_nodes)

    output_path = Path(args.output) if args.output else kg_path.parent / f"{kg_path.stem}_graph.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    draw_graph(G, output_path)


if __name__ == "__main__":
    main()
