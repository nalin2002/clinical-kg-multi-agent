"""Graph operations shared across training and evaluation:

* negative-triple / graph corruptions (training signal),
* fully-connected candidate-graph construction (the FullyConnected+WM ablation),
* graph-comparison metrics (edge P/R/F1, relation accuracy, GED approximation).

All corruptions are pure functions returning new objects; inputs are untouched.
"""

from __future__ import annotations

import itertools
import random
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

from .graph_schema import (
    RELATION_SCHEMA,
    RELATION_TYPES,
    Edge,
    Graph,
    Node,
    is_clinically_plausible,
)

Triple = Tuple[str, str, str]  # (source_id, relation, target_id)


# --------------------------------------------------------------------------- #
# Negative-triple generation (edge_plausibility_classifier)
# --------------------------------------------------------------------------- #
def corrupt_triple(
    graph: Graph,
    edge: Edge,
    strategy: str,
    rng: random.Random,
) -> Optional[Edge]:
    """Return a corrupted (negative) edge from a positive one, or None if the
    strategy cannot apply (e.g. too few nodes). Strategies:

    * ``relation_replacement`` — keep endpoints, swap to a different relation.
    * ``source_target_swap``   — reverse direction (often clinically wrong).
    * ``random_target``        — repoint target to a random other node.
    * ``invalid_clinical_relation`` — choose a (relation,target) that violates
      the typed RELATION_SCHEMA for the source type.
    """
    idx = graph.node_index()
    src, tgt = idx.get(edge.source), idx.get(edge.target)
    if not src or not tgt:
        return None

    if strategy == "relation_replacement":
        alts = [r for r in RELATION_TYPES if r != edge.relation]
        if not alts:
            return None
        return Edge(edge.source, edge.target, rng.choice(alts), confidence=0.0)

    if strategy == "source_target_swap":
        if edge.source == edge.target:
            return None
        return Edge(edge.target, edge.source, edge.relation, confidence=0.0)

    if strategy == "random_target":
        others = [n.id for n in graph.nodes if n.id not in (edge.source, edge.target)]
        if not others:
            return None
        return Edge(edge.source, rng.choice(others), edge.relation, confidence=0.0)

    if strategy == "invalid_clinical_relation":
        # Pick a (relation, target_node) pair that is type-implausible.
        candidates = []
        for n in graph.nodes:
            if n.id == edge.source:
                continue
            for r in RELATION_TYPES:
                if not is_clinically_plausible(src.type, r, n.type):
                    candidates.append((r, n.id))
        if not candidates:
            return None
        r, tid = rng.choice(candidates)
        return Edge(edge.source, tid, r, confidence=0.0)

    raise ValueError(f"unknown corruption strategy: {strategy!r}")


def sample_negatives(
    graph: Graph,
    n_per_pos: int,
    corruption_mix: Dict[str, float],
    rng: random.Random,
) -> List[Tuple[Edge, str]]:
    """Generate negative edges for every positive edge in ``graph``.

    Returns a list of ``(edge, strategy)``. Negatives that accidentally coincide
    with a real positive triple are dropped.
    """
    positives = {(e.source, e.relation, e.target) for e in graph.edges}
    strategies = list(corruption_mix.keys())
    weights = [corruption_mix[s] for s in strategies]
    out: List[Tuple[Edge, str]] = []
    for edge in graph.edges:
        made = 0
        attempts = 0
        while made < n_per_pos and attempts < n_per_pos * 6:
            attempts += 1
            strat = rng.choices(strategies, weights=weights, k=1)[0]
            neg = corrupt_triple(graph, edge, strat, rng)
            if neg is None:
                continue
            if (neg.source, neg.relation, neg.target) in positives:
                continue
            out.append((neg, strat))
            made += 1
    return out


# --------------------------------------------------------------------------- #
# Graph-level corruptions (graph_contrastive_scorer)
# --------------------------------------------------------------------------- #
def corrupt_graph(graph: Graph, strategy: str, rng: random.Random) -> Graph:
    """Return a corrupted copy of ``graph`` (a less-plausible whole graph)."""
    g = Graph(graph.patient_id, f"corrupt:{strategy}", list(graph.nodes), list(graph.edges))
    edges = list(g.edges)
    if not edges:
        return g

    if strategy == "edge_deletion":
        i = rng.randrange(len(edges))
        edges.pop(i)
    elif strategy == "invalid_edge_addition":
        a, b = rng.choice(graph.nodes), rng.choice(graph.nodes)
        edges.append(Edge(a.id, b.id, rng.choice(RELATION_TYPES), confidence=0.0))
    elif strategy == "relation_label_replacement":
        i = rng.randrange(len(edges))
        e = edges[i]
        alts = [r for r in RELATION_TYPES if r != e.relation]
        edges[i] = Edge(e.source, e.target, rng.choice(alts), e.evidence, 0.0)
    elif strategy == "direction_flip":
        i = rng.randrange(len(edges))
        e = edges[i]
        edges[i] = Edge(e.target, e.source, e.relation, e.evidence, 0.0)
    else:
        raise ValueError(f"unknown graph corruption: {strategy!r}")
    g.edges = edges
    return g


# --------------------------------------------------------------------------- #
# Fully-connected candidate graph (FullyConnected+WM ablation)
# --------------------------------------------------------------------------- #
def fully_connected_candidates(graph: Graph) -> List[Edge]:
    """Every type-plausible directed (src, relation, tgt) pair over the nodes.

    Restricting to RELATION_SCHEMA-plausible pairs keeps the candidate set
    tractable while still testing whether the world model can recover the true
    relations from nodes alone. ``confidence=0`` marks them as unscored.
    """
    edges: List[Edge] = []
    for src, tgt in itertools.permutations(graph.nodes, 2):
        for relation in RELATION_TYPES:
            if is_clinically_plausible(src.type, relation, tgt.type):
                edges.append(Edge(src.id, tgt.id, relation, evidence="", confidence=0.0))
    return edges


def candidate_relations_for(src: Node, tgt: Node) -> List[str]:
    """Type-plausible relations for an ordered node pair (for relabeling)."""
    return [r for r in RELATION_TYPES if is_clinically_plausible(src.type, r, tgt.type)]


# --------------------------------------------------------------------------- #
# Node matching + metrics (evaluation)
# --------------------------------------------------------------------------- #
def fuzzy_match(a: str, b: str, threshold: float = 0.80) -> bool:
    a, b = a.lower().strip(), b.lower().strip()
    if a == b:
        return True
    return SequenceMatcher(None, a, b).ratio() >= threshold


def _match_node(name: str, candidates: List[str], threshold: float) -> Optional[str]:
    for c in candidates:
        if fuzzy_match(name, c, threshold):
            return c
    return None


def edge_prf(pred: Graph, ref: Graph, threshold: float = 0.80) -> Dict[str, float]:
    """Edge precision/recall/F1 of ``pred`` vs reference ``ref``.

    Edges match when both endpoints fuzzy-match by normalized node name AND the
    relation is identical. This is direction-sensitive.
    """
    pred_t = pred.triples()
    ref_t = ref.triples()
    if not pred_t and not ref_t:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "pred": 0, "ref": 0, "matched": 0}

    ref_names = sorted({n for t in ref_t for n in (t[0], t[2])})
    matched_ref = set()
    tp = 0
    for s, r, t in pred_t:
        sm = _match_node(s, ref_names, threshold)
        tm = _match_node(t, ref_names, threshold)
        if sm is None or tm is None:
            continue
        hit = next(((i, rt) for i, rt in enumerate(ref_t)
                    if i not in matched_ref and rt[1] == r
                    and fuzzy_match(rt[0], sm, threshold) and fuzzy_match(rt[2], tm, threshold)), None)
        if hit is not None:
            matched_ref.add(hit[0])
            tp += 1
    precision = tp / len(pred_t) if pred_t else 0.0
    recall = tp / len(ref_t) if ref_t else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "pred": len(pred_t), "ref": len(ref_t), "matched": tp}


def relation_accuracy(pred: Graph, ref: Graph, threshold: float = 0.80) -> Dict[str, float]:
    """Among edges whose endpoints match the reference (ignoring relation), the
    fraction whose relation label is also correct."""
    ref_t = ref.triples()
    ref_names = sorted({n for t in ref_t for n in (t[0], t[2])})
    endpoint_hits = 0
    relation_hits = 0
    for s, r, t in pred.triples():
        sm = _match_node(s, ref_names, threshold)
        tm = _match_node(t, ref_names, threshold)
        if sm is None or tm is None:
            continue
        same = [rt for rt in ref_t
                if fuzzy_match(rt[0], sm, threshold) and fuzzy_match(rt[2], tm, threshold)]
        if not same:
            continue
        endpoint_hits += 1
        if any(rt[1] == r for rt in same):
            relation_hits += 1
    acc = relation_hits / endpoint_hits if endpoint_hits else 0.0
    return {"relation_accuracy": acc, "endpoint_matched": endpoint_hits, "relation_correct": relation_hits}


def graph_edit_distance_approx(pred: Graph, ref: Graph, threshold: float = 0.80) -> Dict[str, float]:
    """Cheap, order-free GED proxy: edge insertions + deletions to turn ``pred``
    into ``ref`` (substitutions counted as one delete + one insert). Lower is
    better. Normalised by reference size for cross-patient comparability."""
    prf = edge_prf(pred, ref, threshold)
    matched = prf["matched"]
    inserts = prf["ref"] - matched      # edges in ref missing from pred
    deletes = prf["pred"] - matched     # edges in pred not in ref
    ged = inserts + deletes
    denom = max(prf["ref"], 1)
    return {"ged": float(ged), "ged_normalized": ged / denom,
            "edge_inserts": float(inserts), "edge_deletes": float(deletes)}
