"""Patch construction for Graph-JEPA v2.

The upstream Graph-JEPA implementation works over graph patches/subgraphs.  For
clinical KGs we build those patches on the fly with balanced multi-source BFS,
then derive a small coarsened patch graph and random-walk style positional
features.  The code is deliberately dependency-light: no METIS or torch-scatter
is required, because patient KGs are small enough for direct tensor operations.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import List, Sequence

import torch


@dataclass(frozen=True)
class PatchData:
    """A patched view of one patient graph tensor object."""

    assignment: torch.Tensor       # [num_nodes], node -> patch id
    patch_nodes: List[List[int]]
    patch_edge_index: torch.Tensor # [2, num_patch_edges]
    patch_adj: torch.Tensor        # [num_patches, num_patches], dense 0/1
    patch_pos: torch.Tensor        # [num_patches, patch_pe_dim]
    patch_mask: torch.Tensor       # [num_patches], all True for unpadded v2

    @property
    def num_patches(self) -> int:
        return len(self.patch_nodes)

    def to(self, device: torch.device | str) -> "PatchData":
        return PatchData(
            assignment=self.assignment.to(device),
            patch_nodes=self.patch_nodes,
            patch_edge_index=self.patch_edge_index.to(device),
            patch_adj=self.patch_adj.to(device),
            patch_pos=self.patch_pos.to(device),
            patch_mask=self.patch_mask.to(device),
        )


@dataclass(frozen=True)
class PatchTask:
    """A single context-to-target patch prediction task."""

    context_idx: torch.Tensor
    target_idx: torch.Tensor

    def to(self, device: torch.device | str) -> "PatchTask":
        return PatchTask(
            context_idx=self.context_idx.to(device),
            target_idx=self.target_idx.to(device),
        )


def _edge_lists(edge_index: torch.Tensor, num_nodes: int) -> List[List[int]]:
    adj: List[List[int]] = [[] for _ in range(num_nodes)]
    if edge_index.numel() == 0:
        return adj
    src = edge_index[0].detach().cpu().tolist()
    dst = edge_index[1].detach().cpu().tolist()
    for s, t in zip(src, dst):
        if 0 <= s < num_nodes and 0 <= t < num_nodes:
            adj[s].append(t)
            adj[t].append(s)
    return adj


def _randperm(n: int, generator: torch.Generator | None) -> List[int]:
    return torch.randperm(n, generator=generator).tolist()


def _bfs_distances(adj: Sequence[Sequence[int]], seed: int) -> List[int]:
    dist = [-1] * len(adj)
    dist[seed] = 0
    q: deque[int] = deque([seed])
    while q:
        cur = q.popleft()
        for nb in adj[cur]:
            if dist[nb] < 0:
                dist[nb] = dist[cur] + 1
                q.append(nb)
    return dist


def _choose_seeds(
    adj: Sequence[Sequence[int]],
    num_patches: int,
    generator: torch.Generator | None,
) -> List[int]:
    n = len(adj)
    if num_patches >= n:
        return list(range(n))

    tie_rank = {node: i for i, node in enumerate(_randperm(n, generator))}
    degrees = [len(a) for a in adj]
    first = max(range(n), key=lambda u: (degrees[u], -tie_rank[u]))
    seeds = [first]
    best_dist = _bfs_distances(adj, first)
    best_dist = [d if d >= 0 else n + 1 for d in best_dist]

    while len(seeds) < num_patches:
        selected = set(seeds)
        nxt = max(
            (u for u in range(n) if u not in selected),
            key=lambda u: (best_dist[u], degrees[u], -tie_rank[u]),
        )
        seeds.append(nxt)
        dist = _bfs_distances(adj, nxt)
        for i, d in enumerate(dist):
            if d >= 0:
                best_dist[i] = min(best_dist[i], d)
    return seeds


def balanced_bfs_partition(
    edge_index: torch.Tensor,
    num_nodes: int,
    num_patches: int,
    generator: torch.Generator | None = None,
) -> List[List[int]]:
    """Partition nodes into connected-ish, balanced patches.

    Multi-source BFS keeps patches local.  Disconnected components are handled by
    reseeding any unassigned node into the currently smallest patch.
    """
    if num_nodes <= 0:
        return []
    p = max(1, min(num_patches, num_nodes))
    if p == num_nodes:
        return [[i] for i in range(num_nodes)]

    adj = _edge_lists(edge_index, num_nodes)
    seeds = _choose_seeds(adj, p, generator)
    assignment = [-1] * num_nodes
    queues: List[deque[int]] = [deque() for _ in range(p)]
    counts = [0] * p
    target_size = max(1, (num_nodes + p - 1) // p)

    for pid, seed in enumerate(seeds):
        assignment[seed] = pid
        queues[pid].append(seed)
        counts[pid] = 1

    remaining = num_nodes - p
    while remaining > 0:
        progressed = False
        for pid in range(p):
            if counts[pid] >= target_size and any(c < target_size for c in counts):
                continue
            while queues[pid]:
                cur = queues[pid].popleft()
                neighbors = list(adj[cur])
                if neighbors:
                    order = _randperm(len(neighbors), generator)
                    neighbors = [neighbors[i] for i in order]
                for nb in neighbors:
                    if assignment[nb] == -1:
                        assignment[nb] = pid
                        queues[pid].append(nb)
                        counts[pid] += 1
                        remaining -= 1
                        progressed = True
                        break
                if progressed:
                    break
        if not progressed:
            unassigned = [i for i, a in enumerate(assignment) if a == -1]
            if not unassigned:
                break
            node = unassigned[0]
            pid = min(range(p), key=lambda j: counts[j])
            assignment[node] = pid
            queues[pid].append(node)
            counts[pid] += 1
            remaining -= 1

    patches: List[List[int]] = [[] for _ in range(p)]
    for node, pid in enumerate(assignment):
        patches[pid].append(node)
    return [nodes for nodes in patches if nodes]


def _coarsen_edges(
    edge_index: torch.Tensor,
    assignment: torch.Tensor,
    num_patches: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    adj = torch.zeros((num_patches, num_patches), dtype=torch.float32)
    if edge_index.numel():
        src = edge_index[0].detach().cpu()
        dst = edge_index[1].detach().cpu()
        assn = assignment.detach().cpu()
        for s, t in zip(src.tolist(), dst.tolist()):
            ps = int(assn[s])
            pt = int(assn[t])
            if ps != pt:
                adj[ps, pt] = 1.0
                adj[pt, ps] = 1.0
    edges = adj.nonzero(as_tuple=False).t().contiguous()
    if edges.numel() == 0:
        edges = torch.zeros((2, 0), dtype=torch.long)
    return edges.long(), adj


def _patch_positional_features(
    patch_adj: torch.Tensor,
    patch_nodes: Sequence[Sequence[int]],
    num_nodes: int,
    pe_dim: int,
) -> torch.Tensor:
    p = len(patch_nodes)
    if pe_dim <= 0:
        return torch.zeros((p, 0), dtype=torch.float32)

    pos = torch.zeros((p, pe_dim), dtype=torch.float32)
    sizes = torch.tensor([len(nodes) for nodes in patch_nodes], dtype=torch.float32)
    pos[:, 0] = sizes / max(1, num_nodes)

    if pe_dim > 1:
        degree = patch_adj.sum(dim=-1)
        pos[:, 1] = degree / max(1, p - 1)

    if pe_dim > 2:
        walk_adj = patch_adj + torch.eye(p)
        denom = walk_adj.sum(dim=-1, keepdim=True).clamp_min(1.0)
        transition = walk_adj / denom
        power = transition.clone()
        for col in range(2, pe_dim):
            pos[:, col] = torch.diagonal(power)
            power = power @ transition
    return pos


def build_patch_data(
    data,
    *,
    num_patches: int,
    patch_pe_dim: int,
    generator: torch.Generator | None = None,
) -> PatchData:
    """Build patches for a single v2 ``GraphData`` object."""
    device = data.x.device
    n = int(data.num_nodes)
    patches = balanced_bfs_partition(
        data.edge_index.detach().cpu(), n, num_patches, generator
    )
    if not patches:
        empty_long = torch.zeros((0,), dtype=torch.long, device=device)
        empty_edges = torch.zeros((2, 0), dtype=torch.long, device=device)
        empty_adj = torch.zeros((0, 0), dtype=torch.float32, device=device)
        empty_pos = torch.zeros((0, patch_pe_dim), dtype=torch.float32, device=device)
        return PatchData(empty_long, [], empty_edges, empty_adj, empty_pos, empty_long.bool())

    assignment = torch.empty(n, dtype=torch.long)
    for pid, nodes in enumerate(patches):
        assignment[nodes] = pid
    patch_edges, patch_adj = _coarsen_edges(data.edge_index, assignment, len(patches))
    patch_pos = _patch_positional_features(patch_adj, patches, n, patch_pe_dim)
    patch_mask = torch.ones(len(patches), dtype=torch.bool)

    return PatchData(
        assignment=assignment.to(device),
        patch_nodes=patches,
        patch_edge_index=patch_edges.to(device),
        patch_adj=patch_adj.to(device),
        patch_pos=patch_pos.to(device),
        patch_mask=patch_mask.to(device),
    )


def pool_nodes_to_patches(node_z: torch.Tensor, patch_data: PatchData) -> torch.Tensor:
    """Mean-pool node embeddings into patch embeddings."""
    p = patch_data.num_patches
    if p == 0:
        return node_z.new_zeros((0, node_z.size(-1)))
    out = node_z.new_zeros((p, node_z.size(-1)))
    counts = node_z.new_zeros((p, 1))
    assignment = patch_data.assignment.to(node_z.device)
    out.index_add_(0, assignment, node_z)
    counts.index_add_(0, assignment, node_z.new_ones((node_z.size(0), 1)))
    return out / counts.clamp_min(1.0)


def _connected_patch_sample(
    patch_adj: torch.Tensor,
    count: int,
    generator: torch.Generator | None,
) -> List[int]:
    p = patch_adj.size(0)
    count = max(1, min(count, p))
    seed = int(torch.randint(0, p, (1,), generator=generator).item())
    selected = [seed]
    seen = {seed}
    q: deque[int] = deque([seed])
    adj = patch_adj.detach().cpu()
    while q and len(selected) < count:
        cur = q.popleft()
        neighbors = adj[cur].nonzero(as_tuple=False).flatten().tolist()
        if neighbors:
            order = _randperm(len(neighbors), generator)
            neighbors = [neighbors[i] for i in order]
        for nb in neighbors:
            if nb not in seen:
                seen.add(nb)
                selected.append(nb)
                q.append(nb)
                if len(selected) >= count:
                    break
    while len(selected) < count:
        candidate = int(torch.randint(0, p, (1,), generator=generator).item())
        if candidate not in seen:
            seen.add(candidate)
            selected.append(candidate)
    return selected


def sample_patch_task(
    patch_data: PatchData,
    *,
    context_patches: int,
    target_patches: int,
    generator: torch.Generator | None = None,
) -> PatchTask:
    """Sample one context-to-target patch prediction task."""
    p = patch_data.num_patches
    if p == 0:
        z = torch.zeros((0,), dtype=torch.long, device=patch_data.patch_adj.device)
        return PatchTask(z, z)
    if p == 1:
        one = torch.zeros((1,), dtype=torch.long, device=patch_data.patch_adj.device)
        return PatchTask(one, one)

    target = _connected_patch_sample(patch_data.patch_adj, target_patches, generator)
    target_set = set(target)

    adj = patch_data.patch_adj.detach().cpu()
    neighbors = sorted({
        int(nb)
        for t in target
        for nb in adj[t].nonzero(as_tuple=False).flatten().tolist()
        if int(nb) not in target_set
    })
    candidates = neighbors or [i for i in range(p) if i not in target_set]
    if not candidates:
        candidates = list(range(p))

    c = max(1, min(context_patches, len(candidates)))
    order = _randperm(len(candidates), generator)
    context = [candidates[i] for i in order[:c]]

    device = patch_data.patch_adj.device
    return PatchTask(
        context_idx=torch.tensor(context, dtype=torch.long, device=device),
        target_idx=torch.tensor(target, dtype=torch.long, device=device),
    )


def visible_mask(num_patches: int, context_idx: torch.Tensor) -> torch.Tensor:
    mask = torch.zeros(num_patches, dtype=torch.bool, device=context_idx.device)
    if context_idx.numel():
        mask[context_idx] = True
    return mask
