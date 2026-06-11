"""Patch construction for Graph-JEPA v3.

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
    """A patched view of one graph or a PyG mini-batch of graphs."""

    assignment: torch.Tensor       # [num_nodes], node -> patch id
    patch_nodes: List[List[int]]
    patch_edge_index: torch.Tensor # [2, num_patch_edges]
    patch_adj: torch.Tensor        # [num_patches, num_patches], dense 0/1
    patch_pos: torch.Tensor        # [num_patches, patch_pe_dim]
    patch_mask: torch.Tensor       # [num_patches], all True for unpadded v3
    patch_ptr: torch.Tensor | None = None # [num_graphs + 1], patch offsets

    @property
    def num_patches(self) -> int:
        return len(self.patch_nodes)

    @property
    def num_graphs(self) -> int:
        if self.patch_ptr is None:
            return 1
        return max(0, int(self.patch_ptr.numel()) - 1)

    def to(self, device: torch.device | str) -> "PatchData":
        return PatchData(
            assignment=self.assignment.to(device),
            patch_nodes=self.patch_nodes,
            patch_edge_index=self.patch_edge_index.to(device),
            patch_adj=self.patch_adj.to(device),
            patch_pos=self.patch_pos.to(device),
            patch_mask=self.patch_mask.to(device),
            patch_ptr=self.patch_ptr.to(device) if self.patch_ptr is not None else None,
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


def _build_local_patch_tensors(
    edge_index: torch.Tensor,
    num_nodes: int,
    num_patches: int,
    patch_pe_dim: int,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, List[List[int]], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    patches = balanced_bfs_partition(
        edge_index.detach().cpu(), num_nodes, num_patches, generator
    )
    if not patches:
        empty_long = torch.zeros((0,), dtype=torch.long)
        empty_edges = torch.zeros((2, 0), dtype=torch.long)
        empty_adj = torch.zeros((0, 0), dtype=torch.float32)
        empty_pos = torch.zeros((0, patch_pe_dim), dtype=torch.float32)
        return empty_long, [], empty_edges, empty_adj, empty_pos, empty_long.bool()

    assignment = torch.empty(num_nodes, dtype=torch.long)
    for pid, nodes in enumerate(patches):
        assignment[nodes] = pid
    patch_edges, patch_adj = _coarsen_edges(edge_index, assignment, len(patches))
    patch_pos = _patch_positional_features(patch_adj, patches, num_nodes, patch_pe_dim)
    patch_mask = torch.ones(len(patches), dtype=torch.bool)

    return assignment, patches, patch_edges, patch_adj, patch_pos, patch_mask


def _build_batched_patch_data(
    data,
    *,
    num_patches: int,
    patch_pe_dim: int,
    generator: torch.Generator | None,
) -> PatchData:
    device = data.x.device
    ptr = data.ptr.detach().cpu()
    edge_index = data.edge_index.detach().cpu()
    assignment = torch.empty(int(data.num_nodes), dtype=torch.long)
    patch_nodes: List[List[int]] = []
    patch_edges: List[torch.Tensor] = []
    patch_adjs: List[torch.Tensor] = []
    patch_pos: List[torch.Tensor] = []
    patch_masks: List[torch.Tensor] = []
    patch_ptr = [0]
    patch_offset = 0

    for graph_idx in range(max(0, ptr.numel() - 1)):
        start = int(ptr[graph_idx])
        end = int(ptr[graph_idx + 1])
        n = end - start
        if n <= 0:
            patch_ptr.append(patch_offset)
            continue

        src, dst = edge_index[0], edge_index[1]
        edge_mask = (src >= start) & (src < end) & (dst >= start) & (dst < end)
        local_edge_index = edge_index[:, edge_mask] - start
        local_assignment, local_nodes, local_edges, local_adj, local_pos, local_mask = (
            _build_local_patch_tensors(
                local_edge_index,
                n,
                num_patches,
                patch_pe_dim,
                generator,
            )
        )

        assignment[start:end] = local_assignment + patch_offset
        patch_nodes.extend([[start + node for node in nodes] for nodes in local_nodes])
        if local_edges.numel():
            patch_edges.append(local_edges + patch_offset)
        patch_adjs.append(local_adj)
        patch_pos.append(local_pos)
        patch_masks.append(local_mask)
        patch_offset += len(local_nodes)
        patch_ptr.append(patch_offset)

    if patch_offset == 0:
        empty_long = torch.zeros((0,), dtype=torch.long, device=device)
        empty_edges = torch.zeros((2, 0), dtype=torch.long, device=device)
        empty_adj = torch.zeros((0, 0), dtype=torch.float32, device=device)
        empty_pos = torch.zeros((0, patch_pe_dim), dtype=torch.float32, device=device)
        return PatchData(
            empty_long,
            [],
            empty_edges,
            empty_adj,
            empty_pos,
            empty_long.bool(),
            torch.tensor(patch_ptr, dtype=torch.long, device=device),
        )

    if patch_edges:
        batched_edges = torch.cat(patch_edges, dim=1)
    else:
        batched_edges = torch.zeros((2, 0), dtype=torch.long)
    batched_adj = torch.block_diag(*patch_adjs)

    return PatchData(
        assignment=assignment.to(device),
        patch_nodes=patch_nodes,
        patch_edge_index=batched_edges.to(device),
        patch_adj=batched_adj.to(device),
        patch_pos=torch.cat(patch_pos, dim=0).to(device),
        patch_mask=torch.cat(patch_masks, dim=0).to(device),
        patch_ptr=torch.tensor(patch_ptr, dtype=torch.long, device=device),
    )


def build_patch_data(
    data,
    *,
    num_patches: int,
    patch_pe_dim: int,
    generator: torch.Generator | None = None,
) -> PatchData:
    """Build patches for a v3 graph object or PyG mini-batch."""
    batch = getattr(data, "batch", None)
    ptr = getattr(data, "ptr", None)
    if batch is not None and ptr is not None:
        return _build_batched_patch_data(
            data,
            num_patches=num_patches,
            patch_pe_dim=patch_pe_dim,
            generator=generator,
        )

    device = data.x.device
    n = int(data.num_nodes)
    assignment, patches, patch_edges, patch_adj, patch_pos, patch_mask = (
        _build_local_patch_tensors(
            data.edge_index,
            n,
            num_patches,
            patch_pe_dim,
            generator,
        )
    )
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
    if patch_data.patch_ptr is not None:
        contexts: List[int] = []
        targets: List[int] = []
        ptr = patch_data.patch_ptr.detach().cpu().tolist()
        adj = patch_data.patch_adj.detach().cpu()
        for lo, hi in zip(ptr[:-1], ptr[1:]):
            local_context, local_target = _sample_patch_indices(
                adj[lo:hi, lo:hi],
                context_patches=context_patches,
                target_patches=target_patches,
                generator=generator,
            )
            contexts.extend(lo + idx for idx in local_context)
            targets.extend(lo + idx for idx in local_target)
        device = patch_data.patch_adj.device
        return PatchTask(
            context_idx=torch.tensor(contexts, dtype=torch.long, device=device),
            target_idx=torch.tensor(targets, dtype=torch.long, device=device),
        )

    context, target = _sample_patch_indices(
        patch_data.patch_adj,
        context_patches=context_patches,
        target_patches=target_patches,
        generator=generator,
    )
    device = patch_data.patch_adj.device
    return PatchTask(
        context_idx=torch.tensor(context, dtype=torch.long, device=device),
        target_idx=torch.tensor(target, dtype=torch.long, device=device),
    )


def _sample_patch_indices(
    patch_adj: torch.Tensor,
    *,
    context_patches: int,
    target_patches: int,
    generator: torch.Generator | None = None,
) -> tuple[List[int], List[int]]:
    p = patch_adj.size(0)
    if p < 2:
        return [], []

    target = _connected_patch_sample(patch_adj, target_patches, generator)
    target_set = set(target)

    adj = patch_adj.detach().cpu()
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
    return context, target


def visible_mask(num_patches: int, context_idx: torch.Tensor) -> torch.Tensor:
    mask = torch.zeros(num_patches, dtype=torch.bool, device=context_idx.device)
    if context_idx.numel():
        mask[context_idx] = True
    return mask
