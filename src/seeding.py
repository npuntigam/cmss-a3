from __future__ import annotations
import random
from collections import deque
import networkx as nx


def select_high_degree_seeds(graph: nx.Graph, seed_size: int) -> list[int]:
    _validate_seed_size(graph, seed_size)

    nodes_by_degree = sorted(
        graph.nodes,
        key=lambda node: (-graph.degree[node], node),
    )

    return list(nodes_by_degree[:seed_size])


def select_clustered_seeds(graph: nx.Graph, seed_size: int, rng: random.Random) -> list[int]:
    _validate_seed_size(graph, seed_size)

    focal_node = rng.choice(list(graph.nodes))

    if seed_size == 1:
        return [focal_node]

    local_candidates = _collect_local_candidate_pool(
        graph=graph,
        focal_node=focal_node,
        required_count=seed_size - 1,
    )

    selected_neighbors = rng.sample(local_candidates, seed_size - 1)

    return [focal_node, *selected_neighbors]


def _collect_local_candidate_pool(graph: nx.Graph, focal_node: int, required_count: int) -> list[int]:
    visited = {focal_node}
    queue: deque[tuple[int, int]] = deque([(focal_node, 0)])

    candidates: list[int] = []
    current_distance = 0

    while queue:
        current_node, distance = queue.popleft()

        if distance > current_distance:
            if len(candidates) >= required_count:
                break

            current_distance = distance

        for neighbor in sorted(graph.neighbors(current_node)):
            if neighbor in visited:
                continue

            visited.add(neighbor)
            queue.append((neighbor, distance + 1))
            candidates.append(neighbor)

    if len(candidates) < required_count:
        raise ValueError(
            "Could not collect enough local candidates. "
            "Check whether the graph is connected or reduce seed_size."
        )

    return candidates


def _validate_seed_size(graph: nx.Graph, seed_size: int) -> None:
    if seed_size <= 0:
        raise ValueError("seed_size must be > 0.")

    if seed_size > graph.number_of_nodes():
        raise ValueError(
            f"seed_size={seed_size} is larger than the number of nodes "
            f"n={graph.number_of_nodes()}."
        )