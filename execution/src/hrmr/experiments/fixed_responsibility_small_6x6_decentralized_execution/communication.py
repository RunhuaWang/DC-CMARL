"""同步可靠邻居通信下的带标签成本信息洪泛。"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
Graph = tuple[tuple[int, ...], ...]


def build_undirected_graph(num_agents: int, edges: Sequence[Sequence[int]]) -> Graph:
    """由 0-based 边表建立静态、无向、连通且无自环的通信图。"""

    if isinstance(num_agents, bool) or not isinstance(num_agents, Integral):
        raise TypeError("num_agents must be an integer")
    checked_agents = int(num_agents)
    if checked_agents <= 0:
        raise ValueError("num_agents must be positive")
    if isinstance(edges, (str, bytes)):
        raise TypeError("edges must be a sequence of endpoint pairs")
    neighbors = [set() for _ in range(checked_agents)]
    seen: set[tuple[int, int]] = set()
    for edge in edges:
        pair = tuple(edge)
        if len(pair) != 2 or any(
            isinstance(value, bool) or not isinstance(value, Integral) for value in pair
        ):
            raise ValueError("every edge must contain two integer endpoints")
        left, right = (int(pair[0]), int(pair[1]))
        if not 0 <= left < checked_agents or not 0 <= right < checked_agents:
            raise ValueError("edge endpoint is outside the agent range")
        if left == right:
            raise ValueError("communication graph cannot contain self loops")
        normalized = (min(left, right), max(left, right))
        if normalized in seen:
            raise ValueError("communication graph contains a duplicate edge")
        seen.add(normalized)
        neighbors[left].add(right)
        neighbors[right].add(left)
    graph = tuple(tuple(sorted(items)) for items in neighbors)
    if graph_diameter(graph) == 0 and checked_agents > 1:
        raise ValueError("communication graph must be connected")
    return graph


def _validate_graph(graph: Graph) -> None:
    if not isinstance(graph, tuple) or not graph:
        raise ValueError("graph must be a non-empty tuple of neighbor tuples")
    num_agents = len(graph)
    for agent, neighbors in enumerate(graph):
        if not isinstance(neighbors, tuple):
            raise TypeError("every neighbor collection must be a tuple")
        if len(set(neighbors)) != len(neighbors):
            raise ValueError("neighbor collections cannot contain duplicates")
        for neighbor in neighbors:
            if isinstance(neighbor, bool) or not isinstance(neighbor, Integral):
                raise TypeError("neighbor indices must be integers")
            checked = int(neighbor)
            if not 0 <= checked < num_agents or checked == agent:
                raise ValueError("neighbor index is invalid")
            if agent not in graph[checked]:
                raise ValueError("communication graph must be undirected")


def graph_diameter(graph: Graph) -> int:
    """返回连通无向图直径；单节点图直径为零，非连通图也返回零。"""

    _validate_graph_structure_only(graph)
    num_agents = len(graph)
    if num_agents == 1:
        return 0
    diameter = 0
    for source in range(num_agents):
        distances = [-1] * num_agents
        distances[source] = 0
        queue = deque((source,))
        while queue:
            current = queue.popleft()
            for neighbor in graph[current]:
                if distances[neighbor] < 0:
                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)
        if any(distance < 0 for distance in distances):
            return 0
        diameter = max(diameter, max(distances))
    return diameter


def _validate_graph_structure_only(graph: Graph) -> None:
    if not isinstance(graph, tuple) or not graph:
        raise ValueError("graph must be a non-empty tuple")
    num_agents = len(graph)
    for agent, neighbors in enumerate(graph):
        if not isinstance(neighbors, tuple):
            raise TypeError("every neighbor collection must be a tuple")
        for neighbor in neighbors:
            if isinstance(neighbor, bool) or not isinstance(neighbor, Integral):
                raise TypeError("neighbor indices must be integers")
            checked = int(neighbor)
            if not 0 <= checked < num_agents or checked == agent:
                raise ValueError("neighbor index is invalid")
            if agent not in graph[checked]:
                raise ValueError("communication graph must be undirected")


def flood_tagged_values(
    local_values: ArrayLike,
    graph: Graph,
    rounds: int,
) -> tuple[tuple[Mapping[int, float], ...], ...]:
    """同步执行指定轮数的集合并集洪泛，并保留每轮每个 agent 的信息集。"""

    _validate_graph(graph)
    raw = np.asarray(local_values)
    if raw.shape != (len(graph),) or raw.dtype.kind not in {"i", "u", "f"}:
        raise ValueError("local_values must be a numeric vector with one value per agent")
    values = raw.astype(np.float64, copy=True)
    if not np.all(np.isfinite(values)):
        raise ValueError("local_values must be finite")
    if isinstance(rounds, bool) or not isinstance(rounds, Integral) or int(rounds) < 0:
        raise ValueError("rounds must be a non-negative integer")

    information: tuple[dict[int, float], ...] = tuple(
        {agent: float(values[agent])} for agent in range(len(graph))
    )
    history: list[tuple[Mapping[int, float], ...]] = [information]
    for _ in range(int(rounds)):
        next_information: list[dict[int, float]] = []
        for agent, neighbors in enumerate(graph):
            merged: dict[int, float] = {}
            for source in (agent, *neighbors):
                for owner, value in information[source].items():
                    previous = merged.get(owner)
                    if previous is not None and not np.isclose(
                        previous, value, rtol=0.0, atol=1e-12
                    ):
                        raise ValueError("conflicting values received for the same agent tag")
                    merged[owner] = value
            next_information.append(merged)
        information = tuple(next_information)
        history.append(information)
    return tuple(history)


def aggregate_global_costs(
    local_cost_estimates: ArrayLike,
    graph: Graph,
    rounds: int,
) -> FloatArray:
    """由每个 agent 的洪泛结果独立计算 global cost estimate。"""

    history = flood_tagged_values(local_cost_estimates, graph, rounds)
    final = history[-1]
    num_agents = len(graph)
    expected_tags = set(range(num_agents))
    if any(set(information) != expected_tags for information in final):
        raise ValueError("communication rounds are insufficient to recover all local costs")
    estimates = np.asarray(
        [sum(information.values()) for information in final],
        dtype=np.float64,
    )
    estimates.setflags(write=False)
    return estimates


__all__ = [
    "Graph",
    "aggregate_global_costs",
    "build_undirected_graph",
    "flood_tagged_values",
    "graph_diameter",
]
