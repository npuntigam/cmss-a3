from __future__ import annotations
from typing import Iterable
import networkx as nx
from mesa import Agent, Model


class ThresholdAgent(Agent):
    def __init__(self, model: "ThresholdNetworkModel", node_id: int, initially_active: bool = False) -> None:
        super().__init__(model)

        self.node_id = node_id
        self.active = initially_active
        self.next_active = initially_active

    def determine_next_state(self) -> None:
        if self.active:
            self.next_active = True
            return

        neighbors = list(self.model.graph.neighbors(self.node_id))

        if not neighbors:
            self.next_active = False
            return

        active_neighbors = sum(
            1
            for neighbor_id in neighbors
            if self.model.node_to_agent[neighbor_id].active
        )

        active_fraction = active_neighbors / len(neighbors)
        self.next_active = active_fraction >= self.model.threshold

    def advance_state(self) -> None:
        self.active = self.next_active


class ThresholdNetworkModel(Model):
    def __init__(self, graph: nx.Graph, threshold: float, initial_seeds: Iterable[int], random_seed: int | None = None) -> None:
        super().__init__(rng=random_seed)

        self.graph = graph
        self.threshold = threshold
        self.initial_seeds = set(initial_seeds)
        self._validate_inputs()
        self.node_to_agent: dict[int, ThresholdAgent] = {}

        for node_id in self.graph.nodes:
            agent = ThresholdAgent(
                model=self,
                node_id=node_id,
                initially_active=node_id in self.initial_seeds,
            )
            self.node_to_agent[node_id] = agent

        self.steps = 0
        self.running = True

    def step(self) -> None:
        active_before = self.active_count

        for agent in self.node_to_agent.values():
            agent.determine_next_state()

        for agent in self.node_to_agent.values():
            agent.advance_state()

        self.steps += 1

        active_after = self.active_count

        if active_after == active_before:
            self.running = False

    def run_until_stable(self, max_steps: int) -> None:
        if max_steps <= 0:
            raise ValueError("max_steps must be > 0.")

        while self.running and self.steps < max_steps:
            self.step()

    @property
    def active_count(self) -> int:
        return sum(agent.active for agent in self.node_to_agent.values())

    @property
    def active_share(self) -> float:
        return self.active_count / self.graph.number_of_nodes()

    @property
    def adoption_states(self) -> dict[int, bool]:
        return {
            node_id: agent.active
            for node_id, agent in self.node_to_agent.items()
        }

    def _validate_inputs(self) -> None:
        if not 0 <= self.threshold <= 1:
            raise ValueError(
                f"threshold must be between 0 & 1, got {self.threshold}."
            )

        graph_nodes = set(self.graph.nodes)

        if not self.initial_seeds:
            raise ValueError("initial_seeds must contain at least one node.")

        unknown_seeds = self.initial_seeds - graph_nodes

        if unknown_seeds:
            raise ValueError(
                f"initial_seeds contains nodes that are not in the graph: "
                f"{sorted(unknown_seeds)}"
            )