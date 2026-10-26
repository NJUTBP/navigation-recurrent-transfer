from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import networkx as nx
import numpy as np

from .common import GraphWalkDataset, random_walk_labeled_graph


@dataclass
class SocialHierarchyTask:
    n_levels: int = 4
    seed: int = 42

    def build_graph(self) -> Tuple[nx.DiGraph, Dict[str, int], Dict[str, int], Dict[str, np.ndarray]]:
        graph = nx.DiGraph()
        node_to_id: Dict[str, int] = {}
        node_coords: Dict[str, np.ndarray] = {}
        actions = {
            "up_competence": 0,
            "down_competence": 1,
            "up_popularity": 2,
            "down_popularity": 3,
        }
        idx = 0
        for i in range(self.n_levels):
            for j in range(self.n_levels):
                name = f"p_{i}_{j}"
                graph.add_node(name)
                node_to_id[name] = idx
                node_coords[name] = np.asarray([float(i), float(j)], dtype=np.float32)
                idx += 1

        for i in range(self.n_levels):
            for j in range(self.n_levels):
                src = f"p_{i}_{j}"
                if i + 1 < self.n_levels:
                    graph.add_edge(src, f"p_{i+1}_{j}", action="up_competence")
                if i - 1 >= 0:
                    graph.add_edge(src, f"p_{i-1}_{j}", action="down_competence")
                if j + 1 < self.n_levels:
                    graph.add_edge(src, f"p_{i}_{j+1}", action="up_popularity")
                if j - 1 >= 0:
                    graph.add_edge(src, f"p_{i}_{j-1}", action="down_popularity")
        return graph, node_to_id, actions, node_coords

    def make_dataset(self, num_walks: int, seq_len: int) -> GraphWalkDataset:
        graph, node_to_id, action_to_id, node_coords = self.build_graph()
        return random_walk_labeled_graph(
            graph=graph,
            node_to_id=node_to_id,
            action_to_id=action_to_id,
            node_coords=node_coords,
            num_walks=num_walks,
            seq_len=seq_len,
            seed=self.seed,
        )
