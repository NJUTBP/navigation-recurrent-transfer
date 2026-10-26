from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import networkx as nx
import numpy as np

from .common import GraphWalkDataset, random_walk_labeled_graph


@dataclass
class TransitiveInferenceTask:
    n_items: int = 7
    seed: int = 42

    def build_graph(self) -> Tuple[nx.DiGraph, Dict[str, int], Dict[str, int], Dict[str, np.ndarray]]:
        graph = nx.DiGraph()
        actions = {"higher": 0, "lower": 1}
        nodes = [chr(ord("A") + i) for i in range(self.n_items)]
        node_to_id = {name: idx for idx, name in enumerate(nodes)}
        node_coords = {name: np.asarray([float(idx), 0.0], dtype=np.float32) for idx, name in enumerate(nodes)}
        graph.add_nodes_from(nodes)
        for i in range(self.n_items - 1):
            high = nodes[i]
            low = nodes[i + 1]
            graph.add_edge(low, high, action="higher")
            graph.add_edge(high, low, action="lower")
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
