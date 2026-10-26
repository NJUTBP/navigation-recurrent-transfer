"""Family tree task with optional path-length-based train/test split.

Changes vs original
-------------------
- Added ``make_split_dataset(max_train_len, test_len_range)`` that generates
  short-path episodes for training and long-path episodes for testing,
  enabling generalization tests to indirect relations.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import networkx as nx
import numpy as np
import torch
from .common import GraphEpisodeSample, GraphWalkDataset, random_walk_labeled_graph


@dataclass
class FamilyTreeTask:
    seed: int = 42

    def build_graph(self) -> Tuple[nx.DiGraph, Dict[str, int], Dict[str, int], Dict[str, np.ndarray]]:
        graph = nx.DiGraph()
        actions = {
            "parent": 0, "child": 1, "sibling": 2,
            "spouse": 3, "cousin": 4, "uncle_aunt": 5,
        }
        nodes = [
            "grandfather", "grandmother", "father", "mother", "uncle", "aunt",
            "self", "sibling", "cousin1", "cousin2",
        ]
        node_to_id = {name: idx for idx, name in enumerate(nodes)}
        node_coords = {
            name: np.asarray([idx % 4, idx // 4], dtype=np.float32)
            for idx, name in enumerate(nodes)
        }
        graph.add_nodes_from(nodes)

        edges = [
            ("father", "grandfather", "parent"),
            ("father", "grandmother", "parent"),
            ("mother", "grandfather", "parent"),
            ("mother", "grandmother", "parent"),
            ("uncle", "grandfather", "parent"),
            ("uncle", "grandmother", "parent"),
            ("self", "father", "parent"),
            ("self", "mother", "parent"),
            ("sibling", "father", "parent"),
            ("sibling", "mother", "parent"),
            ("cousin1", "uncle", "parent"),
            ("cousin2", "uncle", "parent"),
        ]
        for child, parent, action in edges:
            graph.add_edge(child, parent, action=action)
            graph.add_edge(parent, child, action="child")

        sibling_pairs = [("self", "sibling"), ("cousin1", "cousin2"), ("father", "uncle")]
        for a, b in sibling_pairs:
            graph.add_edge(a, b, action="sibling")
            graph.add_edge(b, a, action="sibling")

        spouse_pairs = [("father", "mother"), ("uncle", "aunt"), ("grandfather", "grandmother")]
        for a, b in spouse_pairs:
            graph.add_edge(a, b, action="spouse")
            graph.add_edge(b, a, action="spouse")

        graph.add_edge("self", "cousin1", action="cousin")
        graph.add_edge("self", "cousin2", action="cousin")
        graph.add_edge("cousin1", "self", action="cousin")
        graph.add_edge("cousin2", "self", action="cousin")
        graph.add_edge("self", "uncle", action="uncle_aunt")
        graph.add_edge("sibling", "uncle", action="uncle_aunt")
        graph.add_edge("cousin1", "father", action="uncle_aunt")

        return graph, node_to_id, actions, node_coords

    # ---- original random-walk dataset (unchanged) -----------------------

    def make_dataset(self, num_walks: int, seq_len: int) -> GraphWalkDataset:
        graph, node_to_id, action_to_id, node_coords = self.build_graph()
        return random_walk_labeled_graph(
            graph=graph, node_to_id=node_to_id, action_to_id=action_to_id,
            node_coords=node_coords, num_walks=num_walks, seq_len=seq_len,
            seed=self.seed,
        )

    # ---- NEW: path-length-split dataset ---------------------------------

    def make_split_dataset(
        self,
        num_walks_train: int = 500,
        num_walks_test: int = 200,
        max_train_len: int = 2,
        test_len_min: int = 3,
        test_len_max: int = 4,
    ) -> Tuple[GraphWalkDataset, GraphWalkDataset]:
        """Generate train (short paths) and test (long paths) datasets.

        Training episodes have ``seq_len = max_train_len`` (default 2).
        Test episodes have ``seq_len`` uniformly sampled from
        ``[test_len_min, test_len_max]``.

        This tests whether the model can **generalize** from direct relations
        to indirect, multi-hop relation chains.
        """
        graph, node_to_id, action_to_id, node_coords = self.build_graph()
        train_ds = random_walk_labeled_graph(
            graph=graph, node_to_id=node_to_id, action_to_id=action_to_id,
            node_coords=node_coords, num_walks=num_walks_train,
            seq_len=max_train_len, seed=self.seed,
        )

        # for test set, generate walks with varying lengths; pad shorter ones
        rng = np.random.default_rng(self.seed + 1000)
        nodes = list(graph.nodes())
        samples: List[GraphEpisodeSample] = []
        for _ in range(num_walks_test):
            seq_len = int(rng.integers(test_len_min, test_len_max + 1))
            current = rng.choice(nodes)
            obs_ids, action_ids, target_ids, coords = [], [], [], []
            for _ in range(seq_len):
                successors = list(graph.successors(current))
                if not successors:
                    current = rng.choice(nodes)
                    successors = list(graph.successors(current))
                nxt = rng.choice(successors)
                edge_data = graph.get_edge_data(current, nxt)
                obs_ids.append(node_to_id[current])
                action_ids.append(action_to_id[edge_data["action"]])
                target_ids.append(node_to_id[nxt])
                coords.append(node_coords[current])
                current = nxt
            # Repeat values for rectangular batching, but expose an explicit mask
            # so the repeated tail is never treated as additional supervision or
            # neural activity.
            valid_mask = [True] * len(obs_ids)
            while len(obs_ids) < test_len_max:
                obs_ids.append(obs_ids[-1])
                action_ids.append(action_ids[-1])
                target_ids.append(target_ids[-1])
                coords.append(coords[-1])
                valid_mask.append(False)
            samples.append(GraphEpisodeSample(
                observations=torch.tensor(obs_ids, dtype=torch.long),
                actions=torch.tensor(action_ids, dtype=torch.long),
                targets=torch.tensor(target_ids, dtype=torch.long),
                coordinates=torch.tensor(np.asarray(coords), dtype=torch.float32),
                valid_mask=torch.tensor(valid_mask, dtype=torch.bool),
            ))
        test_ds = GraphWalkDataset(samples)
        return train_ds, test_ds
