from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import networkx as nx
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class GraphEpisodeSample:
    observations: torch.Tensor
    actions: torch.Tensor
    targets: torch.Tensor
    coordinates: torch.Tensor
    valid_mask: torch.Tensor | None = None


class GraphWalkDataset(Dataset):
    def __init__(self, samples: List[GraphEpisodeSample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.samples[idx]
        valid_mask = item.valid_mask
        if valid_mask is None:
            valid_mask = torch.ones_like(item.targets, dtype=torch.bool)
        return {
            "observations": item.observations,
            "actions": item.actions,
            "targets": item.targets,
            "coordinates": item.coordinates,
            "valid_mask": valid_mask,
            "length": valid_mask.sum(dtype=torch.long),
        }


def random_walk_labeled_graph(
    graph: nx.DiGraph,
    node_to_id: Dict[str, int],
    action_to_id: Dict[str, int],
    node_coords: Dict[str, np.ndarray],
    num_walks: int,
    seq_len: int,
    seed: int = 42,
) -> GraphWalkDataset:
    rng = np.random.default_rng(seed)
    nodes = list(graph.nodes())
    samples: List[GraphEpisodeSample] = []

    for _ in range(num_walks):
        current = rng.choice(nodes)
        obs_ids = []
        action_ids = []
        target_ids = []
        coords = []
        for _step in range(seq_len):
            successors = list(graph.successors(current))
            if not successors:
                current = rng.choice(nodes)
                successors = list(graph.successors(current))
            nxt = rng.choice(successors)
            edge_data = graph.get_edge_data(current, nxt)
            action_name = edge_data["action"]

            obs_ids.append(node_to_id[current])
            action_ids.append(action_to_id[action_name])
            target_ids.append(node_to_id[nxt])
            coords.append(node_coords[current])
            current = nxt

        samples.append(
            GraphEpisodeSample(
                observations=torch.tensor(obs_ids, dtype=torch.long),
                actions=torch.tensor(action_ids, dtype=torch.long),
                targets=torch.tensor(target_ids, dtype=torch.long),
                coordinates=torch.tensor(np.asarray(coords), dtype=torch.float32),
                valid_mask=torch.ones(seq_len, dtype=torch.bool),
            )
        )
    return GraphWalkDataset(samples)
