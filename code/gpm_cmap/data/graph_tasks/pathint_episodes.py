"""Path-integration episodes for abstract graph reasoning.

This generator produces *blind walk* episodes:

    Input :  start_node_id,  velocity_sequence  (2-D displacement per step)
    Target:  final_node_id   (single prediction at the end)

The model must maintain an internal representation of its abstract position
purely via path integration.  The original implementation only supported the
2-D social hierarchy.  This version supports social_hierarchy, family_tree and
transitive graphs by deriving velocity vectors from each graph's structural
coordinates.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class PathIntEpisode:
    start_node: int
    velocities: np.ndarray
    target_node: int
    start_coords: np.ndarray
    target_coords: np.ndarray
    path_coords: np.ndarray
    net_l1_distance: float
    path_unique_nodes: int


def episode_fingerprint(episode: PathIntEpisode) -> str:
    """Return a stable identity for a complete supervised episode.

    Random i.i.d. slicing can place duplicated generated episodes on both sides
    of a split.  Grouping by this fingerprint prevents an exact input/label
    sequence from leaking into validation or test.
    """
    digest = hashlib.sha256()
    digest.update(np.asarray([episode.start_node, episode.target_node], dtype=np.int64).tobytes())
    for array in (episode.velocities, episode.start_coords, episode.target_coords, episode.path_coords):
        arr = np.ascontiguousarray(array, dtype=np.float32)
        digest.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
        digest.update(arr.tobytes())
    return digest.hexdigest()


def split_manifest(
    train_eps: List[PathIntEpisode],
    validation_eps: List[PathIntEpisode],
    test_eps: List[PathIntEpisode],
) -> Dict[str, object]:
    """Summarize split identities and assert exact-episode isolation."""
    split_sets = {
        "train": {episode_fingerprint(ep) for ep in train_eps},
        "validation": {episode_fingerprint(ep) for ep in validation_eps},
        "test": {episode_fingerprint(ep) for ep in test_eps},
    }
    overlaps = {
        "train_validation": len(split_sets["train"] & split_sets["validation"]),
        "train_test": len(split_sets["train"] & split_sets["test"]),
        "validation_test": len(split_sets["validation"] & split_sets["test"]),
    }
    if any(overlaps.values()):
        raise RuntimeError(f"Exact episode leakage detected across splits: {overlaps}")
    return {
        "counts": {
            "train": len(train_eps),
            "validation": len(validation_eps),
            "test": len(test_eps),
        },
        "unique_fingerprint_counts": {k: len(v) for k, v in split_sets.items()},
        "overlaps": overlaps,
    }


class PathIntDataset(Dataset):
    """PyTorch dataset wrapping a list of PathIntEpisodes."""

    def __init__(self, episodes: List[PathIntEpisode]) -> None:
        self.episodes = episodes
        self.prevalidated = True
        self.min_node_id = min(
            (min(ep.start_node, ep.target_node) for ep in episodes),
            default=0,
        )
        self.max_node_id = max(
            (max(ep.start_node, ep.target_node) for ep in episodes),
            default=-1,
        )
        self._tensor_samples = tuple(self._tensorize_episode(ep) for ep in episodes)

    @staticmethod
    def _tensorize_episode(ep: PathIntEpisode) -> Dict[str, torch.Tensor]:
        velocities = np.asarray(ep.velocities, dtype=np.float32)
        start_coords = np.asarray(ep.start_coords, dtype=np.float32)
        target_coords = np.asarray(ep.target_coords, dtype=np.float32)
        path_coords = np.asarray(ep.path_coords, dtype=np.float32)

        if velocities.ndim != 2 or velocities.shape[-1] != 2:
            raise ValueError(
                f"episode velocities must have shape [T,2]; got {velocities.shape}"
            )
        if start_coords.shape != (2,) or target_coords.shape != (2,):
            raise ValueError(
                "episode start/target coordinates must both have shape [2]; "
                f"got start={start_coords.shape}, target={target_coords.shape}"
            )
        if path_coords.shape != (velocities.shape[0] + 1, 2):
            raise ValueError(
                "episode path_coords must have shape [T+1,2]; "
                f"got velocities={velocities.shape}, path_coords={path_coords.shape}"
            )
        for name, value in (
            ("velocities", velocities),
            ("start_coords", start_coords),
            ("target_coords", target_coords),
            ("path_coords", path_coords),
        ):
            if not np.isfinite(value).all():
                raise FloatingPointError(f"episode {name} contains NaN or Inf")
        if not np.isfinite(ep.net_l1_distance):
            raise FloatingPointError("episode net_l1_distance contains NaN or Inf")

        return {
            "start_node": torch.tensor(ep.start_node, dtype=torch.long),
            "velocities": torch.from_numpy(velocities),
            "target_node": torch.tensor(ep.target_node, dtype=torch.long),
            "start_coords": torch.from_numpy(start_coords),
            "target_coords": torch.from_numpy(target_coords),
            "path_coords": torch.from_numpy(path_coords),
            "net_l1_distance": torch.tensor(ep.net_l1_distance, dtype=torch.float32),
            "path_unique_nodes": torch.tensor(ep.path_unique_nodes, dtype=torch.long),
        }

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self._tensor_samples[idx]


def pad_collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate variable-length path-integration episodes (T13).

    Pads ``velocities`` (and ``path_coords``) with zeros to the batch-max length
    and returns a ``lengths`` tensor with the true number of velocity steps per
    row. The model's forward uses ``lengths`` to read the final hidden state at
    each row's true step (zero padding is NOT a no-op for the RNN). Fixed-size
    fields are stacked normally. Safe for equal-length batches too (lengths are
    then all identical), so it can be the default collate without changing
    behaviour for fixed-length data.
    """
    lengths = torch.tensor([b["velocities"].shape[0] for b in batch], dtype=torch.long)
    max_len = int(lengths.max().item())
    coord_dim = batch[0]["velocities"].shape[-1]

    vel = torch.zeros(len(batch), max_len, coord_dim, dtype=torch.float32)
    # path_coords has length (steps+1); pad to max_len+1 to keep the start row.
    pc = torch.zeros(len(batch), max_len + 1, coord_dim, dtype=torch.float32)
    for i, b in enumerate(batch):
        L = b["velocities"].shape[0]
        vel[i, :L] = b["velocities"]
        pcl = b["path_coords"].shape[0]
        pc[i, :pcl] = b["path_coords"]

    out: Dict[str, torch.Tensor] = {
        "start_node": torch.stack([b["start_node"] for b in batch]),
        "velocities": vel,
        "lengths": lengths,
        "target_node": torch.stack([b["target_node"] for b in batch]),
        "start_coords": torch.stack([b["start_coords"] for b in batch]),
        "target_coords": torch.stack([b["target_coords"] for b in batch]),
        "path_coords": pc,
        "net_l1_distance": torch.stack([b["net_l1_distance"] for b in batch]),
        "path_unique_nodes": torch.stack([b["path_unique_nodes"] for b in batch]),
    }
    return out


# ---------------------------------------------------------------------------
# Task / coordinate helpers
# ---------------------------------------------------------------------------


def _make_task(task_name: str, seed: int, n_levels: int = 4, n_items: int = 7):
    """Instantiate an abstract graph task by name."""
    if task_name == "social_hierarchy":
        from gpm_cmap.data.graph_tasks.social_hierarchy import SocialHierarchyTask
        return SocialHierarchyTask(n_levels=n_levels, seed=seed)
    if task_name in {"family_tree", "family_tree_split"}:
        from gpm_cmap.data.graph_tasks.family_tree import FamilyTreeTask
        return FamilyTreeTask(seed=seed)
    if task_name == "transitive":
        from gpm_cmap.data.graph_tasks.transitive import TransitiveInferenceTask
        return TransitiveInferenceTask(n_items=n_items, seed=seed)
    raise ValueError(
        f"Unsupported path-integration task_name={task_name!r}. "
        "Choose from: social_hierarchy, family_tree, transitive."
    )


def _as_coord_dim(coord: np.ndarray, coord_dim: int = 2) -> np.ndarray:
    """Pad/truncate coordinates to coord_dim and return float32."""
    arr = np.asarray(coord, dtype=np.float32).reshape(-1)
    out = np.zeros(coord_dim, dtype=np.float32)
    n = min(coord_dim, arr.shape[0])
    out[:n] = arr[:n]
    return out


def _prepare_node_coords(
    task_name: str,
    graph: nx.DiGraph,
    raw_node_coords: Dict[str, np.ndarray],
    coord_strategy: Optional[str] = None,
    coord_dim: int = 2,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Return structural coordinates used by the path-int episode generator.

    Defaults mirror ``models.graph_positional``:
    - social_hierarchy -> real 2-D coordinates
    - transitive       -> rank coordinate padded to 2-D
    - family_tree      -> spectral coordinates

    Using structural coordinates is important for family-tree path integration:
    the task's raw layout is only a drawing convention, whereas spectral coords
    encode the graph topology.
    """
    from gpm_cmap.models.graph_positional import compute_structural_coords

    coords = compute_structural_coords(
        task_name="family_tree" if task_name == "family_tree_split" else task_name,
        graph=graph,
        node_coords=raw_node_coords,
        embed_dim=coord_dim,
        strategy=coord_strategy,
        seed=seed,
    )
    return {name: _as_coord_dim(coord, coord_dim=coord_dim) for name, coord in coords.items()}


# ---------------------------------------------------------------------------
# Velocity / displacement helpers
# ---------------------------------------------------------------------------


def _normalize_or_zero(vec: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm < eps:
        return np.zeros_like(vec, dtype=np.float32)
    return (vec / norm).astype(np.float32)


def _edge_displacement(
    src: str,
    dst: str,
    node_coords: Dict[str, np.ndarray],
    normalize: bool = True,
) -> np.ndarray:
    disp = np.asarray(node_coords[dst] - node_coords[src], dtype=np.float32)
    return _normalize_or_zero(disp) if normalize else disp.astype(np.float32)


def _infer_action_displacements(
    graph: nx.DiGraph,
    action_to_id: Dict[str, int],
    node_coords: Dict[str, np.ndarray],
    normalize: bool = True,
) -> Dict[int, np.ndarray]:
    """Infer a representative displacement vector for each action label.

    This is mainly used for the shuffled action-mapping control.  For the main
    aligned condition, per-edge displacements are used, which avoids ambiguity
    in relations such as sibling/spouse/child that can branch in a family tree.
    """
    coord_dim = next(iter(node_coords.values())).shape[0]
    buckets: Dict[int, List[np.ndarray]] = {aid: [] for aid in action_to_id.values()}

    for src, dst, data in graph.edges(data=True):
        action = data.get("action")
        if action not in action_to_id:
            continue
        aid = action_to_id[action]
        buckets[aid].append(_edge_displacement(src, dst, node_coords, normalize=False))

    out: Dict[int, np.ndarray] = {}
    for aid, vecs in buckets.items():
        if not vecs:
            out[aid] = np.zeros(coord_dim, dtype=np.float32)
            continue
        mean_vec = np.mean(np.stack(vecs, axis=0), axis=0).astype(np.float32)
        out[aid] = _normalize_or_zero(mean_vec) if normalize else mean_vec
    return out


def _apply_velocity_transform(vec: np.ndarray, transform: str) -> np.ndarray:
    """Apply a fixed geometric transform to a 2-D velocity (T9c controls).

    - ``none``     : identity.
    - ``rotate90`` : rotate +90 deg in the coordinate plane.
    - ``scale2x``  : double the magnitude.
    - ``signflip`` : negate (180 deg rotation).

    These transforms MOVE the integrated endpoint (unlike within-episode order
    permutation, which is inert by commutativity), so a model that relies on
    veridical velocity geometry must be affected.
    """
    v = np.asarray(vec, dtype=np.float32)
    if transform == "none":
        return v
    if transform == "rotate90":
        if v.shape[0] < 2:
            return v
        out = v.copy()
        out[0], out[1] = -v[1], v[0]
        return out.astype(np.float32)
    if transform == "scale2x":
        return (2.0 * v).astype(np.float32)
    if transform == "signflip":
        return (-v).astype(np.float32)
    raise ValueError(f"Unsupported velocity_transform: {transform!r}")


def _build_shuffled_action_map(
    graph: nx.DiGraph,
    action_to_id: Dict[str, int],
    node_coords: Dict[str, np.ndarray],
    seed: int,
    normalize: bool = True,
) -> Dict[int, np.ndarray]:
    """Build an action-level shuffled displacement map for negative control."""
    base = _infer_action_displacements(
        graph=graph,
        action_to_id=action_to_id,
        node_coords=node_coords,
        normalize=normalize,
    )
    sorted_ids = sorted(base.keys())
    disps = [base[aid].copy() for aid in sorted_ids]
    rng = np.random.default_rng(seed)
    rng.shuffle(disps)
    return {aid: disps[i] for i, aid in enumerate(sorted_ids)}


# ---------------------------------------------------------------------------
# Episode generation
# ---------------------------------------------------------------------------


def generate_pathint_episodes(
    graph: nx.DiGraph,
    node_to_id: Dict[str, int],
    action_to_id: Dict[str, int],
    node_coords: Dict[str, np.ndarray],
    num_episodes: int,
    seq_len: int,
    velocity_scale: float = 0.02,
    seed: int = 42,
    action_map_mode: str = "aligned",
    normalize_displacements: bool = True,
    velocity_transform: str = "none",
    seq_len_range: Optional[Tuple[int, int]] = None,
) -> List[PathIntEpisode]:
    """Generate blind-walk path-integration episodes.

    ``aligned`` uses the actual edge displacement in the structural coordinate
    system.  This supports arbitrary graphs and avoids making family-tree
    relations artificially deterministic at the action-label level.

    ``shuffled`` uses a shuffled action-level displacement map and is retained
    as a negative control for action/velocity alignment.

    ``mismatched_target`` builds true aligned episodes, then breaks the
    velocity-sequence -> target pairing across episodes by deriangement: each
    episode keeps its (start_node, velocities, path_coords) but is reassigned a
    DIFFERENT episode's (target_node, target_coords). This is a strong negative
    control that is immune to the commutativity of displacement summation (which
    makes within-episode order permutation inert for endpoint prediction): if
    the model truly uses the velocity stream to reach the target, mismatched
    pairing must destroy accuracy; a model exploiting start-node/readout
    statistics survives.
    """
    if action_map_mode not in {"aligned", "shuffled", "mismatched_target"}:
        raise ValueError(f"Unsupported action_map_mode: {action_map_mode!r}")
    if velocity_transform not in {"none", "rotate90", "scale2x", "signflip"}:
        raise ValueError(f"Unsupported velocity_transform: {velocity_transform!r}")

    rng = np.random.default_rng(seed)
    nodes = list(graph.nodes())
    coord_dim = next(iter(node_coords.values())).shape[0]
    shuffled_action_map = None
    if action_map_mode == "shuffled":
        shuffled_action_map = _build_shuffled_action_map(
            graph=graph,
            action_to_id=action_to_id,
            node_coords=node_coords,
            seed=seed,
            normalize=normalize_displacements,
        )

    episodes: List[PathIntEpisode] = []
    if seq_len_range is not None:
        lo, hi = seq_len_range
        if not (1 <= lo <= hi):
            raise ValueError(f"seq_len_range must satisfy 1<=lo<=hi; got {seq_len_range}")
    for _ in range(num_episodes):
        current = rng.choice(nodes)
        visited_nodes = [current]
        start_node = node_to_id[current]
        start_coords = node_coords[current].copy()

        velocities: List[np.ndarray] = []
        coords_path = [node_coords[current].copy()]

        # variable-length episodes: sample this episode's length when a
        # range is given (enables a held-out path-LENGTH split: train short,
        # test long). Fixed seq_len otherwise (unchanged default).
        this_len = int(rng.integers(lo, hi + 1)) if seq_len_range is not None else seq_len

        for _step in range(this_len):
            successors = list(graph.successors(current))
            if not successors:
                velocities.append(np.zeros(coord_dim, dtype=np.float32))
                coords_path.append(node_coords[current].copy())
                visited_nodes.append(current)
                continue

            nxt = rng.choice(successors)
            edge_data = graph.get_edge_data(current, nxt)
            action_name = edge_data["action"]
            action_id = action_to_id[action_name]

            if action_map_mode in {"aligned", "mismatched_target"}:
                disp = _edge_displacement(
                    current,
                    nxt,
                    node_coords=node_coords,
                    normalize=normalize_displacements,
                )
            else:
                assert shuffled_action_map is not None
                disp = shuffled_action_map[action_id]

            step_vel = (disp * velocity_scale).astype(np.float32)
            # optional fixed geometric transform of the velocity. The
            # target remains the true aligned endpoint, so a transformed
            # velocity stream no longer integrates to it -> tests reliance on
            # veridical velocity geometry (rotation/scale/sign equivariance).
            step_vel = _apply_velocity_transform(step_vel, velocity_transform)
            velocities.append(step_vel)
            coords_path.append(node_coords[nxt].copy())
            current = nxt
            visited_nodes.append(current)

        target_node = node_to_id[current]
        target_coords = node_coords[current].copy()
        net_l1_distance = float(np.abs(target_coords - start_coords).sum())
        path_unique_nodes = len(set(visited_nodes))

        episodes.append(
            PathIntEpisode(
                start_node=start_node,
                velocities=np.asarray(velocities, dtype=np.float32),
                target_node=target_node,
                start_coords=start_coords,
                target_coords=target_coords,
                path_coords=np.asarray(coords_path, dtype=np.float32),
                net_l1_distance=net_l1_distance,
                path_unique_nodes=path_unique_nodes,
            )
        )

    # mismatched_target negative control: break the velocity->target
    # pairing across episodes via a derangement (no episode keeps its own
    # target). Each episode retains its start_node / velocities / path_coords
    # but is reassigned another episode's target_node + target_coords. This is
    # immune to the commutativity of displacement summation and directly tests
    # whether the velocity stream is actually used to reach the target.
    if action_map_mode == "mismatched_target" and len(episodes) > 1:
        n = len(episodes)
        # Build a derangement (permutation with no fixed point) deterministically.
        perm = rng.permutation(n)
        for i in range(n):
            if perm[i] == i:
                j = (i + 1) % n
                perm[i], perm[j] = perm[j], perm[i]
        # Snapshot original targets FIRST. Reassigning in place while reading
        # episodes[perm[i]] would copy already-overwritten values when
        # perm[i] < i, corrupting the permutation. Snapshotting guarantees a
        # true permutation of the original targets (rule #12).
        orig_target_node = [ep.target_node for ep in episodes]
        orig_target_coords = [ep.target_coords.copy() for ep in episodes]
        for i, ep in enumerate(episodes):
            ep.target_node = orig_target_node[perm[i]]
            ep.target_coords = orig_target_coords[perm[i]].copy()
            # net_l1_distance is defined relative to the (now mismatched) target;
            # recompute so any difficulty bucketing stays internally consistent.
            ep.net_l1_distance = float(np.abs(ep.target_coords - ep.start_coords).sum())

    return episodes


def _grouped_random_partition(
    episodes: List[PathIntEpisode],
    split_ratio: float,
    seed: int,
) -> Tuple[List[PathIntEpisode], List[PathIntEpisode]]:
    """Randomly split fingerprint groups so exact duplicates stay together."""
    groups: Dict[str, List[PathIntEpisode]] = {}
    for episode in episodes:
        groups.setdefault(episode_fingerprint(episode), []).append(episode)
    group_list = list(groups.values())
    rng = np.random.default_rng(seed)
    rng.shuffle(group_list)

    target_train = int(round(len(episodes) * float(split_ratio)))
    train_eps: List[PathIntEpisode] = []
    test_eps: List[PathIntEpisode] = []
    for group in group_list:
        # Greedy assignment keeps the requested sample ratio close while never
        # splitting an exact duplicate group.
        if len(train_eps) < target_train:
            train_eps.extend(group)
        else:
            test_eps.extend(group)
    if not test_eps and len(group_list) > 1:
        moved = group_list[-1]
        train_ids = {id(ep) for ep in moved}
        train_eps = [ep for ep in train_eps if id(ep) not in train_ids]
        test_eps.extend(moved)
    return train_eps, test_eps


def _split_episodes(
    episodes: List[PathIntEpisode],
    split_strategy: str,
    split_ratio: float,
    seed: int,
) -> Tuple[List[PathIntEpisode], List[PathIntEpisode]]:
    """Partition episodes into two leakage-audited subsets."""
    if not 0.0 < float(split_ratio) < 1.0:
        raise ValueError(f"split_ratio must lie strictly between 0 and 1; got {split_ratio}")
    rng = np.random.default_rng(seed)

    if split_strategy == "random":
        train_eps, test_eps = _grouped_random_partition(episodes, split_ratio, seed)
    elif split_strategy in {"held_out_start_nodes", "held_out_target_nodes"}:
        key = (lambda ep: ep.start_node) if "start" in split_strategy else (lambda ep: ep.target_node)
        keys = sorted({key(ep) for ep in episodes})
        rng.shuffle(keys)
        n_train_keys = max(1, min(len(keys) - 1, int(len(keys) * split_ratio)))
        train_keys = set(keys[:n_train_keys])
        train_eps = [ep for ep in episodes if key(ep) in train_keys]
        test_eps = [ep for ep in episodes if key(ep) not in train_keys]
    elif split_strategy == "held_out_pairs":
        pairs = sorted({(ep.start_node, ep.target_node) for ep in episodes})
        rng.shuffle(pairs)
        n_train_pairs = max(1, min(len(pairs) - 1, int(len(pairs) * split_ratio)))
        train_pairs = set(pairs[:n_train_pairs])
        train_eps = [ep for ep in episodes if (ep.start_node, ep.target_node) in train_pairs]
        test_eps = [ep for ep in episodes if (ep.start_node, ep.target_node) not in train_pairs]
    elif split_strategy == "held_out_length":
        lengths = sorted({ep.velocities.shape[0] for ep in episodes})
        if len(lengths) < 2:
            raise RuntimeError(
                "held_out_length needs >=2 distinct episode lengths; pass "
                "seq_len_range=(lo,hi) with hi>lo when generating episodes."
            )
        n_train_lengths = max(1, min(len(lengths) - 1, int(len(lengths) * split_ratio)))
        max_train_len = lengths[n_train_lengths - 1]
        train_eps = [ep for ep in episodes if ep.velocities.shape[0] <= max_train_len]
        test_eps = [ep for ep in episodes if ep.velocities.shape[0] > max_train_len]
    else:
        raise ValueError(
            f"Unsupported split_strategy={split_strategy!r}. Choose from: random, "
            "held_out_pairs, held_out_start_nodes, held_out_target_nodes, "
            "held_out_length."
        )

    if not train_eps or not test_eps:
        raise RuntimeError(
            f"split_strategy={split_strategy!r} (ratio={split_ratio}) produced an "
            f"empty partition: |train|={len(train_eps)}, |test|={len(test_eps)}."
        )
    # Exact episodes must never cross any split, independent of strategy.
    overlap = {episode_fingerprint(ep) for ep in train_eps} & {episode_fingerprint(ep) for ep in test_eps}
    if overlap:
        raise RuntimeError(f"split_strategy={split_strategy!r} leaked {len(overlap)} exact episodes")
    return train_eps, test_eps


def split_train_validation_test(
    episodes: List[PathIntEpisode],
    *,
    split_strategy: str,
    train_ratio: float,
    validation_ratio: float,
    seed: int,
) -> Tuple[List[PathIntEpisode], List[PathIntEpisode], List[PathIntEpisode]]:
    """Create train/validation/test splits without using test for selection."""
    if validation_ratio <= 0 or train_ratio <= 0 or train_ratio + validation_ratio >= 1:
        raise ValueError(
            "train_ratio and validation_ratio must be positive and sum to less than 1; "
            f"got train_ratio={train_ratio}, validation_ratio={validation_ratio}"
        )
    train_val, test_eps = _split_episodes(
        episodes, split_strategy=split_strategy, split_ratio=train_ratio + validation_ratio, seed=seed
    )
    relative_train = train_ratio / (train_ratio + validation_ratio)
    # A second held-out-node/pair split could change the semantic test partition.
    # Within train_val, use grouped random assignment solely for model selection.
    train_eps, validation_eps = _grouped_random_partition(train_val, relative_train, seed + 1)
    if not train_eps or not validation_eps:
        raise RuntimeError("train/validation partition is empty; increase num_episodes")
    split_manifest(train_eps, validation_eps, test_eps)
    return train_eps, validation_eps, test_eps


def build_pathint_loaders(
    task_name: str = "social_hierarchy",
    num_episodes: int = 5000,
    seq_len: int = 8,
    batch_size: int = 64,
    split_ratio: float = 0.8,
    velocity_scale: float = 0.02,
    seed: int = 42,
    n_levels: int = 4,
    n_items: int = 7,
    action_map_mode: str = "aligned",
    coord_strategy: Optional[str] = None,
    coord_dim: int = 2,
    normalize_displacements: bool = True,
    coord_scale: float = 1.0,
    split_strategy: str = "random",
    velocity_transform: str = "none",
    test_only_velocity_transform: Optional[str] = None,
    seq_len_range: Optional[Tuple[int, int]] = None,
    validation_ratio: float = 0.0,
    return_validation: bool = False,
    num_workers: int = 0,
) -> Tuple[
    DataLoader,
    DataLoader,
    object,
    Dict[str, int],
    Dict[str, int],
    Dict[str, np.ndarray],
]:
    """Build train/test DataLoaders for abstract path integration.

    Supported tasks
    ---------------
    - ``social_hierarchy``: n_levels x n_levels lattice.
    - ``family_tree``: kinship graph; coordinates default to spectral layout.
    - ``transitive``: 1-D ordinal chain padded to 2-D coordinates.

    ``coord_scale`` rescales start/target/path coordinates after structural
    coordinates are computed.  When ``normalize_displacements=True`` on a unit
    lattice, setting ``coord_scale == velocity_scale`` makes the supervised
    coordinate target satisfy target_coords - start_coords == sum(velocities).
    This is essential for interpreting the task as literal path integration
    rather than as an arbitrary sequence-to-node classification problem.
    """
    if coord_dim != 2:
        raise ValueError(
            "AbstractPathIntModel/evaluate_pathint currently expect 2-D coordinates; "
            f"got coord_dim={coord_dim}."
        )

    task = _make_task(task_name=task_name, seed=seed, n_levels=n_levels, n_items=n_items)
    graph, node_to_id, action_to_id, raw_node_coords = task.build_graph()
    node_coords = _prepare_node_coords(
        task_name=task_name,
        graph=graph,
        raw_node_coords=raw_node_coords,
        coord_strategy=coord_strategy,
        coord_dim=coord_dim,
        seed=seed,
    )
    coord_scale = float(coord_scale)
    if coord_scale <= 0:
        raise ValueError(f"coord_scale must be positive; got {coord_scale}")
    if coord_scale != 1.0:
        node_coords = {k: (v * coord_scale).astype(np.float32) for k, v in node_coords.items()}

    episodes = generate_pathint_episodes(
        graph=graph,
        node_to_id=node_to_id,
        action_to_id=action_to_id,
        node_coords=node_coords,
        num_episodes=num_episodes,
        seq_len=seq_len,
        velocity_scale=velocity_scale,
        seed=seed,
        action_map_mode=action_map_mode,
        normalize_displacements=normalize_displacements,
        velocity_transform=velocity_transform,
        seq_len_range=seq_len_range,
    )

    validation_eps: List[PathIntEpisode] = []
    if return_validation or validation_ratio > 0:
        train_eps, validation_eps, test_eps = split_train_validation_test(
            episodes,
            split_strategy=split_strategy,
            train_ratio=split_ratio,
            validation_ratio=validation_ratio,
            seed=seed,
        )
    else:
        train_eps, test_eps = _split_episodes(
            episodes, split_strategy=split_strategy, split_ratio=split_ratio, seed=seed
        )

    # Test-only velocity transforms probe out-of-distribution equivariance. A
    # transform applied while generating both train and test episodes instead
    # defines a consistently relabelled task. Targets remain unchanged here so
    # the perturbation genuinely tests sensitivity to the velocity stream.
    if test_only_velocity_transform not in {None, "none"}:
        if velocity_transform not in {"none", None}:
            raise ValueError(
                "Set only one of velocity_transform (train+test, relabels the task) "
                "or test_only_velocity_transform (test-only, true OOD probe); got "
                f"velocity_transform={velocity_transform!r} and "
                f"test_only_velocity_transform={test_only_velocity_transform!r}."
            )
        # Validate eagerly so an unknown name fails loud before any training.
        _apply_velocity_transform(np.zeros(2, dtype=np.float32), test_only_velocity_transform)
        transformed_test: List[PathIntEpisode] = []
        for ep in test_eps:
            vel = np.asarray(
                [_apply_velocity_transform(v, test_only_velocity_transform) for v in ep.velocities],
                dtype=np.float32,
            ).reshape(ep.velocities.shape)
            transformed_test.append(
                PathIntEpisode(
                    start_node=ep.start_node,
                    velocities=vel,
                    target_node=ep.target_node,          # target intentionally UNCHANGED
                    start_coords=ep.start_coords.copy(),
                    target_coords=ep.target_coords.copy(),
                    path_coords=ep.path_coords.copy(),
                    net_l1_distance=ep.net_l1_distance,
                    path_unique_nodes=ep.path_unique_nodes,
                )
            )
        test_eps = transformed_test

    train_ds = PathIntDataset(train_eps)
    validation_ds = PathIntDataset(validation_eps) if validation_eps else None
    test_ds = PathIntDataset(test_eps)

    # Use explicit random generators so a DataLoader's order is a function of the
    # run seed rather than the process-global RNG call order.
    from gpm_cmap.utils.seed import make_torch_generator, make_worker_init_fn

    collate = pad_collate if seq_len_range is not None else None
    loader_kwargs = {
        "batch_size": batch_size,
        "collate_fn": collate,
        "num_workers": int(num_workers),
        "worker_init_fn": make_worker_init_fn(seed),
    }
    train_loader = DataLoader(
        train_ds, shuffle=True, generator=make_torch_generator(seed), **loader_kwargs
    )
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)
    if validation_ds is not None:
        validation_loader = DataLoader(validation_ds, shuffle=False, **loader_kwargs)
        return (
            train_loader, validation_loader, test_loader, task, node_to_id,
            action_to_id, node_coords,
        )
    return train_loader, test_loader, task, node_to_id, action_to_id, node_coords
