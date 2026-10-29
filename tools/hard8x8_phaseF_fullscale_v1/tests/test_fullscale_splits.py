"""Phase F splits: manifest hashes, pool sizes and the held-out-composition design."""
import json
import importlib
import sys
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / "scripts"))
CFG = {"hidden_dim": 32, "content_dim": 16, "nonnegative": True, "cell_type": "rnn", "use_bias": False,
       "dropout": 0.0, "periodic_scales": 0, "use_position_head": False, "start_init": "embedding", "coord_dim": 2}


def phasef_module(name):
    """Import a Phase F script module (the scripts import PyTorch at module level)."""
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            pytest.skip("requires PyTorch")
        raise


def test_fullscale_splits():
    common = phasef_module("fullscale_common")
    p = json.loads((PACKAGE / "PROTOCOL.json").read_text())
    for seed in p["target_split_seeds"]:
        pools, m = common.build_fullscale_split(p, seed)
        assert m["manifest_hash"] == p["expected_manifest_hashes"][str(seed)]
        assert {k: len(v) for k, v in pools.items()} == {
            "train_8000": 8000, "validation_seen_short": 1000, "id_seen_short": 2000,
            "heldout_short": 2000, "seen_long": 2000, "heldout_long": 2000}
        held = {tuple(x) for x in p["held_out_action_bigrams"]}
        assert all(not common._contains_heldout(x, held) for x in pools["train_8000"])
        assert all(not common._contains_heldout(x, held) for x in pools["validation_seen_short"])
        assert all(not common._contains_heldout(x, held) for x in pools["id_seen_short"])
        assert all(common._contains_heldout(x, held) for x in pools["heldout_short"])
        assert all(not common._contains_heldout(x, held) for x in pools["seen_long"])
        assert all(common._contains_heldout(x, held) for x in pools["heldout_long"])


if __name__ == "__main__":
    test_fullscale_splits()
    print("PASS full-scale fresh orthogonal split hashes")
