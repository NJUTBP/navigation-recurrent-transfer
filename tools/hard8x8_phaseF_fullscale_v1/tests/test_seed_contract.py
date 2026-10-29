"""Phase F seeds: run, loader, matched-core and fresh-core seeds are valid uint32 values."""
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


def test_seed_contract():
    stable_seed32 = phasef_module("fullscale_common").stable_seed32
    p = json.loads((PACKAGE / "PROTOCOL.json").read_text())
    for s in p["all_sources"]:
        for t in p["target_split_seeds"]:
            vals = [stable_seed32("phaseF_run", t, 8000), stable_seed32("phaseF_loader", t, 8000),
                    stable_seed32("phaseF_fresh_random_core", s, t), stable_seed32("matched_random", t)]
            assert all(0 <= x <= 2**32 - 1 for x in vals)


if __name__ == "__main__":
    test_seed_contract()
    print("PASS Phase F uint32 seed contract")
