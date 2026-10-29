"""Phase F conditions: every condition builds and only full-core conditions train the recurrent core."""
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


def test_condition_construction_and_trainability():
    torch = pytest.importorskip("torch")
    common = phasef_module("fullscale_common")
    run_group = phasef_module("run_group")
    h = common.import_historical()
    device = torch.device("cpu")
    h["set_seed"](11)
    donor = h["build_hard_model"](CFG, n_nodes=64, device=device)
    nav_hh = donor.grid_rnn.cell.h_proj.weight.detach().cpu().clone()
    nav_ih = donor.grid_rnn.cell.in_proj.weight.detach().cpu().clone()
    matched_hh, matched_ih, _ = h["make_matched_random_weights"](
        nav_hh, nav_ih, seed=22, iterations=4, singular_rank=4, probe_count=8, strict=False)
    cell = h["FactorialCell"]
    for i, (name, spec) in enumerate(run_group.condition_specs().items()):
        h["set_seed"](1234)
        base = h["build_hard_model"](CFG, n_nodes=64, device=device)
        run_group.configure_origin(base, spec, h, nav_hh=nav_hh, nav_ih=nav_ih, matched_hh=matched_hh,
                                   matched_ih=matched_ih, cell_cls=cell, model_cfg=CFG, device=device,
                                   random_core_seed=100 + i)
        model = run_group.wrap_model(base, spec, h, device)
        grid = any(p.requires_grad for p in model.grid_rnn.parameters())
        assert grid == (spec["core"] == "trainable"), (name, grid, spec)
        if spec["kind"] == "scalar":
            assert abs(float(model.gain().detach()) - 1.0) < 1e-8


if __name__ == "__main__":
    test_condition_construction_and_trainability()
    print("PASS condition construction/trainability contracts")
