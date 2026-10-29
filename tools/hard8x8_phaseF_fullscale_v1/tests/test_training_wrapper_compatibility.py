"""Phase F training wrapper: train_fixed_updates runs for representative condition kinds."""
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


def test_train_fixed_updates_for_condition_kinds():
    torch = pytest.importorskip("torch")
    common = phasef_module("fullscale_common")
    run_group = phasef_module("run_group")
    h = common.import_historical()
    device = torch.device("cpu")
    nodes = []  # small deterministic episode set
    for s in range(16):
        r, c = divmod(s, 8)
        if c < 7:
            nodes.append((s, s + 1))
        if r < 7:
            nodes.append((s, s + 8))
    eps = common.to_historical_episodes(nodes, h)
    train, val = eps[:20], eps[20:]
    h["set_seed"](11)
    donor = h["build_hard_model"](CFG, n_nodes=64, device=device)
    nav_hh = donor.grid_rnn.cell.h_proj.weight.detach().cpu().clone()
    nav_ih = donor.grid_rnn.cell.in_proj.weight.detach().cpu().clone()
    matched_hh, matched_ih, _ = h["make_matched_random_weights"](
        nav_hh, nav_ih, seed=22, iterations=4, singular_rank=4, probe_count=8, strict=False)
    cell = h["FactorialCell"]
    specs = run_group.condition_specs()
    check = ["navigation_fixed_gain3_frozen", "navigation_trainable_scalar_gain", "navigation_trainable_input_adapter",
             "navigation_full_core_trainable", "fresh_random_full_core_trainable"]
    for i, name in enumerate(check):
        spec = specs[name]
        h["set_seed"](100)
        base = h["build_hard_model"](CFG, n_nodes=64, device=device)
        run_group.configure_origin(base, spec, h, nav_hh=nav_hh, nav_ih=nav_ih, matched_hh=matched_hh,
                                   matched_ih=matched_ih, cell_cls=cell, model_cfg=CFG, device=device,
                                   random_core_seed=500 + i)
        model = run_group.wrap_model(base, spec, h, device)
        h["set_seed"](100)
        tr = common.make_loader(train, h, 8, True, 777)
        va = common.make_loader(val, h, 8, False, 778)
        res = h["train_fixed_updates"](model, tr, va, device, {
            "max_gradient_updates": 2, "minimum_updates": 2, "eval_every_updates": 1,
            "early_stopping_patience_evals": 99, "lr": 1e-3, "weight_decay": 0, "whh_weight_decay": 1e-4,
            "grad_clip": 1, "position_loss_weight": 0, "calibration_bins": 5}, seed=100)
        assert res.stopped_update == 2, (name, res.stopped_update)


if __name__ == "__main__":
    test_train_fixed_updates_for_condition_kinds()
    print("PASS train_fixed_updates across representative Phase F condition kinds")
