"""Phase F wrappers: gain 1 reproduces the base model and the scalar wrapper freezes the core."""
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


def test_fixed_and_scalar_wrappers():
    torch = pytest.importorskip("torch")
    common = phasef_module("fullscale_common")
    run_group = phasef_module("run_group")
    h = common.import_historical()
    device = torch.device("cpu")
    h["set_seed"](123)
    base = h["build_hard_model"](CFG, n_nodes=64, device=device)
    base.freeze_grid_rnn_all()
    x = torch.tensor([0, 1])
    v = torch.tensor([[[.02, 0.], [0., .02]], [[.02, 0.], [-.02, 0.]]])
    L = torch.tensor([2, 2])
    base.eval()
    with torch.no_grad():
        y0 = base(x, v, lengths=L, validate_inputs=False)["node_logits"]
    fixed = run_group.FixedGainPathIntModel(base, 1.0)
    fixed.eval()
    with torch.no_grad():
        y1 = fixed(x, v, lengths=L, validate_inputs=False)["node_logits"]
    assert float((y0 - y1).abs().max()) < 1e-7
    scalar = run_group.ScalarGainPathIntModel(base)
    scalar.configure_trainable()
    scalar.eval()
    with torch.no_grad():
        y2 = scalar(x, v, lengths=L, validate_inputs=False)["node_logits"]
    assert float((y0 - y2).abs().max()) < 1e-7
    assert not any(p.requires_grad for p in scalar.grid_rnn.parameters())
    assert scalar.log_gain.requires_grad


if __name__ == "__main__":
    test_fixed_and_scalar_wrappers()
    print("PASS fixed/scalar wrapper identity and freeze contract")
