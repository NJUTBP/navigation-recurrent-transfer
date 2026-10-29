"""Phase F protocol: scale of the design and the primary contrast."""
import json
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]


def test_protocol_scale_and_primary_contrast():
    p = json.loads((PACKAGE / "PROTOCOL.json").read_text())
    conds = sum(p["conditions_by_group"].values(), [])
    assert len(p["all_sources"]) == 10
    assert len(p["target_split_seeds"]) == 5
    assert len(conds) == 15 and len(set(conds)) == 15
    assert len(p["all_sources"]) * len(p["target_split_seeds"]) * len(conds) == 750
    assert p["train"]["max_gradient_updates"] == 8000
    assert p["train"]["minimum_updates"] == 8000
    assert p["training_examples"] == 8000
    assert set(p["generalization_family"]) == {"id_seen_short", "heldout_short", "seen_long", "heldout_long"}
    assert p["primary_condition_left"] == "navigation_trainable_scalar_gain"
    assert p["primary_condition_right"] == "matched_random_trainable_scalar_gain"


if __name__ == "__main__":
    test_protocol_scale_and_primary_contrast()
    print("PASS Phase F frozen protocol scale and primary contrast")
