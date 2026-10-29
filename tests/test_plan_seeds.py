"""Seed blocks of the experimental plan: navigation and control sources, target seeds, statistics seed."""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_source_and_target_seed_blocks():
    plan = json.loads((REPO / "plan.json").read_text(encoding="utf-8"))
    pairs = plan["source_pairs"]
    assert [p["source_seed"] for p in pairs] == list(range(6101, 6111))
    assert [p["control_seed"] for p in pairs] == list(range(7101, 7111))
    for i, p in enumerate(pairs):
        base = plan["config"]["target_seed_base"] + 100 * i
        assert p["main_target_seeds"] == [base, base + 1, base + 2]
        assert p["hard_target_seeds"] == [base + 30, base + 31, base + 32]
        assert p["factorial_target_seeds"] == [base + 60, base + 61, base + 62]
    assert plan["config"]["statistics_seed"] == 982451653
    assert len(plan["factorial_contrasts"]) == 8
