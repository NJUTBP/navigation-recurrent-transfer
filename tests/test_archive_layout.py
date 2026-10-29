"""The paths used to load released checkpoints match the layout of the data packages P1 and P2."""
import fnmatch
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools" / "reevaluation"))

import reevaluation_common as rc  # noqa: E402

ENTRIES = {e["name"]: e for e in json.loads(
    (REPO / "release_tools" / "cluster" / "release_manifest.json").read_text())["entries"]}
SCALAR = ("navigation_trainable_scalar_gain", "matched_random_trainable_scalar_gain")


def check(paths, entry):
    paths = {Path(p).relative_to(rc.PROJECT_ROOT).as_posix() for p in paths}
    for path in paths:
        assert any(fnmatch.fnmatch(path, g) for g in ENTRIES[entry]["globs"]), (entry, path)
    assert len(paths) == ENTRIES[entry].get("expect", len(paths)), entry


def test_pretrained_checkpoints():
    check([rc.nav_checkpoint(s) for s in rc.SOURCES], "nav_pretrained")
    check([rc.dr_checkpoint(s) for s in rc.SOURCES], "dr_pretrained")


def test_target_checkpoints():
    check([rc.phasef_checkpoint(s, t, c) for s in rc.SOURCES for t in rc.TARGETS
           for c in SCALAR + ("fresh_random_trainable_scalar_gain",)], "phasef_scalar_models")
    check([rc.e09a_checkpoint(s, t, c) for s in rc.SOURCES for t in rc.TARGETS_32K for c in SCALAR],
          "e09a_scalar_models")
    check([rc.e6_checkpoint(s, t, r) for s in rc.SOURCES for t in rc.TARGETS_32K for r in rc.RHO_SCAN + (0.9,)],
          "e6_rho_models")
    check([rc.e6i_checkpoint(s, t, a) for s in rc.SOURCES for t in rc.TARGETS_32K for a in rc.RESCALE_ARMS],
          "e6i_rescaling_models")
