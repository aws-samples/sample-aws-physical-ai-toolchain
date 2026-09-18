"""Verify family_schemas + canonical task-ids come from the
registry (single source of truth), injected into the staged validator; strict
validation still passes a good report and hard-fails an incomplete one.

Byte-identity checks prove the move introduced no behavior change:
  * suite_canonical_task_ids() == validate_entry's former inline dict.
  * family_schemas() == stage_validate_code's former inline glob of defaults.json.
The injection check proves stage_validate_code now embeds BOTH maps.
"""
from __future__ import annotations

import base64
import json
import os
import sys

import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if os.path.join(_REPO_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from vla_pipeline.registry import family_schemas, suite_canonical_task_ids  # noqa: E402

_LEGACY_SUITE_CANONICAL = {
    "libero_spatial": list(range(10)),
    "libero_object": list(range(10)),
    "libero_goal": list(range(10)),
    "libero_10": list(range(10)),
    "arena_gr1_fridge": [0],
}


def _legacy_family_glob():
    """Reproduce stage_validate_code's former inline glob of steps/*/defaults.json."""
    import glob as _glob
    out = {}
    for dj in sorted(_glob.glob(os.path.join(_REPO_ROOT, "entrypoints", "train", "*", "defaults.json"))):
        with open(dj) as fh:
            d = json.load(fh)
        fam = d.get("family") or os.path.basename(os.path.dirname(dj))
        out[fam] = {
            "input_config_schema": d.get("input_config_schema", {}),
            "provenance_keys": d.get("provenance_keys", {}),
        }
    return out


def test_suite_canonical_matches_legacy_map():
    assert suite_canonical_task_ids() == _LEGACY_SUITE_CANONICAL


def test_family_schemas_matches_legacy_glob():
    assert family_schemas() == _legacy_family_glob()


def test_stage_validate_code_injects_both_maps():
    from vla_pipeline.runner import stage_validate_code
    staged = stage_validate_code(repo_root=_REPO_ROOT)
    with open(staged) as fh:
        code = fh.read()
    assert "VLA_FAMILY_SCHEMAS_JSON" in code
    assert "VLA_SUITE_CANONICAL_TASK_IDS_JSON" in code
    # The injected suite map must decode to the registry's canonical map.
    import re
    m = re.search(r'VLA_SUITE_CANONICAL_TASK_IDS_JSON"\]\s*=\s*_b64\.b64decode\("([^"]+)"\)', code)
    assert m, "suite-canonical base64 injection not found in staged validator"
    decoded = json.loads(base64.b64decode(m.group(1)).decode())
    assert decoded == suite_canonical_task_ids()


def test_strict_validator_still_passes_and_fails_incomplete():
    sys.path.insert(0, os.path.dirname(__file__))
    from test_v2_validator import EXPECT, good_report  # noqa: E402
    from validator import ReportInvalid, validate_report  # noqa: E402

    validate_report(good_report(), **EXPECT)  # complete report passes

    incomplete = good_report()
    del incomplete["model_family"]
    with pytest.raises(ReportInvalid):
        validate_report(incomplete, **EXPECT)
