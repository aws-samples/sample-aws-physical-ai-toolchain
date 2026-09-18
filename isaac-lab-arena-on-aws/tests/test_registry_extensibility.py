"""Isolated test: registry extensibility.

Adding a model = manifest + entry scripts only; adding a sim = manifest only.
Proven by: the dummy family + dummy simulator (added purely as config/ +
steps/dummy stubs, with NO edits to registry.py or pipeline.py) resolve()
cleanly, and the param-driven pipeline definition still builds/serializes.
"""
from __future__ import annotations

import json
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if os.path.join(_REPO_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from vla_pipeline.registry import (  # noqa: E402
    family_schemas,
    list_models,
    list_simulators,
    list_supported_pairs,
    resolve,
    suite_canonical_task_ids,
)


def test_dummy_model_and_sim_registered_via_manifests_only():
    assert "dummy" in list_models()
    assert "dummy_sim" in list_simulators()
    assert ("dummy", "dummy_sim") in list_supported_pairs()


def test_dummy_pair_resolves():
    spec = resolve("dummy", "dummy_sim")
    assert spec.model_family == "dummy"
    assert spec.simulator == "dummy_sim"
    assert spec.train_image_repo == "vla/dummy:v0"
    assert spec.eval_image_repo == "vla/dummy:v0"
    assert spec.registry_group == "vla-demo-dummy"
    assert spec.default_suite == "dummy_suite"
    assert spec.delivery_mode == "sourcedir"
    assert "dummy" in family_schemas()


def test_dummy_sim_adds_no_canonical_suites():
    # dummy_suite has a real manifest now (the demo pair's default_suite pointed
    # at an undeclared suite before), but it is `status: experimental`, so it stays
    # out of the canonical map and the real simulators are unaffected -- no
    # accidental coupling from adding a demo cell.
    canonical = suite_canonical_task_ids()
    assert "dummy_suite" not in canonical
    assert "arena_gr1" not in canonical  # experimental -- excluded
    assert canonical["arena_gr1_fridge"] == [0]  # real sim map intact


def test_pipeline_builds_with_no_orchestration_edits():
    # The pipeline is param-driven: a NEW pair needs zero orchestration code, so
    # the definition still serializes with pipeline.py untouched for the dummy.
    from vla_pipeline.config import PipelineConfig
    from vla_pipeline.pipeline import build_pipeline
    cfg = PipelineConfig(
        account_id="000000000000", region="us-east-1",
                          training_role_arn="arn:aws:iam::000000000000:role/train", workload_role_arn="arn:aws:iam::000000000000:role/wl", validation_role_arn="arn:aws:iam::000000000000:role/val", trust_bucket="trust-bucket", handoff_bucket="handoff-bucket",
        role_arn="arn:aws:iam::000000000000:role/r", bucket="b")
    definition = build_pipeline(cfg).definition()
    json.loads(definition)  # valid, unchanged pipeline handles any resolved pair


def test_require_rejects_empty_and_missing():
    # Review #12: a required field that is missing OR empty/whitespace must fail
    # loud, not slip through (a common manifest typo).
    import pytest

    from vla_pipeline.registry import RegistryError, _require
    with pytest.raises(RegistryError):
        _require({}, "x", "p")            # missing
    with pytest.raises(RegistryError):
        _require({"x": ""}, "x", "p")     # empty string
    with pytest.raises(RegistryError):
        _require({"x": "  "}, "x", "p")   # whitespace only
    assert _require({"x": "ok"}, "x", "p") == "ok"
