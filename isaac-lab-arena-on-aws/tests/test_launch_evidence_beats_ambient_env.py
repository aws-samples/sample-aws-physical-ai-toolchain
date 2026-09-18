"""R4#I2b: the per-launch evidence RECORD must beat an inherited environment variable.

Nothing in eval_entry ever writes GR00T_SERVER_SEED_EVIDENCE into its OWN os.environ -- both
launchers set it only on the child server's env -- so a value visible to the reporting path is
inherited from outside the run and cannot describe the launch being reported. Two launchers
record (start_groot_server for N1.7, start_groot_server_n16 for the N1.6 positive control), so
an ambient value winning attributes one arm's seeding proof to the other.

Discriminator: each evidence file records a DIFFERENT seed, so the validator's own complaint
names which file was read. This test fails if the precedence is inverted.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from unittest.mock import patch

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO_ROOT, "entrypoints", "eval", "isaac_arena", "gr00t", "eval_entry.py")
_ARENA_ONLY = {"isaaclab", "omni", "isaacsim", "arena", "gr00t"}


@pytest.fixture
def eval_entry():
    """Load eval_entry with its required env, restoring ALL global state afterwards.

    Importing this module seeds SM_HP_EMBODIMENT_TAG, SM_HP_TASK_NAME, EVAL_POLICY_CONFIG_YAML,
    EVAL_ARENA_EMBODIMENT and EVAL_OBJECT into os.environ from EVAL_SIM_CONFIG. monkeypatch
    cannot undo those: it only restores keys the TEST set, not keys the module under test set
    itself. Leaving them behind changes which parametrisations of
    tests/test_validate_entry_gate.py::test_arena_generated_report_registration_gate fail, since
    that test stages a copy of eval_entry.py whose behaviour depends on ambient environment.

    patch.dict(os.environ) is the pattern tests/test_eval_sim_config.py already uses for exactly
    this reason. A fixture that perturbs another test's outcome is worse than no fixture.
    """
    saved_modules = dict(sys.modules)
    saved_path = list(sys.path)
    with patch.dict(os.environ):
        os.environ["EVAL_SIM_CONFIG"] = json.dumps({
            "embodiment_tag": "new_embodiment",
            "task_name": "fixture_task",
            "policy_config_yaml": "/workspace/x.yaml",
            "arena_embodiment": "gr1_joint",
            "object": "brown_box",
        })
        spec = importlib.util.spec_from_file_location(f"ee_i2b_{len(sys.modules)}", _SRC)
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        except ModuleNotFoundError as exc:
            if (exc.name or "").split(".")[0] not in _ARENA_ONLY:
                raise
            pytest.skip(f"eval_entry needs an Arena-only module absent here: {exc!r}")
        try:
            yield mod
        finally:
            sys.path[:] = saved_path
            for name in list(sys.modules):
                if name not in saved_modules:
                    del sys.modules[name]
            sys.modules.update(saved_modules)


def _write_evidence(path, checkpoint, seed):
    with open(path, "w") as handle:
        json.dump({
            "evidence_schema": "gr00t_server_evidence_v1",
            "processor_validated": checkpoint,
            "seed": seed,
            "policy_rng_bound": True,
            "server_port": 5555,
        }, handle)


def test_each_launch_reads_its_own_evidence_not_the_ambient_var(eval_entry, tmp_path, monkeypatch):
    n17 = tmp_path / "ckpt_n17"
    n16 = tmp_path / "ckpt_n16"
    n17.mkdir()
    n16.mkdir()
    ev_n17 = tmp_path / "ev_n17.json"
    ev_n16 = tmp_path / "ev_n16.json"
    decoy = tmp_path / "DECOY.json"
    _write_evidence(ev_n17, str(n17), 7001)
    _write_evidence(ev_n16, str(n16), 7002)
    _write_evidence(decoy, "/elsewhere", 9999)

    # The inherited-from-container case the finding is about.
    monkeypatch.setenv("GR00T_SERVER_SEED_EVIDENCE", str(decoy))
    # Both launchers record their per-launch path.
    eval_entry._LAUNCH_EVIDENCE[os.path.realpath(str(n17))] = str(ev_n17)
    eval_entry._LAUNCH_EVIDENCE[os.path.realpath(str(n16))] = str(ev_n16)

    for checkpoint, own_seed in ((n17, 7001), (n16, 7002)):
        blob = json.dumps(eval_entry._seed_scope_for_report(
            checkpoint_path=str(checkpoint), port=5555))
        assert "9999" not in blob, (
            f"{checkpoint.name} read the ambient decoy instead of its own launch record")
        assert str(own_seed) in blob, (
            f"{checkpoint.name} read neither its own evidence nor the decoy: {blob[:300]}")


def test_ambient_var_is_still_the_fallback_when_no_launch_was_recorded(eval_entry, tmp_path, monkeypatch):
    """The bypass case: with no record and no derivable path, the env var is legitimately used."""
    baked = tmp_path / "baked.json"
    _write_evidence(baked, "/baked/layout", 4242)
    monkeypatch.setenv("GR00T_SERVER_SEED_EVIDENCE", str(baked))
    eval_entry._LAUNCH_EVIDENCE.clear()
    # No checkpoint_path -> nothing recorded and nothing derivable, so the env var must be used.
    blob = json.dumps(eval_entry._seed_scope_for_report(checkpoint_path=None, port=None))
    assert "4242" in blob, f"the env var fallback was not consulted: {blob[:300]}"
