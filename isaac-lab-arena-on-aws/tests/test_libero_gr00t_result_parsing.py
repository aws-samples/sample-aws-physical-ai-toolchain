"""C1: the LIBERO/GR00T success vector must come from ONE identified result record.

The parser used to search the whole rollout output for any bracketed boolean list and
take the last match, with the task identity supplied by the wrapper rather than
recovered from the result. Unrelated log output could therefore manufacture the score
while the checkpoint itself was genuine, so no digest check could detect it.
"""
from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import sys
import types

import pytest

from vla_pipeline.common.sourcedir import stage

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_ENTRY = _REPO_ROOT / "entrypoints/eval/libero/gr00t/eval_entry.py"
# stage() flattens the entry script, the shared modules and the family's
# defaults.json into ONE directory, which is why the entry reads defaults.json from
# beside itself even though the repo keeps it under entrypoints/train/gr00t/.

_staged_module = None


def _load():
    """Load the real entrypoint module from a directory mirroring the staged layout.

    Loading the production module rather than re-implementing the parser is the point:
    a test that pasted its own copy of the regex would keep passing with production
    broken, which is the defect class this suite is meant to catch.
    """
    global _staged_module
    if _staged_module is not None:
        return _staged_module
    os.environ["EVAL_GR00T_VERSION"] = "n17"
    for name, attrs in (("digest", {"weights_digest": lambda *a, **k: "0" * 64}),
                        ("validator", {"validate_report": lambda *a, **k: 0.0})):
        if name not in sys.modules:
            sys.modules[name] = types.SimpleNamespace(**attrs)
    staged = pathlib.Path(stage("gr00t", repo_root=str(_REPO_ROOT)))
    spec = importlib.util.spec_from_file_location(
        "libero_gr00t_eval_entry", staged / "eval_entry.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _staged_module = mod
    return mod


ENV = "libero_spatial_1"


def _ENTRY_SOURCE() -> str:
    return _ENTRY.read_text()


def _load_with_baked(baked: bool):
    """Load the entry with the baked marker present or absent.

    `_BAKED` is decided at import from the filesystem, so the marker must be faked BEFORE
    the module executes.
    """
    os.environ["EVAL_GR00T_VERSION"] = "n17"
    for name, attrs in (("digest", {"weights_digest": lambda *a, **k: "0" * 64}),
                        ("validator", {"validate_report": lambda *a, **k: 0.0})):
        if name not in sys.modules:
            sys.modules[name] = types.SimpleNamespace(**attrs)
    staged = pathlib.Path(stage("gr00t", repo_root=str(_REPO_ROOT)))
    real_isfile = os.path.isfile

    def fake_isfile(path):
        if str(path) == "/opt/vla/.baked_env":
            return baked
        return real_isfile(path)

    os.path.isfile = fake_isfile
    try:
        spec = importlib.util.spec_from_file_location(
            f"libero_gr00t_baked_{baked}", staged / "eval_entry.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        os.path.isfile = real_isfile
    return mod


def test_the_image_bakes_and_verifies_the_libero_client_environment():
    """C7: a baked image had the GR00T server env but no LIBERO CLIENT interpreter.

    The root `uv sync` does not create it -- upstream's setup_libero.sh creates a separate
    libero_uv/.venv, and the evaluator requires it unconditionally. So a "baked" image
    either reinstalled everything at runtime on a GPU node or, once the evaluator started
    trusting the marker and skipping setup, failed outright on the missing interpreter.
    """
    dockerfile = (pathlib.Path(__file__).resolve().parents[1]
                  / "docker/gr00t/Dockerfile").read_text()
    assert "setup_libero.sh" in dockerfile, (
        "the image must run LIBERO setup; the root uv sync does not create its venv")
    # Verified before the marker is written, so the marker cannot claim what is absent.
    assert dockerfile.index("libero_uv/.venv/bin/python") < dockerfile.index(
        "> /opt/vla/.baked_env")
    # The marker names the capability the evaluator checks for.
    assert "libero-client-verified" in dockerfile


def test_a_baked_image_without_the_client_fails_with_the_named_cause():
    """An incorrect image must say so, not report an opaque missing file."""
    source = _ENTRY.read_text()
    assert "LIBERO CLIENT" in source
    assert "its marker overstates what it contains" in source
    assert "do not fall back to installing the client at evaluation" in source


def test_a_baked_image_uses_the_directory_the_build_created():
    """I17: the evaluator looked in Isaac-GR00T while the image bakes gr00t.

    In a baked image that path does not exist, so the baked environment was never used and
    the evaluator reinstalled everything over the network on an expensive GPU node. The
    trainer in the same image already resolves this correctly.
    """
    mod = _load_with_baked(True)
    assert mod._BAKED is True
    assert mod.WORK == "/opt/vla"
    assert mod.GR00T_DIR == "/opt/vla/gr00t"


def test_an_unbaked_image_uses_its_own_runtime_clone():
    """Control: without a marker the evaluator still clones to its own directory."""
    mod = _load_with_baked(False)
    assert mod._BAKED is False
    assert mod.GR00T_DIR.endswith("Isaac-GR00T")


def test_the_installation_is_skipped_when_baked():
    """The expensive network operations must be guarded, not merely reachable."""
    source = _ENTRY.read_text()
    assert "if not _BAKED:" in source
    guarded = source[source.index("if not _BAKED:"):]
    for network_op in ('"apt-get", "update"', '"git", "clone"',
                       "setup_libero.sh", "uv_sync_resilient"):
        assert network_op in guarded, network_op
    # A baked marker whose directory is absent must fail rather than reinstall.
    assert "baked marker present but" in source


def test_the_pinned_commit_is_checked_on_both_paths():
    """A baked image must be the pinned commit too, not merely present."""
    source = _ENTRY.read_text()
    assert "The commit check runs on BOTH paths" in source
    assert source.count("repo commit mismatch") == 1


def _result_file(tmp_path, **overrides):
    payload = {"schema": "gr00t_rollout_result_v1", "env_name": ENV,
               "episode_successes": [True, False, True], "episodes": 3, "successes": 2}
    payload.update(overrides)
    path = tmp_path / "result.json"
    path.write_text(json.dumps(payload))
    return str(path)


def test_the_rollouts_own_result_file_is_the_source_of_the_score(tmp_path):
    """I5: stdout is no longer a trust path at all.

    The previous parser scraped stdout. It was neither anchored -- `results:` matched inside
    `debug anticipated results:` -- nor a parser for the complete tuple: it stopped at the
    closing bracket, so a line truncated by a log flush parsed as a finished result. stdout
    interleaves with every library that logs and anything may print text resembling a
    result, so arbitrary log output could set the reported score. No digest detects that:
    the checkpoint is genuine and only the outcome is fabricated.
    """
    mod = _load()
    assert mod._read_result_file(_result_file(tmp_path), ENV, 3) == [True, False, True]


def test_no_result_file_means_the_outcome_is_unknown(tmp_path):
    mod = _load()
    with pytest.raises(ValueError, match="produced no structured result file"):
        mod._read_result_file(str(tmp_path / "absent.json"), ENV, 3)


def test_a_truncated_result_file_is_not_interpreted(tmp_path):
    """The exact failure the stdout regex could not detect."""
    mod = _load()
    path = tmp_path / "result.json"
    path.write_text('{"schema": "gr00t_rollout_result_v1", "env_name": "libero')
    with pytest.raises(ValueError, match="not valid JSON"):
        mod._read_result_file(str(path), ENV, 3)


def test_a_result_for_another_environment_is_rejected(tmp_path):
    mod = _load()
    with pytest.raises(ValueError, match="describes a different task"):
        mod._read_result_file(_result_file(tmp_path, env_name="libero_sim/other"), ENV, 3)


def test_a_vector_of_the_wrong_length_is_rejected(tmp_path):
    mod = _load()
    with pytest.raises(ValueError, match="but 5 episodes were requested"):
        mod._read_result_file(_result_file(tmp_path), ENV, 5)


def test_an_empty_vector_is_not_an_outcome(tmp_path):
    mod = _load()
    with pytest.raises(ValueError, match="no episode_successes list"):
        mod._read_result_file(
            _result_file(tmp_path, episode_successes=[], episodes=0, successes=0), ENV, 0)


def test_non_boolean_outcomes_are_rejected(tmp_path):
    """1 and 0 are truthy/falsy but are not measurements of success."""
    mod = _load()
    with pytest.raises(ValueError, match="non-boolean episode outcomes"):
        mod._read_result_file(
            _result_file(tmp_path, episode_successes=[1, 0, 1]), ENV, 3)


def test_a_payload_that_disagrees_with_itself_is_rejected(tmp_path):
    """The file carries both a vector and counts; disagreement discredits both."""
    mod = _load()
    with pytest.raises(ValueError, match="its vector sums to 2"):
        mod._read_result_file(_result_file(tmp_path, successes=3), ENV, 3)
    with pytest.raises(ValueError, match="carries 3 outcomes"):
        mod._read_result_file(_result_file(tmp_path, episodes=7), ENV, 3)


def test_an_unrecognised_schema_is_rejected(tmp_path):
    mod = _load()
    with pytest.raises(ValueError, match="unrecognised result format"):
        mod._read_result_file(_result_file(tmp_path, schema="something_else"), ENV, 3)


def test_the_stdout_parser_is_gone(tmp_path):
    """It must not linger as a fallback -- a fallback would restore the whole defect."""
    mod = _load()
    assert not hasattr(mod, "_parse_successes")
    assert not hasattr(mod, "_RESULT_RECORD")
    source = _ENTRY_SOURCE()
    assert '"".join(buf)' not in source, "stdout buffer must not feed the score"


def test_the_wrapper_clears_a_stale_result_before_each_task():
    """Otherwise a task whose rollout wrote nothing inherits the previous task's score."""
    source = _ENTRY_SOURCE()
    assert "if os.path.exists(result_json):" in source
    assert "os.remove(result_json)" in source
    assert source.index("os.remove(result_json)") < source.index(
        "successes = _read_result_file(")


def test_a_changed_upstream_stops_the_run():
    mod = _load()
    pinned = (pathlib.Path(__file__).with_name("data")
              / "rollout_policy_pinned.py").read_text()
    with pytest.raises(RuntimeError, match="result-file patch anchor occurs 0 times"):
        mod._patch_result_file(pinned.replace('    print("results: ", results)\n', "", 1))


def test_the_rollout_refuses_to_finish_without_a_result_path():
    """A missing path must fail, not silently skip writing the authoritative outcome."""
    mod = _load()
    pinned = (pathlib.Path(__file__).with_name("data")
              / "rollout_policy_pinned.py").read_text()
    patched = mod._patch_result_file(pinned)
    assert "GR00T_RESULT_JSON is not set" in patched
    assert "os.replace(_tmp_path, _result_path)" in patched, "must publish atomically"
