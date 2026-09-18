"""Offline unit tests for the Arena eval_entry `_patch_policy_config_model_path`
helper (the model_path reconciliation added for the reproducible stock-Arena base).
No AWS, no GPU. Loads the baked eval_entry module by file path."""
import importlib.util
import io
import json
import os
import pathlib
import runpy
import shlex
import shutil
import sys
import threading
import types
import zipfile
from unittest.mock import Mock

import pytest

_EVAL_ENTRY = (pathlib.Path(__file__).resolve().parents[1]
               / "entrypoints/eval/isaac_arena/gr00t/eval_entry.py")


@pytest.mark.parametrize("marker", ["null", "NULL", "None"])
def test_s3_identity_rejects_an_unversioned_bucket_marker(monkeypatch, marker):
    """An unversioned bucket does not omit VersionId -- it returns the string "null".

    The producer's falsy check passed that straight through, so its docstring promise
    never to record "null" was defeated by S3 supplying the value itself rather than by
    the function fabricating it. Downstream then treated it as versioned identity.
    """
    module = _load()

    class _Client:
        def head_object(self, Bucket, Key):
            return {"VersionId": marker, "ETag": '"abc"'}

    fake_boto3 = types.SimpleNamespace(client=lambda name: _Client())
    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)
    with pytest.raises(RuntimeError, match="UNVERSIONED"):
        module._s3_head_identity("s3://bucket/key/model.tar.gz")


def test_s3_identity_accepts_a_real_version_id(monkeypatch):
    """Control: a genuine version id still produces the identity record."""
    module = _load()

    class _Client:
        def head_object(self, Bucket, Key):
            return {"VersionId": "3sL4kqtJlcpXroDTDmJ.O1", "ETag": '"abc"'}

    monkeypatch.setitem(sys.modules, "boto3",
                        types.SimpleNamespace(client=lambda name: _Client()))
    ident = module._s3_head_identity("s3://bucket/key/model.tar.gz")
    assert ident["version_id"] == "3sL4kqtJlcpXroDTDmJ.O1"
    assert ident["bucket"] == "bucket" and ident["key"] == "key/model.tar.gz"
    assert ident["etag"] == "abc"


def _ENTRY_SOURCE() -> str:
    return _EVAL_ENTRY.read_text()


_REPO = pathlib.Path(__file__).resolve().parents[1]


def _load():
    os.environ.setdefault("SM_HP_TASK_NAME", "fixture_task")
    # The baked image places the shared modules FLAT in /workspace beside eval_entry.py, and the
    # evaluator imports them lazily by bare name inside the functions that need them. Put the real
    # ones on sys.path so those call sites resolve exactly as they do in the image.
    for _d in ("src/vla_pipeline/common", "entrypoints/eval/isaac_arena/_shared"):
        _p = str(_REPO / _d)
        if _p not in sys.path:
            sys.path.insert(0, _p)
    spec = importlib.util.spec_from_file_location("arena_eval_entry", _EVAL_ENTRY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mod = _load()
patch = mod._patch_policy_config_model_path


def _write(tmp_path, text):
    p = tmp_path / "cfg.yaml"
    p.write_text(text)
    return str(p)


def test_top_level_model_path_replaced(tmp_path):
    src = "model_path: /models/placeholder/checkpoint-20000\nembodiment_tag: GR1\n"
    out = patch(_write(tmp_path, src), "/tmp/checkpoint")
    body = pathlib.Path(out).read_text()
    assert "model_path: /tmp/checkpoint\n" in body
    assert "embodiment_tag: GR1" in body            # other keys preserved
    assert "placeholder" not in body


def test_indentation_preserved(tmp_path):
    src = "server:\n  model_path: /models/placeholder\n  port: 5555\n"
    out = patch(_write(tmp_path, src), "/tmp/ckpt")
    body = pathlib.Path(out).read_text()
    assert "  model_path: /tmp/ckpt\n" in body       # 2-space indent kept
    assert "  port: 5555" in body


def test_duplicate_model_path_keys_rejected(tmp_path):
    """An ambiguous config must be refused, not silently half-patched.

    This previously asserted that only the FIRST occurrence is rewritten and the
    second is "left intact" -- but a line rewriter cannot know which duplicate the
    YAML loader honours. If a later one wins, the served checkpoint is the unpatched
    placeholder and the evaluation measures a different model than the one being
    registered.
    """
    src = "model_path: /a\nother: x\nmodel_path: /b\n"
    with pytest.raises(RuntimeError, match="declares 'model_path' 2 times"):
        patch(_write(tmp_path, src), "/tmp/ckpt")


def test_no_model_path_key_raises(tmp_path):
    src = "embodiment_tag: GR1\naction_horizon: 16\n"
    with pytest.raises(RuntimeError, match="no 'model_path:' key"):
        patch(_write(tmp_path, src), "/tmp/ckpt")


def test_unreadable_config_raises(tmp_path):
    with pytest.raises(RuntimeError, match="cannot read policy config"):
        patch(str(tmp_path / "does_not_exist.yaml"), "/tmp/ckpt")


def test_does_not_false_match_substring_key(tmp_path):
    # `base_model_path:` must NOT be treated as the model_path key.
    src = "base_model_path: /keep/me\nmodel_path: /models/placeholder\n"
    out = patch(_write(tmp_path, src), "/tmp/ckpt")
    body = pathlib.Path(out).read_text()
    assert "base_model_path: /keep/me" in body       # untouched
    assert "model_path: /tmp/ckpt\n" in body


@pytest.mark.parametrize("seed", [0, 100, 2026])
@pytest.mark.parametrize("policy_type", ["zero_action", "gr00t_remote"])
def test_requested_seed_reaches_runner(monkeypatch, tmp_path, seed, policy_type):
    monkeypatch.setenv("EVAL_SEED", str(seed))
    monkeypatch.setenv("SM_HP_EVAL_SEED", "42")
    policy_config = tmp_path / "policy.yaml"
    policy_config.write_text("model_path: /fixture/checkpoint\n")
    monkeypatch.setenv("EVAL_POLICY_CONFIG_YAML", str(policy_config))
    module = _load()
    monkeypatch.setattr(module.shutil, "which", lambda name: None)
    timer = Mock()
    monkeypatch.setattr(threading, "Timer", Mock(return_value=timer))
    process = Mock(stdout=io.StringIO(
        "[Rank 0/1] Metrics: {'num_episodes': 1, 'success_rate': 1.0}\n"))
    process.wait.return_value = 0
    popen = Mock(return_value=process)
    monkeypatch.setattr(module.subprocess, "Popen", popen)

    result = module.run_arena_eval("fixture_task", 1, num_episodes=1,
                                   policy_type=policy_type)

    command = popen.call_args.args[0]
    expected_policy = (
        "isaaclab_arena_gr00t.policy.gr00t_remote_closedloop_policy.Gr00tRemoteClosedloopPolicy"
        if policy_type == "gr00t_remote" else "zero_action"
    )
    assert command[command.index("--policy_type") + 1] == expected_policy
    popen.assert_called_once()
    # The budget is an episode count, never a step count: a step budget below the
    # environment's episode length silently produced zero completed episodes.
    assert command[command.index("--num_episodes") + 1] == "1"
    assert "--num_steps" not in command
    assert command.count("--seed") == 1
    assert command[command.index("--seed") + 1] == str(seed)
    assert command.index("--seed") < command.index("fixture_task")
    assert module.EVAL_SEED == seed
    assert result["episodes"] == 1
    assert result["success_rate"] == 1.0
    assert result["task_name"] == "fixture_task"
    assert result["policy_type"] == ("checkpoint" if policy_type == "gr00t_remote" else policy_type)
    timer.cancel.assert_called_once()


@pytest.mark.parametrize("ambient_version", [None, "n16", "n17"])
def test_server_environment_uses_explicit_version(monkeypatch, ambient_version):
    module = _load()
    if ambient_version is None:
        monkeypatch.delenv("EVAL_GR00T_VERSION", raising=False)
    else:
        monkeypatch.setenv("EVAL_GR00T_VERSION", ambient_version)
    monkeypatch.setenv("LD_LIBRARY_PATH", "/isaac-sim/libs:/omni/libs:/usr/local/lib")
    for key in module._ISAAC_ENV_KEYS:
        monkeypatch.setenv(key, "/isaac-sim/python")
    discover = Mock(return_value=["/fixture/nvidia/cudnn/lib"])
    monkeypatch.setattr(module.glob, "glob", discover)

    native = module._build_server_env("n16")
    discover.assert_not_called()
    assert native["LD_LIBRARY_PATH"] == "/usr/local/lib"
    assert native["PYTHONNOUSERSITE"] == "1"
    assert all(key not in native for key in module._ISAAC_ENV_KEYS)

    modern = module._build_server_env("n17")
    discover.assert_called_once()
    assert modern["LD_LIBRARY_PATH"] == "/fixture/nvidia/cudnn/lib:/usr/local/lib"


def test_n17_server_requires_its_libraries(monkeypatch):
    module = _load()
    monkeypatch.setattr(module.glob, "glob", Mock(return_value=[]))
    with pytest.raises(SystemExit):
        module._build_server_env("n17")


def test_n16_server_launch_without_version_environment(monkeypatch, tmp_path):
    module = _load()
    monkeypatch.delenv("EVAL_GR00T_VERSION", raising=False)
    monkeypatch.setattr(module, "GR00T_N16_SERVER_LOG", str(tmp_path / "server.log"))
    discover = Mock(return_value=[])
    monkeypatch.setattr(module.glob, "glob", discover)
    launch = Mock()
    monkeypatch.setattr(module.subprocess, "Popen", launch)

    # A real N1.6 checkpoint: the evaluator refuses one whose architecture it cannot determine, after
    # a run spent a full server-startup timeout discovering an N1.6 checkpoint on the n17 loader. The
    # previous "/fixture/checkpoint" did not exist at all.
    _ckpt = tmp_path / "checkpoint"
    _ckpt.mkdir()
    (_ckpt / "config.json").write_text('{"model_type": "Gr00tN1d6"}')
    module.start_groot_server_n16(str(_ckpt), 5555)

    discover.assert_not_called()
    assert launch.call_args.args[0][0] == module.GR00T_N16_VENV
    # I11: the pinned server never read EVAL_SEED, so the policy's inference noise was
    # unbound while Arena's client-side seeding made the scene sequence look
    # reproducible. Both servers must launch THROUGH the seeding wrapper.
    assert launch.call_args.args[0][1] == module.SEEDED_SERVER_ENTRY
    assert "-m" not in launch.call_args.args[0], (
        "the wrapper supplies the server module itself; passing -m would forward it as "
        "a server argument")
    launch.call_args.kwargs["stdout"].close()


@pytest.fixture
def checkpoint_run(tmp_path, monkeypatch):
    repo = _EVAL_ENTRY.parents[4]
    packaged = tmp_path / "packaged"
    packaged.mkdir()
    for source in (
        _EVAL_ENTRY,
        repo / "entrypoints/eval/isaac_arena/_shared/digest.py",
        repo / "src/vla_pipeline/common/validator.py",
        # The evaluator refuses a checkpoint whose architecture the selected loader cannot load, so this
        # fixture -- a stand-in for the baked /workspace -- must carry the module. Its absence
        # surfaced as ModuleNotFoundError from the n16 launcher, which is this fixture correctly
        # modelling an image that lacks the COPY line.
        repo / "src/vla_pipeline/common/checkpoint_compat.py",
        repo / "entrypoints/train/gr00t/defaults.json",
    ):
        shutil.copy2(source, packaged / source.name)
    for name in ("digest", "validator", "checkpoint_compat"):
        spec = importlib.util.spec_from_file_location(name, packaged / f"{name}.py")
        dependency = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(dependency)
        monkeypatch.setitem(sys.modules, name, dependency)
    spec = importlib.util.spec_from_file_location("packaged_arena", packaged / "eval_entry.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv("EVAL_MODEL_FAMILY", "gr00t")
    root = tmp_path / "checkpoint"
    points = [(root, 300)] + [
        (root / ".dose_checkpoints" / f"checkpoint-{step}", step)
        for step in (100, 200, 300)
    ]
    for point, step in points:
        point.mkdir(parents=True, exist_ok=True)
        (point / "weights.bin").write_bytes(f"checkpoint-{step}".encode())
        manifest = {
            "manifest_version": 1, "model_family": "gr00t",
            "base_checkpoint": "fixture/policy", "base_revision": "a" * 40,
            "train_seed": 42,
            "input_config": {
                "embodiment_tag": "GR1", "n_action_steps": "8", "max_episode_steps": "720",
            },
            "train_recipe": {"repo": "fixture", "commit": "b" * 40, "max_steps": step},
            "weights_digest": sys.modules["digest"].weights_digest(str(point)),
            "dataset_manifest": {"source": "fixture", "revision": "c" * 40, "episode_count": 10},
        }
        (point / "checkpoint_manifest.json").write_text(json.dumps(manifest))
    output = tmp_path / "output"
    output.mkdir()
    monkeypatch.setattr(module, "MODEL_DIR", str(output))
    monkeypatch.setattr(module, "extract_checkpoint", Mock(return_value=str(root)))
    monkeypatch.setenv("EVAL_POSCTRL_N16", "false")
    monkeypatch.setenv("SM_HP_USE_GROOT_SERVER", "true")
    monkeypatch.setenv("EVAL_DOSE_STEPS", "100,200")
    policy_config = tmp_path / "policy.yaml"
    policy_config.write_text("model_path: /placeholder\n")
    monkeypatch.setenv("EVAL_POLICY_CONFIG_YAML", str(policy_config))
    monkeypatch.setattr(module, "ensure_gr00t_venv", Mock())
    monkeypatch.setattr(module, "ensure_gr00t_venv_n16", Mock())
    monkeypatch.setattr(module, "resolve_hf_token", Mock(return_value="fixture"))
    monkeypatch.setattr(module, "precache_cosmos", Mock())
    for name in ("start_groot_server", "start_groot_server_n16"):
        monkeypatch.setattr(module, name, Mock(side_effect=lambda *args: Mock(poll=lambda: None)))
    monkeypatch.setattr(module, "wait_for_server", Mock(return_value=True))
    # Episode counts MUST equal the requested EVAL_TRIALS: under the fixed-trials
    # protocol the evaluated sample is exactly what was asked for, and the dose point's
    # `ok` flag is derived from that equality. Request 2 so the middle point's 0.5 is a
    # realizable outcome (1 success of 2) -- success is binary per episode, so 0.5 out
    # of a single episode is half a successful episode and cannot happen.
    monkeypatch.setattr(module, "EVAL_TRIALS", 2)
    monkeypatch.setattr(module, "run_arena_eval", Mock(side_effect=[
        {"success_rate": 1.0, "episodes": 2},
        {"success_rate": 0.5, "episodes": 2},
        {"success_rate": 0.0, "episodes": 2},
    ]))
    monkeypatch.setattr(module, "write_metrics", Mock(side_effect=lambda results, *args: results))
    return module, root, output


@pytest.mark.parametrize("version", ["n16", "n17"])
def test_checkpoint_curve_evaluates_requested_points(checkpoint_run, monkeypatch, version):
    module, root, output = checkpoint_run
    # EVAL_GR00T_VERSION is normalized and validated ONCE at module import (so a value
    # like "N16" cannot make the guards, the launcher and the report disagree), which
    # means setting the env after import has no effect. Patch the module constant.
    monkeypatch.setattr(module, "EVAL_GR00T_VERSION", version)
    module.main()
    expected = [str(root / ".dose_checkpoints" / f"checkpoint-{step}") for step in (100, 200)]
    expected.append(str(root))
    assert [call.kwargs["checkpoint_path"] for call in module.run_arena_eval.call_args_list] == expected
    launcher = module.start_groot_server_n16 if version == "n16" else module.start_groot_server
    assert [call.args[0] for call in launcher.call_args_list] == expected
    other = module.start_groot_server if version == "n16" else module.start_groot_server_n16
    other.assert_not_called()
    module.write_metrics.assert_called_once_with(
        {"success_rate": 0.0, "episodes": 2}, str(output), str(root))
    curve = json.loads((output / "dose_curve.json").read_text())
    assert [point["dose_steps"] for point in curve["points"]] == [100, 200, 300]
    # `ok` is derived from the evidence (observed episodes == requested trials, rate in
    # range), not hardcoded. It was previously always True, so a point that produced no
    # usable measurement was still stamped successful in a sidecar that gates nothing.
    assert all(point["ok"] for point in curve["points"])
    assert not any("not_ok_reason" in point for point in curve["points"])
    assert [point["canonical"] for point in curve["points"]] == [False, False, True]
    assert [point["success_rate"] for point in curve["points"]] == [1.0, 0.5, 0.0]
    assert module.precache_cosmos.call_count == (1 if version == "n17" else 0)


@pytest.mark.parametrize("version", ["n16", "n17"])
def test_requested_checkpoint_failure_stops_run(checkpoint_run, monkeypatch, version):
    module, _, output = checkpoint_run
    monkeypatch.setenv("EVAL_GR00T_VERSION", version)
    failure = RuntimeError("checkpoint rollout failed")
    module.run_arena_eval.side_effect = failure
    with pytest.raises(RuntimeError, match="checkpoint rollout failed") as exc:
        module.main()
    assert exc.value is failure
    module.write_metrics.assert_not_called()
    assert not (output / "dose_curve.json").exists()


@pytest.mark.parametrize("selection", ["999", "100,999", "100,,200", "0", "-1", "100,100", "bad"])
def test_invalid_checkpoint_selection_fails_before_setup(checkpoint_run, monkeypatch, selection):
    module, _, _ = checkpoint_run
    monkeypatch.setenv("EVAL_GR00T_VERSION", "n16")
    monkeypatch.setenv("EVAL_DOSE_STEPS", selection)
    with pytest.raises(RuntimeError, match="EVAL_DOSE_STEPS"):
        module.main()
    module.ensure_gr00t_venv_n16.assert_not_called()
    module.run_arena_eval.assert_not_called()


@pytest.mark.parametrize("version", ["n16", "n17"])
def test_final_only_selection_writes_no_curve(checkpoint_run, monkeypatch, version):
    module, root, output = checkpoint_run
    monkeypatch.setenv("EVAL_GR00T_VERSION", version)
    monkeypatch.setenv("EVAL_DOSE_STEPS", "300")
    module.main()
    assert module.run_arena_eval.call_count == 1
    assert module.run_arena_eval.call_args.kwargs["checkpoint_path"] == str(root)
    assert not (output / "dose_curve.json").exists()


@pytest.mark.parametrize("selection", ["all", "[0]", " [0] "])
def test_single_task_selection_accepted(monkeypatch, selection):
    monkeypatch.setenv("EVAL_TASK_IDS", selection)
    assert _load().requested_task_ids() == [0]


@pytest.mark.parametrize("selection", ["", "[]", "[0,1]", "[0,0]", "[false]", "[0.0]", "[1]", "null", "bad"])
def test_invalid_task_selection_rejected_before_setup(checkpoint_run, monkeypatch, selection):
    module, _, _ = checkpoint_run
    monkeypatch.setenv("EVAL_TASK_IDS", selection)
    with pytest.raises(ValueError, match="EVAL_TASK_IDS"):
        module.main()
    module.extract_checkpoint.assert_not_called()
    module.ensure_gr00t_venv.assert_not_called()
    module.ensure_gr00t_venv_n16.assert_not_called()
    module.run_arena_eval.assert_not_called()
    module.write_metrics.assert_not_called()


@pytest.mark.parametrize("version", ["n16", "n17"])
@pytest.mark.parametrize("point_name", ["canonical", "intermediate"])
@pytest.mark.parametrize("defect", ["weights", "family", "schema", "missing_manifest"])
def test_checkpoint_preflight_rejects_before_server_setup(
    checkpoint_run, monkeypatch, version, point_name, defect
):
    module, root, output = checkpoint_run
    monkeypatch.setenv("EVAL_GR00T_VERSION", version)
    point = root if point_name == "canonical" else root / ".dose_checkpoints/checkpoint-200"
    manifest_path = point / "checkpoint_manifest.json"
    if defect == "weights":
        (point / "weights.bin").write_bytes(b"corrupted weights")
        expected = "checkpoint digest differs"
    elif defect == "missing_manifest":
        manifest_path.rename(point / "missing.json")
        expected = "checkpoint_manifest.json"
    else:
        manifest = json.loads(manifest_path.read_text())
        if defect == "family":
            manifest["model_family"] = "openvla"
            expected = "model_family differs"
        else:
            del manifest["input_config"]["embodiment_tag"]
            expected = "missing required key embodiment_tag"
        manifest_path.write_text(json.dumps(manifest))
    with pytest.raises((ValueError, FileNotFoundError), match=expected):
        module.main()
    for name in ("ensure_gr00t_venv", "ensure_gr00t_venv_n16", "resolve_hf_token",
                 "precache_cosmos", "start_groot_server", "start_groot_server_n16",
                 "run_arena_eval", "write_metrics"):
        getattr(module, name).assert_not_called()
    assert not list(output.iterdir())


@pytest.mark.parametrize("mode", ["n16", "n17", "positive_control"])
@pytest.mark.parametrize("config", [None, "", "AUTO", "missing", "empty", "directory", "unreadable"])
def test_policy_config_rejected_before_setup(checkpoint_run, monkeypatch, tmp_path, mode, config):
    module, _, output = checkpoint_run
    monkeypatch.setenv("EVAL_GR00T_VERSION", "n17" if mode == "n17" else "n16")
    monkeypatch.setenv("EVAL_POSCTRL_N16", str(mode == "positive_control").lower())
    download = Mock()
    monkeypatch.setattr(module, "download_n16_posctrl_ckpt", download)
    if config is None:
        monkeypatch.delenv("EVAL_POLICY_CONFIG_YAML", raising=False)
    elif config in ("", "AUTO"):
        monkeypatch.setenv("EVAL_POLICY_CONFIG_YAML", config)
    else:
        path = tmp_path / "invalid-policy.yaml"
        if config == "empty":
            path.write_text(" \n")
        elif config == "directory":
            path.mkdir()
        elif config == "unreadable":
            original_read = pathlib.Path.read_text

            def read_text(self, *args, **kwargs):
                if self == path:
                    raise PermissionError("injected policy read failure")
                return original_read(self, *args, **kwargs)

            monkeypatch.setattr(pathlib.Path, "read_text", read_text)
        monkeypatch.setenv("EVAL_POLICY_CONFIG_YAML", str(path))
    with pytest.raises(ValueError, match="EVAL_POLICY_CONFIG_YAML"):
        module.main()
    for name in ("extract_checkpoint", "ensure_gr00t_venv", "ensure_gr00t_venv_n16",
                 "resolve_hf_token", "precache_cosmos", "start_groot_server",
                 "start_groot_server_n16", "run_arena_eval", "write_metrics"):
        getattr(module, name).assert_not_called()
    download.assert_not_called()
    assert not list(output.iterdir())


@pytest.mark.parametrize("version", ["n16", "n17"])
def test_missing_baked_environment_stops_without_installing(tmp_path, monkeypatch, version):
    module = _load()
    repo = tmp_path / "repo"
    prefix = "GR00T_N16" if version == "n16" else "GR00T"
    monkeypatch.setattr(module, f"{prefix}_DIR", str(repo))
    monkeypatch.setattr(module, f"{prefix}_VENV", str(repo / ".venv/bin/python"))
    run = Mock(side_effect=AssertionError("runtime installation is forbidden"))
    monkeypatch.setattr(module.subprocess, "run", run)
    setup = module.ensure_gr00t_venv_n16 if version == "n16" else module.ensure_gr00t_venv
    with pytest.raises(RuntimeError, match="baked .* GR00T venv not found"):
        setup()
    run.assert_not_called()
    assert not repo.exists()


def test_connector_archive_contains_dockerfile_inputs():
    root = _EVAL_ENTRY.parents[4]
    builder = runpy.run_path(str(root / "scripts/build_arena_connector.py"))
    payload = builder["create_source_zip"]("gr00t")
    toolchain_root = pathlib.Path(builder["REPO_ROOT"])
    dockerfile = toolchain_root / "containers/isaac-lab-arena/Dockerfile"
    sources = {
        source
        for line in dockerfile.read_text().splitlines()
        if line.startswith("COPY ")
        for source in shlex.split(line)[1:-1]
    }
    assert sources
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = archive.namelist()
        assert len(names) == len(set(names))
        assert sources <= set(names)
        assert builder["BUILDSPEC"]["gr00t"] in names
        assert archive.testzip() is None
        for name in names:
            assert archive.read(name) == (toolchain_root / name).read_bytes()


@pytest.mark.parametrize("field,value", [
    ("num_episodes", True), ("num_episodes", 1.5), ("num_episodes", "2"),
    ("num_episodes", None), ("num_episodes", -1),
    # 0 episodes = the step budget never produced an episode boundary, so every rate
    # was averaged over an empty sample. Restored: 461f7b4 deleted this case to let a
    # zero-episode rollout report success, which is exactly the defect it must catch.
    ("num_episodes", 0),
    ("success_rate", True), ("success_rate", "0.5"), ("success_rate", None),
    ("success_rate", -0.1), ("success_rate", 1.1),
])
def test_malformed_arena_measurements_rejected(field, value):
    metrics = {"num_episodes": 2, "success_rate": 0.5}
    metrics[field] = value
    with pytest.raises(RuntimeError):
        _load().parse_arena_output(f"[Rank 0/1] Metrics: {metrics!r}")


def test_observed_episode_count_must_equal_requested():
    """The printed count comes from the recorder dataset, not the loop counter, so an
    over- or under-count (e.g. num_envs > 1 ending several episodes in one step) must
    be rejected rather than trusted."""
    out = "[Rank 0/1] Metrics: {'num_episodes': 4, 'success_rate': 0.5}"
    assert _load().parse_arena_output(out, expected_episodes=4)["episodes"] == 4
    for requested in (3, 5):
        with pytest.raises(RuntimeError):
            _load().parse_arena_output(out, expected_episodes=requested)


def test_an_impossible_score_is_not_published(monkeypatch, tmp_path):
    """I6: the producer published reports its own consumer considers impossible.

    Three episodes at 0.5 is one and a half successful episodes. The shared validator
    rejects it, but Arena published without calling that validator -- so a standalone
    SimEval completed successfully while writing invalid metrics to disk, and only the
    pipeline's Validate step would have caught it.
    """
    module = _load()
    monkeypatch.setattr(module, "EVAL_TRIALS", 3)
    # Publication refuses without a source identity and that guard fires first, so build a real
    # one; this test is about realizability, not identity.
    module._RESOLVED = _resolved_over(tmp_path, "impossible_score")
    # write_metrics imports these from the staged sourcedir; only the realizability check
    # matters here, and it runs before either is used.
    monkeypatch.setitem(sys.modules, "digest", types.SimpleNamespace(
        weights_digest=lambda *a, **k: "0" * 64))
    monkeypatch.setitem(sys.modules, "validator", types.SimpleNamespace(
        validate_report=lambda *a, **k: 0.0, validate_manifest=lambda *a, **k: None))
    results = {"eval_backend": "isaac_lab_arena", "episodes": 3, "success_rate": 0.5,
               "policy_type": "checkpoint",
               "task_name": "put_item_in_fridge_and_close_door"}
    output = tmp_path / "out"
    with pytest.raises(RuntimeError, match="not realizable from 3 episode"):
        module.write_metrics(results, str(output), str(tmp_path / "checkpoint"))
    assert not (output / "metrics.json").exists(), (
        "an impossible score must not reach disk")


def test_nonfinite_tokens_do_not_corrupt_identifiers_or_strings():
    """A blind str.replace('inf'/'nan') mangles any key containing those substrings.
    Only bare tokens in value position may be substituted."""
    result = _load().parse_arena_output(
        "[Rank 0/1] Metrics: {'num_episodes': 3, 'success_rate': 0.6666666666666666, "
        "'inference_latency': 1.25, 'label': 'nan', "
        "'object_moved_rate_subtask_0': nan}"
    )
    aux = result["aux_metrics"]
    assert aux["inference_latency"] == 1.25          # not 'Noneerence_latency'
    assert aux["label"] == "nan"                      # quoted string preserved
    assert aux["object_moved_rate_subtask_0"] is None  # explicit "not finite"


def test_unrelated_later_metrics_line_does_not_displace_the_record():
    """Arena emits a cumulative record at every episode boundary AND at completion, so
    a sequence is expected and the last rank-qualified record wins -- but a stray line
    that merely contains 'Metrics:' must not be mistaken for one."""
    result = _load().parse_arena_output(
        "[Rank 0/1] Metrics: {'num_episodes': 1, 'success_rate': 1.0}\n"
        "[Rank 0/1] Metrics: {'num_episodes': 3, 'success_rate': 0.6666666666666666}\n"
        "GPU Metrics: {'mem': 123}\n"
    )
    assert result["episodes"] == 3
    # 2 of 3 -- a realizable outcome. Three episodes at 0.5 would be one and a
    # half successful episodes, which write_metrics now rejects (I6).
    assert abs(result["success_rate"] - 2 / 3) < 1e-9


def test_suffixed_subtask_metrics_are_preserved():
    """Arena spells this metric with a per-subtask suffix; an unsuffixed lookup
    silently matched nothing and the reported measurement was dropped."""
    result = _load().parse_arena_output(
        "[Rank 0/1] Metrics: {'num_episodes': 3, 'success_rate': 0.0, "
        "'revolute_joint_moved_rate_subtask_1': 1.0, 'subtask_success_rate': [0.0]}"
    )
    assert result["aux_metrics"]["revolute_joint_moved_rate_subtask_1"] == 1.0
    assert result["aux_metrics"]["subtask_success_rate"] == [0.0]


@pytest.mark.parametrize("output", [
    "success_rate=0.5", "Metrics: {}", "Metrics: {'success_rate': 0.5}",
    "Metrics: {'num_episodes': 2}",
    "Metrics: {'num_episodes': 2, 'success_rate': 0.5}\nMetrics: truncated",
])
def test_incomplete_arena_measurements_rejected(output):
    with pytest.raises(RuntimeError):
        _load().parse_arena_output(output)


def test_arena_measurements_preserve_final_summary():
    result = _load().parse_arena_output(
        "startup\nMetrics: {'num_episodes': 1, 'success_rate': 1.0}\n"
        "[Rank 0/1] Metrics: {'num_episodes': 3, 'success_rate': 0.6666666666666666, "
        "'revolute_joint_moved_rate_subtask_1': 0.0}\nshutdown"
    )
    assert result == {
        "eval_backend": "isaac_lab_arena", "episodes": 3, "success_rate": 2 / 3,
        "aux_metrics": {"revolute_joint_moved_rate_subtask_1": 0.0},
    }


def test_malformed_seed_fails_before_launch(monkeypatch):
    monkeypatch.setenv("EVAL_SEED", "not-an-integer")
    with pytest.raises(ValueError):
        _load()


def test_a_prefixed_line_is_not_a_metrics_record():
    """C5: the pattern anchored only the END, so any line CONTAINING the prefix matched.

    Astra's probe was accepted as a completed successful evaluation:
      debug anticipated [Rank 0/1] Metrics: {"num_episodes": 1, "success_rate": 1.0}
    Upstream emits the record at the start of its own line, so requiring that costs nothing.
    """
    with pytest.raises(RuntimeError, match="no rank-qualified"):
        _load().parse_arena_output(
            'debug anticipated [Rank 0/1] Metrics: '
            '{"num_episodes": 1, "success_rate": 1.0}\n')


def test_a_genuine_record_at_line_start_is_still_accepted():
    result = _load().parse_arena_output(
        "[Rank 0/1] Metrics: {'num_episodes': 3, 'success_rate': 0.6666666666666666}\n")
    assert result["episodes"] == 3


def test_a_prefixed_line_cannot_displace_a_genuine_record():
    """The last GENUINE record wins, not the last line that resembles one."""
    result = _load().parse_arena_output(
        "[Rank 0/1] Metrics: {'num_episodes': 3, 'success_rate': 0.0}\n"
        'debug anticipated [Rank 0/1] Metrics: {"num_episodes": 3, "success_rate": 1.0}\n')
    assert result["success_rate"] == 0.0, "a decoy line must not override the real result"


# --- Cycle-6 I4: dose points marked valid while impossible -------------------------------

@pytest.mark.parametrize("rate,episodes,expected", [
    (0.0, 3, 0),
    (1 / 3, 3, 1),
    (2 / 3, 3, 2),
    (1.0, 3, 3),
    (0.5, 2, 1),
])
def test_a_realizable_rate_yields_its_success_count(rate, episodes, expected):
    assert _load().realizable_successes(rate, episodes) == expected


@pytest.mark.parametrize("rate,episodes", [
    (0.5, 3),      # one and a half successful episodes
    (0.4, 3),
    (1.5, 3),      # out of range
    (-0.1, 3),
    (0.5, 0),      # no episodes
    (0.5, -1),
    ("0.5", 3),    # not a number
    (0.5, "3"),
    (True, 3),     # bool is not a rate
])
def test_an_impossible_measurement_has_no_success_count(rate, episodes):
    assert _load().realizable_successes(rate, episodes) is None


def test_the_dose_path_uses_the_same_check_as_the_canonical_report():
    """I4: realizability was checked for the canonical report only.

    A dose point recording three episodes at 0.5 was marked ok=True. Fixing one of two paths
    and leaving the other has been the recurring shape of defects here, so both now call one
    helper rather than each carrying its own arithmetic.
    """
    source = _ENTRY_SOURCE()
    assert source.count("realizable_successes(") >= 3, (
        "the helper must be defined and used by BOTH the canonical and dose paths")
    assert "_point_successes = realizable_successes(_sr, _eps)" in source
    assert "_successes = realizable_successes(success_rate, episodes)" in source


def test_a_dose_point_records_its_success_count_and_reason():
    source = _ENTRY_SOURCE()
    assert '"successes": _point_successes,' in source
    assert "rate is not realizable from that episode count" in source


def test_the_report_digests_the_config_the_runner_received():
    """I5: the digest always hashed the DECLARED template.

    Arena rewrites model_path into a patched copy and hands THAT to the runner, so the report
    described bytes the evaluation never consumed -- the same class of defect as reporting module
    globals while executing checkpoint-overridden locals.
    """
    import pathlib
    import re
    source = (pathlib.Path(__file__).resolve().parents[1]
              / "entrypoints/eval/isaac_arena/gr00t/eval_entry.py").read_text()
    code = "\n".join(l for l in source.splitlines() if not l.strip().startswith("#"))
    assert "_EXECUTED_POLICY_CONFIG = dst" in code, (
        "the patched config path is never recorded at the launch site")
    assert '_policy_config_digest(_EXECUTED_POLICY_CONFIG or None)' in code, (
        "the report does not digest the executed config")
    # The template digest must survive as its own field so the rewrite stays auditable.
    assert '"policy_config_template_digest"' in code


def test_the_positive_control_report_names_only_the_served_policy():
    """cycle-15 I6, finished by cycle-14 I4's variant.

    The report used to carry the MOUNTED archive, which did not supply the inference weights. My first
    fix could only ADD a weights_served_by block beside it, because the validator required an archive on
    every report. With the snapshot variant the archive is gone and the report names only what served
    the policy -- so both stopgaps are gone too, and their absence is asserted.
    """
    import pathlib
    source = (pathlib.Path(__file__).resolve().parents[1]
              / "entrypoints/eval/isaac_arena/gr00t/eval_entry.py").read_text()
    # Slice the WHOLE positive-control metrics dict: source_snapshot sits ABOVE
    # positive_control_repo, so anchoring on that key missed it and the test failed against correct
    # code. Anchor on the branch instead.
    block = source[source.index('EVAL_POSCTRL_N16'):]
    block = block[:block.index('"note"')]
    # CODE ONLY. The absence assertions below were matching my own explanatory comments, which name
    # the very fields they say are gone -- the ninth time in this review an assertion has matched its
    # own prose. Every one was a false negative or, here, a false alarm on correct code.
    code = "\n".join(line for line in block.splitlines()
                     if not line.strip().startswith("#"))
    # These three asserted the literal spelling of hand-written fields. That spelling was itself the
    # Critical: weights_digest() returns a PREFIXED "sha256:<hex>" and tree_sha256 must be BARE hex, so
    # the hand-built snapshot would have failed publication after a full inference budget. The fields
    # now come from the shared factory, whose report_source_fields() applies bare_hex(), and the
    # assertions bind that instead.
    assert "report_identity_fields()" in code, (
        "the positive-control report does not derive its identity from the shared resolved object, so "
        "it can carry a source variant that disagrees with its own checkpoint")
    assert "from_snapshot(" in code, (
        "the positive control does not build a SNAPSHOT identity; the served policy is a published HF "
        "snapshot, not an archive")
    assert '"tree_sha256"' not in code, (
        "the positive control hand-writes tree_sha256 again -- the exact Critical: a prefixed digest in "
        "a bare-hex field")
    assert '"source_archive"' not in code, (
        "the mounted archive is still reported, so a diagnostic can attribute the score to bytes that "
        "had no part in producing it")
    assert "weights_served_by" not in code, "the stopgap block outlived the archive it explained"


def _resolved_over(tmp_path, name="seed"):
    """A real ResolvedCheckpoint over a real tiny archive, for tests that drive write_metrics directly.

    These used to seed a module-level dict with an invented measurement. The constructor now measures
    the archive it is given and refuses a disagreement, so an invented value cannot build an identity
    -- which is the point: a fixture able to fabricate one is how a producer bug stays invisible.
    """
    import tarfile
    from vla_pipeline.common.digest import measure_archive
    from vla_pipeline.common.source_identity import from_archive
    root = tmp_path / f"{name}_tree"
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text('{"model_type": "gr00t"}')
    arc = tmp_path / f"{name}.tar.gz"
    with tarfile.open(arc, "w:gz") as tf:
        tf.add(root, arcname="ckpt")
    sha, size = measure_archive(str(arc))
    return from_archive(load_root=str(root), archive_path=str(arc),
                        checkpoint=str(root), sha256=sha, size_bytes=size)
