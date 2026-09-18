"""Fail-loud audit smoke: each staged sourcedir is self-contained.

DELIBERATE NAME. "smoke" here is the standard software sense -- a fast sanity check that the
thing imports at all -- and NOT the project's "sample run" concept, which means a real
evaluation at reduced counts. The two were conflated in prose and the sample-run uses were
renamed; this one is correct as it stands.

The layout refactor's chief risk is that stage() drops a co-staged sibling
(digest.py / validator.py / a family data-config) after the sourcedir move, so an
entrypoint's top-level `from digest import ...` would break at runtime. This
smoke asserts, offline and deterministically, that for every family:
  1. every staged .py compiles (py_compile in a clean subprocess), and
  2. every TOP-LEVEL first-party import in a staged file resolves to a file
     present in the staged dir. A "first-party" module = one that has a .py
     somewhere in the repo tree; this deliberately excludes stdlib and
     third-party deps (torch/msgpack/numpy/zmq) that aren't installed offline.

Why AST resolution and not a bare `import`: some entrypoints read a REQUIRED
env var at module top (e.g. openvla eval_entry reads EVAL_CHECKPOINT), so a
runtime import needs the SageMaker env. The AST check is the offline proxy for
"imports without a path break" and targets exactly the dropped-sibling risk.
"""
import ast
import builtins
import importlib.util
import io
import json
import pathlib
import os
import subprocess
import sys
import tarfile
import shutil
from pathlib import Path

import pytest

from vla_pipeline.common import sourcedir

_REPO = Path(__file__).resolve().parents[1]


def _first_party_module_stems():
    """Top-level module names that are first-party (have a .py in the repo)."""
    names = set()
    for base in ("entrypoints", "src/vla_pipeline"):
        root = _REPO / base
        if root.is_dir():
            for p in root.rglob("*.py"):
                names.add(p.stem)
    return names


_FIRST_PARTY = _first_party_module_stems()


@pytest.mark.parametrize("mode", ["default", "published", "local"])
def test_gr00t_checkpoint_revision_is_not_subdirectory(mode):
    staged = sourcedir.stage("gr00t")
    defaults = json.loads((Path(staged) / "defaults.json").read_text())
    env = {key: value for key, value in os.environ.items() if not key.startswith("EVAL_")}
    revision = "a" * 40
    if mode == "published":
        env.update(EVAL_CHECKPOINT="fixture/policy", EVAL_CKPT_REV=revision)
    elif mode == "local":
        env.update(EVAL_CHECKPOINT="/fixture/model", EVAL_MODEL_SOURCE_URI="s3://fixture/model.tar.gz")
    result = subprocess.run(
        [sys.executable, "-c",
         "import json, runpy, sys; m = runpy.run_path(sys.argv[1]); "
         "print(json.dumps([m['HF_REPO'], m['HF_REVISION'], m['CKPT_SUBDIR'], m['PIPELINE_MODE']]))",
         str(Path(staged) / "eval_entry.py")],
        env=env, capture_output=True, text=True, check=True,
    )
    repo, actual_revision, subdirectory, pipeline_mode = json.loads(result.stdout)
    assert repo == {
        "default": defaults["checkpoint"], "published": "fixture/policy", "local": "/fixture/model",
    }[mode]
    assert actual_revision == (revision if mode == "published" else defaults["checkpoint_hf_revision"])
    assert subdirectory == ("" if mode == "local" else defaults["checkpoint_subdir"])
    assert pipeline_mode is (mode == "local")


@pytest.mark.parametrize("success_counts", [(1, 0), (1, 2), (0, 0), (3, 3)])
@pytest.mark.parametrize("pipeline_mode", [False, True])
@pytest.mark.parametrize("missing_backbone", [False, True])
def test_gr00t_report_preserves_measurement_precision(
    tmp_path, monkeypatch, success_counts, pipeline_mode, missing_backbone
):
    from unittest.mock import Mock

    staged = Path(sourcedir.stage("gr00t"))
    monkeypatch.syspath_prepend(str(staged))
    for name in ("digest", "validator"):
        spec = importlib.util.spec_from_file_location(name, staged / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        monkeypatch.setitem(sys.modules, name, module)
    for key in list(os.environ):
        if key.startswith(("EVAL_", "HF_")):
            monkeypatch.delenv(key)
    monkeypatch.setenv("EVAL_TRIALS", "3")
    monkeypatch.setenv("EVAL_TASK_IDS", "[0,1]")
    if pipeline_mode:
        monkeypatch.setenv("EVAL_CHECKPOINT", str(tmp_path / "channel"))
        monkeypatch.setenv("SM_CHANNEL_MODEL", str(tmp_path / "channel"))
        monkeypatch.setenv("EVAL_MODEL_SOURCE_URI", "s3://fixture/model.tar.gz")
    spec = importlib.util.spec_from_file_location("gr00t_report", staged / "eval_entry.py")
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    work = tmp_path / "work"
    repo = work / "Isaac-GR00T"
    client = repo / "gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python"
    client.parent.mkdir(parents=True)
    client.touch()
    wheel = repo / "scripts/deployment/dgpu/wheels/torchcodec-0.8.0-cp312-cp312-linux_aarch64.whl"
    wheel.parent.mkdir(parents=True)
    wheel.write_bytes(b"x" * 100_001)
    # I2/I3: the evaluator patches the pinned rollout loop so that UNOBSERVED success
    # telemetry raises at the episode boundary. Use the REAL vendored pinned file rather
    # than a hand-written stub: a stub only carrying the anchors the patch happens to look
    # for would keep passing when the patch changes shape, and production refuses to run
    # if any anchor is missing.
    rollout = repo / "gr00t/eval/rollout_policy.py"
    rollout.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(
        Path(__file__).parent / "data" / "rollout_policy_pinned.py", rollout)
    checkpoint = work / "checkpoints/GR00T-N1.7-LIBERO" / entry.CKPT_SUBDIR
    checkpoint.mkdir(parents=True)
    (checkpoint / "weights.bin").write_bytes(b"fixture weights")
    # A real config.json declaring the architecture. The evaluator refuses a checkpoint whose
    # architecture it cannot read before launching the server, because an N1.6 checkpoint on the N1.7
    # loader fails only after the full startup timeout with an error naming transformers. This fixture
    # stands in for an N1.7 checkpoint, which is the only kind this evaluator serves.
    (checkpoint / "config.json").write_text('{"model_type": "Gr00tN1d7"}')
    # Real server seeding evidence, because the report now derives backbone_identity from the audit that
    # records what inference LOADED rather than from the precache value. A fixture without it is refused,
    # correctly: a run whose backbone commit cannot be established is unattributable, and the backbone
    # determines preprocessing.
    evidence = work / "seed_evidence.json"
    # The FULL schema the evaluator accepts, read from its own checks: schema tag, matching seed, both
    # reseed stages applied, the audit installed, and a completed audit list. A partial record is
    # rejected -- correctly -- and my first version supplied only the audits, so the report fell back to
    # "unverified" and the backbone binding refused.
    evidence.write_text(json.dumps({
        "evidence_schema": "gr00t_server_evidence_v1",
        "seed": int(entry.SEED),
        "process_start": True,
        "inference_ready": True,
        "strict_load_audit": "installed",
        # The path the SERVER was launched with. In PIPELINE mode that is the EXTRACTED checkpoint at
        # work/model -- not the mounted channel it came from, and not the snapshot directory. I named
        # both of those before getting here; each refused every pipeline-mode run for a different wrong
        # reason, which is the point of comparing realpaths rather than trusting a name.
        "processor_validated": str(work / "model") if pipeline_mode else str(checkpoint),
        "strict_load_audits": [{"path": "nvidia/Cosmos-Reason2-2B",
                                "consumed_commit": "9ce19a195e423419c349abfc86fd07178b230561",
                                "consumed_realpath": None,
                                "missing_keys": 0, "unexpected_keys": 0,
                                "mismatched_keys": 0, "error_msgs": 0}],
    }))
    if missing_backbone:
        incomplete = json.loads(evidence.read_text())
        incomplete["strict_load_audits"][0].update(
            path=incomplete["processor_validated"], consumed_commit=None)
        evidence.write_text(json.dumps(incomplete))
    monkeypatch.setenv("GR00T_SERVER_SEED_EVIDENCE", str(evidence))
    settings = {
        "embodiment_tag": "LIBERO_PANDA",
        "n_action_steps": "4",
        "max_episode_steps": "360",
    }
    if pipeline_mode:
        manifest = {
            "manifest_version": 1, "model_family": "gr00t",
            "base_checkpoint": entry._D["base_checkpoint"],
            "base_revision": entry._D["base_checkpoint_hf_revision"],
            "train_seed": 42, "input_config": settings,
            "train_recipe": {"repo": entry.GR00T_REPO, "commit": entry.GR00T_COMMIT,
                             "max_steps": 100},
            "weights_digest": entry.weights_digest(str(checkpoint)),
            "dataset_manifest": {"source": "fixture", "revision": "a" * 40,
                                 "episode_count": 10},
        }
        (checkpoint / "checkpoint_manifest.json").write_text(json.dumps(manifest))
        channel = tmp_path / "channel"
        channel.mkdir()
        with tarfile.open(channel / "model.tar.gz", "w:gz") as archive:
            for path in checkpoint.iterdir():
                archive.add(path, arcname=path.name)
        import boto3
        s3 = Mock()
        s3.head_object.return_value = {"VersionId": "fixture-version", "ETag": '"fixture"'}
        monkeypatch.setattr(boto3, "client", Mock(return_value=s3))
    for name, value in {
        "WORK": str(work), "GR00T_DIR": str(repo),
        "OUT_DIR": str(tmp_path / "evidence"), "METRICS_DIR": str(tmp_path / "metrics"),
    }.items():
        monkeypatch.setattr(entry, name, value)
    monkeypatch.setattr(entry, "run", Mock())
    monkeypatch.setattr(entry, "uv_sync_resilient", Mock())
    monkeypatch.setattr(entry, "clean_env", lambda: {"MUJOCO_GL": "egl", "PATH": ""})
    monkeypatch.setattr(entry, "wait_for_port", Mock())
    monkeypatch.setattr(entry, "enumerate_tasks", lambda *args: [(0, "task0"), (1, "task1")])
    monkeypatch.setattr(entry.os, "chmod", Mock())
    monkeypatch.setattr(entry.glob, "glob", lambda pattern: ["/fixture/nvidia/lib"])
    real_open = builtins.open

    def open_file(path, *args, **kwargs):
        if path == "/tmp/gr00t_server.log":
            path = tmp_path / "server.log"
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(entry, "open", open_file, raising=False)

    def run_process(command, **kwargs):
        if command[0] == "nvidia-smi" or "--no-env-file" in command:
            return subprocess.CompletedProcess(command, 0)
        if command[0] == "git" and command[-1] == "HEAD":
            return subprocess.CompletedProcess(command, 0, entry.GR00T_COMMIT, "")
        if command[0] == str(client):
            return subprocess.CompletedProcess(command, 0, "/fixture/ffmpeg", "")
        if command[-1] == "-encoders":
            return subprocess.CompletedProcess(command, 0, "libx264", "")
        raise AssertionError(f"unexpected subprocess: {command}")

    monkeypatch.setattr(entry.subprocess, "run", run_process)
    server = Mock()
    # Each record must name the environment it belongs to. This fixture previously
    # emitted 'task' for every rollout while enumerate_tasks yields task0/task1, and
    # nothing noticed, because the identity was taken from the wrapper instead of being
    # recovered from the result (C1). The parser now requires them to agree.
    # I5: the score now comes from the rollout's own structured result FILE, never from
    # stdout, so this fake producer must write that file the way the patched rollout does.
    # Writing it at wait() time mirrors the real ordering -- the wrapper reads only after
    # the process exits.
    def _make_client(index, count):
        vector = [True] * count + [False] * (3 - count)
        client_process = Mock(stdout=io.StringIO(
            f"rollout for task{index} finished\n"))

        def _wait(_index=index, _vector=vector):
            path = pathlib.Path(work) / "task-logs" / f"task_{_index}_result.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({
                "schema": "gr00t_rollout_result_v1",
                "env_name": f"task{_index}",
                "episode_successes": _vector,
                "episodes": len(_vector),
                "successes": sum(_vector),
            }))
            return 0

        client_process.wait.side_effect = _wait
        return client_process

    clients = [_make_client(index, count)
               for index, count in enumerate(success_counts)]
    launch = Mock(side_effect=[server, *clients])
    monkeypatch.setattr(entry.subprocess, "Popen", launch)

    # This test stubs subprocesses, so the real resolver never runs. It used to compensate by
    # seeding entry._SOURCE_ARCHIVE with a plain dict -- unconditionally, in BOTH legs of the
    # pipeline_mode parametrization. Because the producer decided its mode by asking whether that
    # dict was populated, seeding it sent both legs down the archive branch and the HF snapshot
    # branch never executed. That is what kept I6 -- a prefixed digest in a bare-hex field -- green
    # for as long as it existed.
    #
    # Now the identity is CONSTRUCTED over a real directory, per leg, so the object cannot exist
    # without the values agreeing and the HF leg actually exercises the snapshot path.
    from vla_pipeline.common.digest import measure_archive
    from vla_pipeline.common.source_identity import from_archive, from_snapshot
    real_tree = tmp_path / "resolved_ckpt"
    real_tree.mkdir()
    (real_tree / "config.json").write_text('{"model_type": "gr00t"}')
    if pipeline_mode:
        # A REAL archive with DERIVED measurements. The first version of this fixture wrote two bytes
        # and declared a 2048-byte all-"d" measurement; the constructor now measures the named file
        # and refuses a disagreement, so an invented measurement cannot construct an identity.
        real_archive = tmp_path / "model.tar.gz"
        with tarfile.open(real_archive, "w:gz") as tf:
            tf.add(real_tree, arcname="ckpt")
        sha, size = measure_archive(str(real_archive))
        entry._RESOLVED = from_archive(
            load_root=str(real_tree), archive_path=str(real_archive),
            checkpoint=str(real_tree), sha256=sha, size_bytes=size)
    else:
        entry._RESOLVED = from_snapshot(
            load_root=str(real_tree), repo_id=entry.HF_REPO, resolved_commit="a" * 40)
    if missing_backbone:
        with pytest.raises(RuntimeError, match="no strict-load audit"):
            entry.main()
        assert launch.call_count == 1  # Server only; no rollout client was launched.
        server.terminate.assert_called_once()
        assert not (tmp_path / "metrics/metrics.json").exists()
        return
    entry.main()

    assert entry.PIPELINE_MODE is pipeline_mode
    server_command = launch.call_args_list[0].args[0]
    effective = settings if pipeline_mode else {
        "embodiment_tag": entry.EMBODIMENT_TAG,
        "n_action_steps": entry.N_ACTION_STEPS,
        "max_episode_steps": entry.MAX_EPISODE_STEPS,
    }
    assert server_command[server_command.index("--embodiment-tag") + 1] == effective["embodiment_tag"]
    for call in launch.call_args_list[1:]:
        command = call.args[0]
        assert command[command.index("--n-action-steps") + 1] == effective["n_action_steps"]
        assert command[command.index("--max-episode-steps") + 1] == effective["max_episode_steps"]

    report = json.loads((tmp_path / "metrics/metrics.json").read_text())
    assert report["backbone_identity"] == {
        "repo_id": "nvidia/Cosmos-Reason2-2B",
        "resolved_commit": "9ce19a195e423419c349abfc86fd07178b230561",
    }
    assert report["episodes"] == 6
    assert report["success_rate"] == sum(success_counts) / 6
    assert [task["success_rate"] for task in report["per_task"]] == [
        count / 3 for count in success_counts
    ]
    server.terminate.assert_called_once()


def _top_level_imported_names(pyfile: str):
    """Top-level import module roots only (not deferred/in-function imports)."""
    tree = ast.parse(Path(pyfile).read_text())
    mods = set()
    for node in tree.body:  # module body == top level
        if isinstance(node, ast.Import):
            for alias in node.names:
                mods.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:  # absolute import
                mods.add(node.module.split(".")[0])
    return mods


@pytest.mark.parametrize("family", ["gr00t", "openvla", "molmoact2", "dummy"])
def test_staged_sourcedir_self_contained(family):
    d = sourcedir.stage(family)
    staged = {f for f in os.listdir(d) if f.endswith(".py")}
    assert staged, f"{family}: stage() produced no .py files"

    for fname in sorted(staged):
        fpath = os.path.join(d, fname)
        # (1) compiles cleanly in a clean subprocess (catches syntax/parse breaks)
        r = subprocess.run(
            [sys.executable, "-m", "py_compile", fpath],
            capture_output=True, text=True)
        assert r.returncode == 0, f"{family}/{fname} failed py_compile:\n{r.stderr}"

        # (2) every top-level first-party import is co-staged (catches a dropped
        # sibling after the sourcedir move -- the whole point of stage()).
        for mod in _top_level_imported_names(fpath):
            if mod in _FIRST_PARTY:
                assert f"{mod}.py" in staged, (
                    f"{family}/{fname} top-level-imports first-party module "
                    f"{mod!r}, but {mod}.py is NOT in the staged sourcedir "
                    f"({sorted(staged)}). stage() must co-locate it.")
