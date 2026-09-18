"""R1: Tests for baked GR00T venv behavior.

Verifies that ensure_gr00t_venv() and ensure_gr00t_venv_n16() require
pre-existing venvs (built at image time) and fail immediately when the
venv is absent. No runtime dependency installation is allowed.
"""
import importlib.util
import os
import pathlib
import sys

import pytest

_EVAL_ENTRY = (pathlib.Path(__file__).resolve().parents[1]
               / "entrypoints/eval/isaac_arena/gr00t/eval_entry.py")

_load_counter = 0


def _load(**env_overrides):
    global _load_counter
    _load_counter += 1
    os.environ.setdefault("SM_HP_TASK_NAME", "fixture_task")
    saved = {}
    for key in ("GR00T_N17_DIR", "GR00T_N16_DIR"):
        if key in os.environ:
            saved[key] = os.environ.pop(key)
    for key, val in env_overrides.items():
        os.environ[key] = val
    try:
        name = f"arena_eval_entry_r1_{_load_counter}"
        spec = importlib.util.spec_from_file_location(name, _EVAL_ENTRY)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        for key in env_overrides:
            os.environ.pop(key, None)
        for key, val in saved.items():
            os.environ[key] = val


@pytest.mark.parametrize("version", ["n17", "n16"])
def test_missing_venv_fails_immediately(tmp_path, monkeypatch, version):
    """When the baked venv does not exist, ensure_gr00t_venv{,_n16} must raise
    RuntimeError immediately -- no subprocess calls, no network, no installs."""
    module = _load()
    prefix = "GR00T_N16" if version == "n16" else "GR00T"
    fake_dir = str(tmp_path / "nonexistent")
    monkeypatch.setattr(module, f"{prefix}_DIR", fake_dir)
    monkeypatch.setattr(module, f"{prefix}_VENV", str(tmp_path / "nonexistent/.venv/bin/python"))

    calls = []
    monkeypatch.setattr(module.subprocess, "run",
                        lambda *a, **kw: calls.append(a) or None)
    monkeypatch.setattr(module.subprocess, "Popen",
                        lambda *a, **kw: calls.append(a) or None)

    setup = module.ensure_gr00t_venv_n16 if version == "n16" else module.ensure_gr00t_venv
    with pytest.raises(RuntimeError, match="baked .* GR00T venv not found"):
        setup()

    assert not calls, "ensure_gr00t_venv must not spawn any subprocess when venv is missing"


@pytest.mark.parametrize("version", ["n17", "n16"])
def test_present_venv_succeeds_without_subprocess(tmp_path, monkeypatch, version):
    """When the baked venv exists on disk, ensure_gr00t_venv{,_n16} must return
    the path without spawning any subprocess."""
    module = _load()
    prefix = "GR00T_N16" if version == "n16" else "GR00T"
    venv_dir = tmp_path / "gr00t" / ".venv" / "bin"
    venv_dir.mkdir(parents=True)
    venv_python = venv_dir / "python"
    venv_python.write_text("#!/bin/sh\n")

    monkeypatch.setattr(module, f"{prefix}_DIR", str(tmp_path / "gr00t"))
    monkeypatch.setattr(module, f"{prefix}_VENV", str(venv_python))

    calls = []
    monkeypatch.setattr(module.subprocess, "run",
                        lambda *a, **kw: calls.append(a) or None)

    setup = module.ensure_gr00t_venv_n16 if version == "n16" else module.ensure_gr00t_venv
    result = setup()

    assert result == str(venv_python)
    assert not calls, "ensure_gr00t_venv must not spawn any subprocess when venv exists"


@pytest.mark.parametrize("version", ["n17", "n16"])
def test_no_runtime_install_code_in_ensure_functions(version):
    """Structural check: the ensure functions must not contain runtime install
    commands (apt-get, pip install, git clone, uv sync/lock) outside docstrings."""
    import ast
    import inspect
    module = _load()
    fn = module.ensure_gr00t_venv_n16 if version == "n16" else module.ensure_gr00t_venv
    source = inspect.getsource(fn)
    tree = ast.parse(source)
    body_lines = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Constant,)) and isinstance(node.value, str):
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue
        if hasattr(node, 'lineno'):
            line = source.splitlines()[node.lineno - 1]
            body_lines.append(line)
    code_text = "\n".join(body_lines)
    forbidden = ["apt-get", "pip install", "git clone", "uv sync", "uv lock"]
    for term in forbidden:
        assert term not in code_text, (
            f"ensure_gr00t_venv{'_n16' if version == 'n16' else ''}() contains "
            f"runtime install code: '{term}' found in function body")


def test_venv_paths_use_opt_prefix():
    """The default venv paths must point to /opt/gr00t-{n17,n16} (the Dockerfile
    bake location), not /tmp/Isaac-GR00T (the old runtime clone location)."""
    module = _load()
    assert "/opt/gr00t-n17" in module.GR00T_DIR or "GR00T_N17_DIR" in os.environ
    assert "/opt/gr00t-n16" in module.GR00T_N16_DIR or "GR00T_N16_DIR" in os.environ
    assert "/tmp/Isaac-GR00T" not in module.GR00T_DIR
    assert "/tmp/Isaac-GR00T" not in module.GR00T_N16_DIR


def test_venv_dir_overridable_via_env():
    """GR00T_N17_DIR and GR00T_N16_DIR must be overridable via environment
    variables (for testing and non-standard image layouts)."""
    module = _load(GR00T_N17_DIR="/custom/n17", GR00T_N16_DIR="/custom/n16")
    assert module.GR00T_DIR == "/custom/n17"
    assert module.GR00T_N16_DIR == "/custom/n16"
    assert module.GR00T_VENV == "/custom/n17/.venv/bin/python"
    assert module.GR00T_N16_VENV == "/custom/n16/.venv/bin/python"


@pytest.mark.parametrize("version", ["n17", "n16"])
def test_main_fails_on_missing_venv(tmp_path, monkeypatch, version):
    """Integration: main() must fail at the ensure_gr00t_venv stage when the
    venv is missing, without reaching run_arena_eval."""
    import json
    module = _load()
    prefix = "GR00T_N16" if version == "n16" else "GR00T"
    monkeypatch.setattr(module, f"{prefix}_DIR", str(tmp_path / "absent"))
    monkeypatch.setattr(module, f"{prefix}_VENV", str(tmp_path / "absent/.venv/bin/python"))
    monkeypatch.setenv("SM_HP_USE_GROOT_SERVER", "true")
    monkeypatch.setenv("EVAL_GR00T_VERSION", version)
    monkeypatch.setenv("EVAL_POSCTRL_N16", "false")
    policy_config = tmp_path / "policy.yaml"
    policy_config.write_text("model_path: /placeholder\n")
    monkeypatch.setenv("EVAL_POLICY_CONFIG_YAML", str(policy_config))

    ckpt = tmp_path / "checkpoint"
    ckpt.mkdir()
    (ckpt / "weights.bin").write_bytes(b"data")

    manifest = {
        "manifest_version": 1, "model_family": "gr00t",
        "base_checkpoint": "fixture/policy", "base_revision": "a" * 40,
        "train_seed": 42,
        "input_config": {
            "embodiment_tag": "GR1", "n_action_steps": "8", "max_episode_steps": "720",
        },
        "train_recipe": {"repo": "fixture", "commit": "b" * 40, "max_steps": 300},
        "weights_digest": "fixture_digest",
    }
    (ckpt / "checkpoint_manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(module, "extract_checkpoint", lambda *a: str(ckpt))
    monkeypatch.setattr(module, "verify_checkpoint", lambda root: (manifest, manifest["weights_digest"]))
    monkeypatch.setattr(module, "_enumerate_dose_points", lambda root: [(300, root)])

    monkeypatch.setattr(module, "run_arena_eval", lambda *a, **kw: (_ for _ in ()).throw(
        AssertionError("run_arena_eval must not be called when venv is missing")))
    monkeypatch.setattr(module, "resolve_hf_token", lambda: "fixture")
    monkeypatch.setattr(module, "precache_cosmos", lambda *a: None)

    with pytest.raises(RuntimeError, match="baked .* GR00T venv not found"):
        module.main()


def test_dockerfile_contains_baked_venv_stages():
    """The Dockerfile must contain the R1 bake stages for both N1.7 and N1.6."""
    dockerfile = (pathlib.Path(__file__).resolve().parents[1]
                  / "../containers/isaac-lab-arena/Dockerfile")
    content = dockerfile.read_text()
    assert "GR00T_N17_COMMIT" in content, "Dockerfile missing N1.7 bake stage"
    assert "GR00T_N16_COMMIT" in content, "Dockerfile missing N1.6 bake stage"
    assert "/opt/gr00t-n17" in content, "Dockerfile must bake N1.7 to /opt/gr00t-n17"
    assert "/opt/gr00t-n16" in content, "Dockerfile must bake N1.6 to /opt/gr00t-n16"
    commands = "\n".join(line for line in content.splitlines()
                         if not line.lstrip().startswith("#"))
    assert commands.count("uv sync --frozen --no-dev --no-install-package deepspeed") == 2
    assert "rm -f uv.lock" not in commands
    assert "uv lock" not in commands
    assert "sed -i" not in commands
    assert "376ba890" in content, "Dockerfile must pin N1.7 commit"
    assert "5dc80c4" in content, "Dockerfile must pin N1.6 commit"


@pytest.mark.parametrize("defect", [None, "numpy_2x", "missing_numpy"])
def test_build_probe_fails_closed(monkeypatch, capsys, defect):
    import runpy
    import types

    modules = {}
    for name in ("numpy", "torch", "transformers"):
        modules[name] = types.ModuleType(name)
        monkeypatch.setitem(sys.modules, name, modules[name])
    modules["numpy"].__version__ = "2.5.3" if defect == "numpy_2x" else "1.26.4"
    modules["torch"].__version__ = "2.5.1"
    modules["transformers"].__version__ = "4.57.6"
    if defect == "missing_numpy":
        monkeypatch.setitem(sys.modules, "numpy", None)
    probe = _EVAL_ENTRY.with_name("_verify_gr00t.py")
    if defect is None:
        runpy.run_path(str(probe), run_name="__main__")
        assert "system Python deps OK" in capsys.readouterr().out
    else:
        expected = {"numpy_2x": AssertionError, "missing_numpy": ImportError}[defect]
        with pytest.raises(expected):
            runpy.run_path(str(probe), run_name="__main__")
        assert "system Python deps OK" not in capsys.readouterr().out


def test_no_runtime_git_clone_in_eval_entry():
    """eval_entry.py must not contain GR00T_REPO or git clone commands."""
    source = _EVAL_ENTRY.read_text()
    assert 'GR00T_REPO' not in source, "eval_entry.py still contains GR00T_REPO constant"
    assert 'git clone' not in source.lower(), "eval_entry.py still contains git clone"
    assert 'git", "clone' not in source, "eval_entry.py still contains git clone subprocess"
