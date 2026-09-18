"""The positive control must name the commit it actually scored.

Three defects, all in the diagnostic path that scores a known-good published policy:

  N16_POSCTRL_CKPT_REV defaulted to "main" and was published as `resolved_commit`. A branch name is a
  MOVING reference: the attestation claimed to pin bytes while naming something that can point
  elsewhere tomorrow. The component's own validator rejects a non-40-hex resolved_commit, so this
  report could never publish -- after a full evaluation had been paid for.

  The download treated a NON-EMPTY DIRECTORY as cache evidence. That says only that something was
  written there, not that it is this repo at this revision: a partial download, or a different
  revision from an earlier run, satisfied it equally.

  weights_digest was wrapped in try/except that set `tree_sha256: None` on failure. That records an
  absent measurement as though it were a value, and the validator requires 64-hex -- so again, a run
  that had spent its whole budget could not publish.
"""
import ast
import pathlib
import re

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_ARENA = _ROOT / "entrypoints/eval/isaac_arena/gr00t/eval_entry.py"


def test_no_moving_reference_is_a_usable_default():
    """"main" as a DEFAULT is the defect; as an explicit request it is fine, because it gets resolved."""
    tree = ast.parse(_ARENA.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                getattr(t, "id", "") == "N16_POSCTRL_CKPT_REV" for t in node.targets):
            call = node.value
            assert isinstance(call, ast.Call), "expected os.environ.get(...)"
            default = call.args[1] if len(call.args) > 1 else None
            assert isinstance(default, ast.Constant) and default.value == "", (
                f"N16_POSCTRL_CKPT_REV defaults to {getattr(default, 'value', None)!r}. A branch or "
                f"tag default is published as resolved_commit, so the report attests a moving "
                f"reference as the identity of the weights it scored")
            return
    raise AssertionError("N16_POSCTRL_CKPT_REV assignment not found")


def test_write_metrics_requires_a_resolved_commit_for_a_positive_control(tmp_path):
    """Behavioural: the guard must refuse the requested reference and demand the resolved one."""
    import importlib.util
    import os
    import sys
    import types

    os.environ.setdefault("SM_HP_TASK_NAME", "fixture_task")
    # write_metrics selects the positive-control branch on this env var, and that branch is what
    # carries the guard under test. results["policy_type"] alone does not select it.
    os.environ["EVAL_POSCTRL_N16"] = "true"
    spec = importlib.util.spec_from_file_location("arena_posctrl_probe", _ARENA)
    module = importlib.util.module_from_spec(spec)
    stub = types.ModuleType("digest")
    stub.weights_digest = lambda path=None: "sha256:" + "0" * 64
    stub.measure_archive = lambda path: ("0" * 64, 1)
    saved = {k: sys.modules.get(k) for k in ("digest", "validator", "capped_reader", "source_identity")}
    sys.modules["digest"] = stub
    for name in ("validator", "capped_reader", "source_identity"):
        sys.modules.setdefault(name, types.ModuleType(name))
    try:
        spec.loader.exec_module(module)
        results = {"success_rate": 1.0, "policy_type": "positive_control",
                   "task_name": "fixture_task", "episodes": 3, "successes": 3}
        # An ISOLATED directory: /tmp carries a checkpoint_manifest.json from other tests, and the
        # digest cross-check fires on it before the guard under test is reached.
        out = tmp_path / "out"; out.mkdir()
        ckpt = tmp_path / "ckpt"; ckpt.mkdir()
        # No commit at all.
        with pytest.raises(RuntimeError, match="RESOLVED positive-control commit"):
            module.write_metrics(results, str(out), str(ckpt))
        # A moving reference is not a resolved commit.
        with pytest.raises(RuntimeError, match="RESOLVED positive-control commit"):
            module.write_metrics(results, str(out), str(ckpt), posctrl_commit="main")
        # Nor is a short one.
        with pytest.raises(RuntimeError, match="RESOLVED positive-control commit"):
            module.write_metrics(results, str(out), str(ckpt), posctrl_commit="a" * 39)
    finally:
        os.environ.pop("EVAL_POSCTRL_N16", None)
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def test_the_digest_has_no_none_fallback():
    """A digest that cannot be computed is a MISSING measurement, not a value of None.

    Asserted structurally because the alternative -- driving a digest failure through the real
    producer -- needs the whole Arena harness. What matters is that no handler assigns the digest a
    fallback: the validator requires 64-hex, so `None` guaranteed an unpublishable report after a
    fully paid evaluation.
    """
    tree = ast.parse(_ARENA.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        for handler in node.handlers:
            for sub in ast.walk(handler):
                if (isinstance(sub, ast.Assign)
                        and any(getattr(t, "id", "") == "posctrl_digest" for t in sub.targets)):
                    raise AssertionError(
                        f"posctrl_digest is assigned inside an exception handler at line "
                        f"{sub.lineno}; a fallback records an absent measurement as a value")


def test_a_nonempty_directory_is_not_cache_evidence():
    """The cache must be keyed on a RECORDED identity, not on the directory being non-empty."""
    source = _ARENA.read_text()
    assert "N16_POSCTRL_RESOLVED_FILE" in source, (
        "no recorded-commit file: the cache cannot distinguish this repo at this revision from a "
        "partial download or a different revision left by an earlier run")
    # The recorded value must be format-checked before it is trusted.
    assert re.search(r"fullmatch\(r?[\"']\[0-9a-f\]\{40\}[\"']", source), (
        "the recorded commit is not validated as 40-hex before the cache is trusted")


def test_the_submitter_forwards_every_variable_the_posctrl_evaluator_requires():
    """The regression this file's own subject caused.

    Making N16_POSCTRL_CKPT_REV required -- correct, because the positive control once published
    resolved_commit="main", a moving tag that pins nothing -- broke the only submitter that launches it,
    because scripts/submit_simeval.py forwarded the repo and not the revision. The positive control was
    the path deliberately left unconverted, which is exactly why nothing noticed.

    Binds the two sides: every N16_POSCTRL_* variable the evaluator READS must be one the submitter can
    SET. Derived from the evaluator's source rather than listed here, so a new requirement fails this
    test instead of silently making the control unlaunchable.
    """
    import re

    root = pathlib.Path(__file__).resolve().parents[1]
    evaluator = (root / "entrypoints/eval/isaac_arena/gr00t/eval_entry.py").read_text()
    submitter = (root / "scripts/submit_simeval.py").read_text()

    required = set(re.findall(r'environ(?:\.get)?\(?\s*\[?["\'](N16_POSCTRL_[A-Z_]+)["\']', evaluator))
    assert required, "no N16_POSCTRL_* reads found in the evaluator; the extraction is wrong"

    # The names the submitter actually ASSIGNS, via the AST -- not a substring search over the file.
    # A substring search passes on any mention, including the explanatory comment I wrote next to this
    # very fix, so deleting the assignment left the test green. Prose must not be able to satisfy it.
    tree = ast.parse(submitter)
    assigned = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (isinstance(target, ast.Subscript)
                        and isinstance(target.slice, ast.Constant)
                        and isinstance(target.slice.value, str)):
                    assigned.add(target.slice.value)
        elif isinstance(node, ast.Dict):
            for key in node.keys:
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    assigned.add(key.value)

    missing = sorted(name for name in required if name not in assigned)
    assert not missing, (
        f"the evaluator reads {missing} but scripts/submit_simeval.py never sets them, so a positive "
        f"control launched from there cannot satisfy its own evaluator. Required {sorted(required)}, "
        f"assigns {sorted(n for n in assigned if n.startswith('N16_POSCTRL'))}")


def test_the_submitter_refuses_a_positive_control_without_an_immutable_revision():
    """Refused at submit, not after a GPU node is provisioned.

    The evaluator fails closed on an empty revision, but only inside the container -- after capacity,
    after the image pull. The operator who omitted the flag should learn it from the command they typed.
    """
    root = pathlib.Path(__file__).resolve().parents[1]
    submitter = (root / "scripts/submit_simeval.py").read_text()
    assert "--posctrl-revision" in submitter, (
        "there is no way to supply the revision the evaluator requires")
    assert "[0-9a-f]{40}" in submitter, (
        "the submitter does not validate the revision is an immutable 40-hex commit, so a moving tag "
        "like 'main' would be forwarded -- the defect this requirement exists to prevent")
