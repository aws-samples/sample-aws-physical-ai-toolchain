"""Negative tests for the suite-manifest loader and the launcher/build gates.

The whole point of ``_parse_dataset`` / ``_parse_arena`` /
``_parse_family_overrides`` / ``resolve_suite``'s guards is raising
``RegistryError``, and the happy path is already covered by the eight real
manifests in ``tests/test_suites.py``. Without these tests a guard could be
deleted or inverted with the suite green -- and every one of them exists to stop a
misconfigured GPU run.

``resolve_suite`` is ``lru_cache``d, so every test that repoints ``_CONFIG_DIR``
must clear the caches (the ``suite_dir`` fixture does it on both sides).
"""
from __future__ import annotations

import io
import os
import sys

import pytest
import yaml

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if os.path.join(_REPO_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from vla_pipeline import registry as reg  # noqa: E402

_GOOD = {
    "name": "t_suite",
    "simulator": "isaac_arena",
    "canonical_task_ids": [0],
    "dataset": {"repo_id": "org/ds", "subdir": "", "revision": None,
                "copy_libero_modality": False},
    "family_overrides": {"gr00t": {"n17": {"embodiment_tag": "new_embodiment"}}},
    "arena": {"task": "t", "embodiment": None, "object": None,
              "policy_config": "/w/c.yaml"},
}


@pytest.fixture
def suite_dir(tmp_path, monkeypatch):
    """A throwaway config dir with real families/simulators and one writable suite.

    Families and simulators are symlinked from the repo so cross-manifest checks
    (`_load_simulator`, the known-family check) exercise the real thing.
    """
    for kind in ("families", "simulators", "pairs"):
        (tmp_path / kind).symlink_to(os.path.join(_REPO_ROOT, "config", kind))
    (tmp_path / "suites").mkdir()
    reg.clear_caches()
    monkeypatch.setattr(reg, "_CONFIG_DIR", str(tmp_path))
    yield tmp_path / "suites"
    reg.clear_caches()


def _write(suite_dir, **overrides):
    """Write t_suite.yaml as _GOOD + overrides; a None value deletes the key."""
    m = {k: (v.copy() if isinstance(v, dict) else v) for k, v in _GOOD.items()}
    for k, v in overrides.items():
        if v is _DELETE:
            m.pop(k, None)
        else:
            m[k] = v
    with open(suite_dir / "t_suite.yaml", "w") as fh:
        yaml.safe_dump(m, fh)
    return "t_suite"


_DELETE = object()


def test_the_baseline_manifest_resolves(suite_dir):
    """Control: without this, every rejection below could be a fixture defect."""
    s = reg.resolve_suite(_write(suite_dir))
    assert s.arena.task == "t"
    assert s.dataset.repo_id == "org/ds"


@pytest.mark.parametrize("overrides,expect", [
    # --- canonical_task_ids -------------------------------------------------
    ({"canonical_task_ids": [1, 0]},                 "sorted"),
    ({"canonical_task_ids": []},                     "non-empty"),
    ({"canonical_task_ids": [0, 0]},                 "duplicate-free"),
    ({"canonical_task_ids": [True]},                 "ints"),
    ({"canonical_task_ids": "all"},                  "non-empty"),
    ({"canonical_task_ids": _DELETE},                "canonical_task_ids"),
    # --- status -------------------------------------------------------------
    ({"status": "beta"},                             "status must be one of"),
    # --- arena --------------------------------------------------------------
    # The retired step budget must be REJECTED, not ignored: a manifest still
    # declaring it belongs to the old contract, where the budget and the trial
    # count were transported separately and could disagree.
    ({"arena": {"task": "t", "num_steps": 280, "policy_config": "/w/c.yaml"}},
     "unknown key"),
    ({"arena": {"task": "t", "objct": "x"}},
     "unknown key"),
    ({"arena": {}},                                  "missing required field 'task'"),
    ({"arena": {"task": "t", "embodiment": 7}},
     "must be a string or null"),
    # An isaac_arena suite with no arena block cannot produce a policy_runner argv.
    ({"arena": None},                                "no 'arena' block"),
    # ...and a non-Arena suite must not carry one.
    ({"simulator": "libero"},                        "only meaningful"),
    # --- dataset ------------------------------------------------------------
    ({"dataset": {"repo_id": "org/ds"}},             "copy_libero_modality"),
    ({"dataset": {"repo_id": "org/ds", "copy_libero_modality": "no"}},
     "copy_libero_modality"),
    ({"dataset": {"copy_libero_modality": False}},   "missing required field 'repo_id'"),
    ({"dataset": {"repo_id": "org/ds", "copy_libero_modality": False,
                  "revision": 3}},                   "revision must be a string"),
    ({"dataset": "org/ds"},                          "must be a mapping or null"),
    # --- family_overrides ---------------------------------------------------
    ({"family_overrides": {"nope": {"n17": {}}}},    "unknown family"),
    ({"family_overrides": {"gr00t": {"n17": {"embodment_tag": "x"}}}},
     "unknown key"),
    ({"family_overrides": {"gr00t": {"n17": {"embodiment_tag": 7}}}},
     "must be a string or null"),
    ({"family_overrides": {"gr00t": "n17"}},         "must be a mapping"),
    # --- simulator ----------------------------------------------------------
    ({"simulator": "no_such_sim", "arena": None},    "manifest not found"),
    ({"simulator": _DELETE},                         "missing required field 'simulator'"),
    # --- unknown top-level key (a typo must not be silently ignored) --------
    ({"datasets": {"repo_id": "x"}},                 "unknown key"),
])
def test_suite_manifest_defects_fail_loud(suite_dir, overrides, expect):
    name = _write(suite_dir, **overrides)
    with pytest.raises(reg.RegistryError, match=expect):
        reg.resolve_suite(name)


def test_name_must_match_filename(suite_dir):
    with open(suite_dir / "t_suite.yaml", "w") as fh:
        yaml.safe_dump({**_GOOD, "name": "other"}, fh)
    with pytest.raises(reg.RegistryError, match="declares name"):
        reg.resolve_suite("t_suite")


def test_missing_manifest_fails_loud(suite_dir):
    with pytest.raises(reg.RegistryError, match="manifest not found"):
        reg.resolve_suite("nonexistent_suite")


def test_malformed_yaml_fails_loud(suite_dir):
    (suite_dir / "t_suite.yaml").write_text("- just\n- a\n- list\n")
    with pytest.raises(reg.RegistryError, match="must be a YAML mapping"):
        reg.resolve_suite("t_suite")


@pytest.mark.parametrize("bad", [
    "../pairs/gr00t--isaac_arena",   # path traversal out of config/suites/
    "/etc/passwd",
    "Upper_Case",
    "has-dash",
    "",
    "_leading",
])
def test_suite_id_charset_is_enforced(suite_dir, bad):
    """--suite is user input and becomes a path component; reject it directly.

    Without the charset guard, traversal was blocked only INCIDENTALLY, by the
    name-vs-filename cross-check happening to fail.
    """
    with pytest.raises(reg.RegistryError, match="invalid suite id"):
        reg.resolve_suite(bad)


# --------------------------------------------------------------------------
# Invariants the real manifests must satisfy (I7)
# --------------------------------------------------------------------------

def test_pair_default_suites_are_not_experimental():
    """A pair's default_suite must be registrable, or its default cell can't register.

    `dummy--dummy_sim` intentionally defaults to the experimental demo suite, so it
    is exempted explicitly rather than by silence.
    """
    for family, simulator in reg.list_supported_pairs():
        spec = reg.resolve(family, simulator)
        suite = reg.resolve_suite(spec.default_suite)
        if (family, simulator) == ("dummy", "dummy_sim"):
            assert not suite.supported, (
                "the dummy demo pair is expected to point at the experimental "
                "demo suite; update this exemption if that changed")
            continue
        assert suite.supported, (
            f"pair {family}--{simulator} defaults to suite "
            f"{spec.default_suite!r}, which is status: {suite.status} -- Validate "
            f"would reject the report as an unknown suite")


# --------------------------------------------------------------------------
# Launcher + build-script gates (I9)
# --------------------------------------------------------------------------

def test_gate_suite_rejects_a_wrong_simulator_suite():
    from vla_pipeline.arena import ArenaConfigError, gate_suite
    with pytest.raises(ArenaConfigError, match="runs on simulator 'libero'"):
        gate_suite(reg.resolve_suite("libero_spatial"), allow_experimental=False)


def test_gate_suite_rejects_an_experimental_suite_without_optin():
    from vla_pipeline.arena import ArenaConfigError, gate_suite
    with pytest.raises(ArenaConfigError, match="experimental"):
        gate_suite(reg.resolve_suite("arena_g1"), allow_experimental=False)


def test_gate_suite_allows_an_experimental_suite_with_optin():
    from vla_pipeline.arena import gate_suite
    gate_suite(reg.resolve_suite("arena_g1"), allow_experimental=True)   # no raise


def test_gate_suite_accepts_a_supported_arena_suite():
    from vla_pipeline.arena import gate_suite
    gate_suite(reg.resolve_suite("arena_gr1_fridge"), allow_experimental=False)


def _build_script():
    import importlib.util
    path = os.path.join(_REPO_ROOT, "scripts", "build_arena_connector.py")
    spec = importlib.util.spec_from_file_location("build_arena_connector", path)
    mod = importlib.util.module_from_spec(spec)
    prev = os.getcwd()
    os.chdir(_REPO_ROOT)          # the script does sys.path.insert(0, "src")
    try:
        spec.loader.exec_module(mod)
    finally:
        os.chdir(prev)
    return mod


@pytest.mark.parametrize("bad", ["repo", "repo:", ":tag", "", "repo:tag:extra "])
def test_split_repo_tag_rejects_malformed_values(bad):
    mod = _build_script()
    if bad == "repo:tag:extra ":
        # rsplit means this IS well-formed (repo="repo:tag", tag="extra ").
        assert mod._split_repo_tag(bad, "f") == ("repo:tag", "extra ")
        return
    with pytest.raises(SystemExit, match="must be 'repo:tag'"):
        mod._split_repo_tag(bad, "f")


def test_connector_archive_contains_its_selected_entrypoint():
    import io
    import json
    import shlex
    import zipfile
    from pathlib import Path

    mod = _build_script()
    with zipfile.ZipFile(io.BytesIO(mod.create_source_zip("gr00t"))) as archive:
        dockerfile = archive.read(
            "containers/isaac-lab-arena/Dockerfile").decode()
        copies = {}
        entrypoint = None
        for line in dockerfile.splitlines():
            if line.startswith("COPY "):
                _, source, destination = shlex.split(line)
                copies[destination] = source
                assert archive.read(source) == (Path(mod.REPO_ROOT) / source).read_bytes()
            elif line.startswith("ENTRYPOINT "):
                entrypoint = json.loads(line.removeprefix("ENTRYPOINT "))
        assert entrypoint == ["/bin/bash", "/workspace/docker_entrypoint_multi.sh"]
        assert copies[entrypoint[1]] == (
            "isaac-lab-arena-on-aws/entrypoints/eval/isaac_arena/_shared/docker_entrypoint_multi.sh")


def test_split_repo_tag_accepts_a_registry_qualified_value():
    mod = _build_script()
    repo, tag = mod._split_repo_tag("111122223333.dkr.ecr.us-east-1.amazonaws.com/"
                                    "vla/isaac-arena:base-test", "eval_base_image")
    assert tag == "base-test"


class _FakeEcr:
    """Minimal ECR stub: botocore exposes error classes off client.exceptions."""
    class RepositoryNotFoundException(Exception):
        pass

    class ImageNotFoundException(Exception):
        pass

    def __init__(self, raises=None):
        self._raises = raises
        self.exceptions = self


    def describe_images(self, **kw):
        if self._raises:
            raise self._raises


def test_verify_base_image_reports_a_missing_repository():
    mod = _build_script()
    ecr = _FakeEcr()
    ecr._raises = _FakeEcr.RepositoryNotFoundException()
    with pytest.raises(SystemExit) as ei:
        mod.verify_base_image(ecr, "vla/isaac-arena:base-test", "us-east-1")
    assert "README.md, 'Build the container images'" in str(ei.value)


def test_verify_base_image_reports_a_missing_tag():
    mod = _build_script()
    ecr = _FakeEcr()
    ecr._raises = _FakeEcr.ImageNotFoundException()
    with pytest.raises(SystemExit) as ei:
        mod.verify_base_image(ecr, "vla/isaac-arena:base-test", "us-east-1")
    msg = str(ei.value)
    assert "not present in ECR" in msg
    assert "README.md, 'Build the container images'" in msg


def test_verify_base_image_passes_when_present():
    mod = _build_script()
    mod.verify_base_image(_FakeEcr(), "vla/isaac-arena:base-test", "us-east-1")  # no raise


# --------------------------------------------------------------------------
# Launcher gates must actually RUN (I18)
#
# Round 2 found that five of round 1's fixes could each be deleted with the whole
# suite green: `test_gate_suite_*` proves the FUNCTION raises, but nothing proved a
# launcher CALLS it. These tests drive each launcher's main() to the point of the
# gate, so deleting the gate fails here.
# --------------------------------------------------------------------------

def _load_script(name: str):
    import importlib.util
    path = os.path.join(_REPO_ROOT, "scripts", f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    prev = os.getcwd()
    os.chdir(_REPO_ROOT)          # launchers do sys.path.insert(0, "src")
    try:
        spec.loader.exec_module(mod)
    finally:
        os.chdir(prev)
    return mod


def _argv(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", list(args))


#: A fully-specified run_arena invocation: every offline gate satisfied, so any
#: SystemExit a test sees must come from the specific thing it is varying.
_ARENA_OK = ("run_arena.py", "--family", "gr00t", "--train-steps", "5",
             "--suite", "arena_gr1_fridge", "--policy-config-yaml", "/w/c.yaml")


class _ReachedAws(Exception):
    """Raised in place of load_config() so a test can never touch real AWS.

    The control tests below drive `main()` past the offline gates on purpose. On a
    machine WITH credentials, letting `main()` continue would upload sourcedirs to
    S3 and upsert a pipeline -- a test suite advertised as "no AWS" must not be able
    to do that by accident. Patching the first AWS-touching call gives a positive,
    deterministic signal that the gates were cleared, with no network.
    """


def _block_aws(monkeypatch, mod):
    monkeypatch.setattr(mod, "load_config",
                        lambda *a, **k: (_ for _ in ()).throw(
                            _ReachedAws("reached load_config")))
    # resolve_image_digest calls ECR, and the submitter now resolves the evaluator image to a DIGEST
    # before load_config -- the attestation must name the bytes that ran, and a tag cannot. Blocking only
    # load_config let that reach real ECR and fail on credentials, which is the harness missing an AWS
    # boundary rather than a product defect.
    if hasattr(mod, "resolve_image_digest"):
        monkeypatch.setattr(mod, "resolve_image_digest",
                            lambda uri, **k: (uri if "@sha256:" in uri
                                              else f"{uri.rsplit(':', 1)[0]}@sha256:{'ab' * 32}"))

def _write_libero_experimental(suite_dir):
    """An experimental suite on the LIBERO simulator, so only the status gate fires."""
    with open(suite_dir / "t_libero_exp.yaml", "w") as fh:
        yaml.safe_dump({
            "name": "t_libero_exp",
            "simulator": "libero",
            "status": "experimental",
            "canonical_task_ids": [0],
            "dataset": {"repo_id": "org/ds", "subdir": "", "revision": None,
                        "copy_libero_modality": True},
            # C2: libero suites must declare the protocol their numbers were produced under.
            # Resolution rejects one that does not, so this fixture supplies it -- the test is
            # about the STATUS gate, and a resolution error would mask that.
            "evaluation_protocol": {"n_action_steps": 8, "max_episode_steps": 720},
        }, fh)
    return "t_libero_exp"



def test_run_libero_refuses_an_arena_suite(monkeypatch):
    """The I5 fix must be invoked, not merely present."""
    mod = _load_script("run_libero")
    _argv(monkeypatch, "run_libero.py", "--family", "gr00t",
          "--suite", "arena_gr1", "--train-steps", "5")
    with pytest.raises(SystemExit, match="not 'libero'"):
        mod.main()


def test_run_libero_refuses_an_experimental_suite(monkeypatch, suite_dir):
    """The EXPERIMENTAL gate specifically -- not the simulator gate.

    `arena_g1` is both experimental AND a non-LIBERO suite, so pointing this test at
    it exercised the simulator check and proved nothing about the status check.
    Declare a LIBERO-simulator experimental suite so only one gate can fire.
    """
    _write_libero_experimental(suite_dir)
    mod = _load_script("run_libero")
    _argv(monkeypatch, "run_libero.py", "--family", "gr00t",
          "--suite", "t_libero_exp", "--train-steps", "5")
    with pytest.raises(SystemExit, match="experimental"):
        mod.main()


def test_run_libero_allow_experimental_bypasses_the_status_gate(
        monkeypatch, suite_dir):
    """...and the opt-in flag must actually let it through that gate."""
    _write_libero_experimental(suite_dir)
    mod = _load_script("run_libero")
    _block_aws(monkeypatch, mod)
    _argv(monkeypatch, "run_libero.py", "--family", "gr00t",
          "--suite", "t_libero_exp", "--train-steps", "5", "--allow-experimental")
    with pytest.raises(_ReachedAws):
        mod.main()


def test_run_libero_refuses_a_dataset_null_suite_for_train_default(monkeypatch):
    """S13: libero_goal is supported but eval-only; FineTune would reject it."""
    mod = _load_script("run_libero")
    _argv(monkeypatch, "run_libero.py", "--family", "gr00t",
          "--suite", "libero_goal", "--train-steps", "5")
    with pytest.raises(SystemExit, match="no fine-tune dataset"):
        mod.main()


def test_run_arena_valid_invocation_reaches_past_the_gates(monkeypatch):
    """Control for every run_arena test below.

    Without this, a test asserting SystemExit proves nothing: `arena_gr1_fridge` has
    `policy_config: null`, so ANY invocation without --policy-config-yaml exits on
    that check. Round 3 caught exactly this --
    test_run_arena_rejects_a_retired_step_budget_flag covers the retired override.
    A fully-specified run must get past the offline gates and only then fail on
    AWS config, proving the gates are not what is firing.
    """
    mod = _load_script("run_arena")
    _block_aws(monkeypatch, mod)
    _argv(monkeypatch, *_ARENA_OK, "--eval-trials", "3")
    # _ReachedAws means every offline gate passed and nothing hit the network.
    with pytest.raises(_ReachedAws):
        mod.main()


@pytest.mark.parametrize("version,tag", [("n16", "GR1"), ("n17", "new_embodiment")])
def test_default_arena_suite_resolves_without_policy_override(monkeypatch, version, tag):
    from vla_pipeline.arena import resolve_runtime

    mod = _load_script("run_arena")
    _block_aws(monkeypatch, mod)
    _argv(monkeypatch, "run_arena.py", "--train-steps", "5",
          "--gr00t-version", version)
    with pytest.raises(_ReachedAws):
        mod.main()
    suite = reg.resolve_suite(reg.resolve("gr00t", "isaac_arena").default_suite)
    knobs = resolve_runtime(suite, "gr00t", version)
    assert knobs["embodiment_tag"] == tag
    assert knobs["policy_config_yaml"] == (
        "/workspace/isaaclab_arena_gr00t/policy/config/"
        "gr1_manip_ranch_bottle_gr00t_closedloop_config.yaml")
    assert knobs["task_name"] == "put_item_in_fridge_and_close_door"


def test_run_arena_rejects_a_retired_step_budget_flag(monkeypatch):
    """--num-steps is retired. It must be REJECTED, not silently ignored: a caller
    passing it believes they are setting the sample size."""
    mod = _load_script("run_arena")
    _argv(monkeypatch, *_ARENA_OK, "--num-steps", "600")
    with pytest.raises(SystemExit):
        mod.main()


@pytest.mark.parametrize("bad", ["0", "-1", "abc"])
def test_run_arena_rejects_a_non_positive_eval_trials(monkeypatch, bad):
    """The episode count IS the sample size, so it is validated at submit time
    rather than failing inside a GPU job."""
    mod = _load_script("run_arena")
    _argv(monkeypatch, *_ARENA_OK, "--eval-trials", bad)
    with pytest.raises(SystemExit):
        mod.main()


def test_run_arena_calls_gate_suite(monkeypatch):
    """I22: the primary Arena launcher must INVOKE the gate, not just import it.

    The gate_suite() call in run_arena.main() was deletable with the whole suite
    green -- round 2's I18 asked for both launchers and only run_libero got pinned.
    """
    mod = _load_script("run_arena")
    _argv(monkeypatch, "run_arena.py", "--family", "gr00t", "--train-steps", "5",
          "--suite", "arena_g1", "--policy-config-yaml", "/w/c.yaml")
    with pytest.raises(SystemExit, match="experimental"):
        mod.main()


def test_run_arena_rejects_a_wrong_simulator_suite(monkeypatch):
    mod = _load_script("run_arena")
    _argv(monkeypatch, "run_arena.py", "--family", "gr00t", "--train-steps", "5",
          "--suite", "libero_spatial", "--policy-config-yaml", "/w/c.yaml")
    with pytest.raises(SystemExit, match="runs on simulator"):
        mod.main()


@pytest.mark.parametrize("bad", [
    "s3://b/p/x/other.tar.gz",      # sibling archive: would evaluate model.tar.gz
    "s3://b/p/x/",                  # prefix form: names no object to HEAD
    "s3://b/p/x",                   # prefix form without the trailing slash
    "s3://b/model.tar.gz.bak",      # not the canonical artifact
])
def test_submit_simeval_requires_the_canonical_checkpoint_object(monkeypatch, bad):
    """I16: mounting the PARENT PREFIX let evaluated bytes differ from recorded ones.

    The evaluator opens the canonical `model.tar.gz` from the mount, while the
    recorded source identity is whatever URI was supplied. Given
    `.../other.tar.gz` in a prefix that also holds a `model.tar.gz`, the sibling was
    evaluated under the other object's identity. Only the canonical object is
    accepted now, and it is mounted as that exact key.
    """
    mod = _load_script("submit_simeval")
    _argv(monkeypatch, "submit_simeval.py", "--eval-image", "acct/repo:tag",
          "--checkpoint-s3", bad,
          "--eval-source-dir", "s3://b/src.tar.gz",
          "--policy-config-yaml", "/w/c.yaml")
    with pytest.raises(SystemExit, match="canonical checkpoint OBJECT|no key component"):
        mod.main()


def test_submit_simeval_refuses_a_wrong_simulator_suite(monkeypatch):
    """Part of the I2 fix: submit_simeval must CALL gate_suite."""
    mod = _load_script("submit_simeval")
    _argv(monkeypatch, "submit_simeval.py", "--eval-image", "acct/repo:tag",
          "--checkpoint-s3", "s3://b/p/x/model.tar.gz",
          "--eval-source-dir", "s3://b/src.tar.gz",
          "--suite", "libero_spatial",
          "--policy-config-yaml", "/w/c.yaml")
    with pytest.raises(SystemExit, match="not 'isaac_arena'"):
        mod.main()


def test_submit_simeval_requires_n16_for_the_positive_control(monkeypatch):
    """I19 follow-on: posctrl serves an N1.6 checkpoint under GR1.

    The match= pattern is specific to the posctrl guard's own message, so this
    cannot pass on some earlier unrelated SystemExit.
    """
    mod = _load_script("submit_simeval")
    _argv(monkeypatch, "submit_simeval.py", "--eval-image", "acct/repo:tag",
          "--checkpoint-s3", "s3://b/p/x/model.tar.gz",
          "--eval-source-dir", "s3://b/src.tar.gz",
          "--posctrl-n16", "true", "--posctrl-revision", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
          "--policy-config-yaml", "/w/c.yaml")
    with pytest.raises(SystemExit,
                       match="serves an N1.6 checkpoint under embodiment"):
        mod.main()


def test_submit_simeval_accepts_the_positive_control_with_n16(monkeypatch):
    """Control: the guard must not fire on the correct combination."""
    mod = _load_script("submit_simeval")
    _argv(monkeypatch, "submit_simeval.py", "--eval-image", "acct/repo:tag",
          "--checkpoint-s3", "s3://b/p/x/model.tar.gz",
          "--eval-source-dir", "s3://b/src.tar.gz",
          "--posctrl-n16", "true", "--posctrl-revision", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "--gr00t-version", "n16",
          "--policy-config-yaml", "/w/c.yaml")
    _block_aws(monkeypatch, mod)
    with pytest.raises(_ReachedAws):
        mod.main()


# --------------------------------------------------------------------------
# BEHAVIOURAL replacements for the source-grep tests (C5)
#
# Round 3 reintroduced the exact I19, I6 and C3 bugs with the whole suite green,
# because five tests asserted `"<literal>" in open(script).read()`. A bug added as a
# COMMENT leaves the literal in place. The property "a byte sequence appears in this
# file" survives the bug -- and breaks on a harmless rename. These call the code.
# --------------------------------------------------------------------------

def _simeval_args(**over):
    import types
    base = dict(model_family="gr00t", gr00t_version="n17", use_groot_server="true",
                arena_connector="groot", eval_seed="100", eval_trials="1",
                eval_task_ids="all", checkpoint_s3="s3://b/p/x/model.tar.gz",
                hf_secret="s", posctrl_n16="false", posctrl_repo="",
                posctrl_revision="")
    base.update(over)
    return types.SimpleNamespace(**base)


_KNOBS = {"policy_config_yaml": "/w/c.yaml", "embodiment_tag": "GR1",
          "task_name": "put_item_in_fridge_and_close_door",
          "arena_embodiment": "gr1_joint", "object": "NONE"}


def test_submit_simeval_environment_sets_eval_gr00t_version():
    """I19: without this key the n16 route silently starts the n17 server."""
    mod = _load_script("submit_simeval")
    env = mod.build_environment(_simeval_args(gr00t_version="n16"),
                                reg.resolve_suite("arena_gr1_fridge"), _KNOBS)
    assert env["EVAL_GR00T_VERSION"] == "n16"
    # ...and the embodiment tag it ships must be the n16 one, or the two disagree.
    assert env["SM_HP_EMBODIMENT_TAG"] == "GR1"


def test_submit_simeval_environment_carries_the_resolved_knobs():
    mod = _load_script("submit_simeval")
    # eval_trials=3 (not the fixture default of 1) so the assertion proves the value
    # is carried through rather than coincidentally matching a default.
    env = mod.build_environment(_simeval_args(eval_trials="3"),
                                reg.resolve_suite("arena_gr1_fridge"), _KNOBS)
    assert env["SM_HP_TASK_NAME"] == "put_item_in_fridge_and_close_door"
    assert env["EVAL_POLICY_CONFIG_YAML"] == "/w/c.yaml"
    assert env["EVAL_ARENA_EMBODIMENT"] == "gr1_joint"
    assert env["EVAL_OBJECT"] == "NONE"
    # The episode count is the only sample-size knob and travels as EVAL_TRIALS.
    assert env["EVAL_TRIALS"] == "3"
    assert "SM_HP_NUM_STEPS" not in env
    assert env["EVAL_SUITE"] == "arena_gr1_fridge"


def test_submit_simeval_environment_posctrl_repo_is_optional():
    mod = _load_script("submit_simeval")
    suite = reg.resolve_suite("arena_gr1_fridge")
    assert "POSCTRL_N16_REPO" not in mod.build_environment(
        _simeval_args(), suite, _KNOBS)
    env = mod.build_environment(_simeval_args(posctrl_repo="nvidia/x"), suite, _KNOBS)
    assert "POSCTRL_N16_REPO" not in env
    assert env["N16_POSCTRL_CKPT_REPO"] == "nvidia/x"


# --- run_matrix.gate_cell (S14, behaviourally) ----------------------------

def test_run_matrix_gates_a_wrong_simulator_cell():
    mod = _load_script("run_matrix")
    with pytest.raises(SystemExit, match="not 'libero'"):
        mod.gate_cell({"family": "gr00t", "suite": "arena_gr1"}, 0)


def test_run_matrix_gates_an_experimental_cell(suite_dir):
    _write_libero_experimental(suite_dir)
    mod = _load_script("run_matrix")
    with pytest.raises(SystemExit, match="experimental"):
        mod.gate_cell({"family": "gr00t", "suite": "t_libero_exp"}, 0)


def test_run_matrix_gates_a_dataset_null_cell():
    mod = _load_script("run_matrix")
    with pytest.raises(SystemExit, match="no fine-tune dataset"):
        mod.gate_cell({"family": "gr00t", "suite": "libero_goal"}, 0)


def test_run_matrix_accepts_its_shipped_cells():
    """Control: the real manifest must pass, or the gates are simply too strict."""
    import json
    mod = _load_script("run_matrix")
    with open(os.path.join(_REPO_ROOT, "config", "matrix_manifest.json")) as fh:
        manifest = json.load(fh)
    assert manifest["cells"], "matrix manifest has no cells"
    for i, cell in enumerate(manifest["cells"]):
        assert mod.gate_cell(cell, i).name == cell["suite"]


# --- build_arena_connector.resolve_build_target (C3, I6) ------------------

def _spec(eval_image="vla/isaac-arena-gr00t:v1",
          eval_base_image="vla/isaac-arena:base-test"):
    import dataclasses
    return dataclasses.replace(reg.resolve("gr00t", "isaac_arena"),
                               eval_image_repo=eval_image,
                               eval_base_image_repo=eval_base_image)


_REGISTRY = "000000000000.dkr.ecr.us-east-1.amazonaws.com"
_OTHER_URI = "111122223333.dkr.ecr.us-east-1.amazonaws.com/vla/isaac-arena:base-test"


def test_relative_base_image_is_prefixed_with_our_registry():
    mod = _build_script()
    t = mod.resolve_build_target(_spec(), "gr00t", _REGISTRY)
    assert t.base_image_uri == f"{_REGISTRY}/vla/isaac-arena:base-test"


def test_full_uri_base_image_is_not_double_prefixed():
    """I6: prefixing our registry onto a full URI yields an unpullable FROM."""
    mod = _build_script()
    t = mod.resolve_build_target(_spec(), "gr00t", _REGISTRY,
                                 base_image=_OTHER_URI, skip_base_check=True)
    assert t.base_image_uri == _OTHER_URI
    assert not t.base_image_uri.startswith(_REGISTRY)


def test_full_uri_base_image_requires_skip_base_check():
    mod = _build_script()
    with pytest.raises(SystemExit, match="explicit registry"):
        mod.resolve_build_target(_spec(), "gr00t", _REGISTRY, base_image=_OTHER_URI)


def test_tag_defaults_to_the_pair_manifest_tag():
    """C3: the documented build must produce the tag SimEval actually runs."""
    mod = _build_script()
    t = mod.resolve_build_target(_spec(eval_image="vla/isaac-arena-gr00t:v7"),
                                 "gr00t", _REGISTRY)
    assert t.tag == "v7"
    assert t.ecr_repo == f"{_REGISTRY}/vla/isaac-arena-gr00t"


def test_explicit_tag_overrides_the_manifest():
    mod = _build_script()
    assert mod.resolve_build_target(_spec(), "gr00t", _REGISTRY, tag="v99").tag == "v99"


def test_default_build_produces_the_shipped_manifest_image():
    """Against the REAL manifest: --tag omitted builds exactly `eval_image`."""
    mod = _build_script()
    spec = reg.resolve("gr00t", "isaac_arena")
    t = mod.resolve_build_target(spec, "gr00t", _REGISTRY)
    assert f"{t.repo}:{t.tag}" == spec.eval_image_repo


def test_connector_refuses_to_push_into_its_own_base_repo():
    """The fresh-account regression fence, exercised rather than grepped."""
    mod = _build_script()
    with pytest.raises(SystemExit, match="SAME repo"):
        mod.resolve_build_target(
            _spec(eval_image="vla/isaac-arena:connector-test",
                  eval_base_image="vla/isaac-arena:base-test"), "gr00t", _REGISTRY)


def test_missing_base_image_declaration_fails_loud():
    mod = _build_script()
    with pytest.raises(SystemExit, match="no Arena base image declared"):
        mod.resolve_build_target(_spec(eval_base_image=None), "gr00t", _REGISTRY)


# --------------------------------------------------------------------------
# main() WIRING (I24)
#
# Round 4: the extracted functions were pinned, but nothing drove main(), so the
# CALL SITES were free. I6, C3, I19 and I23 were all reintroducible at the call site
# with the suite green -- including by deleting run_matrix's entire up-front gating
# block. Testing a pure function proves the function; it does not prove anyone uses
# its result.
# --------------------------------------------------------------------------

class _FakeCodeBuild:
    """Captures start_build kwargs instead of calling CodeBuild."""

    def __init__(self):
        self.started = None

    def start_build(self, **kw):
        self.started = kw
        return {"build": {"id": "vla-image-build:fake"}}


class _FakeS3:
    def __init__(self):
        self.put = None
        # C3: the publisher reads the object back and compares digests, because a same-key
        # replacement between the write and the build is the attack it addresses. A fake that
        # only records the PUT cannot exercise that, so it stores the bytes.
        self._objects: dict[tuple[str, str], bytes] = {}

    def put_object(self, **kw):
        self.put = kw
        self._objects[(kw["Bucket"], kw["Key"])] = kw["Body"]

    def get_object(self, Bucket, Key, **kw):
        return {"Body": io.BytesIO(self._objects[(Bucket, Key)])}


class _PresentEcr:
    def __init__(self):
        self.exceptions = self

    class RepositoryNotFoundException(Exception):
        pass

    class ImageNotFoundException(Exception):
        pass

    def batch_get_image(self, **kw):
        """The capability label lives in the image CONFIG blob, not in describe_images.

        Launchers now verify the evaluator image declares whatever its simulator requires, which reads
        the config blob via batch_get_image -> the manifest's config digest -> get_download_url_for_layer.
        A stub without these fails for a reason unrelated to what the test is checking.
        """
        import json
        return {"images": [{"imageManifest": json.dumps(
            {"config": {"digest": "sha256:" + "ef" * 32}})}]}

    def get_download_url_for_layer(self, **kw):
        return {"downloadUrl": "https://example.invalid/config-blob"}

    @staticmethod
    def capability_opener(_url):
        import json
        return json.dumps({"config": {"Labels": {
            "dev.physical-ai.baked-capabilities":
                "gr00t-baked:376ba890cff8c9de64d71d982772a9c36185fdd7:"
                "torchcodec-verified:libero-client-verified"}}}).encode()

    def describe_images(self, **kw):
        # A real digest, because the launchers now resolve tag -> digest before submitting: the
        # attestation must name image BYTES, and a stub returning no digest makes every submission
        # fail for a reason that has nothing to do with what the test is checking.
        return {"imageDetails": [{"imageDigest": "sha256:" + "cd" * 32}]}


def _fake_cfg():
    from vla_pipeline.config import PipelineConfig
    return PipelineConfig(account_id="000000000000", region="us-east-1",
                          training_role_arn="arn:aws:iam::000000000000:role/train", workload_role_arn="arn:aws:iam::000000000000:role/wl", validation_role_arn="arn:aws:iam::000000000000:role/val", trust_bucket="trust-bucket", handoff_bucket="handoff-bucket",
                          role_arn="arn:aws:iam::000000000000:role/r", bucket="b")


def _run_build_main(monkeypatch, *argv, repository=None):
    """Drive build_arena_connector.main() with every AWS client faked.

    No network: boto3.client is replaced wholesale, so a missed patch surfaces as a
    KeyError on an unexpected service name rather than a real API call.
    """
    mod = _build_script()
    from unittest.mock import Mock

    cb, s3 = _FakeCodeBuild(), _FakeS3()
    ssm = Mock()
    ssm.get_parameter.return_value = {
        "Parameter": {"Value": repository or f"{_REGISTRY}/physical-ai/isaac-lab-arena"}}
    clients = {"codebuild": cb, "s3": s3, "ecr": _PresentEcr(), "ssm": ssm}
    monkeypatch.setattr(mod, "load_config", lambda *a, **k: _fake_cfg())
    monkeypatch.setattr(mod.boto3, "client",
                        lambda service, **k: clients[service])
    _argv(monkeypatch, "build_arena_connector.py", *argv)
    mod.main()
    return {e["name"]: e["value"]
            for e in cb.started["environmentVariablesOverride"]}


def test_build_main_passes_the_resolved_base_uri_and_tag(monkeypatch):
    """C3 + I6 at the CALL SITE: what actually reaches CodeBuild.

    Pins that main() forwards resolve_build_target()'s values rather than
    recomputing them -- the round-4 mutation recomputed base_image_uri at the call
    site (reintroducing the double-prefix bug) with the whole suite green.
    """
    env = _run_build_main(monkeypatch, "--connector", "gr00t")
    spec = reg.resolve("gr00t", "isaac_arena")
    _, declared_tag = spec.eval_image_repo.rsplit(":", 1)
    assert env["IMAGE_TAG"] == declared_tag          # C3
    assert env["ARENA_BASE_IMAGE"] == f"{_REGISTRY}/{spec.eval_base_image_repo}"
    assert env["ECR_REPO_URI"] == f"{_REGISTRY}/{spec.eval_image_repo.rsplit(':', 1)[0]}"
    assert env["AWS_DEFAULT_REGION"] == "us-east-1"


def test_build_main_does_not_double_prefix_a_full_uri_base(monkeypatch):
    """I6 at the call site: a full-URI base must reach the build unprefixed."""
    env = _run_build_main(monkeypatch, "--connector", "gr00t",
                          "--base-image", _OTHER_URI, "--skip-base-check")
    assert env["ARENA_BASE_IMAGE"] == _OTHER_URI
    assert not env["ARENA_BASE_IMAGE"].startswith(_REGISTRY)


def test_build_main_uses_foundation_repository_instead_of_manifest(monkeypatch):
    repository = f"{_REGISTRY}/custom/arena-evaluation"
    env = _run_build_main(monkeypatch, "--connector", "gr00t", repository=repository)
    assert env["ECR_REPO_URI"] == repository
    assert env["IMAGE_TAG"] == reg.resolve("gr00t", "isaac_arena").eval_image_repo.rsplit(":", 1)[1]


def test_build_main_rejects_foundation_repository_in_another_account(monkeypatch):
    repository = "111122223333.dkr.ecr.us-east-1.amazonaws.com/custom/arena"
    with pytest.raises(ValueError, match="configured account and region"):
        _run_build_main(monkeypatch, "--connector", "gr00t", repository=repository)


def test_build_main_honours_an_explicit_tag(monkeypatch):
    env = _run_build_main(monkeypatch, "--connector", "gr00t", "--tag", "v42")
    assert env["IMAGE_TAG"] == "v42"


@pytest.mark.parametrize("deploy", [False, True])
def test_matrix_submits_family_groups_and_training_identity(monkeypatch, tmp_path, deploy):
    import json
    from unittest.mock import Mock

    mod = _load_script("run_matrix")
    cfg = _fake_cfg()
    sm = Mock()
    sm.start_pipeline_execution.return_value = {"PipelineExecutionArn": "execution"}
    sm.describe_pipeline_execution.return_value = {"PipelineExecutionStatus": "Succeeded"}
    monkeypatch.setattr(mod, "load_config", lambda: cfg)
    # ECR too: the launcher resolves the eval image to a digest before submitting, so a stub that
    # knows only "sagemaker" fails with KeyError('ecr') -- a harness gap, not a product defect.
    monkeypatch.setattr(mod.boto3, "client",
                        lambda service, **kw: {"sagemaker": sm, "ecr": _PresentEcr()}[service])
    # The capability label lives in the image config BLOB, fetched over a pre-signed URL. Stubbing only
    # boto3 left that fetch reaching the real network for a fake host, so the test failed on DNS rather
    # than on anything it was checking.
    import vla_pipeline.registry as _registry
    monkeypatch.setattr(_registry, "_fetch_config_blob", _PresentEcr.capability_opener)
    monkeypatch.setattr(mod.boto3, "Session", Mock())
    monkeypatch.setattr("sagemaker.workflow.pipeline_context.PipelineSession", Mock())
    pipeline = Mock()
    pipeline.upsert.return_value = {"PipelineArn": "pipeline", "PipelineVersionId": 17}
    monkeypatch.setattr(mod, "build_pipeline", Mock(return_value=pipeline))
    monkeypatch.setattr(mod, "stage_validate_code", lambda: "validator.py")
    monkeypatch.setattr(mod, "upload_code", lambda *a: "s3://b/validate.py")
    monkeypatch.setattr(mod, "stage", lambda family: family)
    monkeypatch.setattr(mod, "upload_directory", lambda cfg, family: f"s3://b/{family}.tar.gz")
    monkeypatch.setattr(mod.time, "sleep", lambda seconds: None)
    cells = [{"family": family, "suite": suite}
             for family in ("gr00t", "openvla", "molmoact2")
             for suite in ("libero_spatial", "libero_object")]
    manifest = tmp_path / "matrix.json"
    manifest.write_text(json.dumps({
        "protocol": {"train_steps": 200, "eval_trials": 1, "eval_seed": 100,
                     "eval_task_ids": "all", "success_threshold": 0.0},
        "cells": cells,
    }))
    selection = ["--deploy"] if deploy else ["--pipeline-version-id", "11"]
    _argv(monkeypatch, "run_matrix.py", "--manifest", str(manifest), *selection)
    with pytest.raises(SystemExit) as exit_info:
        mod.main()
    assert exit_info.value.code == 0
    calls = sm.start_pipeline_execution.call_args_list
    assert len(calls) == len(cells)
    parameters = []
    for cell, call in zip(cells, calls):
        assert call.kwargs["PipelineVersionId"] == (17 if deploy else 11)
        params = {p["Name"]: p["Value"] for p in call.kwargs["PipelineParameters"]}
        parameters.append(params)
        assert params["ModelPackageGroupName"] == reg.resolve(cell["family"], "libero").registry_group
        assert params["TrainSuite"] == ("unified" if cell["family"] == "molmoact2" else cell["suite"])
        assert params["Suite"] == cell["suite"]
    assert {k: v for k, v in parameters[-2].items() if k != "Suite"} == {
        k: v for k, v in parameters[-1].items() if k != "Suite"}
    assert mod.build_pipeline.call_count == int(deploy)


def test_run_matrix_main_gates_before_touching_aws(monkeypatch, tmp_path):
    """I23 + S23: a bad cell must abort BEFORE load_config or any upload.

    Round 4 deleted the entire up-front gating block with the suite green. Driving
    main() with load_config patched to raise proves the gate runs first: if the block
    is removed, _ReachedAws is raised instead of the gate's SystemExit.
    """
    import json as _json
    mod = _load_script("run_matrix")
    _block_aws(monkeypatch, mod)
    manifest = tmp_path / "m.json"
    manifest.write_text(_json.dumps({
        "protocol": {"train_steps": 5, "eval_trials": 1, "eval_seed": 1000,
                     "eval_task_ids": "all", "success_threshold": 0.0},
        # cell 1 is invalid: an Arena suite in a LIBERO-only matrix.
        "cells": [{"family": "gr00t", "suite": "libero_spatial"},
                  {"family": "gr00t", "suite": "arena_gr1"}],
    }))
    _argv(monkeypatch, "run_matrix.py", "--manifest", str(manifest), "--deploy")
    with pytest.raises(SystemExit, match="matrix cell 1"):
        mod.main()


def test_run_matrix_main_reaches_aws_only_after_all_cells_pass(monkeypatch, tmp_path):
    """Control: a valid manifest clears the gate and then stops at load_config."""
    import json as _json
    mod = _load_script("run_matrix")
    _block_aws(monkeypatch, mod)
    manifest = tmp_path / "m.json"
    manifest.write_text(_json.dumps({
        "protocol": {"train_steps": 5, "eval_trials": 1, "eval_seed": 1000,
                     "eval_task_ids": "all", "success_threshold": 0.0},
        "cells": [{"family": "gr00t", "suite": "libero_spatial"}],
    }))
    _argv(monkeypatch, "run_matrix.py", "--manifest", str(manifest), "--deploy")
    with pytest.raises(_ReachedAws):
        mod.main()
