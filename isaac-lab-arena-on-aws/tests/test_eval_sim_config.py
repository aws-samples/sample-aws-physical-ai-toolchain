"""The Arena params collapse to one EvalSimConfig blob, resolved from the suite.

Three checks:
  1. run_arena.build_eval_sim_config() derives every Arena knob from the SUITE
     MANIFEST (config/suites/<suite>.yaml), with CLI flags as overrides, and
     refuses to submit a run whose policy config is unresolved.
  2. The pipeline definition surface: EvalSimConfig param IN, the 6 discrete
     params OUT; SimEval env has EVAL_SIM_CONFIG and none of the 6 discrete
     SM_HP_*/EVAL_* knobs, while retaining the routing vars ARENA_CONNECTOR +
     SM_HP_USE_GROOT_SERVER (shell entrypoint reads these).
  3. eval_entry._seed_env_from_eval_sim_config seeds the 6 discrete env keys from
     a blob, and blob values win over image-baked discrete env.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import types

import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if os.path.join(_REPO_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))


def _load_script(module_name: str):
    path = os.path.join(_REPO_ROOT, "scripts", f"{module_name}.py")
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run_arena():
    prev = os.getcwd()
    os.chdir(_REPO_ROOT)
    try:
        return _load_script("run_arena")
    finally:
        os.chdir(prev)


def _args(**over):
    """An argparse namespace with every Arena override unset (= use the manifest)."""
    base = dict(family="gr00t", gr00t_version="n17", embodiment_tag=None,
                policy_config_yaml=None, arena_embodiment=None, object=None)
    base.update(over)
    return types.SimpleNamespace(**base)


def _blob(suite_name, **over):
    from vla_pipeline.registry import resolve_suite
    run_arena = _run_arena()
    suite = resolve_suite(suite_name)
    return json.loads(run_arena.build_eval_sim_config(suite, _args(**over)))


def test_eval_sim_config_derives_from_suite_manifest():
    """Every Arena knob comes from the suite, not from a CLI default.

    This is the regression guard for the train/eval mismatch: the launcher used
    to default to `task_name=cube_goal_pose` / `embodiment_tag=LIBERO_PANDA`
    regardless of --suite, so `--suite arena_gr1_fridge` fine-tuned on the fridge
    dataset and then evaluated a different task on a Franka embodiment tag.
    """
    blob = _blob("arena_gr1_fridge", policy_config_yaml="/workspace/gr1.yaml")
    assert blob == {
        "embodiment_tag": "new_embodiment",           # gr00t n17 -> custom slot
        "task_name": "put_item_in_fridge_and_close_door",
        "policy_config_yaml": "/workspace/gr1.yaml",
        "arena_embodiment": "gr1_joint",              # GR1 joint-control twin
        "object": "NONE",                             # task takes no --object
        # No budget key: the sample size is an episode count carried as
        # EvalTrials -> EVAL_TRIALS -> --num_episodes, so it cannot disagree
        # with a separately transported step budget.
    }
    # None of the old cross-cell defaults can appear for an Arena suite.
    assert "LIBERO_PANDA" not in blob.values()
    assert "cube_goal_pose" not in blob.values()
    assert "AUTO" not in blob.values()


def test_eval_sim_config_embodiment_tag_tracks_gr00t_version():
    """n16 has a native GR1 head; n17 must fall back to the custom slot.

    Train and eval read the SAME manifest field, so the eval server's
    --embodiment-tag cannot drift from what FineTune trained under.
    """
    n17 = _blob("arena_gr1_fridge", policy_config_yaml="/w/x.yaml",
                gr00t_version="n17")
    n16 = _blob("arena_gr1_fridge", policy_config_yaml="/w/x.yaml",
                gr00t_version="n16")
    assert n17["embodiment_tag"] == "new_embodiment"
    assert n16["embodiment_tag"] == "GR1"


def test_eval_sim_config_object_flag_from_manifest():
    """A task WITH an --object selector emits it; one without emits the sentinel."""
    g1 = _blob("arena_g1", policy_config_yaml="/w/x.yaml")
    assert g1["task_name"] == "galileo_g1_locomanip_pick_and_place"
    assert g1["object"] == "brown_box"
    assert g1["arena_embodiment"] == "g1_wbc_joint"


def test_eval_sim_config_cli_overrides_win():
    blob = _blob("arena_gr1_fridge", policy_config_yaml="/w/x.yaml",
                 arena_embodiment="NONE", embodiment_tag="GR1")
    assert "num_steps" not in blob
    assert blob["arena_embodiment"] == "NONE"
    assert blob["embodiment_tag"] == "GR1"


def test_eval_sim_config_carries_no_budget_key():
    """The blob must not transport a sample size. Two independently transported
    budgets (steps here, trials elsewhere) is the defect that produced a
    zero-episode evaluation."""
    blob = _blob("arena_gr1_fridge", policy_config_yaml="/w/x.yaml")
    assert "num_steps" not in blob and "num_episodes" not in blob


def test_eval_sim_config_refuses_a_suite_with_no_embodiment_tag():
    """No declared tag -> fail at submit, never emit the unconsumable "NONE".

    `embodiment_tag` goes straight to the GR00T server's --embodiment-tag, which has
    no "omit" sentinel, so "NONE" would die on an opaque tyro enum error after the
    venv build. `family_override()` treats an absent entry as "use the family
    default", which is right for LIBERO but not for Arena.
    """
    with pytest.raises(SystemExit) as ei:
        _blob("arena_gr1_fridge", policy_config_yaml="/w/x.yaml",
              gr00t_version="n99")          # no such family_overrides entry
    assert "embodiment_tag" in str(ei.value)


def test_eval_sim_config_refuses_unresolved_policy_config():
    """No policy config -> fail at SUBMIT, not 20 minutes into SimEval.

    The eval entry refuses to auto-discover one (Arena's packaged examples are
    embodiment placeholders and yield an invalid rollout), so a run submitted
    without a config is guaranteed to burn a GPU job and fail.
    """
    from dataclasses import replace

    from vla_pipeline.registry import resolve_suite

    suite = resolve_suite("arena_gr1_fridge")
    suite = replace(suite, arena=replace(suite.arena, policy_config=None))
    with pytest.raises(SystemExit, match="policy config"):
        _run_arena().build_eval_sim_config(suite, _args())


def test_pipeline_surface_collapsed():
    from vla_pipeline.config import PipelineConfig
    from vla_pipeline.pipeline import build_pipeline
    cfg = PipelineConfig(
        account_id="000000000000", region="us-east-1",
                          training_role_arn="arn:aws:iam::000000000000:role/train", workload_role_arn="arn:aws:iam::000000000000:role/wl", validation_role_arn="arn:aws:iam::000000000000:role/val", trust_bucket="trust-bucket", handoff_bucket="handoff-bucket",
        role_arn="arn:aws:iam::000000000000:role/r", bucket="b")
    definition = build_pipeline(cfg).definition()

    # Parameter surface: new param in, the 6 old ones gone.
    assert "EvalSimConfig" in definition
    for gone in ("EvalEmbodimentTag", "EvalTaskName", "EvalPolicyConfigYaml",
                 "EvalArenaEmbodiment", "EvalObject", "EvalNumSteps"):
        assert gone not in definition, f"stale param {gone} still in definition"

    # SimEval env: single blob var in; 6 discrete knobs gone; routing vars kept.
    assert "EVAL_SIM_CONFIG" in definition
    for gone in ("SM_HP_EMBODIMENT_TAG", "SM_HP_TASK_NAME",
                 "EVAL_POLICY_CONFIG_YAML", "EVAL_ARENA_EMBODIMENT",
                 "EVAL_OBJECT", "SM_HP_NUM_STEPS"):
        assert gone not in definition, f"stale env {gone} still in definition"
    assert "ARENA_CONNECTOR" in definition        # shell-routed, must stay
    assert "SM_HP_USE_GROOT_SERVER" in definition  # shell-routed, must stay


#: Top-level modules that only exist inside the Arena eval image. Anything else
#: failing to import is a real regression, not an environment gap.
_ARENA_ONLY_MODULES = {"isaacsim", "isaaclab", "isaaclab_arena", "omni", "carb",
                       "gr00t", "zmq", "torch", "numpy"}


@pytest.fixture(autouse=True)
def restore_environment():
    from unittest.mock import patch

    with patch.dict(os.environ):
        yield


def _load_eval_entry(env: dict):
    """Load steps/isaac_arena/eval_entry.py fresh with a controlled environ."""
    path = os.path.join(_REPO_ROOT, "entrypoints", "eval", "isaac_arena", "gr00t", "eval_entry.py")
    # scrub the keys this test cares about, then apply the provided env
    for k in ("EVAL_SIM_CONFIG", "SM_HP_EMBODIMENT_TAG", "SM_HP_TASK_NAME",
              "EVAL_POLICY_CONFIG_YAML", "EVAL_ARENA_EMBODIMENT", "EVAL_OBJECT",
              "SM_HP_NUM_STEPS"):
        os.environ.pop(k, None)
    os.environ.update(env)
    spec = importlib.util.spec_from_file_location(
        f"eval_entry_{len(sys.modules)}", path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except ModuleNotFoundError as e:
        # Narrow: skip ONLY for the Arena/Isaac-only deps that genuinely cannot be
        # installed here. A blanket skip would silently turn every test in this
        # file green the moment eval_entry grew any new top-level import.
        if (e.name or "").split(".")[0] not in _ARENA_ONLY_MODULES:
            raise
        pytest.skip(f"eval_entry.py needs an Arena-only module absent here: {e!r}")
    return mod


def test_eval_entry_seeds_discrete_env_from_blob():
    blob = json.dumps({
        "embodiment_tag": "new_embodiment",
        "task_name": "gr1_open_microwave",
        "policy_config_yaml": "/workspace/x.yaml",
        "arena_embodiment": "gr1_joint",
        "object": "brown_box",
    })
    _load_eval_entry({"EVAL_SIM_CONFIG": blob})
    assert os.environ["SM_HP_EMBODIMENT_TAG"] == "new_embodiment"
    assert os.environ["SM_HP_TASK_NAME"] == "gr1_open_microwave"
    assert os.environ["EVAL_POLICY_CONFIG_YAML"] == "/workspace/x.yaml"
    assert os.environ["EVAL_ARENA_EMBODIMENT"] == "gr1_joint"
    assert os.environ["EVAL_OBJECT"] == "brown_box"
    # The retired step budget must not be seeded from the blob at all.
    assert "SM_HP_NUM_STEPS" not in os.environ


def test_eval_entry_blob_is_authoritative_over_baked_env():
    # Review fix (#2): blob values must WIN over a pre-existing (e.g. image-baked)
    # discrete env, so the pipeline's requested config can never be shadowed.
    blob = json.dumps({"task_name": "gr1_open_microwave",
                       "arena_embodiment": "gr1_joint"})
    _load_eval_entry({"EVAL_SIM_CONFIG": blob,
                      "SM_HP_TASK_NAME": "stale_baked_task",
                      "EVAL_ARENA_EMBODIMENT": "stale_baked_embodiment"})
    assert os.environ["SM_HP_TASK_NAME"] == "gr1_open_microwave"
    assert os.environ["EVAL_ARENA_EMBODIMENT"] == "gr1_joint"


def test_eval_entry_has_no_cross_cell_task_default():
    """The container must not carry a hardcoded Arena task.

    `TASK_NAME = os.environ.get("SM_HP_TASK_NAME", "cube_goal_pose")` meant an
    absent (or task_name-less) EVAL_SIM_CONFIG rolled out `cube_goal_pose` for ANY
    suite -- the pre-suite-manifest bug, living inside the image where no launcher
    check can reach it. `per_task[].task` is written from TASK_NAME, so it would
    also be the value the coherence gate compares.
    """
    path = os.path.join(_REPO_ROOT, "entrypoints", "eval", "isaac_arena", "gr00t",
                        "eval_entry.py")
    with open(path) as fh:
        src = fh.read()
    assert 'os.environ.get("SM_HP_TASK_NAME", "cube_goal_pose")' not in src
    assert "SM_HP_TASK_NAME is unset" in src, (
        "expected a fail-loud guard when SM_HP_TASK_NAME is absent")


def test_eval_entry_requires_a_task_name(monkeypatch):
    """With no task in the blob and no discrete env, the entry refuses to run."""
    with pytest.raises(RuntimeError, match="SM_HP_TASK_NAME is unset"):
        _load_eval_entry({"EVAL_SIM_CONFIG": json.dumps(
            {"arena_embodiment": "gr1_joint"})})


def test_eval_entry_rejects_unknown_blob_key():
    # Review fix (#5): a typo'd/unknown key hard-fails instead of silently
    # falling back to eval-entry defaults.
    bad = json.dumps({"task_name": "t", "bogus_key": "x"})
    with pytest.raises(ValueError):
        _load_eval_entry({"EVAL_SIM_CONFIG": bad})


def test_eval_entry_rejects_the_retired_step_budget_key():
    """A control plane still sending a step budget belongs to the old contract
    and must fail loud rather than run with a silently ignored sample size."""
    bad = json.dumps({"task_name": "t", "num_steps": "280"})
    with pytest.raises(ValueError, match="unknown key"):
        _load_eval_entry({"EVAL_SIM_CONFIG": bad})
