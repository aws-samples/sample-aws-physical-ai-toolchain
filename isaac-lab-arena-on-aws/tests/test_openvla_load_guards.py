"""Criticals 4 and 5: OpenVLA must not evaluate an incomplete or substituted model.

Two independent defects in the same upstream helper, both verified against pinned
moojink/openvla-oft@e4287e94:

  4. get_vla() calls AutoModelForVision2Seq.from_pretrained without requesting or checking
     loading diagnostics. Transformers initializes missing tensors, warns, and returns the
     model, so an incomplete checkpoint runs and is reported as a checkpoint evaluation.
     Rejecting logged episode exceptions does not cover it: successful construction is not
     an exception.

  5. Immediately before loading, update_auto_map() rewrites config.json and
     check_model_logic_mismatch() copies the CURRENT checkout's modeling_prismatic.py and
     configuration_prismatic.py over the checkpoint's. The wrapper hashes the checkpoint
     beforehand, so the reported digest describes a tree that was never evaluated -- and a
     supplied checkpoint can carry different model logic from the logic that earned its score.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import os
import pathlib
import re
import sys
import types
from unittest.mock import MagicMock

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_ENTRY = _REPO_ROOT / "entrypoints/eval/libero/openvla/eval_entry.py"
# Verbatim from moojink/openvla-oft@e4287e94541f459edc4feabc4e181f537cd569a8,
# experiments/robot/openvla_utils.py. Production refuses to run if its anchors are absent.
_PINNED = pathlib.Path(__file__).with_name("data") / "openvla_utils_pinned.py"


def _module():
    for name in ("digest", "validator"):
        sys.modules.setdefault(name, MagicMock())
    os.environ.update({
        "EVAL_CHECKPOINT": "/opt/ml/input/data/model", "EVAL_SUITE": "libero_spatial",
        "EVAL_SEED": "1000", "EVAL_TRIALS": "3",
        "EVAL_MODEL_SOURCE_URI": "s3://bucket/key/model.tar.gz",
    })
    spec = importlib.util.spec_from_file_location("openvla_guards_under_test", _ENTRY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _patched():
    return _module()._patch_openvla_load_guards(_PINNED.read_text())


def _guards():
    """Execute the injected guard definitions with the modules upstream provides.

    The guards run inside openvla_utils.py, which already imports these; a bare namespace
    would fail on a NameError that says nothing about the behaviour under test.
    """
    namespace = {"os": os, "json": json}
    exec(compile(_module()._OPENVLA_GUARDS, "<guards>", "exec"), namespace)
    return namespace


def test_the_patched_source_is_valid_python():
    ast.parse(_patched())


def test_loading_diagnostics_are_requested_at_the_real_constructor():
    patched = _patched()
    assert "output_loading_info=True" in patched
    assert "vla, _loading_info = AutoModelForVision2Seq.from_pretrained(" in patched
    # The keyword must follow the positional checkpoint argument, or the call is a SyntaxError.
    call = patched[patched.index("vla, _loading_info ="):]
    assert call.index("cfg.pretrained_checkpoint,") < call.index("output_loading_info=True")


def test_the_audit_runs_after_construction_and_before_use():
    patched = _patched()
    assert "_vla_strict_load_audit(_loading_info, cfg.pretrained_checkpoint)" in patched


def test_a_clean_load_passes():
    _guards()["_vla_strict_load_audit"](
        {"missing_keys": [], "unexpected_keys": [], "mismatched_keys": [], "error_msgs": []},
        "/ckpt")


@pytest.mark.parametrize("problem", [
    "missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"])
def test_an_incomplete_load_is_refused(problem):
    """A COMPLETE diagnostic structure with one problem populated -- the realistic case.

    Passing a single-key dict would now trip the completeness check instead, which would test
    a different thing than intended.
    """
    info = {"missing_keys": [], "unexpected_keys": [], "mismatched_keys": [], "error_msgs": []}
    info[problem] = ["something"]
    with pytest.raises(RuntimeError, match="strict load audit FAILED"):
        _guards()["_vla_strict_load_audit"](info, "/ckpt")


@pytest.mark.parametrize("absent", [
    "missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"])
def test_absent_diagnostics_are_not_a_clean_load(absent):
    """I7: an absent field was read as empty, so a load reporting NO diagnostics passed.

    Absence means the diagnostics could not be read, which is not the same as there being no
    problems -- and it is exactly what a changed upstream structure would look like.
    """
    info = {"missing_keys": [], "unexpected_keys": [], "mismatched_keys": [], "error_msgs": []}
    del info[absent]
    with pytest.raises(RuntimeError, match="diagnostics for .* are incomplete"):
        _guards()["_vla_strict_load_audit"](info, "/ckpt")


def test_a_none_valued_diagnostic_is_also_incomplete():
    """Present-but-None is unreadable too, not empty."""
    info = {"missing_keys": None, "unexpected_keys": [], "mismatched_keys": [],
            "error_msgs": []}
    with pytest.raises(RuntimeError, match="incomplete"):
        _guards()["_vla_strict_load_audit"](info, "/ckpt")


def test_model_code_substitution_is_refused_rather_than_performed():
    """Critical 5: upstream would replace the checkpoint's model code before loading."""
    with pytest.raises(RuntimeError, match="checkpoint model code differs"):
        _guards()["_refuse_model_code_substitution"](
            "/curr/modeling_prismatic.py", "/ckpt/modeling_prismatic.py")


def test_no_silent_copy_of_model_code_survives():
    patched = _patched()
    assert "shutil.copy2(curr_filepath, checkpoint_filepath)" not in patched, (
        "a remaining copy site would still substitute model code before loading")
    assert patched.count("_refuse_model_code_substitution(") == 3, (
        "both copy sites plus the helper definition")


def test_a_missing_checkpoint_model_file_is_also_refused():
    """Upstream copies its own file in when the checkpoint lacks one.

    A checkpoint missing its model code cannot be attested either, so that path is refused
    too rather than quietly completed from the current checkout.
    """
    patched = _patched()
    assert "A checkpoint MISSING its model code" in patched


def test_a_changed_upstream_stops_the_run():
    module = _module()
    moved = _PINNED.read_text().replace(
        "    vla = AutoModelForVision2Seq.from_pretrained(\n", "", 1)
    with pytest.raises(RuntimeError, match="openvla load-guard anchor 1 occurs 0 times"):
        module._patch_openvla_load_guards(moved)


def test_re_patching_is_refused():
    module = _module()
    with pytest.raises(RuntimeError):
        module._patch_openvla_load_guards(_patched())


def test_the_patch_is_applied_before_the_evaluation_subprocess():
    import ast

    source = _ENTRY.read_text()
    calls = [node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Call)]
    patches = [node for node in calls if isinstance(node.func, ast.Name)
               and node.func.id == "_patch_openvla_load_guards"]
    evaluations = [
        node for node in calls
        if isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name) and node.func.value.id == "subprocess"
        and node.func.attr in {"run", "Popen"}
        and node.args and isinstance(node.args[0], ast.Name) and node.args[0].id == "eval_cmd"
    ]
    assert len(patches) == len(evaluations) == 1
    assert patches[0].lineno < evaluations[0].lineno


def _ckpt(tmp_path, files):
    directory = tmp_path / "ckpt"
    directory.mkdir(exist_ok=True)
    for name, content in files.items():
        (directory / name).write_text(content)
    return str(directory)


# --- Cycle-5 C1: checkpoint-supplied processor code ------------------------------------

# Verbatim from openvla/openvla-7b, which the pinned OFT finetune also saves into its output.
# It DOES declare an auto_map, naming the same processor classes the image registers.
_REAL_PREPROCESSOR = (pathlib.Path(__file__).with_name("data")
                      / "openvla_preprocessor_config_real.json")


def test_the_components_own_trainer_output_is_accepted(tmp_path):
    """Cycle-6 I2: a blanket auto_map ban rejected the component's normal trainer output.

    The pinned finetune loads the base processor and saves it into the checkpoint, so a real
    checkpoint carries preprocessor_config.json with an auto_map pointing at
    processing_prismatic -- the same classes get_vla registers from the image. With
    trust_remote_code=False the registry supplies them and the checkpoint's copy is not
    imported, so this is legitimate.

    The previous "clean checkpoint" fixture contained NO processor configuration, which is why
    it could not detect the regression.
    """
    _guards()["_reject_checkpoint_processor_code"](_ckpt(tmp_path, {
        "config.json": "{}",
        "preprocessor_config.json": _REAL_PREPROCESSOR.read_text(),
        "modeling_prismatic.py": "x = 1",
        "configuration_prismatic.py": "y = 1",
        "processing_prismatic.py": "z = 1"}))


def test_a_checkpoint_with_no_processor_configuration_is_also_accepted(tmp_path):
    """Absence is fine; it is a NAMED UNTRUSTED class that is refused."""
    _guards()["_reject_checkpoint_processor_code"](_ckpt(tmp_path, {
        "config.json": "{}", "modeling_prismatic.py": "x = 1",
        "configuration_prismatic.py": "y = 1"}))


@pytest.mark.parametrize("name,key", [
    ("processor_config.json", "AutoProcessor"),
    ("preprocessor_config.json", "AutoImageProcessor"),
    ("tokenizer_config.json", "AutoTokenizer"),
])
def test_a_checkpoint_supplied_processor_class_is_refused(tmp_path, name, key):
    """C1: trust_remote_code=True imports and EXECUTES the named class with the job's
    credentials, and the model-code guards cover only the two prismatic files."""
    payload = json.dumps({"auto_map": {key: "sneaky_module.Thing"}})
    with pytest.raises(RuntimeError, match="names processor classes this evaluator does not"):
        _guards()["_reject_checkpoint_processor_code"](_ckpt(tmp_path, {name: payload}))


def test_an_unaudited_python_module_in_the_checkpoint_is_refused(tmp_path):
    """Only the two prismatic files are compared against the evaluation code."""
    with pytest.raises(RuntimeError, match="unaudited python modules"):
        _guards()["_reject_checkpoint_processor_code"](
            _ckpt(tmp_path, {"sneaky.py": "import os"}))


def test_an_unreadable_processor_config_is_refused(tmp_path):
    with pytest.raises(RuntimeError, match="unreadable"):
        _guards()["_reject_checkpoint_processor_code"](
            _ckpt(tmp_path, {"processor_config.json": "{not json"}))


def test_the_processor_no_longer_trusts_remote_code():
    patched = _patched()
    assert "trust_remote_code=False" in patched
    assert "_reject_checkpoint_processor_code(cfg.pretrained_checkpoint)" in patched


# --- Cycle-5 C2: update_auto_map mutated the digested checkpoint -----------------------

def _update_auto_map():
    patched = _patched()
    start = patched.index("def update_auto_map(")
    end = patched.index("\ndef ", start + 10)
    namespace = {"os": os, "json": json, "print": lambda *a, **k: None}
    exec(compile(patched[start:end], "<uam>", "exec"), namespace)
    return namespace["update_auto_map"]


_GOOD_AUTO_MAP = {
    "AutoConfig": "configuration_prismatic.OpenVLAConfig",
    "AutoModelForVision2Seq": "modeling_prismatic.OpenVLAForActionPrediction",
}


def test_a_matching_auto_map_leaves_the_checkpoint_untouched(tmp_path):
    """C2: upstream rewrote config.json and left a timestamped backup UNCONDITIONALLY.

    Both change the tree after its digest was taken -- the rewrite changes config.json's
    bytes, the backup adds a member the digest never saw.
    """
    path = _ckpt(tmp_path, {"config.json": json.dumps(
        {"model_type": "openvla", "auto_map": _GOOD_AUTO_MAP})})
    before = pathlib.Path(path, "config.json").read_bytes()
    _update_auto_map()(path)
    assert pathlib.Path(path, "config.json").read_bytes() == before
    assert sorted(p.name for p in pathlib.Path(path).iterdir()) == ["config.json"], (
        "no backup file may be created beside the digested config")


@pytest.mark.parametrize("auto_map", [None, {"AutoConfig": "other.Thing"}])
def test_a_differing_auto_map_is_refused_without_mutating(tmp_path, auto_map):
    config = {"model_type": "openvla"}
    if auto_map is not None:
        config["auto_map"] = auto_map
    path = _ckpt(tmp_path, {"config.json": json.dumps(config)})
    before = pathlib.Path(path, "config.json").read_bytes()
    with pytest.raises(RuntimeError, match="auto_map does not match"):
        _update_auto_map()(path)
    assert pathlib.Path(path, "config.json").read_bytes() == before
    assert sorted(p.name for p in pathlib.Path(path).iterdir()) == ["config.json"]


def test_no_config_rewrite_or_backup_survives_in_the_patched_source():
    patched = _patched()
    assert "json.dump(config, f, indent=2)" not in patched
    assert "config.json.back" not in patched
    assert "Updated config.json at" not in patched, (
        "narration claiming an update would mislead an operator reading a failure")


# --- I4: execute the patched get_vla rather than reading it ------------------------------

class _Recorder:
    """Records what from_pretrained was asked for, and what happened after it returned."""

    def __init__(self, info):
        self.info = info
        self.kwargs = None
        self.model_used = False

    def from_pretrained(self, checkpoint, **kwargs):
        self.kwargs = kwargs
        return self._model(), self.info

    def register(self, *args, **kwargs):
        pass

    def _model(self):
        recorder = self

        class _Model:
            class vision_backbone:
                @staticmethod
                def set_num_images_in_input(count):
                    recorder.model_used = True

            @staticmethod
            def eval():
                recorder.model_used = True

            def to(self, device):
                recorder.model_used = True
                return self

        return _Model()


def _run_get_vla(info):
    """Execute the PATCHED get_vla with doubles, returning the recorder.

    The tests that shipped with the audit checked source substrings and called the helper
    directly, so they passed while the config-mutation defect was still present and would
    also pass with the audit call inside unreachable code. This runs the real emitted
    function body.
    """
    module = _module()
    patched = _patched()
    start = patched.index("def get_vla(")
    end = patched.index("\ndef ", start + 10)
    guards = {}
    exec(compile(module._OPENVLA_GUARDS, "<guards>", "exec"), guards)

    recorder = _Recorder(info)
    namespace = {
        "print": lambda *a, **k: None,
        "os": os, "json": json,
        "model_is_on_hf_hub": lambda path: True,   # skip the mutation branch
        "AutoModelForVision2Seq": recorder,
        "AutoConfig": recorder, "AutoImageProcessor": recorder, "AutoProcessor": recorder,
        "OpenVLAConfig": object, "PrismaticImageProcessor": object,
        "PrismaticProcessor": object, "OpenVLAForActionPrediction": object,
        "torch": types.SimpleNamespace(bfloat16="bf16"),
        "_apply_film_to_vla": lambda vla, cfg: vla,
        "DEVICE": "cpu",
        "_load_dataset_stats": lambda vla, path: None,
        "_vla_strict_load_audit": guards["_vla_strict_load_audit"],
    }
    exec(compile(patched[start:end], "<get_vla>", "exec"), namespace)
    cfg = types.SimpleNamespace(
        pretrained_checkpoint="/ckpt", load_in_8bit=False, load_in_4bit=False,
        use_film=False, num_images_in_input=1)
    namespace["get_vla"](cfg)
    return recorder


def _clean_info():
    return {"missing_keys": [], "unexpected_keys": [], "mismatched_keys": [], "error_msgs": []}


def test_get_vla_requests_loading_diagnostics():
    """Asserted from the executed call, not from the source text."""
    recorder = _run_get_vla(_clean_info())
    assert recorder.kwargs.get("output_loading_info") is True


def test_get_vla_completes_on_a_clean_load():
    recorder = _run_get_vla(_clean_info())
    assert recorder.model_used, "a clean load must proceed to configure and eval the model"


@pytest.mark.parametrize("problem", [
    "missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"])
def test_get_vla_aborts_before_using_the_model(problem):
    """The audit must stop the run BEFORE the model is configured or returned.

    An audit that raised after the model was already prepared would still be an audit, but a
    caller holding a reference could use it. This asserts the ordering.
    """
    info = _clean_info()
    info[problem] = ["some.tensor"]
    with pytest.raises(RuntimeError, match="strict load audit FAILED"):
        _run_get_vla(info)


def test_the_audit_call_is_reachable():
    """The 'audit runs' source assertion could pass with the call inside dead code."""
    recorder = _run_get_vla(_clean_info())
    assert recorder.kwargs is not None, "from_pretrained was never reached"


def test_the_allowlist_matches_the_real_published_configuration():
    """The allowlist must be the real values, not values I assumed.

    If these drift apart, either legitimate checkpoints are rejected or untrusted classes are
    accepted -- both silent until a run fails or does not.
    """
    module = _module()
    exec(compile(module._OPENVLA_GUARDS, "<guards>", "exec"), {"os": os, "json": json})
    real = json.loads(_REAL_PREPROCESSOR.read_text())
    trusted = re.search(r"_TRUSTED_AUTO_MAP = \{(.*?)\}", module._OPENVLA_GUARDS, re.DOTALL)
    assert trusted, "the allowlist must be declared"
    for key, value in real["auto_map"].items():
        assert f'"{key}": "{value}"' in trusted.group(1), (
            f"the real published config names {key}={value!r}, which the allowlist omits")


def test_processing_prismatic_is_allowed_but_not_executed():
    """It is present in real checkpoints; trust_remote_code=False stops it being imported."""
    module = _module()
    assert '"processing_prismatic.py",' in module._OPENVLA_GUARDS
    assert "trust_remote_code=False" in _patched()
