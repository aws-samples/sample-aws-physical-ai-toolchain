"""The family manifest's declared operational requirements must govern the submitted request.

R4 (a real run): a hand-passed --train-instance of ml.g5.12xlarge (4x22.9 GB) was accepted for a
family whose manifest declares ml.g6e.12xlarge (4x45.8 GB). The job started, was billed, staged its
sourcedir, pulled the image, initialised DDP across four ranks, entered the forward pass, and died
on CUDNN_STATUS_ALLOC_FAILED. Nothing in the launcher had compared the requested hardware to what
the family declares.

The cause was not the override. It was that all three launchers carried their OWN hardcoded
defaults and never read the manifest's train_instance/eval_instance at all -- run_libero sent
ml.g6e.12xlarge for both roles, run_arena sent ml.g6e.2xlarge for both, and run_matrix inlined
ml.g6e.12xlarge per cell. So openvla evaluated on four large GPUs where its manifest asks for one
(ml.g5.2xlarge), and Arena would train GR00T on a single GPU where the manifest asks for four.
Volume had exactly the same shape: fixed in run_libero and run_matrix for cycle-15 I7, left
hardcoded at 100 in run_arena while gr00t.yaml declares 150.

This is the fourth field of the same manifest to be read by nobody, so the fix removes the
per-launcher derivation entirely rather than adding a fourth copy of it: resolve() owns all three
values, and every launcher takes them from the resolved spec.
"""
import pathlib
import re

import pytest

from vla_pipeline.registry import (RegistryError, list_models, list_supported_pairs,
                                   resolve)

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_LAUNCHERS = ("scripts/run_libero.py", "scripts/run_arena.py", "scripts/run_matrix.py")


def test_the_resolved_spec_carries_every_declared_operational_requirement():
    """Fails before the fix: ResolvedAdapterSpec had no instance or volume fields at all."""
    spec = resolve("openvla", "libero")
    assert spec.train_instance == "ml.g6e.12xlarge"
    # The value that matters: openvla's manifest asks for a SMALLER eval instance than train, and
    # every launcher was sending the large one for both.
    assert spec.eval_instance == "ml.g5.2xlarge", (
        "openvla declares a small eval instance; a launcher substituting the training type pays "
        "for four large GPUs to run an evaluation sized for one")
    # Per ROLE. openvla's defaults.json declared 300 train / 200 eval and NOTHING read it -- the
    # single shared figure could not express the distinction, so eval ran with a training volume.
    assert spec.train_volume_gb == 300
    assert spec.eval_volume_gb == 200, (
        "openvla declares a smaller eval volume because eval runs on a smaller instance; one shared "
        "figure cannot be checked against two different storage limits")


def test_every_family_declares_all_three_and_resolve_fails_closed_without_them():
    """Enumerate the DECLARED pairs; never catch RegistryError to find them.

    This loop used to call resolve() over the cross product of families and simulators and `continue`
    on RegistryError, meaning "that pair is not defined". But resolve raises RegistryError for a
    MALFORMED declaration too -- a family declaring a volume as a string, or omitting an instance type
    -- so the handler swallowed the exact failure the test exists to detect. A broken manifest passed
    by being skipped, and the assertions below never ran for it.

    list_supported_pairs() reads the declared pairs from the manifests, so every pair reaching resolve()
    here is one that MUST work. Any exception now propagates.
    """
    pairs = list_supported_pairs()
    assert pairs, "no declared pairs; this test would otherwise pass by iterating nothing"
    for family, simulator in pairs:
        spec = resolve(family, simulator)        # deliberately unguarded
        assert spec.train_instance.startswith("ml."), f"{family}/{simulator}"
        assert spec.eval_instance.startswith("ml."), f"{family}/{simulator}"
        assert isinstance(spec.train_volume_gb, int) and spec.train_volume_gb > 0, (
            f"{family}/{simulator} train_volume_gb={spec.train_volume_gb!r}")
        assert isinstance(spec.eval_volume_gb, int) and spec.eval_volume_gb > 0, (
            f"{family}/{simulator} eval_volume_gb={spec.eval_volume_gb!r}")
    print(f"checked {len(pairs)} declared pair(s): {pairs}")


def test_the_volume_write_is_not_nested_in_a_mode_branch():
    """Carried over from the deleted test_launcher_volume.py, which is superseded otherwise.

    Cycle-15 I7: the whole volume resolution sat inside `if args.checkpoint_s3:`, so a
    train-default run omitted VolumeSizeInGB and silently took the pipeline's 100 GB even with
    --volume-gb passed. That test also asserted the launcher calls a helper BY NAME, which is a
    claim about implementation rather than behaviour -- it went red the moment the derivation moved
    into resolve() even though the behaviour it protected got stricter. The name assertion is gone;
    this structural one remains, because no behavioural test can see a parameter that was never
    written.
    """
    import ast
    tree = ast.parse((_ROOT / "scripts/run_libero.py").read_text())
    writes = [n for n in ast.walk(tree)
              if isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Constant)
              and n.slice.value == "VolumeSizeInGB"]
    assert writes, "run_libero.py never writes VolumeSizeInGB, so every run takes the 100 GB default"
    for branch in ast.walk(tree):
        if isinstance(branch, ast.If) and "checkpoint_s3" in ast.dump(branch.test):
            nested = {id(n) for n in ast.walk(branch)}
            assert not all(id(w) in nested for w in writes), (
                "VolumeSizeInGB is only set when a checkpoint is supplied, so train-default runs "
                "ignore both the family manifest and --volume-gb")


def test_an_undeclared_instance_override_is_refused():
    """The exact command that caused a paid OOM must no longer be accepted.

    A real run passed --train-instance ml.g5.12xlarge (4x22.9 GB) for a family declaring
    ml.g6e.12xlarge (4x45.8 GB). The override won, the job was billed, staged its sourcedir, pulled
    its image, initialised DDP across four ranks, entered the forward pass and died on
    CUDNN_STATUS_ALLOC_FAILED. Changing the DEFAULT does not close that -- the override still wins.

    Deliberately not an approved-instance list per family: both reviewers rejected one as a second
    declaration of the same fact that would drift from the first. The declared value IS the approved
    value, and a departure must be stated rather than passing silently.
    """
    from vla_pipeline.registry import select_instance

    spec = resolve("openvla", "libero")
    assert select_instance(spec, "train", None) == spec.train_instance

    with pytest.raises(RegistryError, match="declares"):
        select_instance(spec, "train", "ml.g5.12xlarge")

    # A stated departure is allowed -- it appears in the command and the log.
    assert select_instance(spec, "train", "ml.g5.12xlarge", acknowledged=True) == "ml.g5.12xlarge"
    # Requesting exactly what is declared needs no acknowledgement.
    assert select_instance(spec, "train", spec.train_instance) == spec.train_instance


def test_a_volume_override_of_zero_is_an_error_not_a_default():
    """Absence is `is None`. A falsy explicit value silently selecting a default is how a declared
    figure stops governing -- `args.volume_size or spec.volume_gb` treated an explicit 0 as absent."""
    from vla_pipeline.registry import select_volume_gb

    spec = resolve("openvla", "libero")
    assert select_volume_gb(spec, "train", None) == 300
    assert select_volume_gb(spec, "eval", None) == 200
    assert select_volume_gb(spec, "train", 500) == 500
    for bad in (0, -1, True, 1.5, "300"):
        with pytest.raises(RegistryError):
            select_volume_gb(spec, "train", bad)


def test_the_dead_resource_keys_are_gone_from_defaults_json():
    """They were read by NOBODY: load_family_schemas() and registry.family_schemas() both project
    only input_config_schema and provenance_keys, so openvla's per-role block there was a declared
    value governing nothing -- while the YAML it duplicated could not express the same distinction.

    Every OTHER key stays: GR00T reads its repo pins, checkpoint ids, embodiment tag and server port
    from the same file, so removing more than the resource keys would break it.
    """
    import json

    dead = {"train_instance_type", "eval_instance_type", "train_volume_gb", "eval_volume_gb",
            "instance_type", "volume_gb"}
    for family in list_models():
        path = _ROOT / f"entrypoints/train/{family}/defaults.json"
        if not path.is_file():
            continue
        keys = set(json.loads(path.read_text()))
        assert not (keys & dead), (
            f"{family}/defaults.json still declares {sorted(keys & dead)}; nothing reads them, and a "
            f"second declaration of a fact the YAML owns is how the two drift apart")
        assert "input_config_schema" in keys, f"{family} lost the schema this file exists for"

    # And GR00T's other live keys survived.
    gr00t = set(json.loads((_ROOT / "entrypoints/train/gr00t/defaults.json").read_text()))
    for key in ("repo_commit", "checkpoint", "embodiment_tag", "server_port"):
        assert key in gr00t, f"gr00t/defaults.json lost {key}, which its evaluator reads"


@pytest.mark.parametrize("launcher", _LAUNCHERS)
def test_no_launcher_carries_its_own_instance_or_volume_default(launcher):
    """Asserts over a GLOB with a required count of zero, not a named list.

    A named list is how three launchers ended up with three different hardcoded instance types:
    each fix addressed the launcher in front of it. Any NEW launcher that hardcodes a default
    fails this without anyone remembering to add it here.
    """
    source = (_ROOT / launcher).read_text()
    # Strip comments and docstrings so the explanatory prose above a fix cannot satisfy the check.
    code = "\n".join(line.split("#")[0] for line in source.splitlines())
    code = re.sub(r'"""[\s\S]*?"""', "", code)

    hardcoded = re.findall(r'default\s*=\s*["\']ml\.[a-z0-9.]+["\']', code)
    assert not hardcoded, (
        f"{launcher} hardcodes an instance type as an argparse default {hardcoded}; it must fall "
        f"through to the family manifest's declared value via the resolved spec")

    inline = re.findall(r'"(?:Train|Eval)InstanceType",\s*"Value":\s*["\']ml\.[a-z0-9.]+["\']', code)
    assert not inline, (
        f"{launcher} inlines an instance type into the submitted parameters {inline}")

    vol = re.findall(r'--volume[-_](?:gb|size)["\'][^)]*?default\s*=\s*(\d+)', code, re.S)
    assert not vol, (
        f"{launcher} defaults its volume argument to a literal {vol}; the manifest declares "
        f"volume_gb and a launcher-side default silently overrides it")
