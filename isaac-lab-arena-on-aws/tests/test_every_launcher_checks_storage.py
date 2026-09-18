"""R4#M2: every launcher must call the storage check, not just the ones that were corrected.

tests/test_storage_fits_the_instance.py covers check_storage_fits itself -- given an instance type
and a volume size, does it accept or refuse correctly. What it does NOT cover is whether each
launcher actually CALLS it. run_matrix.py did not, and nothing failed, because the function's own
tests pass whether or not anybody invokes it.

That is the recurring shape in this component: a shared guard is added, one call site is converted,
and the remaining sites are found later by a reviewer or a real run. This test enumerates the sites
instead, so a launcher that drops the call fails here rather than at submission time.

It parses the source rather than importing it. These launchers read AWS config and argv at import,
so importing them in a test is not free; the question here is only whether the call is present, and
the syntax tree answers that exactly.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

#: Launcher -> the stages it submits. A launcher that submits a training job must check the training
#: instance's storage, and likewise for eval. run_matrix submits both.
_LAUNCHERS = {
    "scripts/run_arena.py": ("train", "eval"),
    "scripts/run_libero.py": ("train", "eval"),
    "scripts/run_matrix.py": ("train", "eval"),
    "scripts/submit_simeval.py": ("eval",),
}

_GUARD = "check_storage_fits"


def _calls_to(tree: ast.AST, name: str) -> list[ast.Call]:
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        target = node.func
        if isinstance(target, ast.Name) and target.id == name:
            found.append(node)
        elif isinstance(target, ast.Attribute) and target.attr == name:
            found.append(node)
    return found


@pytest.mark.parametrize("script,stages", sorted(_LAUNCHERS.items()))
def test_every_launcher_checks_storage_for_every_stage_it_submits(script, stages):
    path = _REPO_ROOT / script
    assert path.exists(), f"{script} is missing; update _LAUNCHERS if it was renamed"
    tree = ast.parse(path.read_text())

    assert _GUARD in {
        alias.asname or alias.name
        for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }, (f"{script} does not import {_GUARD}. A launcher that submits {stages} can then request a "
        f"volume the instance cannot provide, and the job fails after capacity has been allocated.")

    calls = _calls_to(tree, _GUARD)
    assert len(calls) >= len(stages), (
        f"{script} submits {stages} ({len(stages)} stage(s)) but calls {_GUARD} only "
        f"{len(calls)} time(s). Every stage's instance needs its own check -- train and eval can "
        f"be different instance types with different local-NVMe caps, so one call cannot cover "
        f"both. This is how run_matrix.py shipped with no storage check at all.")


def test_the_launcher_list_matches_what_is_on_disk():
    """A launcher added without being listed here would silently escape the check above.

    The trigger is CHOOSING HARDWARE **and** SUBMITTING A JOB. Both are required, because either
    alone produced false positives that were actually observed: submitting alone flagged
    run_plumbing.py, whose instance types are defined in the pipeline rather than the script, and
    verify_evaluation_cell.py, which only reads job state.
    """
    selects_hardware = ("select_instance", "select_volume_gb", "InstanceType", "VolumeSizeInGB")
    submits = ("create_training_job", "create_processing_job", "start_pipeline_execution",
               ".fit(", "Estimator(", "ScriptProcessor(", "create_model_package")

    on_disk = set()
    for path in (_REPO_ROOT / "scripts").glob("*.py"):
        source = path.read_text()
        if any(name in source for name in selects_hardware) and any(name in source for name in submits):
            on_disk.add(f"scripts/{path.name}")
    unlisted = on_disk - set(_LAUNCHERS)
    assert not unlisted, (
        f"these scripts choose an instance type or volume size AND submit a job, but are not in "
        f"_LAUNCHERS, so nothing checks that they verify storage fits: {sorted(unlisted)}. Add them "
        f"with the stages they submit.")
