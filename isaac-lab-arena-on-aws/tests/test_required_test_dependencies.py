"""I6: a skipped guard test is indistinguishable from an absent guard.

The behavioural audit tests call pytest.importorskip("torch") and friends, and [dev] declared none
of those packages. So a clean CI run -- installing only .[dev] and printing skips without failing
on them -- omitted the behavioural tests for the newest live-policy guards and reported success.

These assertions are deliberately NOT importorskip. If the environment lacks a dependency the guard
tests need, that must FAIL here rather than silently reduce coverage somewhere else.
"""
from __future__ import annotations

import importlib

import pytest

# Each entry is a module the behavioural guard tests import, with the guard it underpins.
_REQUIRED = {
    "torch": "MolmoAct2 / GR00T live-policy load audits",
    "numpy": "success-telemetry shape validation",
    "safetensors": "checkpoint tensor enumeration in the policy audits",
    "yaml": "GR1 action-contract permutation derivation from real Arena configs",
    "tqdm": "LIBERO evaluator harness import",
    "botocore": "registration ContentDigest checked against the SageMaker model",
}


@pytest.mark.parametrize("module", sorted(_REQUIRED))
def test_a_guard_dependency_is_installed(module):
    try:
        importlib.import_module(module)
    except ImportError as exc:  # pragma: no cover - the point of the test
        pytest.fail(
            f"{module!r} is not installed, so the guard tests for "
            f"{_REQUIRED[module]} will SKIP rather than run. A skipped guard test is "
            f"indistinguishable from an absent guard. Install the [dev] extra. ({exc})")


def test_every_importorskip_module_is_declared_in_dev_extras():
    """A new importorskip must not be able to reintroduce silent skipping."""
    import pathlib
    import re
    root = pathlib.Path(__file__).resolve().parents[1]
    used = set()
    for path in (root / "tests").glob("test_*.py"):
        used.update(re.findall(r'importorskip\(\s*"([a-z_]+)"', path.read_text()))
    declared = (root / "pyproject.toml").read_text()
    # yaml ships as pyyaml; map the import name to the distribution name.
    dist = {"yaml": "pyyaml"}
    missing = sorted(m for m in used if dist.get(m, m) not in declared)
    assert not missing, (
        f"these modules are importorskip'd by the test suite but not declared in the dev "
        f"extras, so their tests can silently skip in CI: {missing}")
