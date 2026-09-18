"""S8: a fresh clone must resolve the same Terraform provider build.

The README promises a clone works from scratch, but the AWS provider constraint was
">= 5.60" while the lock recorded 6.64.0, and the lock file was git-ignored. A fresh clone
therefore resolved across the 5.x/6.x major boundary to whatever was latest at the time.
"""
from __future__ import annotations

import pathlib
import re
import subprocess

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_VERSIONS_TF = _REPO_ROOT / "infra/versions.tf"
_LOCK = _REPO_ROOT / "infra/.terraform.lock.hcl"



def _provider_constraint() -> str:
    """The aws PROVIDER constraint, not terraform's own required_version.

    Both are spelled `version =`, so an unanchored search finds required_version first.
    """
    source = _VERSIONS_TF.read_text()
    after_source = source[source.index('source = "hashicorp/aws"'):]
    match = re.search(r'version\s*=\s*"([^"]+)"', after_source)
    assert match, "the aws provider must declare a version constraint"
    return match.group(1)


def test_the_lock_file_exists():
    assert _LOCK.is_file(), (
        "the dependency lock file is what makes a clone select the same provider build")


def test_the_lock_file_is_not_git_ignored():
    """It was ignored, which defeated the pinning it exists to provide."""
    result = subprocess.run(
        ["git", "check-ignore", str(_LOCK)],
        cwd=str(_REPO_ROOT), capture_output=True, text=True)
    assert result.returncode != 0, (
        f"the lock file is git-ignored ({result.stdout.strip()}), so a fresh clone would not "
        f"have it and would re-resolve the provider")


def test_the_provider_constraint_pins_a_major_version():
    """An open-ended lower bound allows a major upgrade with breaking changes."""
    constraint = _provider_constraint()
    assert not constraint.startswith(">="), (
        f"constraint {constraint!r} is open-ended, so a fresh clone can cross a major "
        f"version boundary. Use a pessimistic constraint such as '~> 6.64'.")
    assert constraint.startswith("~>"), f"expected a pessimistic constraint, got {constraint!r}"


def test_the_locked_version_satisfies_the_declared_constraint():
    """A lock that disagrees with the constraint means one of them is a lie."""
    constraint = re.match(r'~>\s*([\d.]+)', _provider_constraint())
    assert constraint, "could not read the pessimistic constraint"
    locked = re.search(r'^\s*version\s*=\s*"([\d.]+)"', _LOCK.read_text(), re.MULTILINE)
    assert locked, "could not read the locked provider version"
    floor = [int(part) for part in constraint.group(1).split(".")]
    actual = [int(part) for part in locked.group(1).split(".")]
    assert actual[0] == floor[0], (
        f"locked provider {locked.group(1)} is a different MAJOR version from the "
        f"constraint ~> {constraint.group(1)}")
    assert actual >= floor, (
        f"locked provider {locked.group(1)} is below the constraint floor "
        f"{constraint.group(1)}")


def test_the_lock_records_the_same_constraint_it_was_generated_from():
    """A stale constraints line means the lock predates the current declaration."""
    declared = _provider_constraint()
    recorded = re.search(r'constraints\s*=\s*"([^"]+)"', _LOCK.read_text())
    assert recorded, "the lock must record the constraint it resolved"
    assert recorded.group(1).replace(" ", "") == declared.replace(" ", ""), (
        f"the lock records {recorded.group(1)!r} but versions.tf declares {declared!r}; "
        f"re-run terraform init so they agree")
