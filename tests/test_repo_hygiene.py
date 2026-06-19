"""Repo-hygiene guards that would otherwise rot silently:
- every container buildspec is valid YAML
- no merge-conflict markers anywhere
- no hardcoded AWS account id in runnable code (status docs are exempt)
"""
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
HARDCODED_ACCOUNT = "802782083985"  # the original author's account; must not be in code


def _code_files():
    exts = {".py", ".sh", ".ts", ".yml", ".yaml"}
    # Skip build artifacts and the tests dir itself (this file names the account
    # string as its detection target, which is not a hardcoded-usage offense).
    skip = {"node_modules", ".git", "cdk.out", "tests"}
    for p in REPO.rglob("*"):
        if p.is_file() and p.suffix in exts and not any(s in p.parts for s in skip):
            yield p


def test_buildspecs_are_valid_yaml():
    yaml = pytest.importorskip("yaml")
    specs = list((REPO / "containers").rglob("buildspec.yml"))
    assert specs, "no buildspecs found"
    for s in specs:
        yaml.safe_load(s.read_text())  # raises on invalid YAML


def test_no_conflict_markers():
    bad = []
    for p in _code_files():
        for i, line in enumerate(p.read_text(errors="ignore").splitlines(), 1):
            if line.startswith("<<<<<<< ") or line.startswith(">>>>>>> "):
                bad.append(f"{p}:{i}")
    assert not bad, f"conflict markers found: {bad}"


def test_no_hardcoded_account_in_runnable_code():
    # Status/history docs may mention the old account factually; code must not.
    offenders = []
    for p in _code_files():
        if HARDCODED_ACCOUNT in p.read_text(errors="ignore"):
            offenders.append(str(p.relative_to(REPO)))
    assert not offenders, f"hardcoded account {HARDCODED_ACCOUNT} in: {offenders}"
