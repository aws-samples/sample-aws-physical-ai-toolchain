"""S5: full-suite coverage must come from the task SET, not the selector literal.

`full_suite` was `expected_task_ids_str == "all"`, so an explicit list naming every canonical
task -- which IS full coverage -- was stamped partial, while "all" was trusted without
comparing anything to the canonical set. Both values are pipeline-owned, so this was a
mislabelling rather than a spoofing path, but the label was wrong for a legitimate way of
requesting the whole suite.
"""
from __future__ import annotations

import pathlib

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_ENTRY = _REPO_ROOT / "entrypoints/validate_entry.py"


def _coverage(canonical, selected, selector):
    """Execute the committed derivation with the same names it reads."""
    source = _ENTRY.read_text()
    start = source.index("    coverage_canonical = locals().get")
    end = source.index('        full_suite_basis = "selector_literal"') + len(
        '        full_suite_basis = "selector_literal"')
    block = "\n".join(line[4:] if line.startswith("    ") else line
                      for line in source[start:end].splitlines())
    namespace = {"canonical_ids": canonical, "task_ids": selected,
                 "expected_task_ids_str": selector}
    exec(compile(block, "<coverage>", "exec"), namespace)
    return namespace["full_suite"], namespace["full_suite_basis"]


def test_an_explicit_list_of_every_task_is_full_coverage():
    """The defect: this was stamped partial."""
    full, basis = _coverage([0, 1, 2], [0, 1, 2], "[0,1,2]")
    assert full is True
    assert basis == "task_set_equality"


def test_order_does_not_change_coverage():
    full, _ = _coverage([0, 1, 2], [2, 0, 1], "[2,0,1]")
    assert full is True


def test_a_genuine_subset_is_partial():
    full, basis = _coverage([0, 1, 2], [0, 1], "[0,1]")
    assert full is False
    assert basis == "task_set_equality"


def test_the_all_selector_is_verified_against_the_canonical_set():
    """"all" resolves to the canonical ids upstream, so it compares equal -- but it is now
    COMPARED rather than trusted."""
    full, basis = _coverage([0, 1, 2], [0, 1, 2], "all")
    assert full is True
    assert basis == "task_set_equality"


def test_the_fallback_is_recorded_as_a_literal_not_a_comparison():
    """When the canonical set is unavailable the old behaviour applies, labelled as such."""
    full, basis = _coverage(None, None, "all")
    assert full is True
    assert basis == "selector_literal", (
        "an unverified label must not claim to be a set comparison")
    full, basis = _coverage(None, None, "[0,1]")
    assert full is False and basis == "selector_literal"


def test_the_basis_is_published():
    source = _ENTRY.read_text()
    assert '"full_suite_basis": full_suite_basis,' in source, (
        "a consumer must be able to see HOW coverage was determined")


def test_a_zero_action_run_can_never_be_full_suite():
    """Pre-existing guard that must survive the change."""
    source = _ENTRY.read_text()
    assert source.index("full_suite_basis = \"selector_literal\"") < source.index(
        'if policy_type == "zero_action":')
