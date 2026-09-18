"""Every episode-count default must be at least three.

A single episode is not a measurement: the success rate can only be 0.0 or 1.0. Three
launchers defaulted to 1 and the plumbing graph to 2, so the DEFAULT run produced a rate that
could not express a partial result. A sample run is allowed to be small, but it must still
produce a rate that means something -- reducing counts is the only sanctioned way to make a
run smaller, and three is the floor.
"""
from __future__ import annotations

import pathlib
import re

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
MINIMUM_EPISODES = 3

# Each entry: file, and a regex whose first group is the default episode count.
_DEFAULT_SITES = [
    ("scripts/run_libero.py",
     r'add_argument\("--eval-trials",\s*type=_positive_int,\s*default=(\d+)'),
    ("scripts/run_arena.py",
     r'add_argument\("--eval-trials",\s*type=_positive_int,\s*default=(\d+)'),
    ("scripts/submit_simeval.py",
     r'add_argument\("--eval-trials",\s*default="(\d+)"'),
    ("src/vla_pipeline/runner.py",
     r'ParameterInteger\(name="EvalTrials",\s*default_value=(\d+)\)'),
    ("src/vla_pipeline/pipeline.py",
     r'ParameterInteger\(name="EvalTrials",\s*default_value=(\d+)\)'),
]


@pytest.mark.parametrize("relative,pattern", _DEFAULT_SITES)
def test_the_default_episode_count_is_at_least_three(relative, pattern):
    source = (_REPO_ROOT / relative).read_text()
    match = re.search(pattern, source)
    assert match, f"could not find the episode-count default in {relative}"
    value = int(match.group(1))
    assert value >= MINIMUM_EPISODES, (
        f"{relative} defaults to {value} episodes. One episode makes the rate either 0.0 or "
        f"1.0, so it cannot express a partial result; the floor is {MINIMUM_EPISODES}.")


def test_every_episode_default_site_is_covered():
    """A new launcher must be added here rather than silently defaulting to one."""
    found = set()
    for path in sorted(_REPO_ROOT.glob("scripts/*.py")):
        if "--eval-trials" in path.read_text():
            found.add(f"scripts/{path.name}")
    declared = {relative for relative, _ in _DEFAULT_SITES}
    assert found <= declared, (
        f"these launchers accept --eval-trials but are not checked here: {found - declared}")
