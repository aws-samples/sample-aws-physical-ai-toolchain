"""A checkpoint's embodiment must be one the target simulator can serve.

R7, from a real run. A gr00t checkpoint trained for Isaac Lab Arena declares
`embodiment_tag: GR1`. Handed to the LIBERO evaluator it passed the model_family check -- it IS a
gr00t checkpoint -- and then died inside upstream GR00T code:

    ValueError: Unknown embodiment tag: 'GR1'
      Base model tags (work with nvidia/GR00T-N1.7-3B)
        OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT, REAL_G1, XDOF, ...
    [gr00t-eval] FATAL: server exited early rc=1

after the image had pulled and the 8 GB checkpoint had downloaded. FAMILY IS NOT SIMULATOR, and the
family check I added earlier this session does not cover this.

The information was already in the repo: LIBERO suites declare `embodiment_tag: null` (the GR00T
LIBERO path uses the base model's own tags) while Arena suites declare GR1 or new_embodiment. So the
mismatch is decidable at the same seam as the family check, instead of in a third-party stack trace.
"""
import pathlib

import pytest

import yaml

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_EVALUATOR = _ROOT / "entrypoints/eval/libero/gr00t/eval_entry.py"


def _arena_declared_tags() -> set:
    """Every embodiment tag the ARENA suites declare, read from the manifests themselves."""
    tags = set()
    for path in sorted((_ROOT / "config/suites").glob("*.yaml")):
        data = yaml.safe_load(path.read_text()) or {}
        if data.get("simulator") != "isaac_arena":
            continue
        for value in _walk_values(data):
            if isinstance(value, str) and value:
                tags.add(value)
    return tags


def _walk_values(node, key_filter=("embodiment_tag", "embodiment")):
    if isinstance(node, dict):
        for k, v in node.items():
            if k in key_filter and isinstance(v, str):
                yield v
            else:
                yield from _walk_values(v, key_filter)
    elif isinstance(node, list):
        for item in node:
            yield from _walk_values(item, key_filter)


def test_the_libero_evaluator_rejects_every_arena_embodiment_tag():
    """Derived from the manifests, so a NEW Arena suite with a new tag is covered automatically.

    A hardcoded list in the evaluator plus a hardcoded list here would be two declarations of one
    fact -- the exact shape of the defects this work has been closing -- so this test reads the
    manifests and requires the evaluator to cover what they declare.
    """
    declared = _arena_declared_tags()
    assert declared, "no Arena suite declares an embodiment tag; this test would be vacuous"

    # The tag set ONLY -- not the whole file. My first version searched the entire source, so the
    # explanatory comment above the check (which quotes "GR1") satisfied the assertion and removing
    # GR1 from the rejected set still passed. An assertion matched by its own prose is not a check.
    source = _EVALUATOR.read_text()
    i = source.index("_ARENA_ONLY_TAGS = {")
    tag_set = source[i:source.index("}", i)]
    missing = sorted(t for t in declared if f'"{t}"' not in tag_set)
    assert not missing, (
        f"the LIBERO evaluator does not reject Arena embodiment tag(s) {missing}, which "
        f"config/suites declares. A checkpoint carrying one passes the family check and then fails "
        f"inside upstream GR00T code after the image and checkpoint have downloaded")


def test_libero_suites_declare_no_embodiment_tag():
    """The other half of the claim: if a LIBERO suite declared GR1, rejecting it would be wrong."""
    for path in sorted((_ROOT / "config/suites").glob("libero*.yaml")):
        data = yaml.safe_load(path.read_text()) or {}
        for value in _walk_values(data):
            assert not value, (
                f"{path.name} declares embodiment {value!r}; the LIBERO GR00T path uses the base "
                f"model's own tags, so a declared tag here would make the new rejection incorrect")


def test_the_rejection_is_fatal_rather_than_a_warning():
    source = _EVALUATOR.read_text()
    i = source.index("_ARENA_ONLY_TAGS")
    window = source[i:i + 1600]
    assert "sys.exit(1)" in window, (
        "the embodiment mismatch is detected but does not exit, so the run continues into the GR00T "
        "server that cannot serve it")
