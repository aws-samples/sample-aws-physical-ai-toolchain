

def test_the_baked_check_only_requires_pins_the_image_actually_writes():
    """C2 (cycle 15): the check demanded LIBERO_COMMIT, which the Dockerfile never stamps, so every
    baked run exited 1.

    Derived from the Dockerfile rather than hardcoded: a check that requires a pin the image cannot
    supply is unsatisfiable by construction, and only comparing the two sides catches that.
    """
    import pathlib
    import re
    root = pathlib.Path(__file__).resolve().parents[1]
    dockerfile = (root / "docker/openvla/Dockerfile").read_text()
    evaluator = (root / "entrypoints/eval/libero/openvla/eval_entry.py").read_text()

    marker = re.search(r'echo "openvla-baked:([^"]*)"', dockerfile)
    assert marker, "the Dockerfile no longer writes the baked marker"
    written = set(re.findall(r"\$\{(\w+)\}", marker.group(1)))
    assert written, "the marker interpolates no commit at all"

    block = evaluator[evaluator.index("missing_pins = "):]
    block = block[:block.index("sys.exit(1)")]
    required = set(re.findall(r'\("(\w+)", \w+\)', block))
    assert required, "no required pins parsed; the check may have changed shape"
    assert required <= written, (
        f"the evaluator requires {sorted(required - written)} in the baked stamp, but the Dockerfile "
        f"only writes {sorted(written)} -- every baked run would refuse to evaluate")


def test_a_reported_pin_is_verified_on_every_path_that_reports_it():
    """Cycle-16 I4: the evaluator recorded LIBERO_COMMIT while accepting an importable LIBERO without
    checking anything. My own C2 rationale claimed the runtime checkout established it -- but that
    checkout is skipped on exactly that path, so on an image with LIBERO preinstalled the provenance
    claim was unverified.

    The invariant: if a constant is REPORTED as provenance, some path must compare it against the
    tree that actually ran. Asserted per branch, because the install branch checks out and the
    already-present branch previously did not.
    """
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1]
    body = (root / "entrypoints/eval/libero/openvla/eval_entry.py").read_text()
    branch = body[body.index('__import__("libero")'):]
    branch = branch[:branch.index("except ImportError:")]
    assert "rev-parse" in branch, (
        "the already-importable branch does not read LIBERO's revision, so the reported "
        "LIBERO_COMMIT is an unverified claim")
    assert "!= LIBERO_COMMIT" in branch, (
        "the revision is read but never compared against the reported pin")
    assert "sys.exit(1)" in branch, (
        "a mismatch does not refuse the evaluation, so the run would publish a result attributed to "
        "benchmark definitions it did not use")
