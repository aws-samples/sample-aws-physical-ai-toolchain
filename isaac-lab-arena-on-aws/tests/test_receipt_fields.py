

def test_the_receipt_carries_every_validated_field_it_checks():
    """Cycle-15 I1: provenance was validated and then dropped from the attestation.

    Source-level rather than behavioural, because building a receipt needs the full checkpoint and S3
    surface mocked. The property asserted is the one that failed: a field the raw report is REQUIRED
    to carry, and which validate_entry checks, must also appear in the receipt -- validating a field
    and then discarding it means the check runs and nothing it established survives.
    """
    import pathlib
    import re
    root = pathlib.Path(__file__).resolve().parents[1]
    entry = (root / "entrypoints/validate_entry.py").read_text()
    receipt = entry[entry.index("    validated = {"):]
    receipt = receipt[:receipt.index("\n    }")]
    for field in ("provenance", "aux_metrics", "seed_scope", "effective_eval_config"):
        assert f'"{field}"' in receipt, (
            f"{field} is validated on the raw report but absent from the immutable receipt, so two "
            f"runs differing only in {field} register indistinguishably")


def test_no_optional_report_key_is_silently_dropped():
    """Derived from the validator's own key sets rather than a hand-written list, so a new optional
    key added there cannot be forgotten here."""
    import pathlib
    import re
    root = pathlib.Path(__file__).resolve().parents[1]
    validator = (root / "src/vla_pipeline/common/validator.py").read_text()
    entry = (root / "entrypoints/validate_entry.py").read_text()
    match = re.search(r"TOP_OPTIONAL_KEYS = \{([^}]*)\}", validator)
    assert match, "TOP_OPTIONAL_KEYS no longer parses"
    optional = set(re.findall(r'"(\w+)"', match.group(1)))
    # The source-identity variants are REPORT-level fields describing what the evaluator read; the
    # receipt records model_artifact_identity instead, which validate_entry verifies itself rather
    # than copying from the producer. Excluded by name so a future optional key is still caught.
    optional -= {"source_archive", "source_snapshot"}
    assert optional, "no optional keys parsed"
    receipt = entry[entry.index("    validated = {"):]
    receipt = receipt[:receipt.index("\n    }")]
    missing = [k for k in sorted(optional) if f'"{k}"' not in receipt]
    assert not missing, (
        f"the validator accepts {missing} on the raw report but the receipt drops them, so that "
        f"evidence does not reach the model package")


def test_the_evaluator_sourcedir_reaches_the_receipt_from_the_pipeline():
    """Cycle-15 I1, second half: evaluator_image_uri does not identify the evaluator.

    Ties the two ends together. A receipt field reading an env var the graph never sets would record
    null forever and look like absent evidence rather than a plumbing bug -- the same shape as the
    max_run defect, where a declared value governed nothing.
    """
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1]
    pipeline = (root / "src/vla_pipeline/pipeline.py").read_text()
    entry = (root / "entrypoints/validate_entry.py").read_text()
    assert '"VLA_EVAL_SOURCEDIR_URI": params["eval_source_dir"].to_string()' in pipeline, (
        "the Validate step does not pass the eval sourcedir URI, so the receipt field records null")
    receipt = entry[entry.index("    validated = {"):]
    receipt = receipt[:receipt.index("\n    }")]
    assert '"evaluator_sourcedir_uri"' in receipt, (
        "the receipt does not record the evaluator sourcedir, so the attestation names only the image")
    assert 'environ.get("VLA_EVAL_SOURCEDIR_URI")' in entry, (
        "the receipt value must come from the PIPELINE's environment, not from the report -- a "
        "producer must not name its own provenance")


def test_validate_uses_the_strict_json_parser_and_serialiser():
    """Cycle-16 I6: Validate read metrics.json with ordinary json.load, which ACCEPTS NaN and
    Infinity -- both illegal in standard JSON. The shared module already ships parse_report() to
    reject them, and Validate, the one step whose output is immutable evidence, was the only place
    not using it. A report carrying aux_metrics={"diagnostic": NaN} validated cleanly and the receipt
    then serialised a token no conforming parser can read.
    """
    import pathlib
    import re
    root = pathlib.Path(__file__).resolve().parents[1]
    entry = (root / "entrypoints/validate_entry.py").read_text()
    assert "metrics = parse_report(" in entry, (
        "Validate does not use the strict parser, so NaN in the report reaches the receipt")
    assert not re.search(r"metrics = json\.load\(", entry), "the permissive read is still present"
    # Every serialisation of sealed evidence must refuse to EMIT a non-finite token.
    for match in re.finditer(r"json\.dump(?:s)?\((attestation|validated)[^)]*\)", entry):
        assert "allow_nan=False" in match.group(0), (
            f"{match.group(0)[:60]}: can emit NaN, producing evidence a conforming parser cannot read")


def test_the_strict_parser_is_in_scope_where_it_is_called():
    """A function-local import binds when it EXECUTES, so an import below the call leaves the name
    unbound. cycle-15 C1 was this across functions; the first attempt at this fix repeated it within
    one function and broke 62 tests."""
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1]
    lines = (root / "entrypoints/validate_entry.py").read_text().splitlines()
    imported = next(i for i, l in enumerate(lines) if "from validator import parse_report" in l)
    called = next(i for i, l in enumerate(lines) if "parse_report(f.read())" in l)
    assert imported < called, (
        f"parse_report is imported at line {imported + 1} but called at {called + 1}, so it is "
        f"unbound when the call runs")
