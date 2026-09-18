"""Important 6: a GR00T report must state which evaluator version actually ran.

`effective_eval_config` was required only for Arena, and the version comparison ran only when
the field was present and non-empty. So a LIBERO GR00T report could omit it and pass under any
EXPECTED_FAMILY_VERSION -- a probe accepted a correctly digested report carrying N1.6 base and
recipe provenance under n17. The LIBERO suites' embodiment tags are null, so the embodiment
check does not independently distinguish N1.6 from N1.7 either.
"""
from __future__ import annotations

import pathlib
import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_VALIDATE = _REPO_ROOT / "entrypoints/validate_entry.py"
_LIBERO_GR00T = _REPO_ROOT / "entrypoints/eval/libero/gr00t/eval_entry.py"


def test_the_libero_producer_states_its_evaluator_version():
    """The check is only enforceable if the producer supplies the field."""
    source = _LIBERO_GR00T.read_text()
    assert '"effective_eval_config": {' in source
    assert '"gr00t_version": _gr00t_version' in source, (
        "the version must be the one that RAN, not a literal")


def test_the_producer_refuses_a_version_it_cannot_run():
    """So the recorded version is the version that ran rather than the one requested."""
    source = _LIBERO_GR00T.read_text()
    assert 'if _gr00t_version != "n17":' in source
    assert source.index('if _gr00t_version != "n17":') < source.index(
        '"gr00t_version": _gr00t_version')


def test_an_absent_effective_config_is_refused_for_gr00t():
    source = _VALIDATE.read_text()
    assert "no effective_eval_config, so the version that actually ran is" in source
    assert 'if _efv and metrics.get("model_family") == "gr00t":' in source


def test_an_empty_version_is_refused_rather_than_treated_as_agreement():
    source = _VALIDATE.read_text()
    assert "An absent version cannot be shown to match." in source


def test_the_requirement_is_scoped_to_gr00t():
    """gr00t_version is GR00T-specific and EXPECTED_FAMILY_VERSION is set for every family.

    Demanding it from a MolmoAct2 report would reject valid evidence.
    """
    source = _VALIDATE.read_text()
    guard = source.index('if _efv and metrics.get("model_family") == "gr00t":')
    # The strict block must sit inside that guard, not before it.
    assert source.index("no effective_eval_config, so the version") > guard


def test_a_mismatched_version_is_refused_unconditionally():
    """The pre-existing conditional comparison is retained AND made mandatory."""
    source = _VALIDATE.read_text()
    assert "differs from the requested" in source


# --- Cycle-6 I6: the checkpoint chose the evaluation protocol ----------------------------

def test_the_suites_declare_their_evaluation_protocol():
    """The expectation must be PIPELINE-owned, so it lives in the suite manifest."""
    yaml = __import__("yaml")
    suites = sorted((_REPO_ROOT / "config/suites").glob("libero*.yaml"))
    assert suites, "no LIBERO suite manifests found"
    for path in suites:
        spec = yaml.safe_load(path.read_text())
        protocol = spec.get("evaluation_protocol")
        assert isinstance(protocol, dict), f"{path.name} declares no evaluation_protocol"
        for key in ("n_action_steps", "max_episode_steps"):
            assert isinstance(protocol.get(key), int) and protocol[key] > 0, (
                f"{path.name} evaluation_protocol.{key} must be a positive integer")


def test_the_registry_accepts_the_protocol_key():
    """An undeclared key is rejected by the suite schema, which is an explicit allowlist."""
    source = (_REPO_ROOT / "src/vla_pipeline/registry.py").read_text()
    assert '"evaluation_protocol"' in source


def test_the_producer_records_the_protocol_it_consumed():
    """C3 (cycle 12): this test PINNED the defect it was meant to prevent.

    It asserted the report names the module GLOBALS. But the command passes the LOCALS, which
    pipeline mode replaces with checkpoint-provided values -- so a checkpoint declaring 4 steps
    and a 100-step horizon executed 4/100 while this test demanded the report say 8/720, and
    Validate then compared that report against the suite expectation and passed it.

    Recording "the protocol it consumed" means the values the command actually used, so the
    assertion is on the DATA FLOW rather than on a spelling.
    """
    import re
    source = (_REPO_ROOT / "entrypoints/eval/libero/gr00t/eval_entry.py").read_text()
    code = "\n".join(l for l in source.splitlines() if not l.strip().startswith("#"))
    passed = re.findall(r'"--(?:n-action-steps|max-episode-steps)",\s*(\w+)', code)
    assert sorted(passed) == ["max_episode_steps", "n_action_steps"], passed
    reported = re.findall(
        r'"(?:n_action_steps|max_episode_steps)":\s*(?:int\()?(\w+)', code)
    assert reported, "no protocol report site found"
    assert not [r for r in reported if r.isupper()], (
        f"report names module globals rather than the executed values: {reported}")


def test_validate_compares_the_consumed_protocol_to_the_suite():
    """Recording alone is not a check; the comparison is what makes it load-bearing."""
    source = (_REPO_ROOT / "entrypoints/validate_entry.py").read_text()
    assert 'get("evaluation_protocol")' in source
    assert "evaluation protocol mismatch" in source
    # And an unreported setting must fail rather than be treated as agreement.
    assert "An unreported " in source


def test_the_comparison_is_not_scoped_to_one_family():
    """The protocol belongs to the SUITE, so it applies whatever family produced the report."""
    source = (_REPO_ROOT / "entrypoints/validate_entry.py").read_text()
    guard = source.index('if _efv and metrics.get("model_family") == "gr00t":')
    protocol = source.index('_protocol = (suite_specs.get(expected_suite) or {})')
    guard_indent = len(source[:guard].split("\n")[-1])
    protocol_indent = len(source[:protocol].split("\n")[-1])
    assert protocol_indent <= guard_indent, (
        "the protocol comparison must be a sibling of the gr00t-only guard, not inside it")


# --- C2: the protocol must reach the SERIALIZED table, not just the manifest ---------------

def test_the_serialized_suite_table_carries_the_protocol():
    """This is the test whose absence let the check never run.

    The manifests declared evaluation_protocol and Validate compared it, and my tests asserted
    on both ends -- but resolution discarded the field in between, so the serialized table
    carried nothing and Validate's comparison skipped on absence for every suite. Asserting on
    the manifest proves nothing about what the pipeline actually embeds.
    """
    from vla_pipeline import registry
    table = registry.suites_json()
    libero = {n: s for n, s in table.items() if s["simulator"] == "libero"}
    assert libero, "no libero suites resolved"
    for name, spec in sorted(libero.items()):
        protocol = spec.get("evaluation_protocol")
        assert protocol, f"{name} serializes no evaluation_protocol"
        assert protocol["n_action_steps"] == 8
        assert protocol["max_episode_steps"] == 720


def test_the_protocol_is_scoped_to_the_simulator_it_means_something_for():
    """n_action_steps and max_episode_steps are LIBERO parameters.

    Arena takes its horizon from the Arena task configuration, so a protocol block there would
    look like an enforced constraint while no evaluator reads it.
    """
    from vla_pipeline import registry
    table = registry.suites_json()
    for name, spec in sorted(table.items()):
        if spec["simulator"] != "libero":
            assert spec.get("evaluation_protocol") is None, (
                f"{name} is {spec['simulator']} and must not declare a libero protocol")


def test_a_libero_manifest_without_a_protocol_is_rejected(tmp_path, monkeypatch):
    import yaml
    from vla_pipeline import registry
    suite_dir = tmp_path / "suites"
    suite_dir.mkdir()
    (suite_dir / "t_noproto.yaml").write_text(yaml.safe_dump({
        "name": "t_noproto", "simulator": "libero", "status": "supported",
        "canonical_task_ids": [0],
        "dataset": {"repo_id": "org/ds", "subdir": "", "revision": None,
                    "copy_libero_modality": True},
    }))
    monkeypatch.setattr(registry, "_CONFIG_DIR", str(tmp_path))
    registry.resolve_suite.cache_clear()
    with pytest.raises(registry.RegistryError, match="required for libero suites"):
        registry.resolve_suite("t_noproto")
    registry.resolve_suite.cache_clear()


def test_an_arena_manifest_with_a_protocol_is_rejected(tmp_path, monkeypatch):
    import shutil
    import yaml
    from vla_pipeline import registry
    suite_dir = tmp_path / "suites"
    suite_dir.mkdir()
    # Resolution also reads the simulator manifest, so the real ones come along; otherwise the
    # test would pass on "manifest not found" rather than on the protocol rejection.
    shutil.copytree(pathlib.Path(registry._CONFIG_DIR) / "simulators",
                    tmp_path / "simulators")
    (suite_dir / "t_proto.yaml").write_text(yaml.safe_dump({
        "name": "t_proto", "simulator": "isaac_arena", "status": "experimental",
        "canonical_task_ids": [0],
        "evaluation_protocol": {"n_action_steps": 8, "max_episode_steps": 720},
        # An arena suite must carry its runtime block, which is validated first; without it the
        # test would pass on the missing block rather than on the protocol rejection.
        "arena": {"task": "gr1_open_microwave", "embodiment": "gr1_joint",
                  "object": None, "policy_config": None},
    }))
    monkeypatch.setattr(registry, "_CONFIG_DIR", str(tmp_path))
    registry.resolve_suite.cache_clear()
    with pytest.raises(registry.RegistryError, match="LIBERO protocol parameters"):
        registry.resolve_suite("t_proto")
    registry.resolve_suite.cache_clear()


def test_validate_fails_closed_when_a_libero_suite_carries_no_protocol():
    """Absence must be decided against the simulator, not read as 'nothing to check'."""
    source = (_VALIDATE).read_text()
    # C3 (cycle 10): scoped to gr00t. The declared 8 steps / 720 horizon is a GR00T protocol --
    # pinned OpenVLA-OFT uses per-suite horizons 220/280/300/520 and MolmoAct2 trains with ten
    # action steps -- so requiring it of every libero family rejected two working producers, and
    # adding the fields with these constants would have published a wrong protocol instead.
    assert ('if _simulator == "libero" and _family == "gr00t" and not _protocol'
            in source)
    assert "carries no evaluation_protocol" in source
    # And the fail must come BEFORE the comparison it guards.
    assert source.index("carries no evaluation_protocol") < source.index(
        "declares an evaluation protocol but the")


@pytest.mark.parametrize("bad", [
    # bool is an int subclass, so True would otherwise become a step count of 1.
    True,
    0,
    -8,
    "8",
    8.5,
    None,
])
def test_a_malformed_protocol_value_is_rejected(tmp_path, monkeypatch, bad):
    """Mutation found this validation unprotected: disabling it left the suite green.

    The protocol is what Validate compares the consumed configuration against, so a value that
    is silently accepted here becomes an expectation nothing can meaningfully match.
    """
    import shutil
    import yaml
    from vla_pipeline import registry
    suite_dir = tmp_path / "suites"
    suite_dir.mkdir()
    shutil.copytree(pathlib.Path(registry._CONFIG_DIR) / "simulators", tmp_path / "simulators")
    (suite_dir / "t_badproto.yaml").write_text(yaml.safe_dump({
        "name": "t_badproto", "simulator": "libero", "status": "supported",
        "canonical_task_ids": [0],
        "dataset": {"repo_id": "org/ds", "subdir": "", "revision": None,
                    "copy_libero_modality": True},
        "evaluation_protocol": {"n_action_steps": bad, "max_episode_steps": 720},
    }))
    monkeypatch.setattr(registry, "_CONFIG_DIR", str(tmp_path))
    registry.resolve_suite.cache_clear()
    try:
        with pytest.raises(registry.RegistryError, match="positive integer|is missing"):
            registry.resolve_suite("t_badproto")
    finally:
        registry.resolve_suite.cache_clear()


def test_the_openvla_baked_stamp_is_verified_against_the_pinned_commits():
    """I7: marker EXISTENCE was sufficient and its contents were only logged.

    The report records fixed OFT_COMMIT and LIBERO_COMMIT regardless, so an older or custom image
    could execute while the report attributed the result to the pinned implementation. File
    existence is not provenance.
    """
    source = (_REPO_ROOT / "entrypoints/eval/libero/openvla/eval_entry.py").read_text()
    code = "\n".join(l for l in source.splitlines() if not l.strip().startswith("#"))
    block = code[code.index("baked_marker"):]
    block = block[:block.index("Override paths") if "Override paths" in block else 4000]
    assert "OFT_COMMIT not in marker" in block or "commit not in marker" in block, (
        "the baked stamp is not compared against the pinned commits")
    assert "sys.exit(1)" in block, "a mismatched stamp must be fatal, not a warning"
    # Both pins, not one: attributing to a pinned pair means verifying both.
    assert "OFT_COMMIT" in block and "LIBERO_COMMIT" in block


def test_the_receipt_names_the_evaluator_image_that_produced_the_evidence():
    """I8: validate_entry had ZERO references to any image identity.

    The immutable attestation certified a score without recording what code produced it. An
    immutable tag prevents overwriting a tag; it does not say which image ran, so a current
    launcher and validator could evaluate with an older evaluator image and the receipt would look
    identical. A score whose attestation cannot name its code is not attributable, which is the
    property this component exists to establish.
    """
    validate = (_REPO_ROOT / "entrypoints/validate_entry.py").read_text()
    start = validate.index("All checks passed: emit validated_metrics.json")
    receipt = validate[start:validate.index("json.dump", start)]
    assert '"evaluator_image_uri"' in receipt
    # Read from the PIPELINE, not the report: the producer must not name its own provenance.
    # Was: assert the literal '_require_expectation("EVALUATOR_IMAGE_URI")' appeared in the source.
    # That assertion described one spelling of the requirement rather than the requirement, so it
    # broke the moment the check became digest-aware while the property it cared about still held.
    # Bind the property instead: the receipt must carry the reference AND the resolved digest.
    assert '"evaluator_image_uri"' in receipt, (
        "the attestation does not record which evaluator image produced the evidence")
    assert '"evaluator_image_digest"' in receipt, (
        "the attestation records a reference but not the DIGEST, so it names a pointer rather than "
        "the bytes that ran -- a stale tag would produce an identical-looking receipt")
    assert 'metrics.get("evaluator_image_uri")' not in receipt, (
        "the image identity must not come from the report the producer wrote")
    pipeline = (_REPO_ROOT / "src/vla_pipeline/pipeline.py").read_text()
    assert '"EVALUATOR_IMAGE_URI": params["eval_image"].to_string()' in pipeline
