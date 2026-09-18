"""N1.7 loads Cosmos through a concrete class, bypassing AutoModel's hook."""
import importlib.util
import sys
import types
from pathlib import Path

import pytest

REPO = "nvidia/Cosmos-Reason2-2B"
COMMIT = "b" * 40


def wrapper():
    path = (Path(__file__).resolve().parents[1]
            / "entrypoints/eval/isaac_arena/gr00t/gr00t_seeded_server.py")
    spec = importlib.util.spec_from_file_location("direct_backbone_audit_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def loaders(monkeypatch, *, commit=COMMIT, diagnostics=None):
    clean = {"missing_keys": [], "unexpected_keys": [],
             "mismatched_keys": [], "error_msgs": []}
    info = dict(clean) if diagnostics is None else diagnostics
    backbone = types.SimpleNamespace(config=types.SimpleNamespace(_commit_hash=commit))
    checkpoint = types.SimpleNamespace(config=types.SimpleNamespace(_commit_hash=None))
    calls = []

    class Qwen3VLForConditionalGeneration:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.append((cls.__name__, path, kwargs))
            return (backbone, info) if kwargs.get("output_loading_info") else backbone

    class AutoModel:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            # The pinned GR00T constructor invokes the concrete loader directly.
            checkpoint.backbone = Qwen3VLForConditionalGeneration.from_pretrained(REPO)
            return (checkpoint, clean) if kwargs.get("output_loading_info") else checkpoint

    fake = types.ModuleType("transformers")
    fake.AutoModel = AutoModel
    fake.Qwen3VLForConditionalGeneration = Qwen3VLForConditionalGeneration
    monkeypatch.setitem(sys.modules, "transformers", fake)
    return fake, checkpoint, backbone, calls


def test_nested_direct_load_is_recorded_and_bound_to_the_report(monkeypatch, tmp_path):
    module = wrapper()
    fake, checkpoint, backbone, calls = loaders(monkeypatch)
    monkeypatch.setattr(module, "EVIDENCE_PATH", str(tmp_path / "audit.json"))
    evidence = {"processor_validated": str(tmp_path)}
    module._install_strict_load_audit(evidence)

    assert fake.AutoModel.from_pretrained(str(tmp_path)) is checkpoint
    assert checkpoint.backbone is backbone
    assert calls[0][2].get("output_loading_info") is True
    records = evidence["strict_load_audits"]
    assert [(r["loader"], r["path"], r["consumed_commit"]) for r in records] == [
        ("Qwen3VLForConditionalGeneration", REPO, COMMIT),
        ("AutoModel", str(tmp_path), None),
    ]
    identity = module.consumed_backbone_identity(
        {"server_seeding_evidence": evidence}, str(tmp_path), REPO)
    assert identity == {"repo_id": REPO, "resolved_commit": COMMIT}


@pytest.mark.parametrize("field", [
    "missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs",
])
def test_nested_direct_load_still_rejects_bad_weights(monkeypatch, field):
    module = wrapper()
    info = {"missing_keys": [], "unexpected_keys": [],
            "mismatched_keys": [], "error_msgs": []}
    info[field] = ["bad.weight"]
    fake, _, _, _ = loaders(monkeypatch, diagnostics=info)
    evidence = {}
    module._install_strict_load_audit(evidence)
    with pytest.raises(SystemExit):
        fake.AutoModel.from_pretrained("/checkpoint")
    assert not evidence.get("strict_load_audits")


@pytest.mark.parametrize("commit", [None, "main", "a" * 39])
def test_direct_hub_load_cannot_publish_an_unknown_revision(monkeypatch, commit):
    module = wrapper()
    fake, _, _, _ = loaders(monkeypatch, commit=commit)
    evidence = {}
    module._install_strict_load_audit(evidence)
    with pytest.raises(SystemExit):
        fake.AutoModel.from_pretrained("/checkpoint")
    assert not evidence.get("strict_load_audits")


def test_direct_loader_preserves_explicit_diagnostics_return(monkeypatch, tmp_path):
    module = wrapper()
    fake, _, backbone, _ = loaders(monkeypatch)
    monkeypatch.setattr(module, "EVIDENCE_PATH", str(tmp_path / "audit.json"))
    module._install_strict_load_audit({})
    model, info = fake.Qwen3VLForConditionalGeneration.from_pretrained(
        REPO, output_loading_info=True)
    assert model is backbone
    assert info == {"missing_keys": [], "unexpected_keys": [],
                    "mismatched_keys": [], "error_msgs": []}
