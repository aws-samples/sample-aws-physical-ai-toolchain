import copy
import io
import json

import pytest

from vla_pipeline.common import training_lineage as lineage
from vla_pipeline.common.digest import weights_digest


def test_named_revision_is_resolved_independently(monkeypatch, tmp_path):
    calls = []

    def metadata(url, timeout):
        calls.append((url, timeout))
        return io.BytesIO(json.dumps({"sha": "a" * 40}).encode())

    monkeypatch.setattr(lineage, "urlopen", metadata)
    lineage.write_training_lineage(
        str(tmp_path), train_suite="arena_gr1_fridge",
        source_parameter="__LIBERO_DEFAULT__", revision_parameter="release",
        repo_id="owner/dataset", requested_revision="release",
        resolved_revision="a" * 40, subdirectory="lerobot",
        content_digest="sha256:" + "d" * 64,
    )
    recorded = json.loads((tmp_path / lineage.FILENAME).read_text())
    manifest = {"dataset_manifest": {
        "source": "hf:owner/dataset", "revision": "sha256:" + "d" * 64}}
    kwargs = dict(
        train_suite="arena_gr1_fridge", source_parameter="__LIBERO_DEFAULT__",
        revision_parameter="release",
        suite_spec={"dataset": {"repo_id": "owner/dataset", "subdir": "lerobot"}},
    )
    result = lineage.verify_training_lineage(manifest, recorded, **kwargs)
    assert {field["status"] for field in result} == {"attested", "verified"}
    revision = next(field for field in result if field["field"] == "dataset_revision")
    assert revision["verification_method"] == "independent_huggingface_resolution"
    assert calls == [("https://huggingface.co/api/datasets/owner/dataset/revision/release", 30)]
    forged = copy.deepcopy(recorded)
    forged["dataset"]["resolved_revision"] = "b" * 40
    with pytest.raises(ValueError, match="dataset_revision mismatch"):
        lineage.verify_training_lineage(manifest, forged, **kwargs)


def test_lineage_file_is_covered_by_the_normative_checkpoint_digest(tmp_path):
    (tmp_path / "model.bin").write_bytes(b"weights")
    path = tmp_path / lineage.FILENAME
    path.write_text('{"train_suite":"arena_gr1_fridge"}')
    before = weights_digest(str(tmp_path))
    path.write_text('{"train_suite":"arena_gr1"}')
    assert weights_digest(str(tmp_path)) != before


def test_exact_commit_needs_no_metadata_request(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("Exact commit unexpectedly made a network request")
    monkeypatch.setattr(lineage, "urlopen", unexpected)
    assert lineage.resolve_hf_revision("owner/dataset", "a" * 40) == "a" * 40


def test_invalid_metadata_fails_closed(monkeypatch):
    monkeypatch.setattr(lineage, "urlopen", lambda *a, **k: io.BytesIO(b'{"sha":"main"}'))
    with pytest.raises(ValueError, match="commit SHA"):
        lineage.resolve_hf_revision("owner/dataset", "main")
