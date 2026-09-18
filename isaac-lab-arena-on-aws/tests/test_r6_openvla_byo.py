"""R6 regression tests: OpenVLA BYO dataset path is fail-closed.

Verifies that:
  1. validate_rlds_dataset accepts a well-formed RLDS directory.
  2. validate_rlds_dataset rejects missing dirs, empty dirs, missing
     dataset_info.json, malformed JSON, zero-episode datasets, and datasets
     with no tfrecord files.
  3. The sourcedir stages rlds_validator.py for every family.
  4. The BYO path in train_entry.py never suppresses validation or digest
     errors (no fallback to sentinel episode_count or fabricated revision).
"""
from __future__ import annotations

import ast
import json
import os

import pytest

from vla_pipeline.common.rlds_validator import (
    RLDSValidationError,
    validate_rlds_dataset,
)
from vla_pipeline.common.sourcedir import stage

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_rlds_dir(tmp_path, episodes=3, version="1.0.0", splits=None):
    import numpy as np
    import tensorflow_datasets as tfds

    ver_dir = tmp_path / version
    ver_dir.mkdir(parents=True)
    identity = tfds.core.DatasetIdentity(
        name="libero_spatial_no_noops", version=tfds.core.Version("1.0.0"),
        data_dir=str(ver_dir), module_name="fixture")
    features = tfds.features.FeaturesDict({
        "steps": tfds.features.Dataset({
            "action": tfds.features.Tensor(shape=(7,), dtype=np.float32),
            "observation": {
                "state": tfds.features.Tensor(shape=(8,), dtype=np.float32),
                "image": tfds.features.Image(shape=(8, 8, 3)),
            },
            "language_instruction": tfds.features.Text(),
        })
    })
    info = tfds.core.DatasetInfo(builder=identity, features=features)
    writer = tfds.core.SequentialWriter(info, max_examples_per_shard=2)
    counts = splits if splits is not None else {"train": episodes}
    writer.initialize_splits(list(counts))
    episode = {"steps": [{
        "action": np.zeros(7, np.float32),
        "observation": {"state": np.zeros(8, np.float32),
                        "image": np.zeros((8, 8, 3), np.uint8)},
        "language_instruction": "move",
    }]}
    writer.add_examples({name: [episode] * count for name, count in counts.items()})
    writer.close_all()
    return tmp_path


# ---------------------------------------------------------------------------
# validate_rlds_dataset: happy path
# ---------------------------------------------------------------------------

class TestValidateRLDS:

    def test_valid_dataset_returns_count(self, tmp_path):
        d = _make_rlds_dir(tmp_path, episodes=42)
        assert validate_rlds_dataset(str(d)) == 42

    def test_multi_shard_sums(self, tmp_path):
        _make_rlds_dir(tmp_path, episodes=5)
        assert len(list((tmp_path / "1.0.0").glob("*.tfrecord-*"))) == 3
        assert validate_rlds_dataset(str(tmp_path)) == 5

    def test_multi_split_sums(self, tmp_path):
        _make_rlds_dir(tmp_path, splits={"train": 3, "test": 2})
        assert validate_rlds_dataset(str(tmp_path)) == 5

    @pytest.mark.parametrize("defect", ["missing", "corrupt", "truncated", "count"])
    def test_invalid_shard_rejected(self, tmp_path, defect):
        import tensorflow as tf

        _make_rlds_dir(tmp_path)
        shard = sorted((tmp_path / "1.0.0").glob("*.tfrecord-*"))[0]
        if defect == "missing":
            shard.rename(shard.with_suffix(".missing"))
            expected = tf.errors.NotFoundError
        elif defect == "count":
            info_path = tmp_path / "1.0.0/dataset_info.json"
            info = json.loads(info_path.read_text())
            info["splits"][0]["shardLengths"][0] = "3"
            info_path.write_text(json.dumps(info))
            expected = RLDSValidationError
        else:
            shard.write_bytes(b"x" if defect == "corrupt" else shard.read_bytes()[:-1])
            expected = (tf.errors.DataLossError, RLDSValidationError)
        with pytest.raises(expected):
            validate_rlds_dataset(str(tmp_path))


# ---------------------------------------------------------------------------
# validate_rlds_dataset: rejection paths (fail-closed)
# ---------------------------------------------------------------------------

class TestValidateRLDSRejects:

    def test_missing_directory(self, tmp_path):
        with pytest.raises(RLDSValidationError, match="does not exist"):
            validate_rlds_dataset(str(tmp_path / "no_such_dir"))

    def test_empty_directory(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(RLDSValidationError, match="empty"):
            validate_rlds_dataset(str(empty))

    def test_no_dataset_info_json(self, tmp_path):
        (tmp_path / "data.tfrecord-00000").write_bytes(b"\x00")
        with pytest.raises(RLDSValidationError, match="no dataset_info.json"):
            validate_rlds_dataset(str(tmp_path))

    def test_malformed_json(self, tmp_path):
        ver_dir = tmp_path / "1.0.0"
        ver_dir.mkdir()
        (ver_dir / "dataset_info.json").write_text("{bad json")
        with pytest.raises(RLDSValidationError, match="cannot parse"):
            validate_rlds_dataset(str(tmp_path))

    def test_no_splits(self, tmp_path):
        ver_dir = tmp_path / "1.0.0"
        ver_dir.mkdir()
        (ver_dir / "dataset_info.json").write_text(json.dumps({"other": 1}))
        with pytest.raises(RLDSValidationError, match="no splits"):
            validate_rlds_dataset(str(tmp_path))

    def test_zero_episodes(self, tmp_path):
        ver_dir = tmp_path / "1.0.0"
        ver_dir.mkdir()
        info = {"splits": [{"name": "train", "shardLengths": ["0"]}]}
        (ver_dir / "dataset_info.json").write_text(json.dumps(info))
        (ver_dir / "data.tfrecord-00000").write_bytes(b"\x00")
        with pytest.raises(RLDSValidationError, match="0 episodes"):
            validate_rlds_dataset(str(tmp_path))

    def test_no_tfrecord_files(self, tmp_path):
        ver_dir = tmp_path / "1.0.0"
        ver_dir.mkdir()
        info = {"splits": [{"name": "train", "shardLengths": ["10"]}]}
        (ver_dir / "dataset_info.json").write_text(json.dumps(info))
        (ver_dir / "not_a_record.bin").write_bytes(b"\x00")
        with pytest.raises(RLDSValidationError, match="no tfrecord"):
            validate_rlds_dataset(str(tmp_path))

    def test_non_integer_shard_length(self, tmp_path):
        ver_dir = tmp_path / "1.0.0"
        ver_dir.mkdir()
        info = {"splits": [{"name": "train", "shardLengths": ["abc"]}]}
        (ver_dir / "dataset_info.json").write_text(json.dumps(info))
        with pytest.raises(RLDSValidationError, match="non-integer"):
            validate_rlds_dataset(str(tmp_path))

    def test_root_not_object(self, tmp_path):
        ver_dir = tmp_path / "1.0.0"
        ver_dir.mkdir()
        (ver_dir / "dataset_info.json").write_text(json.dumps([1, 2, 3]))
        with pytest.raises(RLDSValidationError, match="not an object"):
            validate_rlds_dataset(str(tmp_path))


# ---------------------------------------------------------------------------
# Sourcedir includes rlds_validator.py for all families
# ---------------------------------------------------------------------------

class TestSourcedirIncludesValidator:

    @pytest.mark.parametrize("family", ["openvla", "gr00t", "molmoact2", "dummy"])
    def test_rlds_validator_staged(self, family):
        d = stage(family)
        staged = {f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f))}
        assert "rlds_validator.py" in staged, (
            f"{family}: rlds_validator.py not in staged sourcedir: {staged}")


# ---------------------------------------------------------------------------
# Train entry BYO path: no fallbacks (AST check)
# ---------------------------------------------------------------------------

TRAIN_ENTRY = os.path.join(
    os.path.dirname(__file__), os.pardir,
    "entrypoints", "train", "openvla", "train_entry.py")


@pytest.mark.parametrize("failure_stage", ["validation", "digest"])
def test_byo_failure_stops_entrypoint_before_training(tmp_path, monkeypatch, capsys, failure_stage):
    import importlib.util
    import subprocess
    import sys
    from unittest.mock import Mock

    from vla_pipeline.common import digest, rlds_validator

    monkeypatch.setitem(sys.modules, "digest", digest)
    monkeypatch.setitem(sys.modules, "rlds_validator", rlds_validator)
    monkeypatch.setenv("TRAIN_DATASET_S3URI", "s3://fixture/dataset")
    monkeypatch.setenv("TRAIN_BASE_REV", "a" * 40)
    monkeypatch.setenv("TRAIN_SUITE", "libero_spatial")
    monkeypatch.setenv("TRAIN_MAX_STEPS", "5")
    spec = importlib.util.spec_from_file_location("openvla_failure_test", TRAIN_ENTRY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "WORK", str(tmp_path))
    monkeypatch.setattr(module, "RLDS_DIR", str(tmp_path / "rlds"))
    monkeypatch.setattr(module, "OFT_DIR", str(tmp_path / "oft"))
    monkeypatch.setattr(module, "MODEL_DIR", str(tmp_path / "model"))
    dataset = tmp_path / "rlds" / module.DATASET
    _make_rlds_dir(dataset)
    original_exists = module.os.path.exists
    monkeypatch.setattr(module.os.path, "exists", lambda path: False
                        if path == "/opt/vla/.baked_env" else original_exists(path))
    monkeypatch.setattr(module.os, "chmod", Mock())
    run = Mock(return_value=subprocess.CompletedProcess([], 0))
    monkeypatch.setattr(module.subprocess, "run", run)
    failure = RuntimeError(f"injected {failure_stage} failure")
    validator = Mock(wraps=validate_rlds_dataset)
    hasher = Mock(wraps=digest.weights_digest)
    if failure_stage == "validation":
        validator.side_effect = failure
    else:
        hasher.side_effect = failure
    monkeypatch.setattr(module, "validate_rlds_dataset", validator)
    monkeypatch.setattr(module, "weights_digest", hasher)

    with pytest.raises(RuntimeError) as caught:
        module.main()

    assert caught.value is failure
    validator.assert_called_once_with(str(dataset))
    if failure_stage == "validation":
        hasher.assert_not_called()
    else:
        hasher.assert_called_once_with(str(dataset))
    assert all(call.args[0][0] != "torchrun" for call in run.call_args_list)
    assert "BYO dataset OK" not in capsys.readouterr().out
    assert not (tmp_path / "model" / "checkpoint_manifest.json").exists()
    assert not (tmp_path / "finetune_rank_shim.py").exists()


def test_missing_validator_stops_entrypoint_import(monkeypatch):
    import runpy
    import sys
    from unittest.mock import Mock

    from vla_pipeline.common import digest

    monkeypatch.setitem(sys.modules, "digest", digest)
    monkeypatch.setitem(sys.modules, "rlds_validator", None)
    run = Mock(side_effect=AssertionError("no subprocess may run before dependencies load"))
    monkeypatch.setattr("subprocess.run", run)
    with pytest.raises(ModuleNotFoundError, match="rlds_validator"):
        runpy.run_path(TRAIN_ENTRY, run_name="__main__")
    run.assert_not_called()


class TestTrainEntryNoFallbacks:
    """Static analysis: the BYO path must NOT contain try/except blocks that
    suppress validation or digest failures."""

    @pytest.fixture(autouse=True)
    def _load_source(self):
        with open(os.path.normpath(TRAIN_ENTRY)) as fh:
            self.source = fh.read()
        self.tree = ast.parse(self.source)

    def test_no_import_error_catch_for_validate_dataset(self):
        """The old pattern caught ImportError for validate_dataset and set
        episode_count=-1. That must not exist."""
        assert "ImportError" not in self.source or \
            "validate_dataset" not in self._except_bodies_text(), \
            "BYO path still catches ImportError for validate_dataset"

    def test_no_episode_count_sentinel(self):
        """episode_count = -1 must not appear in the BYO path."""
        assert "episode_count = -1" not in self.source, \
            "BYO path still falls back to episode_count = -1"

    def test_no_fabricated_revision(self):
        """The 'empty-' + ... fabricated revision must not exist."""
        assert 'dataset_revision = "empty-"' not in self.source, \
            "BYO path still fabricates a revision on digest failure"
        assert '"empty-" + ' not in self.source, \
            "BYO path still fabricates a revision on digest failure"

    def test_imports_rlds_validator(self):
        """train_entry.py must import validate_rlds_dataset at module level
        (not inside a try/except)."""
        has_top_import = False
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ImportFrom):
                if node.module == "rlds_validator":
                    names = [a.name for a in node.names]
                    if "validate_rlds_dataset" in names:
                        has_top_import = True
        assert has_top_import, \
            "train_entry.py must import validate_rlds_dataset from rlds_validator"

    def _except_bodies_text(self) -> str:
        """Concatenate source text of all except handler bodies."""
        parts = []
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ExceptHandler):
                for child in ast.walk(node):
                    if isinstance(child, ast.Constant) and isinstance(child.value, str):
                        parts.append(child.value)
                    elif isinstance(child, ast.Name):
                        parts.append(child.id)
        return " ".join(parts)
