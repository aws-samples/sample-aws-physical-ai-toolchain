"""RLDS dataset validator for BYO (bring-your-own) datasets.

Uses TensorFlow and TensorFlow Datasets for record inspection. Shipped in the
SageMaker sourcedir alongside digest.py so train_entry.py can import it at runtime. Validates the structural
requirements of an RLDS dataset directory BEFORE training begins --
fail-closed, never returns a sentinel or suppresses errors.
"""
from __future__ import annotations

import json
import os


class RLDSValidationError(ValueError):
    pass


def validate_rlds_dataset(dataset_dir: str) -> int:
    """Validate an RLDS dataset directory and return the episode count.

    Checks:
      1. Directory exists and is non-empty.
      2. A dataset_info.json is locatable (standard RLDS layout).
      3. The JSON is parseable and contains a splits array with shardLengths.
      4. Total episode count across all splits is >= 1.
      5. At least one tfrecord file exists (the actual data).

    Returns the total episode count. Raises RLDSValidationError on any failure.
    """
    if not os.path.isdir(dataset_dir):
        raise RLDSValidationError(
            f"dataset directory does not exist: {dataset_dir}")

    if not any(os.scandir(dataset_dir)):
        raise RLDSValidationError(
            f"dataset directory is empty: {dataset_dir}")

    info_path = os.path.join(dataset_dir, "1.0.0", "dataset_info.json")
    if not os.path.isfile(info_path):
        candidates = []
        for r, _d, fs in os.walk(dataset_dir, onerror=_walk_error):
            if "dataset_info.json" in fs:
                candidates.append(os.path.join(r, "dataset_info.json"))
        if not candidates:
            raise RLDSValidationError(
                f"no dataset_info.json found under {dataset_dir}")
        info_path = candidates[0]

    try:
        with open(info_path) as fh:
            info = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        raise RLDSValidationError(
            f"cannot parse {info_path}: {exc}") from exc

    if not isinstance(info, dict):
        raise RLDSValidationError(
            f"dataset_info.json root is not an object: {info_path}")

    splits = info.get("splits")
    if not isinstance(splits, list) or not splits:
        raise RLDSValidationError(
            f"dataset_info.json has no splits array: {info_path}")

    total = 0
    for split in splits:
        if not isinstance(split, dict):
            raise RLDSValidationError(
                f"split entry is not an object in {info_path}")
        lengths = split.get("shardLengths", [])
        if not isinstance(lengths, list):
            raise RLDSValidationError(
                f"shardLengths is not a list in {info_path}")
        for x in lengths:
            try:
                total += int(x)
            except (ValueError, TypeError) as exc:
                raise RLDSValidationError(
                    f"non-integer shardLength {x!r} in {info_path}") from exc

    if total < 1:
        raise RLDSValidationError(
            f"dataset has 0 episodes (shardLengths sum=0) in {info_path}")

    has_tfrecord = False
    for _r, _d, fs in os.walk(dataset_dir, onerror=_walk_error):
        if any(f.endswith(".tfrecord") or ".tfrecord-" in f for f in fs):
            has_tfrecord = True
            break
    if not has_tfrecord:
        raise RLDSValidationError(
            f"no tfrecord files found under {dataset_dir} -- "
            f"dataset_info.json is present but the actual data is missing")

    return _validate_records(os.path.dirname(info_path), total)


def _validate_records(version_dir: str, expected_count: int) -> int:
    import tensorflow as tf
    import tensorflow_datasets as tfds

    builder = tfds.builder_from_directory(version_dir)
    if "train" not in builder.info.splits or builder.info.splits["train"].num_examples < 1:
        raise RLDSValidationError("dataset must contain a nonempty train split")
    count = 0
    for split in builder.info.splits.values():
        for instruction in split.file_instructions:
            records = tf.data.TFRecordDataset(instruction.filename)
            shard_count = sum(1 for _ in records)
            if shard_count != instruction.examples_in_shard:
                raise RLDSValidationError(
                    f"shard episode count differs: {instruction.filename}: "
                    f"expected {instruction.examples_in_shard}, decoded {shard_count}")
        for episode in builder.as_dataset(split=split.name, shuffle_files=False):
            steps = episode["steps"]
            step_count = 0
            for step in steps:
                tf.debugging.assert_shapes([
                    (step["action"], (7,)),
                    (step["observation"]["state"], (8,)),
                    (step["observation"]["image"], (None, None, 3)),
                ])
                tf.debugging.assert_all_finite(step["action"], "nonfinite action")
                tf.debugging.assert_all_finite(step["observation"]["state"], "nonfinite state")
                if step["language_instruction"].dtype != tf.string:
                    raise RLDSValidationError("language_instruction must be a string")
                step_count += 1
            if not step_count:
                raise RLDSValidationError("episode has no steps")
            count += 1
    if count != expected_count:
        raise RLDSValidationError(
            f"dataset episode count differs: expected {expected_count}, decoded {count}")
    return count


def _walk_error(exc: OSError) -> None:
    raise RLDSValidationError(
        f"cannot traverse dataset directory: {exc}") from exc
