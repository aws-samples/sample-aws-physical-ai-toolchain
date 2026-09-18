"""GR00T training identity, carried in a file covered by the checkpoint digest."""
from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

FILENAME = "training_lineage.json"


def resolve_hf_revision(repo_id: str, revision: str) -> str:
    """Resolve the expected dataset ref independently of the training producer."""
    if re.fullmatch(r"[0-9a-f]{40}", revision):
        return revision
    url = (
        "https://huggingface.co/api/datasets/"
        + quote(repo_id, safe="/") + "/revision/" + quote(revision or "main", safe="")
    )
    with urlopen(url, timeout=30) as response:
        document = json.loads(response.read(1024 * 1024))
    sha = document.get("sha")
    if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError("Hugging Face did not return a dataset commit SHA")
    return sha


def write_training_lineage(
    checkpoint: str, *, train_suite: str, source_parameter: str,
    revision_parameter: str, repo_id: str, requested_revision: str,
    resolved_revision: str, subdirectory: str, content_digest: str,
) -> None:
    if not re.fullmatch(r"[0-9a-f]{40}", resolved_revision):
        raise ValueError("Training must record the immutable dataset revision it downloaded")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", content_digest):
        raise ValueError("Training must measure its consumed dataset tree")
    record = {
        "schema_version": 1,
        "train_suite": train_suite,
        "source_parameter": source_parameter,
        "revision_parameter": revision_parameter,
        "dataset": {
            "source": "hf:" + repo_id,
            "requested_revision": requested_revision,
            "resolved_revision": resolved_revision,
            "subdirectory": subdirectory,
            "content_digest": content_digest,
        },
    }
    (Path(checkpoint) / FILENAME).write_text(json.dumps(record, indent=2) + "\n")


def verify_training_lineage(
    manifest: dict, lineage: dict | None, *, train_suite: str,
    source_parameter: str, revision_parameter: str, suite_spec: dict,
) -> list[dict]:
    if not isinstance(lineage, dict) or lineage.get("schema_version") != 1:
        raise ValueError("GR00T training requires training_lineage.json schema_version=1")
    dataset_spec = suite_spec.get("dataset")
    if not isinstance(dataset_spec, dict) or not dataset_spec.get("repo_id"):
        raise ValueError("Expected training suite has no dataset specification")
    if source_parameter != "__LIBERO_DEFAULT__":
        raise ValueError("GR00T training does not support a supplied S3 dataset")
    expected_source = "hf:" + dataset_spec["repo_id"]
    expected_ref = (
        dataset_spec.get("revision") or ""
        if revision_parameter == "__FROM_SUITE_MANIFEST__" else revision_parameter
    )
    actual = lineage.get("dataset")
    if not isinstance(actual, dict):
        raise ValueError("training_lineage.json has no dataset identity")
    pairs = [
        ("train_suite", lineage.get("train_suite"), train_suite),
        ("dataset_source_parameter", lineage.get("source_parameter"), source_parameter),
        ("dataset_revision_parameter", lineage.get("revision_parameter"), revision_parameter),
        ("dataset_source", actual.get("source"), expected_source),
        ("dataset_source", manifest["dataset_manifest"]["source"], expected_source),
        ("dataset_requested_revision", actual.get("requested_revision"), expected_ref),
        ("dataset_subdirectory", actual.get("subdirectory"), dataset_spec.get("subdir") or ""),
        ("dataset_content_digest", actual.get("content_digest"),
         manifest["dataset_manifest"]["revision"]),
    ]
    for field, observed, expected in pairs:
        if observed != expected:
            raise ValueError(f"{field} mismatch: expected {expected!r}, observed {observed!r}")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", actual.get("content_digest", "")):
        raise ValueError("Dataset content digest is missing or malformed")
    # Named refs are resolved from the dataset service by Validate, not accepted
    # from the checkpoint's own declaration. A moved ref fails closed; an exact
    # DatasetRevision commit avoids both movement and the metadata request.
    expected_commit = resolve_hf_revision(dataset_spec["repo_id"], expected_ref)
    if actual.get("resolved_revision") != expected_commit:
        raise ValueError(
            "dataset_revision mismatch: expected commit "
            f"{expected_commit!r}, observed {actual.get('resolved_revision')!r}")
    return [
        {"field": "dataset_source", "status": "attested", "expected": expected_source,
         "observed": actual["source"], "source_parameter": source_parameter,
         "expectation_matched": True, "binding": "checkpoint_digest",
         "training_consumption_verified": False},
        {"field": "dataset_revision", "status": "verified", "expected": expected_commit,
         "observed": actual["resolved_revision"], "requested_revision": expected_ref,
         "verification_scope": "revision_identity_only",
         "verification_method": (
             "pinned_commit_parameter" if re.fullmatch(r"[0-9a-f]{40}", expected_ref)
             else "independent_huggingface_resolution")},
        {"field": "dataset_subdirectory", "status": "attested",
         "expected": dataset_spec.get("subdir") or "", "observed": actual["subdirectory"],
         "expectation_matched": True, "binding": "checkpoint_digest",
         "training_consumption_verified": False},
        {"field": "dataset_content_digest", "status": "attested",
         "observed": actual["content_digest"], "binding": "checkpoint_digest",
         "manifest_consistency_checked": True, "dataset_recomputed_by_validate": False},
        {"field": "train_suite", "status": "attested", "expected": train_suite,
         "observed": lineage["train_suite"], "expectation_matched": True,
         "binding": "checkpoint_digest", "training_consumption_verified": False},
    ]
