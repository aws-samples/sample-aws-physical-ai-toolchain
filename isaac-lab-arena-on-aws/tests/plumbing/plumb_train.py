#!/usr/bin/env python3
"""Plumbing FineTune: writes a dummy model with a real, verifiable digest.

Writes to SM_MODEL_DIR so the output is exposed as ModelArtifacts (step property).
The checkpoint_manifest.json is contract-valid: it contains a weights_digest
computed over the actual bytes written, so the eval step can recompute and match.

This is NOT real training code. It exists solely to prove the pipeline mechanics
(artifact hand-off, digest chain, PropertyFile resolution) on CPU instances
before spending GPU hours.
"""
import hashlib
import json
import os


def compute_digest(file_paths):
    """Fixture digest over concatenated bytes; production digest.py also frames paths and sizes."""
    h = hashlib.sha256()
    for fp in sorted(file_paths):
        with open(fp, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def main():
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    os.makedirs(model_dir, exist_ok=True)

    # Write dummy weights (small but non-trivial so digest is meaningful)
    weights_path = os.path.join(model_dir, "weights.bin")
    weights_content = b"PLUMBING_WEIGHTS_" + os.urandom(64)
    with open(weights_path, "wb") as f:
        f.write(weights_content)

    # Compute digest over the weights file
    digest = compute_digest([weights_path])
    print(f"[plumb_train] weights_digest: {digest}")

    # Write contract-valid checkpoint_manifest.json
    manifest = {
        "schema_version": 2,
        "model_family": os.environ.get("TRAIN_MODEL_FAMILY", "openvla"),
        "train_recipe": {
            "max_steps": int(os.environ.get("TRAIN_MAX_STEPS", "5")),
            "train_seed": 42,
        },
        "dataset_manifest": {
            "revision": os.environ.get("TRAIN_DATASET_S3URI", "plumbing-test"),
        },
        "weights_digest": digest,
    }

    manifest_path = os.path.join(model_dir, "checkpoint_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[plumb_train] wrote: {weights_path} ({len(weights_content)} bytes)")
    print(f"[plumb_train] wrote: {manifest_path}")
    print("[plumb_train] DONE")


if __name__ == "__main__":
    main()
