"""SSM transport endpoint for the same local execution adapter used on the host.

This is called by the CLI, not a second user-facing recipe. Requests and replies
contain selections and evidence only; runtime credentials stay on the instance.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from .backend import COMPONENT, script
from .deployment import aws_session, discover, local_instance
from .operations import Store, timestamp


def require_host(deployment):
    local = deployment["local"]
    identity = local_instance()
    if (identity["accountId"], identity["instanceId"], identity["region"]) != (
        deployment["account_id"], local["instance_id"], local["host_region"],
    ):
        raise ValueError("The request targets a different EC2 instance")
    session = aws_session(None, deployment["region"])
    caller = session.client("sts").get_caller_identity()
    expected = f"arn:aws:sts::{deployment['account_id']}:assumed-role/{local['expected_role']}/"
    if session.get_credentials().method != "iam-role" or not caller["Arn"].startswith(expected):
        raise ValueError("Remote execution requires the selected instance role")
    return session


def prepare_host_images(deployment):
    session = require_host(deployment)
    cfg = discover(session, deployment["project"])
    from sagemaker import image_uris

    images = {f"{cell}-{step}": uri for cell, selected in deployment["images"].items()
              for step, uri in selected.items()}
    images["Validate"] = image_uris.retrieve(
        framework="sklearn", region=cfg.region, version="1.2-1", py_version="py3",
        instance_type="ml.m5.xlarge")
    script("local/run_local_pipeline.py").ensure_images(session, images, pull=True)
    # This reports real headroom after downloads. Per-cell launch budgets are
    # still checked before training; preparation does not assert they are met.
    subprocess.run(["df", "-h", "/", deployment["local"]["scratch_root"]], check=True)
    print("Host dependencies and selected images are prepared; launch checks remain.")


def run_action(payload, store):
    from .execution import refresh, prepare, source_identity, stop, submit

    deployment = store.load("deployments", payload["deployment"])
    require_host(deployment)
    if source_identity() != payload["source_commit"]:
        raise ValueError("Prepared host checkout differs from the submitting checkout")
    run_id = payload["run_id"]
    action = payload["action"]
    if action == "run":
        request = payload["request"]
        if store.path("runs", run_id).exists():
            record = store.load("runs", run_id)
            if (record["source_commit"] != payload["source_commit"]
                    or record["request"]["parameters"] != request["parameters"]
                    or record["request"]["steps"] != request["steps"]):
                raise ValueError("This run ID already contains a different request")
            # A lost SSM response must never start training a second time.
            return refresh(record, store)
        raw = {**request, "image_overrides": request["images"]}
        raw.pop("transport", None)
        verified = prepare(raw, deployment)
        for key in ("parameters", "steps", "images", "checkpoint_identity"):
            if verified.get(key) != request.get(key):
                raise ValueError(f"Host resolved different {key}; no execution was submitted")
        record = submit(verified, deployment, store, run_id)
    else:
        record = store.load("runs", run_id)
        if action == "stop":
            record = stop(record, store=store)
        elif action == "verify":
            return refresh(record, store, verify=True)
        elif action != "status":
            raise ValueError(f"Unknown transport action: {action}")
    return refresh(record, store)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "exchange"])
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--deployment")
    parser.add_argument("--payload")
    args = parser.parse_args()
    store = Store(args.state_dir)
    if args.action == "prepare":
        from .execution import source_identity

        deployment = store.load("deployments", args.deployment)
        if source_identity() != deployment["source_commit"]:
            raise ValueError("Host preparation source differs from the selected commit")
        prepare_host_images(deployment)
        deployment["local"].update(
            source_commit=deployment["source_commit"], remote_ready=True, prepared_at=timestamp())
        store.save("deployments", deployment)
        return
    payload = json.loads(Path(args.payload).read_text())
    session = aws_session(None, "us-east-1")
    response = payload["response"]
    try:
        with contextlib.redirect_stdout(sys.stderr):
            result = {"ok": True, "record": run_action(payload, store)}
    except Exception as exc:
        result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    result.update(operation_token=payload["operation_token"], run_id=payload["run_id"],
                  source_commit=payload["source_commit"])
    body = (json.dumps(result, default=str) + "\n").encode()
    uploaded = session.client("s3").put_object(
        Bucket=response["bucket"], Key=response["key"], Body=body, IfNoneMatch="*")
    print(json.dumps({"response_version": uploaded.get("VersionId"),
                      "response_sha256": hashlib.sha256(body).hexdigest(),
                      "ok": result["ok"]}), flush=True)


if __name__ == "__main__":
    main()
