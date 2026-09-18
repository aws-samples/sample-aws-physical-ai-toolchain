"""Resume the existing CodeBuild image recipes, retaining every submitted build ID."""
from __future__ import annotations

import datetime as dt
import hashlib
import io
import time
import uuid
import zipfile

from .backend import COMPONENT, script
from .deployment import image_digest
from .operations import timestamp
from .registry import named_cells, require_image_capability, resolve


def stable_zip(body):
    """Git checkout mtimes must not change the identity of otherwise identical inputs."""
    output = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(body)) as source, zipfile.ZipFile(
        output, "w", zipfile.ZIP_DEFLATED
    ) as target:
        for name in sorted(source.namelist()):
            info = zipfile.ZipInfo(name)
            info.external_attr = source.getinfo(name).external_attr
            target.writestr(info, source.read(name), compress_type=zipfile.ZIP_DEFLATED)
    return output.getvalue()


def upload_source(session, bucket, name, body):
    digest = hashlib.sha256(body).hexdigest()
    key = f"code/v1/{digest}/{name}.zip"
    s3 = session.client("s3")
    created = s3.put_object(Bucket=bucket, Key=key, Body=body)
    version = created.get("VersionId")
    if version in (None, "", "null"):
        raise RuntimeError("Build source requires versioned storage")
    stored = s3.get_object(Bucket=bucket, Key=key, VersionId=version)["Body"]
    try:
        if hashlib.sha256(stored.read()).hexdigest() != digest:
            raise RuntimeError("Build-source readback differs from the selected source")
    finally:
        stored.close()
    return f"{bucket}/{key}", digest


def reconcile_build(client, entry):
    """Recover a lost start response by the recorded request's unique environment marker."""
    token = None
    for _ in range(10):
        page = client.list_builds_for_project(
            projectName=entry["request"]["projectName"], sortOrder="DESCENDING",
            **({"nextToken": token} if token else {}),
        )
        ids = page["ids"]
        for offset in range(0, len(ids), 100):
            builds = client.batch_get_builds(ids=ids[offset:offset + 100])["builds"]
            matches = [
                build for build in builds
                if any(variable["name"] == "VLA_DEPLOY_TOKEN"
                       and variable["value"] == entry["token"]
                       for variable in build.get("environment", {}).get("environmentVariables", []))
            ]
            if len(matches) > 1:
                raise RuntimeError("More than one build has this submission token; inspect saved IDs")
            if matches:
                entry["id"] = matches[0]["id"]
                return
        token = page.get("nextToken")
        if not token:
            break
    elapsed = time.time() - dt.datetime.fromisoformat(entry["submitted_at"]).timestamp()
    if elapsed > 180:
        raise RuntimeError(
            "Build submission response was lost and no matching build was found. "
            "Inspect the saved request/token and CodeBuild history before retrying; "
            "automatic resubmission would risk a duplicate."
        )
    # CodeBuild's idempotency window is five minutes; retry only well inside it.


def build(session, record, save, key, repository, make_request, *, retry_failed=False):
    client = session.client("codebuild")
    builds = record.setdefault("builds", {})
    entry = builds.get(key)
    if entry and entry.get("status") in {"FAILED", "FAULT", "STOPPED", "TIMED_OUT"}:
        if not retry_failed:
            raise RuntimeError(f"{key} build {entry.get('id')} failed; inspect it, then deploy --resume")
        record.setdefault("build_history", []).append({"key": key, **entry})
        entry = None
    if entry is None:
        token = uuid.uuid4().hex
        tag = f"vla-{record['source_commit'][:12]}-{token[:12]}"
        request = make_request(tag)
        request.update(idempotencyToken=token, autoRetryLimitOverride=0)
        request.setdefault("environmentVariablesOverride", []).append(
            {"name": "VLA_DEPLOY_TOKEN", "value": token, "type": "PLAINTEXT"})
        entry = {"request": request, "token": token, "tag": tag, "repository": repository,
                 "status": "Prepared"}
        builds[key] = entry
        save()
    if entry.get("digest"):
        return image_digest(session, entry["digest"])
    if not entry.get("id"):
        if entry["status"] == "Submitting":
            reconcile_build(client, entry)
            save()
        if not entry.get("id"):
            entry.update(status="Submitting", submitted_at=timestamp())
            save()  # Durable token/request BEFORE the external submission.
            result = client.start_build(**entry["request"])
            entry.update(id=result["build"]["id"], status=result["build"]["buildStatus"])
            save()  # Durable ID BEFORE waiting.
    while True:
        found = client.batch_get_builds(ids=[entry["id"]])
        if len(found["builds"]) != 1:
            raise RuntimeError(f"Recorded build is unavailable: {entry['id']}")
        current = found["builds"][0]
        entry.update(status=current["buildStatus"], phase=current.get("currentPhase"),
                     logs=current.get("logs"), phases=current.get("phases"),
                     observed_at=timestamp())
        record["activity"] = f"{key}: {entry['status']} / {entry['phase']} ({entry['id']})"
        save()
        print(record["activity"], flush=True)
        if entry["status"] == "SUCCEEDED":
            entry["digest"] = image_digest(session, repository + ":" + entry["tag"])
            save()
            return entry["digest"]
        if entry["status"] != "IN_PROGRESS":
            raise RuntimeError(f"{record['activity']}; logs: {entry.get('logs')}; "
                               f"phase details: {entry.get('phases')}")
        time.sleep(30)


def prepare_images(session, cfg, record, save, *, retry_failed=False):
    """Deduplicate family builds; Arena's connector follows its recorded base build."""
    request = record["request"]
    cells = named_cells()
    images = record.setdefault("images", {})
    registry = f"{cfg.account_id}.dkr.ecr.{cfg.region}.amazonaws.com"
    project = session.client("ssm").get_parameter(
        Name=f"/{cfg.foundation_project}/component/codebuild-project")["Parameter"]["Value"]
    supplied = request.get("images", {})
    families = {}
    connector = None

    def env(values):
        return [{"name": name, "value": value, "type": "PLAINTEXT"}
                for name, value in values.items()]

    for name in request["cells"]:
        cell = cells[name]
        selected = images.setdefault(name, {})
        for step, uri in supplied.get(name, {}).items():
            selected[step] = image_digest(session, uri)
        family = cell["model"]
        if "FineTune" not in selected or (cell["simulator"] == "libero" and "SimEval" not in selected):
            if family not in families:
                helper = script("build_images.py")
                body = stable_zip(helper.create_source_zip(family))
                source, digest = upload_source(session, cfg.trust_bucket, family, body)
                repository = registry + "/vla/" + family
                def family_request(tag):
                    return {
                        "projectName": project, "sourceTypeOverride": "S3",
                        "sourceLocationOverride": source, "buildspecOverride": "buildspec.yml",
                        "computeTypeOverride": "BUILD_GENERAL1_LARGE",
                        "timeoutInMinutesOverride": 60,
                        "environmentVariablesOverride": env({
                            "FAMILY": family, "ECR_REPO": repository,
                            "IMAGE_TAG": tag, "AWS_REGION": cfg.region,
                        }),
                    }
                families[family] = build(session, record, save, family, repository,
                                         family_request, retry_failed=retry_failed)
            selected.setdefault("FineTune", families[family])
            if cell["simulator"] == "libero":
                selected.setdefault("SimEval", families[family])
        if cell["simulator"] == "isaac_arena" and "SimEval" not in selected:
            if connector is None:
                base_repo = registry + "/vla/isaac-arena"
                base_spec = (COMPONENT / "entrypoints/eval/isaac_arena/base/buildspec_base.yml").read_text()
                def base_request(tag):
                    return {
                        "projectName": project, "sourceTypeOverride": "NO_SOURCE",
                        "buildspecOverride": base_spec,
                        "computeTypeOverride": "BUILD_GENERAL1_2XLARGE",
                        "timeoutInMinutesOverride": 180,
                        "environmentVariablesOverride": env({
                            "ECR_REGISTRY": registry, "IMAGE_URI": base_repo + ":" + tag,
                            "NGC_SECRET_ID": request["ngc_secret_name"],
                            "AWS_DEFAULT_REGION": cfg.region,
                        }),
                    }
                base = build(session, record, save, "arena-base", base_repo, base_request,
                             retry_failed=retry_failed)
                helper = script("build_arena_connector.py")
                source, digest = upload_source(
                    session, cfg.trust_bucket, "arena-connector",
                    stable_zip(helper.create_source_zip("gr00t")))
                repository = session.client("ssm").get_parameter(
                    Name=f"/{cfg.foundation_project}/ecr/isaac-lab-arena")["Parameter"]["Value"]
                if not repository.startswith(registry + "/"):
                    raise ValueError("The discovered connector repository belongs to another registry")
                def connector_request(tag):
                    return {
                        "projectName": project, "sourceTypeOverride": "S3",
                        "sourceLocationOverride": source,
                        "buildspecOverride": helper.BUILDSPEC["gr00t"],
                        "computeTypeOverride": "BUILD_GENERAL1_2XLARGE",
                        "timeoutInMinutesOverride": 120,
                        "environmentVariablesOverride": env({
                            "ECR_REPO_URI": repository, "IMAGE_TAG": tag,
                            "AWS_DEFAULT_REGION": cfg.region, "ARENA_BASE_IMAGE": base,
                        }),
                    }
                connector = build(session, record, save, "arena-connector", repository,
                                  connector_request, retry_failed=retry_failed)
            selected["SimEval"] = connector
        capability = resolve(family, cell["simulator"]).required_image_capability
        if capability:
            require_image_capability(selected["SimEval"], capability, ecr_client=session.client("ecr"))
        save()
    return images
