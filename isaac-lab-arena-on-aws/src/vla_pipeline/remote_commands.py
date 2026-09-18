"""Small SSM command adapter shared by host preparation and remote CLI execution."""
from __future__ import annotations

import time
import uuid

from .operations import timestamp

ACTIVE = {"Pending", "InProgress", "Delayed", "Cancelling"}


def submit(client, host, entry, commands, save, *, timeout=3600, bucket=None):
    if entry.get("command_id"):
        return entry["command_id"]
    if entry.get("submitted_at"):
        # SendCommand has no idempotency token. Reconcile a lost response by our
        # unique comment; never blindly send a second mutating command.
        token = None
        matches = []
        for _ in range(10):
            page = client.list_commands(
                InstanceId=host, MaxResults=50,
                Filters=[{"key": "InvokedAfter", "value": entry["submitted_at"]}],
                **({"NextToken": token} if token else {}),
            )
            matches.extend(c for c in page["Commands"] if c.get("Comment") == entry["comment"])
            token = page.get("NextToken")
            if not token:
                break
        if len(matches) != 1 or token:
            raise RuntimeError(
                f"SSM submission needs reconciliation: host={host}, comment={entry['comment']}, "
                f"submitted_at={entry['submitted_at']}, matches={len(matches)}, "
                f"scan_truncated={bool(token)}. No duplicate command was sent. "
                "Inspect Run Command history in the host region using that comment/time, "
                "then follow this saved operation again. Absence from this scan does not "
                "prove that no command was submitted.")
        entry["command_id"] = matches[0]["CommandId"]
        save()
        return entry["command_id"]
    entry.update(comment="vla-" + uuid.uuid4().hex, submitted_at=timestamp(),
                 status="Submitting", timeout_seconds=timeout)
    save()
    request = dict(
        InstanceIds=[host], DocumentName="AWS-RunShellScript", Comment=entry["comment"],
        TimeoutSeconds=600, Parameters={"commands": commands, "executionTimeout": [str(timeout)]},
    )
    if bucket:
        request.update(OutputS3BucketName=bucket, OutputS3Region="us-east-1",
                       OutputS3KeyPrefix="localdev/cli/ssm/" + entry["comment"])
    response = client.send_command(**request)["Command"]
    entry.update(command_id=response["CommandId"], status=response["Status"])
    save()
    return entry["command_id"]


def observe(client, host, entry):
    try:
        result = client.get_command_invocation(CommandId=entry["command_id"], InstanceId=host)
    except client.exceptions.InvocationDoesNotExist:
        return {"Status": "Pending"}
    entry.update(status=result["Status"], response_code=result.get("ResponseCode"),
                 stdout=result.get("StandardOutputContent"), stderr=result.get("StandardErrorContent"),
                 stdout_url=result.get("StandardOutputUrl"), stderr_url=result.get("StandardErrorUrl"),
                 observed_at=timestamp())
    return result


def wait(client, host, entry, save):
    while True:
        result = observe(client, host, entry)
        save()
        print(f"SSM {entry['command_id']}: {result['Status']}; "
              f"{result.get('StatusDetails', 'waiting for agent response')}", flush=True)
        if result["Status"] == "Success":
            return result
        if result["Status"] not in ACTIVE:
            raise RuntimeError(
                f"SSM {entry['command_id']} {result['Status']}: "
                f"{result.get('StandardErrorContent', '')}\n{result.get('StandardOutputContent', '')}\n"
                f"Full output: {result.get('StandardOutputUrl') or entry.get('stdout_url')}")
        time.sleep(30)
