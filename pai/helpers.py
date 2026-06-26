"""Formatting helpers, friendly error handlers, and subprocess wrappers."""

from __future__ import annotations

import subprocess
import sys

import click
from botocore.exceptions import BotoCoreError, ClientError


# Message helpers

def info(text: str) -> None:
    click.secho(text, fg="cyan")


def success(text: str) -> None:
    click.secho(text, fg="green")


def warn(text: str) -> None:
    click.secho(text, fg="yellow")


def error(text: str) -> None:
    click.secho(text, fg="red", err=True)


def heading(text: str) -> None:
    click.secho(f"\n{text}\n", bold=True)


def pass_msg(text: str) -> None:
    click.secho("  [PASS] ", fg="green", nl=False)
    click.echo(text)


def fail_msg(text: str) -> None:
    click.secho("  [FAIL] ", fg="red", nl=False)
    click.echo(text)


def friendly_boto_error(e: Exception) -> None:
    """Convert a boto3 exception to user-friendly guidance and exit(1).

    Special-cases:
    - AccessDenied with iam:PassRole → print the exact policy fix
    - Credential/region errors → point to aws configure
    - Otherwise, a concise error message (no raw traceback)
    """
    if isinstance(e, ClientError):
        code = e.response.get("Error", {}).get("Code", "Unknown")
        message = e.response.get("Error", {}).get("Message", "")

        if code in ("AccessDenied", "AccessDeniedException", "UnauthorizedOperation"):
            error(f"AWS API call denied: {code}")
            if "iam:PassRole" in message:
                # Extract account from message or env (best-effort)
                import boto3
                try:
                    account = boto3.client("sts").get_caller_identity()["Account"]
                except Exception:
                    account = "<YOUR_ACCOUNT_ID>"

                error("\nYour AWS identity lacks iam:PassRole for the SageMaker role.")
                info("Attach this policy to your IAM user/role:\n")
                info("{\n"
                     '  "Version": "2012-10-17",\n'
                     '  "Statement": [{\n'
                     '    "Effect": "Allow",\n'
                     '    "Action": "iam:PassRole",\n'
                     f'    "Resource": "arn:aws:iam::{account}:role/physical-ai-dev-sagemaker-role"\n'
                     "  }]\n"
                     "}")
            else:
                info(f"Details: {message}")
            sys.exit(1)

        elif code in ("InvalidClientTokenId", "SignatureDoesNotMatch", "ExpiredToken"):
            error(f"AWS credential error: {code}")
            info("Run 'aws configure' to set up credentials, or refresh your session.")
            sys.exit(1)

        else:
            error(f"AWS API error: {code} — {message}")
            sys.exit(1)

    elif isinstance(e, BotoCoreError):
        error(f"AWS connection error: {type(e).__name__}")
        if "NoRegionError" in type(e).__name__ or "region" in str(e).lower():
            info("Set AWS_DEFAULT_REGION or run 'aws configure' to configure a region.")
        else:
            info(f"Details: {e}")
        sys.exit(1)

    else:
        # Non-boto error — re-raise so caller can handle or Python shows the traceback
        raise


# Subprocess wrappers

def run(cmd: list[str], cwd: str | None = None, env: dict | None = None, check: bool = False) -> subprocess.CompletedProcess:
    """Run a command, streaming output to the terminal.

    Args:
        cmd: Command as list (e.g., ["aws", "s3", "ls"])
        cwd: Working directory
        env: Environment dict (replaces os.environ)
        check: Raise CalledProcessError if returncode != 0

    Returns:
        CompletedProcess
    """
    return subprocess.run(cmd, cwd=cwd, env=env, check=check)


def run_capture(cmd: list[str], cwd: str | None = None, timeout: int = 30) -> tuple[int, str]:
    """Run a command and capture stdout. Returns (returncode, stdout).

    Args:
        cmd: Command as list
        cwd: Working directory
        timeout: Timeout in seconds

    Returns:
        (returncode, stdout) — stdout is stripped. Empty string on timeout.
    """
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            stdin=subprocess.DEVNULL,
            cwd=cwd,
        )
        return result.returncode, result.stdout.strip()
    except subprocess.TimeoutExpired:
        return 1, ""
