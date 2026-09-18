#!/usr/bin/env python3
"""Wait for the detached local launcher, then run its independent verifier."""
import argparse
import datetime
import json
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path


def reported_activity(line):
    """Describe explicit log markers only; log activity is not a success check."""
    for marker, description in (
        ("-- runtime install", "installing the selected training runtime"),
        (">>> dataset_download", "downloading the training dataset"),
        ("<<< dataset_download OK", "dataset download completed"),
        ("cooling down 120s before base pull", "waiting for the Hugging Face rate-limit window"),
        (">>> base_materialize", "downloading the base model"),
        ("<<< base_materialize OK", "base model download completed"),
        (">>> finetune", "training started"),
        ("<<< finetune OK", "training process completed; step output still being prepared"),
        (">>> checkpoint_extract", "extracting the evaluation checkpoint"),
        ("<<< checkpoint_extract OK", "checkpoint extraction completed"),
        (">>> arena_eval", "starting Arena simulation"),
        ("<<< arena_eval OK", "simulation completed; step output still being prepared"),
    ):
        if marker in line:
            return description
    episodes = re.search(r"Episodes:.*?\|\s*(\d+)/(\d+)\s*\[", line)
    if episodes:
        return f"simulator reports {episodes[1]}/{episodes[2]} episodes"
    loss = re.search(r"\{'loss':\s*([0-9.eE+-]+)", line)
    if loss:
        return f"trainer reported loss {loss[1]}"
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--run-dir", type=Path)
    selection.add_argument("--latest", action="store_true",
                           help="Follow the last run submitted from this clone; never launches a run")
    parser.add_argument("--timeout-seconds", type=int, default=86400)
    parser.add_argument("--heartbeat-seconds", type=int, default=30)
    parser.add_argument("--reconnect-command", help="Display-only reconnect command supplied by vla")
    args = parser.parse_args()
    if args.timeout_seconds <= 0 or args.heartbeat_seconds <= 0:
        parser.error("Timeout and heartbeat must be positive")
    if args.latest:
        runtime = Path(__file__).resolve().parents[2] / "local-dev"
        try:
            run_id = (runtime / "latest-run-id.txt").read_text().strip()
        except FileNotFoundError:
            parser.error("No saved launch in this clone; use --run-dir for an existing run")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]{0,39}", run_id):
            parser.error("Invalid saved run ID")
        args.run_dir = runtime / "runs" / run_id
    root = args.run_dir.resolve()
    if not root.is_dir():
        parser.error(f"Run directory does not exist: {root}")
    reconnect = args.reconnect_command or shlex.join([
        "sudo", sys.executable, str(Path(__file__).resolve()),
        "--run-dir", str(root),
        "--timeout-seconds", str(args.timeout_seconds),
        "--heartbeat-seconds", str(args.heartbeat_seconds),
    ])
    try:
        wait_for_run(root, args.timeout_seconds, args.heartbeat_seconds, reconnect)
    except KeyboardInterrupt:
        print(f"\nStopped watching {root.name}; no cancellation was sent to the run.\n"
              f"Reconnect to the same run:\n{reconnect}", file=sys.stderr, flush=True)
        return 130


def wait_for_run(root, timeout_seconds, heartbeat_seconds, reconnect=None):
    unit = f"vla-local-{root.name}"
    deadline = time.monotonic() + timeout_seconds
    previous = None
    next_heartbeat = 0
    log_position, pipeline_started, step, progress = 0, False, None, None
    print(f"Watching {root}\nCtrl-C disconnects this waiter; it does not cancel the run.", flush=True)
    while time.monotonic() < deadline:
        state = {}
        try:
            state = json.loads((root / "status.json").read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        observed = state.get("status", "Starting")
        if observed == "Failed":
            raise SystemExit(f"{state.get('failure_reason', 'Local execution failed')}\n"
                             f"Evidence: {root}\nRetry requires an explicit launch with a new run ID.")
        active = subprocess.check_output(
            ["systemctl", "show", unit, "--property=ActiveState", "--value"],
            text=True, timeout=15).strip()
        if observed != previous or time.monotonic() >= next_heartbeat:
            activity = state.get("activity", observed)
            log = root / "run.log"
            log_age = "no log yet"
            if log.exists():
                log_age = f"last log write {max(0, time.time() - log.stat().st_mtime):.0f}s ago"
                with log.open(errors="replace") as stream:
                    stream.seek(log_position)
                    while True:
                        line_position = stream.tell()
                        line = stream.readline()
                        if not line:
                            break
                        if not line.endswith("\n"):
                            # The writer may append the rest of a step marker later.
                            stream.seek(line_position)
                            break
                        if ("[local] Starting FineTune → SimEval" in line
                                or "[local] Starting requested pipeline:" in line):
                            pipeline_started = True
                        match = re.search(r"Starting pipeline step: '(FineTune|SimEval|Validate|SuccessGate)'", line)
                        if pipeline_started and match:
                            step = match.group(1)
                            progress = None
                        elif pipeline_started:
                            progress = reported_activity(line) or progress
                    log_position = stream.tell()
            if observed == "Running" and step:
                activity = f"latest step started: {step}"
                if progress:
                    activity += f"; last reported: {progress}"
            if state.get("started_at"):
                elapsed = max(0, time.time() - datetime.datetime.fromisoformat(state["started_at"]).timestamp())
                age = f"{elapsed / 60:.1f}m elapsed"
            else:
                age = "start time not yet recorded"
            print(f"{root.name}: {observed}; {age}; service={active or 'unknown'}; "
                  f"{activity}; {log_age}", flush=True)
            previous = observed
            next_heartbeat = time.monotonic() + heartbeat_seconds
        if active not in {"active", "activating", "deactivating", "reloading"}:
            if observed == "PreflightSucceeded":
                print("Preflight completed; no training/evaluation pipeline was launched.")
                return
            if observed == "Succeeded":
                subprocess.run([
                    sys.executable, str(Path(__file__).with_name("verify_run.py")),
                    "--run-dir", str(root),
                ], check=True)
                return
            raise SystemExit(f"Launcher is {active!r} without a successful terminal result: {state}")
        time.sleep(10)
    raise SystemExit(f"Wait timeout reached; the detached service may still be running.\n"
                     f"Reconnect: {reconnect or '--run-dir ' + str(root)}; "
                     "do not relaunch to resume.")


if __name__ == "__main__":
    raise SystemExit(main())
