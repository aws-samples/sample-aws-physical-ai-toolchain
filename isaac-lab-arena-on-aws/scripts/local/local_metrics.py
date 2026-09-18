"""Export measured timings and trainer diagnostics without inventing missing fields."""
from __future__ import annotations

import ast
import csv
import datetime as dt
import hashlib
import json
import math
import re
from pathlib import Path


def training_records(text):
    records, summary = [], {}
    for line_number, line in enumerate(text.splitlines(), 1):
        start, end = line.find("{"), line.rfind("}")
        if start < 0 or end < start:
            # The pinned MolmoAct2/LeRobot trainer logs fields rather than a dict:
            # "step:200 smpl:2K ... loss:0.134 grdn:0.863 lr:4.5e-05 ...".
            step = re.search(r"\bstep:(\d+)\b", line)
            loss = re.search(r"\bloss:(\S+)", line)
            if not step or not loss:
                continue
            try:
                value = {"step": int(step[1]), "loss": float(loss[1])}
                for label, key in (("grdn", "grad_norm"), ("lr", "learning_rate")):
                    field = re.search(r"\b" + label + r":(\S+)", line)
                    if field:
                        value[key] = float(field[1])
            except ValueError:
                continue
        else:
            try:
                value = ast.literal_eval(line[start:end + 1])
            except (SyntaxError, ValueError):
                continue
        if not isinstance(value, dict):
            continue
        if "train_runtime" in value:
            summary = {key: item for key, item in value.items()
                       if isinstance(item, (int, float)) and not isinstance(item, bool)}
        loss = value.get("loss")
        if isinstance(loss, (int, float)) and not isinstance(loss, bool):
            if not math.isfinite(loss):
                raise RuntimeError("Trainer emitted non-finite loss")
            records.append({
                "record_index": len(records) + 1, "source_line": line_number,
                "optimizer_step": value.get("step", value.get("global_step")),
                "loss": loss, "grad_norm": value.get("grad_norm"),
                "learning_rate": value.get("learning_rate"),
            })
    if any(not math.isfinite(value) for value in summary.values()):
        raise RuntimeError("Trainer summary contains a non-finite value")
    if re.search(r"""['"]loss['"]\s*:\s*[-+]?(?:nan|inf)\b""", text, re.I):
        raise RuntimeError("Trainer emitted non-finite loss")
    return records, summary


def _timestamp(value):
    if isinstance(value, (int, float)):
        return value
    # Docker emits nanoseconds; Python 3.10 accepts at most microseconds here.
    normalized = re.sub(r"(\.\d{6})\d+", r"\1", value).replace("Z", "+00:00")
    return dt.datetime.fromisoformat(normalized).timestamp()


def write_metrics(root, status, steps, containers, parameters, receipt, verification_seconds):
    root = Path(root)
    trained = any(step["StepName"] == "FineTune" for step in steps)
    text = (root / "train.log").read_text(errors="replace") if trained else ""
    records, trainer_summary = training_records(text)
    with (root / "training-metrics.csv").open("w") as output:
        writer = csv.DictWriter(output, fieldnames=[
            "record_index", "source_line", "optimizer_step", "loss", "grad_norm", "learning_rate"])
        writer.writeheader()
        writer.writerows(records)
    observed_steps = re.findall(r"log confirms global_step=(\d+)", text)
    recorded_steps = [
        row["optimizer_step"] for row in records
        if type(row["optimizer_step"]) is int and row["optimizer_step"] >= 0
    ]
    observed_final_step = (
        int(observed_steps[-1]) if observed_steps else
        recorded_steps[-1] if recorded_steps else None
    )
    diagnostics = {
        "evidence_class": "trainer_reported_diagnostics",
        "training_performed": trained,
        "requested_optimizer_steps": parameters.get("TrainSteps") if trained else None,
        "observed_final_step": observed_final_step,
        "observed_step_basis": "trainer_log" if observed_final_step is not None else "not_extracted",
        "logged_loss_records": len(records),
        "first_logged_loss": records[0]["loss"] if records else None,
        "last_logged_loss": records[-1]["loss"] if records else None,
        "first_five_record_mean": (
            sum(row["loss"] for row in records[:5]) / len(records[:5]) if records else None),
        "last_five_record_mean": (
            sum(row["loss"] for row in records[-5:]) / len(records[-5:]) if records else None),
        "trainer_summary": trainer_summary,
        "source": "train.log" if trained else None,
        "source_sha256": hashlib.sha256((root / "train.log").read_bytes()).hexdigest() if trained else None,
        "missing_fields": (
            ["per_record_optimizer_steps"] if records and any(
                row["optimizer_step"] is None for row in records) else []),
        "coverage": ("loss_records_extracted" if records else
                     "raw_log_retained_parser_has_no_loss_records" if trained else "training_not_requested"),
    }
    step_times = {}
    for step in steps:
        duration = _timestamp(step["EndTime"]) - _timestamp(step["StartTime"])
        if duration < 0:
            raise RuntimeError("Invalid step timing")
        step_times[step["StepName"]] = round(duration, 3)
    container_times = {
        row["kind"]: round(
            _timestamp(row["state"]["FinishedAt"]) - _timestamp(row["state"]["StartedAt"]), 3)
        for row in containers
    }
    timings = {
        "exporter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "preparation_seconds": status.get("preparation_seconds"),
        "pipeline_seconds": status.get("pipeline_seconds", round(sum(step_times.values()), 3)),
        "launcher_wall_seconds": status["elapsed_seconds"],
        "verification_seconds": round(verification_seconds, 3),
        "step_wall_seconds": step_times, "container_wall_seconds": container_times,
        "trainer_compute_seconds": trainer_summary.get("train_runtime"),
        "cache_state": "see manifest image identities and retained logs; cache warmth not inferred",
        "scope": "Step wall time includes SDK transfer/packaging; container wall time excludes them.",
    }
    evaluation = {
        key: (receipt or {}).get(key) for key in (
            "model_family", "suite", "task_ids", "episodes", "success_rate",
            "per_task", "policy_type", "model_artifact_identity", "training_contract")
    }
    for name, value in (
        ("training-summary.json", diagnostics), ("timings.json", timings),
        ("evaluation-summary.json", evaluation),
    ):
        (root / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def completion_summary(root, result):
    """Present the completed verifier's evidence without upgrading its claims."""
    root = Path(root)
    training = json.loads((root / "training-summary.json").read_text())
    evaluation = json.loads((root / "evaluation-summary.json").read_text())
    timings = json.loads((root / "timings.json").read_text())
    episodes = evaluation["episodes"]
    tasks = evaluation.get("per_task") or []
    successes = None
    if tasks and all(
        type(row.get("successes")) is int and type(row.get("episodes")) is int
        and 0 <= row["successes"] <= row["episodes"] for row in tasks
    ) and sum(row["episodes"] for row in tasks) == episodes:
        successes = sum(row["successes"] for row in tasks)
    outcome = (f"{successes}/{episodes} successful" if successes is not None
               else f"{episodes} episodes; success count not separately recorded")
    seconds = round(timings["launcher_wall_seconds"])
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    observed = training["observed_final_step"]
    full = result.get("complete_training_workflow", True)
    workflow = ("Local workflow: independently verified" if full
                else "Requested local steps: independently checked; full training pipeline not run")
    training_text = (
        f"{observed if observed is not None else 'not extracted'}/"
        f"{training['requested_optimizer_steps']} steps (trainer log)"
        if training.get("training_performed", True)
        else "omitted; supplied checkpoint, earlier training not verified by this run"
    )
    evaluation_text = (
        f"{outcome}; success rate {evaluation['success_rate']:.1%}"
        if episodes is not None else "not requested"
    )
    text = (
        f"{workflow} ({result['run_id']})\n"
        f"Training: {training_text}\n"
        f"Robot trials: {evaluation_text}\n"
        f"Validation/publication: {'checked' if result.get('validation_performed', True) else 'not requested'}\n"
        f"Run time: {hours}h {minutes:02d}m {seconds:02d}s, including preparation\n"
        "Managed execution: not checked by this local verifier\n"
        "Host: still running; archive results before stopping it\n"
        f"Results: {root.resolve()}\n"
        "A workflow pass does not establish model quality.\n"
    )
    return text
