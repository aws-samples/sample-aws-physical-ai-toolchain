"""Explicit scratch mounts for the pinned SageMaker local SDK."""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path

# Cache/data directories only. Never hide the image's /opt/vla code or venvs.
CACHE_DESTINATIONS = (
    "/opt/vla/hf-cache", "/opt/vla/rlds", "/opt/vla/lerobot-data", "/opt/vla/base",
)
RUN_DESTINATIONS = ("/tmp", "/opt/vla/model", "/opt/vla/ckpt", "/opt/vla/checkpoints")


def free_gib(path):
    return round(shutil.disk_usage(path).free / 1024**3, 1)


def check_space(layout, *, phase, minimums, run_dir):
    """Check each filesystem separately and retain the values behind a refusal."""
    paths = {"root": "/", "scratch": layout["root"]}
    filesystems = {
        name: {"path": path, "free_gib": shutil.disk_usage(path).free / 1024**3,
               "required_gib": minimums[name]}
        for name, path in paths.items()
    }
    failed = [name for name, row in filesystems.items()
              if row["free_gib"] < row["required_gib"]]
    report = {"phase": phase, "status": "Failed" if failed else "Passed",
              "filesystems": filesystems}
    path = Path(run_dir) / "disk-checks.json"
    checks = json.loads(path.read_text()) if path.exists() else []
    checks.append(report)
    path.write_text(json.dumps(checks, indent=2) + "\n")
    detail = "; ".join(
        f"{name} ({row['path']}): {row['free_gib']:.2f} GiB free, "
        f"{row['required_gib']} GiB required"
        for name, row in filesystems.items()
    )
    print(f"[local] Disk check {phase}: {detail}", flush=True)
    if failed:
        raise RuntimeError(
            f"Insufficient disk space {phase}: {detail}. "
            "Reclaim only retained/archived run data on the affected filesystem, "
            "or add storage; freeing scratch does not free root."
        )
    return report


def initial_budget(profile, steps):
    """Planning reserves for the sample, in addition to already downloaded images.

    Retained runs used about 40 GiB root/73 GiB scratch (Arena), 1/87
    (GR00T LIBERO), and 4/218 (Molmo). Preserve the later 40/100 guards
    plus growth/margin, including Arena source working files now on scratch.
    These are conservative launch floors, not predictions for arbitrary doses.
    """
    if "FineTune" not in steps:
        return {"root": 60, "scratch": 220}
    return {
        "arena-gr1": {"root": 90, "scratch": 250},
        "molmoact2-libero": {"root": 60, "scratch": 350},
    }.get(profile, {"root": 60, "scratch": 220})


class DiskMeasurements:
    """Sample both filesystems while retaining the observed minimum free space."""

    def __init__(self, layout, run_dir):
        self.paths = {"root": "/", "scratch": layout["root"]}
        self.output = Path(run_dir) / "disk-usage.json"
        self.samples = []
        self.finished = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def sample(self):
        self.samples.append({
            "timestamp": time.time(),
            **{name + "_free_bytes": shutil.disk_usage(path).free
               for name, path in self.paths.items()},
        })
        report = {
            "sampling_interval_seconds": 15,
            "scope": "Observed filesystem use; activity from other processes is included.",
            "minimum_free_bytes": {
                name: min(row[name + "_free_bytes"] for row in self.samples)
                for name in self.paths},
            "samples": self.samples,
        }
        temporary = self.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(self.output)

    def _run(self):
        while not self.finished.wait(15):
            self.sample()

    def start(self):
        self.sample()
        self.thread.start()
        return self

    def stop(self):
        self.finished.set()
        self.thread.join()
        self.sample()


def prepare(root, run_dir, profile):
    root, run_dir = Path(root).resolve(), Path(run_dir).resolve()
    # Refuse an absent/unmounted scratch disk silently falling back onto root.
    existing = root
    while not existing.exists():
        existing = existing.parent
    if existing.stat().st_dev == Path("/").stat().st_dev:
        raise RuntimeError("--scratch-root must be on a mounted filesystem separate from root")
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    work = root / "runs" / run_dir.name
    if work.exists():
        raise RuntimeError(f"Scratch run already exists: {work}")
    work.mkdir(parents=True, mode=0o700)
    containers = work / "containers"
    containers.mkdir()
    (run_dir / "containers").symlink_to(containers, target_is_directory=True)
    host_tmp = work / "host-tmp"
    host_tmp.mkdir()
    os.environ["TMPDIR"] = str(host_tmp)
    tempfile.tempdir = str(host_tmp)
    cache = root / "cache" / profile
    cache.mkdir(parents=True, exist_ok=True, mode=0o700)
    return {"root": str(root), "work": str(work), "cache": str(cache),
            "root_free_gib": free_gib("/"), "scratch_free_gib": free_gib(root)}


def install_mounts(layout, run_dir):
    """Add mounts through the SDK's real volume interface, preserving generated mounts."""
    import sagemaker.local.image

    original = sagemaker.local.image._SageMakerContainer._generate_compose_file
    volume_type = sagemaker.local.image._Volume
    observations = []

    def compose(container, command, additional_volumes=None, additional_env_vars=None):
        check_space(layout, phase="before the next container",
                    minimums={"root": 40, "scratch": 100}, run_dir=run_dir)
        volumes = list(additional_volumes or [])
        existing_targets = {volume.container_dir for volume in volumes}
        job = Path(container.container_root).name
        paths = []
        destinations = ("/tmp",) if command == "process" else (
            *CACHE_DESTINATIONS, *RUN_DESTINATIONS)
        for target in destinations:
            if target in existing_targets:
                raise RuntimeError(f"Scratch mount conflicts with a generated mount: {target}")
            base = (Path(layout["cache"]) if target in CACHE_DESTINATIONS
                    else Path(layout["work"]) / "runtime" / job)
            source = base / target.lstrip("/")
            source.mkdir(parents=True, exist_ok=True, mode=0o700)
            if target == "/tmp":
                source.chmod(0o1777)
            volumes.append(volume_type(str(source), container_dir=target))
            paths.append({"source": str(source), "target": target})
        result = original(container, command, volumes, additional_env_vars)
        for service in result["services"].values():
            generated = set(service["volumes"])
            if not all(f"{item['source']}:{item['target']}" in generated for item in paths):
                raise RuntimeError("SageMaker omitted a required scratch mount")
        observations.append({
            "job": job, "command": command, "image": container.image, "mounts": paths,
            "root_free_gib": free_gib("/"), "scratch_free_gib": free_gib(layout["root"]),
        })
        (Path(run_dir) / "scratch-mounts.json").write_text(
            json.dumps(observations, indent=2) + "\n")
        return result

    sagemaker.local.image._SageMakerContainer._generate_compose_file = compose
