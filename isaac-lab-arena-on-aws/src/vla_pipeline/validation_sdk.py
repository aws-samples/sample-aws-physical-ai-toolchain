"""Bundle the Python 3.9 Validate SDK at code-staging time, never during a job."""
from __future__ import annotations

import base64
import hashlib
import io
import os
import subprocess
import sys
import tempfile
import zipfile
from functools import lru_cache
from pathlib import Path, PurePosixPath


def _locked_wheels(directory: Path, requirements: str) -> list[Path]:
    expected = {
        line.split(" --hash=sha256:")[1]: line.split(" --hash=")[0]
        for line in requirements.splitlines() if line.strip()
    }
    wheels = sorted(directory.glob("*.whl"))
    observed = {hashlib.sha256(wheel.read_bytes()).hexdigest() for wheel in wheels}
    if len(wheels) != len(expected) or observed != set(expected):
        raise ValueError("Validate SDK wheelhouse does not match the complete hash-locked closure")
    return wheels


@lru_cache(maxsize=1)
def sdk_bundle() -> bytes:
    lock = Path(__file__).with_name("validation_sdk.lock")
    requirements = lock.read_text()
    key = hashlib.sha256(lock.read_bytes()).hexdigest()
    cache = Path(os.environ.get(
        "VLA_VALIDATION_SDK_CACHE", str(Path.home() / ".cache" / "vla-validation-sdk")))
    cache.mkdir(parents=True, exist_ok=True)
    wheelhouse = cache / key
    if not wheelhouse.exists():
        # pip runs on the submitting host. Only pure-Python, hash-locked wheels
        # compatible with sklearn 1.2-1's Python 3.9 can enter the staged source.
        with tempfile.TemporaryDirectory(prefix="download-", dir=cache) as work:
            destination = Path(work) / "wheels"
            subprocess.run([
                sys.executable, "-m", "pip", "download", "--disable-pip-version-check",
                "--no-deps", "--require-hashes", "--only-binary=:all:",
                "--python-version=3.9", "--platform=any", "--abi=none",
                "--dest", str(destination), "-r", str(lock),
            ], check=True, timeout=300)
            _locked_wheels(destination, requirements)
            try:
                destination.rename(wheelhouse)
            except OSError:
                if not wheelhouse.exists():
                    raise
    wheels = _locked_wheels(wheelhouse, requirements)
    output = io.BytesIO()
    paths = set()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as bundle:
        for wheel in wheels:
            with zipfile.ZipFile(wheel) as contents:
                for member in contents.infolist():
                    path = PurePosixPath(member.filename)
                    if path.is_absolute() or ".." in path.parts or "\\" in member.filename:
                        raise ValueError(f"Unsafe Validate SDK wheel member: {member.filename}")
                    if not member.is_dir():
                        if member.filename in paths:
                            raise ValueError(f"Duplicate Validate SDK module: {member.filename}")
                        paths.add(member.filename)
            info = zipfile.ZipInfo(wheel.name, date_time=(1980, 1, 1, 0, 0, 0))
            bundle.writestr(info, wheel.read_bytes())
    return output.getvalue()


def sdk_bootstrap() -> str:
    """The SDK bytes are covered by the staged validate_entry.py's content digest."""
    bundle = sdk_bundle()
    digest = hashlib.sha256(bundle).hexdigest()
    encoded = base64.b64encode(bundle).decode("ascii")
    return f'''
# --- Embedded, hash-locked Validate SDK; no runtime installer or package index ---
import base64 as _sdk_b64, hashlib as _sdk_hash, io as _sdk_io
import os as _sdk_os, sys as _sdk_sys, tempfile as _sdk_tf, zipfile as _sdk_zip
_sdk_bytes = _sdk_b64.b64decode("{encoded}")
if _sdk_hash.sha256(_sdk_bytes).hexdigest() != "{digest}":
    raise RuntimeError("Embedded Validate SDK digest mismatch")
_sdk_dir = _sdk_tf.TemporaryDirectory(prefix="vla-validation-sdk-")
with _sdk_zip.ZipFile(_sdk_io.BytesIO(_sdk_bytes)) as _sdk_bundle:
    for _sdk_wheel in _sdk_bundle.namelist():
        with _sdk_zip.ZipFile(_sdk_io.BytesIO(_sdk_bundle.read(_sdk_wheel))) as _sdk_contents:
            _sdk_contents.extractall(_sdk_dir.name)
_sdk_sys.path.insert(0, _sdk_dir.name)
_sdk_os.environ["VLA_VALIDATION_SDK_SHA256"] = "{digest}"
del _sdk_bytes
# --- END embedded Validate SDK ---
'''
