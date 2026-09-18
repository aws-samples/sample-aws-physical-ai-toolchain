"""Preserve SageMaker tar contents while parallelizing local output compression."""
import os
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path


def create_archive(source_files, target, threads=8):
    started = time.monotonic()
    print(f"[local] Compressing {Path(target).name} with pigz, {threads} workers", flush=True)
    with open(target, "wb") as output, tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(
            ["/usr/bin/pigz", "-1", "-p", str(threads)],
            stdin=subprocess.PIPE, stdout=output, stderr=errors,
        )
        try:
            with tarfile.open(fileobj=process.stdin, mode="w|", dereference=True) as archive:
                for source in source_files:
                    archive.add(source, arcname=os.path.basename(source))
            process.stdin.close()
            code = process.wait()
            if code:
                errors.seek(0)
                raise RuntimeError(f"pigz exited {code}: {errors.read(4096).decode(errors='replace')}")
        except BaseException:
            process.kill()
            process.wait(timeout=10)
            Path(target).unlink(missing_ok=True)
            raise
    print(
        f"[local] Compressed {Path(target).name}: {Path(target).stat().st_size} bytes "
        f"in {time.monotonic() - started:.1f}s", flush=True,
    )
    return target


def install(run_dir):
    """Replace only the SDK's output-archive path for this local execution."""
    import sagemaker.utils

    original = sagemaker.utils.create_tar_file
    container_root = (Path(run_dir) / "containers").resolve()

    def create_local_artifact(source_files, target=None):
        if target is not None:
            path = Path(target).resolve()
            if path.is_relative_to(container_root) and path.parent.name == "compressed_artifacts":
                return create_archive(source_files, target)
        return original(source_files, target)

    sagemaker.utils.create_tar_file = create_local_artifact
