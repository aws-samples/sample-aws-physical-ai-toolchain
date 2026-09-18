"""The full CodeBuild archive must contain the real decoder regression inputs."""
import io
import runpy
import shlex
import zipfile
from pathlib import Path


def test_full_gr00t_archive_contains_every_copy_source_in_its_family_context():
    component = Path(__file__).resolve().parents[1]
    builder = runpy.run_path(str(component / "scripts/build_images.py"))
    with zipfile.ZipFile(io.BytesIO(builder["create_source_zip"]("gr00t"))) as archive:
        dockerfile = archive.read("docker/gr00t/Dockerfile").decode()
        sources = {
            source
            for line in dockerfile.splitlines()
            if line.startswith("COPY ")
            for source in shlex.split(line)[1:-1]
        }
        assert {"build_ffmpeg.sh", "check_video_decode.py", "av1-two-frames.mp4"} <= sources
        for source in sources:
            name = f"docker/gr00t/{source}"
            assert archive.read(name) == (component / name).read_bytes()
        assert archive.testzip() is None
        # The normal build starts from the upstream DLC; it cannot require an
        # application image already published into the new account.
        base = next(line for line in dockerfile.splitlines() if line.startswith("FROM "))
        assert "/pytorch-training:" in base
