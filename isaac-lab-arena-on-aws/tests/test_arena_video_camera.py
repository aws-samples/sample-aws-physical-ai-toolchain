"""The recorded camera is posed from the env's ViewerCfg, not left at Kit's default.

The first recorded rollout was a valid mp4 of a grey frame with the kitchen as a
few-pixel speck: headless, nothing in the pinned Isaac Lab applies ViewerCfg to
/OmniverseKit_Persp (see arena_video_runner.py). Every exit code was zero, so these
tests pin the pieces that make the camera land where the task says:

  1. The look-at math puts the camera at the eye, facing the lookat, upright.
  2. The pose is authored onto the camera prim and wins over any weaker opinion.
  3. The wrapper aims the camera before RecordVideo starts, and runs policy_runner with
     its own argv.
  4. eval_entry routes through the wrapper only when recording, and the image ships it.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys
import types

import numpy as np
import pytest

_REPO = pathlib.Path(__file__).resolve().parents[1]
_WRAPPER = _REPO / "entrypoints/eval/isaac_arena/gr00t/arena_video_runner.py"
_EVAL_ENTRY = _REPO / "entrypoints/eval/isaac_arena/gr00t/eval_entry.py"
_DOCKERFILE = _REPO.parent / "containers/isaac-lab-arena/Dockerfile"

# What the blank-video run logged for the fridge task: PickAndPlaceTask's
# offset=[-1.5, -1.5, 1.5] around the ranch bottle.
EYE = (2.5512890815734863, -2.081841289997101, 2.5184643268585205)
LOOKAT = (4.051289081573486, -0.5818412899971008, 1.0184643268585205)


def _load_wrapper():
    spec = importlib.util.spec_from_file_location("arena_video_runner_probe", _WRAPPER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _forward_and_up(matrix):
    """World-space view direction (-Z) and up (+Y) of a USD row-vector camera matrix."""
    m = np.asarray(matrix, dtype=float)
    return -m[2, :3], m[1, :3]


def test_camera_sits_at_the_eye_and_faces_the_lookat():
    mod = _load_wrapper()
    matrix = mod.camera_to_world(EYE, LOOKAT)

    forward, up = _forward_and_up(matrix)
    expected = np.subtract(LOOKAT, EYE) / np.linalg.norm(np.subtract(LOOKAT, EYE))

    np.testing.assert_allclose(matrix[3, :3], EYE)
    np.testing.assert_allclose(forward, expected, atol=1e-9)
    assert up[2] > 0, "the camera must be upright on a Z-up stage"
    np.testing.assert_allclose(matrix[:3, :3] @ matrix[:3, :3].T, np.eye(3), atol=1e-9)


@pytest.mark.parametrize("eye,lookat", [((1, 1, 1), (1, 1, 1)), ((0, 0, 5), (0, 0, 0))])
def test_degenerate_poses_fail_loud(eye, lookat):
    with pytest.raises(ValueError):
        _load_wrapper().camera_to_world(eye, lookat)


class _Tensor:
    def __init__(self, values):
        self._values = np.asarray(values, dtype=float)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._values


def test_pose_is_resolved_like_the_viewport_camera_controller():
    mod = _load_wrapper()
    origin = (30.0, 0.0, 0.0)
    vcc = types.SimpleNamespace(viewer_origin=_Tensor(origin),
                                default_cam_eye=np.array(EYE), default_cam_lookat=np.array(LOOKAT))

    eye, lookat = mod.viewer_pose(types.SimpleNamespace(viewport_camera_controller=vcc))

    np.testing.assert_allclose(eye, np.add(origin, EYE))
    np.testing.assert_allclose(lookat, np.add(origin, LOOKAT))


def test_without_a_controller_an_env_origin_is_read_from_the_scene():
    mod = _load_wrapper()
    origin = (0.0, 30.0, 0.0)
    env = types.SimpleNamespace(
        viewport_camera_controller=None,
        cfg=types.SimpleNamespace(viewer=types.SimpleNamespace(
            origin_type="env", env_index=1, eye=EYE, lookat=LOOKAT)),
        scene=types.SimpleNamespace(env_origins=[_Tensor((0, 0, 0)), _Tensor(origin)]))

    eye, _ = mod.viewer_pose(env)

    np.testing.assert_allclose(eye, np.add(origin, EYE))


def test_the_authored_pose_is_the_prims_world_transform():
    """Real USD: Kit defines /OmniverseKit_Persp in the session layer with its own ops."""
    pytest.importorskip("pxr")
    from pxr import Gf, Usd, UsdGeom

    mod = _load_wrapper()
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        cam = UsdGeom.Camera.Define(stage, "/OmniverseKit_Persp")
        cam.AddTranslateOp().Set(Gf.Vec3d(500.0, 500.0, 500.0))  # a far default pose
    # A weaker root-layer opinion must not leak back in.
    UsdGeom.Xformable(stage.OverridePrim("/OmniverseKit_Persp")).AddRotateXOp().Set(45.0)

    mod.author_camera_pose(stage, "/OmniverseKit_Persp", mod.camera_to_world(EYE, LOOKAT))

    world = UsdGeom.Xformable(stage.GetPrimAtPath("/OmniverseKit_Persp")) \
        .ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    position = world.Transform(Gf.Vec3d(0, 0, 0))
    target_dir = world.TransformDir(Gf.Vec3d(0, 0, -1)).GetNormalized()
    expected = Gf.Vec3d(*np.subtract(LOOKAT, EYE)).GetNormalized()

    assert Gf.IsClose(position, Gf.Vec3d(*EYE), 1e-6)
    assert Gf.IsClose(target_dir, expected, 1e-6)


def test_missing_camera_prim_fails_loud():
    pytest.importorskip("pxr")
    from pxr import Usd

    mod = _load_wrapper()
    with pytest.raises(RuntimeError, match="does not exist"):
        mod.author_camera_pose(Usd.Stage.CreateInMemory(), "/OmniverseKit_Persp", np.eye(4))


@pytest.fixture
def fake_gymnasium(monkeypatch):
    calls = []

    class RecordVideo:
        def __init__(self, env, *args, **kwargs):
            calls.append(("record_video", env))

    wrappers = types.ModuleType("gymnasium.wrappers")
    wrappers.RecordVideo = RecordVideo
    gym = types.ModuleType("gymnasium")
    gym.wrappers = wrappers
    monkeypatch.setitem(sys.modules, "gymnasium", gym)
    monkeypatch.setitem(sys.modules, "gymnasium.wrappers", wrappers)
    return calls


def test_the_camera_is_aimed_before_recording_starts(fake_gymnasium, monkeypatch):
    mod = _load_wrapper()
    monkeypatch.setattr(mod, "aim_render_camera", lambda env: fake_gymnasium.append(("aim", env)))

    mod.install_camera_aim()
    sys.modules["gymnasium.wrappers"].RecordVideo("the-env", video_folder="v")

    assert fake_gymnasium == [("aim", "the-env"), ("record_video", "the-env")]


def test_policy_runner_runs_with_its_own_argv_and_the_patched_wrapper(
        fake_gymnasium, monkeypatch, tmp_path):
    mod = _load_wrapper()
    monkeypatch.setattr(mod, "aim_render_camera", lambda env: None)
    monkeypatch.setattr(sys, "argv", list(sys.argv))
    monkeypatch.setattr(sys, "path", list(sys.path))
    out = tmp_path / "seen.txt"
    runner = tmp_path / "policy_runner.py"
    runner.write_text(
        "import sys\n"
        "from gymnasium.wrappers import RecordVideo\n"
        "if __name__ == '__main__':\n"
        f"    open({str(out)!r}, 'w').write(RecordVideo.__name__ + ' ' + ' '.join(sys.argv))\n")

    mod.main([str(runner), "--video", "put_item_in_fridge_and_close_door"])

    assert out.read_text() == (
        f"RecordVideoWithAimedCamera {runner} --video put_item_in_fridge_and_close_door")
    assert sys.path[0] == str(tmp_path)


def test_eval_entry_routes_through_the_wrapper_only_when_recording():
    source = _EVAL_ENTRY.read_text()

    guard = source.index('os.environ.get("EVAL_RECORD_VIDEO"')
    insert = source.index("cmd.insert(1, VIDEO_RUNNER_ENTRY)")
    positional = source.index("cmd.append(task_name)")

    assert guard < insert < positional
    assert source.count("VIDEO_RUNNER_ENTRY)") == 1, "only the gated block may use the wrapper"
    # Index 1 is policy_runner.py; the wrapper takes it as its first argument.
    assert 'f"{ARENA_WORKSPACE}/isaaclab_arena/evaluation/policy_runner.py",' in source


def test_the_image_ships_the_wrapper_where_eval_entry_looks():
    source = _EVAL_ENTRY.read_text()
    assert '"ARENA_VIDEO_RUNNER_ENTRY", "/workspace/arena_video_runner.py"' in source
    assert ("COPY isaac-lab-arena-on-aws/entrypoints/eval/isaac_arena/gr00t/"
            "arena_video_runner.py /workspace/arena_video_runner.py") in _DOCKERFILE.read_text()

    sys.path.insert(0, str(_REPO / "scripts"))
    try:
        import build_arena_connector
    finally:
        sys.path.pop(0)
    assert any(m.endswith("gr00t/arena_video_runner.py")
               for m in build_arena_connector._required_zip_members("gr00t")), (
        "a file missing from the build zip never reaches CodeBuild and the COPY fails")
