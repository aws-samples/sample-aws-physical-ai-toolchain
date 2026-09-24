#!/usr/bin/env python
"""Aim the render camera at the task, then run Arena's policy_runner unchanged.

Used ONLY when rollout video is requested (EVAL_RECORD_VIDEO=true). Unrecorded runs
launch policy_runner directly, exactly as before.

WHY THIS EXISTS
---------------
The first recorded rollout was a valid 1305-frame mp4 in which the whole kitchen was a
few grey pixels. The env's camera config was right -- the run logged
``ViewerCfg(eye=(2.55,-2.08,2.52), lookat=<the ranch bottle>)`` -- but nothing ever
applied it to the camera being recorded:

- ``ManagerBasedRLEnv.render()`` in ``rgb_array`` mode reads a render product on
  ``cfg.viewer.cam_prim_path`` (``/OmniverseKit_Persp``).
- In the Isaac Lab this image pins (Arena's submodule ``e57379c6``),
  ``ViewportCameraController.update_view_location()`` computes the right eye/lookat and
  hands it to ``sim.set_camera_view()``, which only loops over ``sim._visualizers``.
- A headless run has none. Requesting one does not help: ``KitVisualizer`` skips viewport
  setup when headless and its ``_set_viewport_camera()`` returns early when there is no
  viewport. Arena's own ``reapply_viewer_cfg()`` goes through the same path.

So ``/OmniverseKit_Persp`` keeps Kit's default pose, far outside the scene. Nothing errors.
Upstream Isaac Lab later fixed this with ``set_kit_renderer_camera_view()``
(``isaaclab_physx/renderers/kit_viewport_utils.py``), which poses the renderer camera
independently of visualizers. This file is that one call, backported for the pinned image.

WHAT THIS DOES
--------------
Wraps ``gymnasium.wrappers.RecordVideo`` so that, when policy_runner constructs it (after
the env is built and seeded, before the first frame), the render camera is posed at the
eye/lookat the env's own ``ViewerCfg`` resolves to. The pose is authored straight onto the
camera prim in the session layer rather than through a viewport API, because a headless
app has no viewport to go through. No coordinate is hardcoded here: the task decides where
the camera goes, as it would in a GUI run.

The camera is posed once. Tracking origins (``asset_root``/``asset_body``) would need a
per-frame update; the Arena tasks this component evaluates use ``origin_type="env"``.
"""
from __future__ import annotations

import os
import runpy
import sys

import numpy as np

LOG_PREFIX = "[arena-video-runner]"
# Distinct suffix so the op never collides with an existing xformOp:transform of another
# precision on the camera prim.
XFORM_OP_SUFFIX = "rolloutVideo"


def camera_to_world(eye, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """USD camera transform (row-vector convention) placing a camera at ``eye`` looking at
    ``target``. A USD camera looks down its local -Z with +Y up; the stage is Z-up."""
    eye = np.asarray(eye, dtype=float)
    back = eye - np.asarray(target, dtype=float)
    norm = np.linalg.norm(back)
    if norm == 0.0:
        raise ValueError(f"camera eye and lookat coincide at {eye.tolist()}")
    back /= norm
    right = np.cross(np.asarray(up, dtype=float), back)
    if np.linalg.norm(right) == 0.0:
        raise ValueError(f"camera looks straight along the up axis from {eye.tolist()}")
    right /= np.linalg.norm(right)
    true_up = np.cross(back, right)
    matrix = np.eye(4)
    matrix[0, :3], matrix[1, :3], matrix[2, :3], matrix[3, :3] = right, true_up, back, eye
    return matrix


def viewer_pose(base_env) -> tuple[np.ndarray, np.ndarray]:
    """World eye/lookat the env's ViewerCfg asks for, resolved the way
    ViewportCameraController does (origin + configured offsets)."""
    vcc = getattr(base_env, "viewport_camera_controller", None)
    if vcc is not None and getattr(vcc, "viewer_origin", None) is not None:
        origin = np.asarray(vcc.viewer_origin.detach().cpu().numpy(), dtype=float)
        eye, lookat = vcc.default_cam_eye, vcc.default_cam_lookat
    else:
        cfg = base_env.cfg.viewer
        if cfg.origin_type == "env":
            origin = np.asarray(base_env.scene.env_origins[cfg.env_index].detach().cpu().numpy(),
                                dtype=float)
        elif cfg.origin_type == "world":
            origin = np.zeros(3)
        else:
            raise RuntimeError(
                f"viewer origin_type={cfg.origin_type!r} needs the viewport camera controller, "
                "which this env did not create; cannot resolve where the camera should be")
        eye, lookat = cfg.eye, cfg.lookat
    return origin + np.asarray(eye, dtype=float), origin + np.asarray(lookat, dtype=float)


def author_camera_pose(stage, camera_path: str, matrix: np.ndarray) -> None:
    """Write ``matrix`` as the camera prim's only xform op, in the session layer (where Kit
    defines ``/OmniverseKit_Persp``, and the strongest layer regardless)."""
    from pxr import Gf, Usd, UsdGeom

    prim = stage.GetPrimAtPath(camera_path)
    if not prim.IsValid():
        raise RuntimeError(f"render camera prim {camera_path} does not exist on the stage")
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        xformable = UsdGeom.Xformable(prim)
        xformable.ClearXformOpOrder()
        xformable.AddTransformOp(opSuffix=XFORM_OP_SUFFIX).Set(Gf.Matrix4d(matrix.tolist()))


def aim_render_camera(env) -> None:
    import omni.usd

    base_env = env.unwrapped
    camera_path = base_env.cfg.viewer.cam_prim_path
    eye, lookat = viewer_pose(base_env)
    author_camera_pose(omni.usd.get_context().get_stage(), camera_path,
                       camera_to_world(eye, lookat))
    print(f"{LOG_PREFIX} posed {camera_path}: eye={eye.round(3).tolist()} "
          f"lookat={lookat.round(3).tolist()}", flush=True)


def install_camera_aim() -> None:
    """Aim the camera whenever RecordVideo wraps an env. Must run before policy_runner is
    imported, since it binds ``from gymnasium.wrappers import RecordVideo`` at import."""
    import gymnasium.wrappers

    record_video = gymnasium.wrappers.RecordVideo

    class RecordVideoWithAimedCamera(record_video):
        def __init__(self, env, *args, **kwargs):
            aim_render_camera(env)
            super().__init__(env, *args, **kwargs)

    gymnasium.wrappers.RecordVideo = RecordVideoWithAimedCamera


def main(argv: list[str]) -> None:
    if not argv:
        raise SystemExit(f"usage: {os.path.basename(__file__)} <policy_runner.py> [args...]")
    policy_runner, runner_args = argv[0], argv[1:]
    install_camera_aim()
    # Match `python policy_runner.py ...`: its directory first on sys.path, its own argv.
    sys.path.insert(0, os.path.dirname(os.path.abspath(policy_runner)))
    sys.argv = [policy_runner] + list(runner_args)
    runpy.run_path(policy_runner, run_name="__main__")


if __name__ == "__main__":
    main(sys.argv[1:])
