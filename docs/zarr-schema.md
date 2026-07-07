# Bring Your Own Data — Zarr episode schema

Lab 1 ships 27 real UR3 pick-and-place demonstrations. To train GR00T on **your
own** teleoperation data, record it in the Zarr layout below, then run the
ingestion pipeline (`training/groot/ingest_customer_data.py`) — it converts your
Zarr episodes to the LeRobot v2 format GR00T reads, uploads to S3, and (optionally)
launches training.

This schema is exactly what `training/groot/convert_zarr_to_lerobot.py` consumes —
it is not aspirational. If your recordings match it, the same conversion applies.

## Directory layout

```
<episodes-dir>/
├── episode_001_<task_slug>/
│   ├── zarr.json                      # group attrs (see below) — zarr v3
│   ├── observations/
│   │   ├── joints                     # (N, 6) float, radians — UR3 joint angles
│   │   ├── gripper_position           # (N,)  int 0..255 — gripper opening
│   │   ├── timestamps                 # (N,)  float, epoch seconds — telemetry clock
│   │   ├── tcp_pose                   # (optional) (N, 6) — not used by the converter
│   │   └── force_torque               # (optional) — not used by the converter
│   ├── images/
│   │   ├── wrist                      # (M, H, W, 3) uint8 RGB — wrist camera frames
│   │   └── wrist_timestamps           # (M,) float, epoch seconds — camera clock
│   └── commands.json                  # list of {"t": <epoch_s>, "prog": "<URScript>"}
├── episode_002_<task_slug>/
└── ...
```

- **Two clocks.** Telemetry (`observations/timestamps`, ~10 Hz) and camera
  (`images/wrist_timestamps`, ~5 Hz) run independently. The converter aligns each
  camera frame to the nearest telemetry sample, so the output is at camera rate.
- **`N` = telemetry samples, `M` = camera frames.** They differ; that's expected.

## `zarr.json` attributes (root group)

```json
{
  "attributes": {
    "task_name": "pick_up_the_red_block...",        // becomes the LeRobot task string
    "task_description": "pick up the red block...",  // human-readable (optional)
    "camera_hz": 5,                                   // output dataset fps
    "telemetry_hz": 10,
    "schema_version": 1
  },
  "zarr_format": 3,
  "node_type": "group"
}
```

Only `task_name` and `camera_hz` are load-bearing for conversion (`camera_hz`
defaults to 5 if absent). The rest are metadata.

## Actions: `commands.json`

Each entry is a timestamped UR controller command. The converter parses URScript:

- `speedl([vx, vy, vz, rx, ry, rz], a, t)` → 6D Cartesian velocity. The converter
  multiplies by the per-step `dt` (1/`camera_hz`) to get position **deltas**.
- `stopl` / `stopj` → zero velocity.
- `socket_set_var("POS", <0..255>)` → gripper position (normalized to 0..1).

```json
[
  {"t": 1776807909.87, "prog": "def cmd():\n  speedl([0,0,-0.035,0,0,0],1.0,0.12)\nend\n"},
  {"t": 1776807910.10, "prog": "def cmd():\n  socket_set_var(\"POS\", 255)\nend\n"}
]
```

If `commands.json` is absent, the converter falls back to joint-position deltas
between consecutive telemetry samples (lower fidelity — commanded velocities are
preferred).

## Resulting GR00T dimensions

The converter emits a 7D state and 7D action (see `meta/modality.json`):

| Field   | Dims  | Source                                              |
|---------|-------|-----------------------------------------------------|
| state   | `[0:6]` arm + `[6:7]` gripper | 6 UR3 joint angles + normalized gripper |
| action  | `[0:6]` arm + `[6:7]` gripper | 6 Cartesian velocity·dt deltas + gripper |
| video   | `wrist` | wrist camera, encoded to MP4 |

These ranges are wired into the GR00T modality config
(`containers/groot-training/ur3_modality_config.py`). **If your robot is not a
6-DOF arm + 1-DOF gripper, you must update both** `convert_zarr_to_lerobot.py`
(STATE_DIM/ACTION_DIM + the feature names) **and** that modality config so the
index ranges match.

## Porting to a different arm

The pipeline splits cleanly into two halves with very different porting cost. Be
honest with yourself about which one you're touching:

| Half | Stages | Coupled to UR3? | Effort |
|------|--------|-----------------|--------|
| **Cloud ML** | convert → train → serve | Only via dimensions/keys | **Easy** — edit a few constants |
| **Hardware** | teleop capture, on-arm control | Deeply (protocol, packets, gripper) | **Hard** — write a new controller |

### Half 1 — the cloud ML path (easy)

GR00T is embodiment-agnostic: a custom robot is just the `NEW_EMBODIMENT` tag plus a
modality config naming your state/action/video keys. If your data already matches the
Zarr schema above (or you produce the LeRobot v2 output some other way), adapting the
ML path is a handful of edits:

1. **`training/groot/convert_zarr_to_lerobot.py`** — set `STATE_DIM` / `ACTION_DIM`,
   the `meta/modality.json` index ranges (`arm`/`gripper` start:end), and the joint
   `names` list to your robot.
2. **`containers/groot-training/ur3_modality_config.py`** — adjust `modality_keys`
   and the per-key `action_configs` to match. (This is the canonical embodiment config;
   `train_entrypoint.py` writes an identical one inline — keep them in sync.)
3. **`training/groot/deploy_endpoint.py`** — the endpoint reads dims from env vars
   (`GROOT_STATE_DIM`, `GROOT_ACTION_DIM`, `GROOT_CAMERAS`), so set those to your robot's
   values. Normalization is already generic (driven by the checkpoint's stats file).

Start from `ur3_modality_config.py`, which mirrors NVIDIA's own
`examples/SO100/so100_config.py`. The numeric ranges live in `meta/modality.json`, not
the Python config.

### Half 2 — the physical robot path (hard, and inherently so)

There is no universal robot-arm protocol, so the hardware layer is genuinely
arm-specific — this is not a shortcoming of this code, it's true of any robot toolkit.
`robot/ur3/safe_controller.py` speaks UR's **URScript** over TCP and parses UR
**CB-series binary state packets** at fixed byte offsets; the gripper driver
(`robot/ur3/robotiq_gripper_control.py` + `robotiq_preamble.py`) is Robotiq-specific;
and the converter parses UR `speedl(...)` commands out of `commands.json`. A Franka
(`libfranka`), a UR e-series (different packet offsets), or a different gripper each
need a **new controller** written against that arm's SDK — not a config change.

The good news: the teleop and control entrypoints depend on a **small, well-defined
method surface**, not on UR internals. Implement a class exposing this contract
(against your arm's SDK) and `pai groot record` / `pai groot control` work unchanged:

```python
class SafeRobotController:            # your arm; mirror robot/ur3/safe_controller.py
    def __enter__(self) / __exit__(self)      # used as `with SafeRobotController(ip) as robot:`
    def connect(self) / disconnect(self)
    def get_joints(self) -> list[float]        # current joint angles (radians)
    def get_tcp_pose(self) -> list[float]      # [x,y,z,rx,ry,rz]
    def get_force(self) -> list[float]         # [fx,fy,fz,...] for the safety stop
    def move_joints(self, angles, vel=..., accel=...)
    def gripper_open(self) / gripper_close(self)
```

You will also need a teleop **command log** in `commands.json`: either emit UR-style
`speedl`/gripper URScript (then the existing converter parses it untouched), or record
actions in your own format and adjust `_parse_commands` in
`convert_zarr_to_lerobot.py` to match. Then re-point `gamepad_teleop.py` /
`keyboard_teleop.py` / `control.py` at your controller class and retune the safety
bounds (`BoundsConfig`) and `START_JOINTS_DEG` home pose for your arm's workspace.

**Bottom line:** if you already have demonstration data (or another way to produce the
Zarr/LeRobot format), reaching a trained, deployable policy on a new arm is easy. Making
*this repo's* teleop-capture and on-arm-control loop drive a non-UR arm requires a new
controller implementation — plan for that, don't expect a flag.
