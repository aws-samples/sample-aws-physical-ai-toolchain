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

## A different robot?

The convention for a non-UR3 robot is GR00T's `NEW_EMBODIMENT` embodiment tag plus
a modality config naming your state/action/video keys. Start from
`containers/groot-training/ur3_modality_config.py` (which mirrors NVIDIA's own
`examples/SO100/so100_config.py`) and adjust `modality_keys` + the `meta/modality.json`
index ranges your converter writes. The numeric ranges live in `meta/modality.json`,
not in the Python config.
