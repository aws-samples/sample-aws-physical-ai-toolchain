# Lab 1b: Close the Loop on a Physical UR3 (Optional Hardware Track)

**Goal:** Run the *complete* end-to-end loop on a real robot — teleoperate a UR3 to record your own demonstrations, fine-tune GR00T on them (Lab 1), deploy the model, then let it drive the arm autonomously.
**Time:** 1–2 hours hands-on (plus the Lab 1 training run)
**Cost:** Same as Lab 1 (training + endpoint) — the robot itself is yours

> ## 🤖 Hardware is optional
> **You do not need a robot to complete this workshop.** Labs 0–6 run entirely in
> the cloud on the 27 bundled demonstrations. This lab is the **bring-your-own-robot
> track**: it documents how the pieces connect to *physical* hardware for teams that
> have a UR3. If you don't, read it as the reference for what "end-to-end on a real
> robot" looks like — every cloud step is identical.

---

## Where this fits

Lab 1 trains and serves a policy using demonstration data that was already recorded.
This lab adds the two hardware-facing bookends around it:

```
        ┌─────────────────────────── Lab 1b (this lab, needs a UR3) ───────────────────────────┐
        │                                                                                       │
   ┌─────────┐      ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌─────────────┐
   │ teleop  │─────▶│  convert │────▶│  upload  │────▶│fine-tune │────▶│  deploy  │────▶│  closed-loop│
   │ record  │ Zarr │ (Lab 1)  │     │ (Lab 1)  │     │ (Lab 1)  │     │ (Lab 1)  │     │  control    │
   │ (record)│      └──────────┘     └──────────┘     └──────────┘     └──────────┘     │  (control)  │
   └─────────┘                    ── cloud, no hardware needed ──                       └─────────────┘
        │                                                                                       │
        └───── pai groot record ──────────────────────────────────────── pai groot control ────┘
```

The middle four steps are exactly Lab 1. This lab covers `pai groot record` (front)
and `pai groot control` (back), which need a physical arm + wrist camera.

---

## Prerequisites

- **Completed [Lab 0](lab-0-prerequisites.md)** (`pip install -e .`, Foundation stack deployed).
- **A Universal Robots UR3** (CB-series or e-Series) reachable over the network, with
  the **URScript port (30002)** and **dashboard port (29999)** open.
- **A wrist-mounted camera** — an Intel RealSense D405 (native support) or any
  UVC/RTSP camera (OpenCV fallback). Only the RGB stream is used.
- **A Robotiq 2F-85 gripper** (or adapt `robot/ur3/robotiq_gripper_control.py`).
- Optional but recommended: a **game controller** (Xbox/PS) for the gamepad teleop UI.

Set the arm's address once:

```bash
export ROBOT_IP=192.168.1.100    # your UR3's IP; 127.0.0.1 targets a local URSim
```

> **Safety first.** The control loop *moves the arm*. Clear the workspace, keep the
> teach-pendant e-stop within reach, and start with slow/small motions. The bundled
> `SafeUR3Controller` clamps joint/workspace/velocity limits and trips on force, but
> it is not a substitute for the physical e-stop.

---

## Step 1: Record demonstrations by teleoperation

`pai groot record` opens a teleop session and writes each demonstration into the
Zarr layout that `pai groot convert` reads — so recorded data flows straight into
the Lab 1 pipeline with no extra conversion.

```bash
# Gamepad (opens a browser UI that reads an HTML5 game controller):
pai groot record --task "pick up the red cube" --mode gamepad

# Or keyboard (WASD + IJKL; works over an SSH tunnel, no browser/gamepad needed):
pai groot record --task "pick up the red cube" --mode keyboard
```

**Controls (gamepad):** left stick = X/Y, right stick = Z + wrist, triggers =
gripper, **A** = start/stop recording, **B** = emergency stop.
**Controls (keyboard):** `WASD` = X/Y, `I/K` = Z, `J/L` = wrist, `Q/E` = gripper,
`R` = record, `X` = e-stop.

Each recording lands in `training/data/episodes/episodes/episode_NNN_<task>/` as a
Zarr store (joint angles + gripper + wrist frames + the URScript command log). Record
**50–100 demonstrations** with natural variation (different object positions, minor
lighting changes) for a policy that generalizes.

```bash
# Preview the command without moving anything:
pai groot record --task "pick up the red cube" --dry-run

# Sanity-check what you captured:
python -m robot.ur3.recorder list
```

> **Schema:** the recorder writes exactly the layout in
> [docs/ZARR_SCHEMA.md](../docs/ZARR_SCHEMA.md). If you record with your own tools
> instead, match that schema and the rest of the pipeline is unchanged.

---

## Step 2: Train, deploy — run Lab 1

Your recorded episodes are now in the same place Lab 1 expects. Run
[Lab 1](lab-1-train-groot.md) from **Step 2 (Convert)** onward — nothing changes:

```bash
pai groot convert                      # Zarr → LeRobot v2 (reads your recordings)
pai groot upload                       # → S3
pai groot launch --max-steps 5000      # fine-tune on SageMaker
# ... wait for training, then deploy (Lab 1 Step 9):
pai groot deploy --model-s3 "$MODEL_S3" --endpoint-name groot-ur3
```

When the endpoint is `InService`, come back here to close the loop.

---

## Step 3: Run the policy on the robot (closed-loop control)

`pai groot control` captures the wrist camera, asks the endpoint for an action
chunk, executes the first few actions on the arm, then re-queries — receding-horizon
control at the rate the model trained on (5 Hz).

```bash
# Preview (no motion, no endpoint call):
pai groot control --task "pick up the red cube" --endpoint-name groot-ur3 --dry-run

# Run it for real — THE ARM WILL MOVE:
pai groot control --task "pick up the red cube" --endpoint-name groot-ur3 --max-queries 20
```

What happens each cycle:

1. Grab a wrist frame + read the 7D robot state (6 joints + gripper).
2. POST them to the endpoint → get a 16-step action chunk
   (`[vx, vy, vz, rx, ry, rz, gripper]` per step).
3. Execute the first `execute_horizon` (4) steps via URScript `speedl`, clamped by
   the safety controller.
4. Re-query and repeat until the task completes or `--max-queries` is hit.

Add `--save-images` to dump the wrist frames the policy saw to `/tmp` for debugging.

---

## ✅ Lab 1b Checkpoint

- [ ] Where do recorded episodes land, and why does `pai groot convert` pick them up
      with no extra flags? (`training/data/episodes/episodes/` — the converter's default input)
- [ ] What are the two teleop modes and when do you use each? (gamepad = browser +
      controller on the local network; keyboard = SSH-tunnel-friendly)
- [ ] How does the control loop decide what to send the arm? (endpoint returns a
      16-step chunk; the loop executes the first 4, then re-queries — receding horizon)
- [ ] Which two commands need physical hardware, and which don't? (`record` +
      `control` need the arm; `convert`/`upload`/`launch`/`deploy` are cloud-only)

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `No UR3 responding at <ip>:30002` | Check `ROBOT_IP`, that the arm is powered and in remote-control mode, and that port 30002 is reachable (`nc -vz $ROBOT_IP 30002`) |
| Browser teleop UI doesn't see the gamepad | Press a button to activate the HTML5 Gamepad API; some browsers require the page to be focused |
| Camera not found | RealSense: check USB + `pyrealsense2`; otherwise set `CAMERA_WRIST` to an RTSP URL or device index (OpenCV fallback) |
| Arm trips a protective stop mid-run | Force limit exceeded — clear obstructions, reduce speed, recover via the dashboard/pendant |
| Endpoint calls fail | Confirm the endpoint is `InService` and in the region `pai` resolves (`pai groot deploy` output); tear down with `pai groot delete` when done |
| Motions look erratic | Usually too few / low-variety demos — record more (50–100) covering the object positions you expect |

---

**Previous:** [← Lab 1: Train from Demonstrations](lab-1-train-groot.md)
**Next:** [Lab 2: Isaac Sim Workstation →](lab-2-isaac-workstation.md)
