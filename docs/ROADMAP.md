# Roadmap: 4 Proposed Features

Status: **code-complete for all 4 (to the hardware boundary).** Every no-hardware
slice is built, verified (cdk synth / parse / dry-run / one full TorchScript
round-trip), and committed. What remains for each is *runtime validation* that
genuinely requires a Jetson, a G-family GPU, or sustained p5 capacity — wired
correctly and labelled "unvalidated", not done.

### Status snapshot
| Feature | Code | Validated without HW | Hardware-gated remainder |
|---|---|---|---|
| Tier 0 bugfixes | ✅ | ✅ cdk synth (3 modes) | — |
| 1 — Robot deploy | ✅ ROS2 pkg, edge scripts, **TorchScript export helper** | ✅ dry-runs + scriptify round-trip | Jetson + UR3 run; object-pose perception |
| 2 — Standalone RL | ✅ launch_rl.py, Lab 4b, **UR3 env registered** | ✅ dry-run + bare-import | UR3 env instantiation on a GPU (Anymal path already proven) |
| 3 — Cosmos | ✅ v2 EC2/NIM runtime, **v3 generation runner** | ✅ dry-runs | p5 capacity + HF license; a real restyle/generate |
| 4 — Isaac Sim demo | ✅ open_demo_scene.py + wiring | ✅ synth + parse | one G-family workstation run |

The rest of this doc is the original scoping (kept for context); the items marked
"build" / "no-hardware slice" below are now done.

---

## Original scoping (for context)

This doc scoped four requested features with honest effort/risk/testability, based
on a code+external investigation.

The single most important finding: **each feature sits on top of pre-existing,
in-repo bugs that ship broken today and need no GPU/hardware to fix.** Do that
"Tier 0" sweep first — it's cheap, unblocks everything, and shrinks what later
(expensive, hardware-gated) validation has to cover.

A hard ceiling applies to all four: the load-bearing validation — Jetson inference,
p5/H100 Cosmos runs, Isaac Sim on a G-family GPU — **cannot be done on the dev
laptop or in plain CI.** Each feature therefore splits into a no-hardware slice
(code + mocked tests + docs, mergeable now) and a hardware-gated slice (explicitly
marked "unvalidated").

---

## Tier 0 — Bugfix sweep (do first; no hardware; ~days)

These are real defects found during investigation, independent of the features:

| Bug | Location | Fix |
|-----|----------|-----|
| Greengrass recipe ships a **literal** `'${ECR_INFERENCE_REPO_URI}:latest'` (inside a `JSON.stringify`, not a CDK token) → component pulls a non-existent image | `cdk/lib/edge-stack.ts:~177` | Use the real CDK token / `repository.repositoryUri` |
| Telemetry recipe references `{artifacts:path}/telemetry_collector.py` but defines **no Artifacts block** and the file is missing | `cdk/lib/edge-stack.ts` | Add Artifacts block + the script (see Feature 1) |
| CDK output describes the cosmos3 repo as **"mirrored from Docker Hub"** — but it's built from source | `cdk/lib/foundation-stack.ts` (cosmos3 output) | Fix the description string |
| Task-id mismatch: `PickAndPlaceUR3-v0` vs `PickAndPlace-UR3-v0` | env/config/docs | Pick one canonical id |
| False comment "registered as Isaac Lab task" (the UR3 env is **not** `gym.register()`-ed) | `containers/isaac-lab/Dockerfile`, `training/envs/` | Remove/realize the claim |
| `PLAN.md` internal contradictions (Labs "0-6 ✅" vs Lab docs 🔲; "200Hz ✅" overstates an unvalidated stub) | `PLAN.md`, `README.md` | Reconcile status to reality |

Land these as **small, independent PRs off `main`** — low risk, easy review.

---

## Feature 4 — Isaac Sim opens a demo scene (not an empty stage)

**Best value-to-effort of the actual features. Recommend doing first (after Tier 0).**

- **Today:** the workstation's `run-isaac-sim-gui.sh` runs a bare `isaacsim` → users
  connect via DCV and land on an empty stage.
- **Goal:** launch with a meaningful example (an NVIDIA-provided sample is fine).
- **Effort:** M · **Files:** `cdk/lib/workstation-stack.ts`, `workshop/lab-2-isaac-workstation.md`

**Must discover on a deployed G-family box before coding** (cannot be guessed):
- exact installed version: `pip show isaacsim`
- the real stage-open mechanism: `isaacsim --help` — **do NOT assume `--open-stage`/`--exec` exist or are spelled that way** (unverified).

**Critical correction from investigation:** the UR3 env's USD paths use
`omniverse://localhost/...` — a **local Nucleus server that is NOT installed** on the
workstation. Reusing those paths will fail to load assets. Use a bundled built-in
example, or a tiny inline Python scene against the version-correct NVIDIA **cloud**
asset root (reachability from EC2 is itself unverified).

**Constraints:** G-family GPU only (P4/P5 lack RT Cores → Isaac Sim crashes).
Cost figures in-repo conflict ($1.62/hr vs $4.53/hr) — reconcile against the actual
instance. The "isaac-lab:2.1.0 bundles Isaac Sim 6.0" claim is unsourced/likely false.

**Next action:** one short GPU discovery session (`pip show` + `--help`), then the
2-file change with graceful fallback to bare `isaacsim`.

---

## Feature 2 — Standalone Isaac RL demo (decouple from GR00T)

- **Today:** RL is packaged inside `training/scripts/groot_to_rl_bridge.py`, framed as
  GR00T-dependent. **In practice it already runs standalone** — this session an RL job
  ran on SageMaker with zero GR00T involvement and produced a real policy + rendered
  video (the built-in `Isaac-Velocity-Flat-Anymal-D-v0` task).
- **Goal:** a clean standalone `training/scripts/launch_rl.py` + a lab that needs no Lab 1.
- **Effort:** L (because of the UR3 gap below) · **Files:** `training/scripts/launch_rl.py` (new), `workshop/` (new/updated lab), `training/envs/__init__.py`

**No-hardware slice (mergeable):** extract `launch_rl.py --dry-run` (precedent:
`groot/launch_training.py`), remove the `groot-models` registry lookup from the standalone
path, edit the **real** entrypoint (`sm-train-entrypoint.sh`, not the dead `train_entrypoint.py`).

**The unsolved blocker (hardware + real risk):** the UR3 env (`pick_and_place_ur3.py`)
is **not registered**, so `parse_env_cfg('PickAndPlace-UR3-v0')` fails today. Even after
registration, the `omniverse://localhost` asset paths likely won't resolve in a headless
SageMaker container (no Nucleus). So "RL trains the **UR3** task" is multi-hour GPU
debugging with a real chance the env isn't instantiable. The standalone demo should ship
on the **proven Anymal task** and be honest that UR3 is unvalidated.

**Next action:** Phase 0 = register the env + reconcile the task id; deliver the
Anymal standalone demo first; treat UR3 as a separate, clearly-unvalidated effort.

---

## Feature 1 — Deploy to robot after training

- **Today:** building blocks exist (`export.py`, inference Dockerfile x86+jetson,
  ROS2 node, `edge-stack.ts`) but **nothing in the edge path has run end-to-end**
  (PLAN Phase 3 all 🔲), and several scripts the lab references are missing.
- **Goal:** a believable train → export → package → deploy (Greengrass/Jetson) → run flow.
- **Effort:** L (XL with hardware) · **Files:** `edge/*` (several new), `containers/inference/*`, `cdk/lib/edge-stack.ts`, `workshop/lab-5-edge-deployment.md`

**Missing on disk (verified):** `edge/deploy.sh`, `create_component.py`,
`deploy_to_fleet.py`, `provision_device.py`, `telemetry_collector.py`; ROS2 package
files (`package.xml`/`setup.py`) — so `colcon build` / `ros2 run` don't work (the node
only runs today because `entrypoint.sh` calls the `.py` by path).

**Other real issues:**
- `export.py` uses `torch.jit.load` → needs a **TorchScript** checkpoint; confirm what
  `train.py` emits or export fails at load. The ">100Hz" is a print-time assertion, not
  a measured result.
- Lab 5 is a placeholder with **wrong commands** (`--output/--format/--precision/--target`
  vs the real `--output-onnx/--output-trt/--target-device/--fp16`) and wrong ROS topics/rate.
- Object-pose input is a **hard-coded zero placeholder** — even a perfect deploy feeds the
  policy garbage for that channel (no perception source).
- ECR IAM perms *do* exist on the Greengrass role; the real gap is **credential delivery
  on-device** (Greengrass Token Exchange Service + ECR credential helper), undocumented.

**No-hardware slice (mergeable):** ROS2 package files, the missing edge scripts with
`--dry-run`, a moto/Stubber mock-AWS harness, and a corrected Lab 5.
**Hardware-gated (undeliverable without a Jetson Orin + UR3 + camera):** TensorRT engine
build, real Greengrass deployment, on-device pull, live inference, sim-to-real.

**Next action:** fix the `edge-stack.ts` recipe literal + add ROS2 package files first,
then scaffold the scripts with mocked tests; scope the on-robot part as code+docs only.

---

## Feature 3 — Cosmos: NIM/runtime v2 → v3

- **Today:** `cosmos_setup.py` describes Cosmos **Transfer 2.5** as a SageMaker real-time
  endpoint — **known broken** (SageMaker GPUs ship driver 470; Cosmos needs 580+). The
  only path that ever ran was a hand-launched EC2 p5 (`docs/cosmos-deployment-guide.md`),
  and even that **timed out at inference** (93 frames/5 steps > 10 min on CP=1) — only the
  health endpoint passed.
- **Build status (this session, ground truth):** both images built successfully in
  CodeBuild and are in ECR — `physical-ai/cosmos-transfer:latest` (2.5, from source) and
  `physical-ai/cosmos3:latest` (cosmos-framework `main`, CUDA 13 base, Python 3.13).
- **Goal:** ambiguous — "move runtime to v3" has no PLAN task. **Confirm intent first.**
- **Effort:** L · **Files:** `training/scripts/cosmos_setup.py`, `scripts/cosmos-userdata.sh`, `containers/cosmos3/*`, `workshop/lab-3*.md`, `docs/cosmos-deployment-guide.md` (+ near-duplicate `.kiro/steering/cosmos-deployment.md`)

**Hard constraint (verified this session):** Cosmos 3 does **not** do controlled
sim-to-real *transfer* (edge/depth/seg/blur control) yet — only generation
(text2image/video2video). Cosmos **Transfer 2.5** remains the only path for the
toolkit's actual transfer use case. So "v3" cannot replace v2 for transfer.

**Recommendation:** keep **v2 = transfer**, add **v3 = generation (opt-in)**; do not
remove v2. Phase 0: resolve the **NGC_API_KEY vs HF_TOKEN** contradiction (deploy code
uses NGC; build comments say HF — a load-bearing conflict), fix the CDK mirror-description
bug, and rewrite `cosmos_setup.py` off the broken SageMaker-endpoint model toward the EC2
reality. Reconcile Lab 3's three mismatched API descriptions (`ai.api.nvidia.com`,
SageMaker deploy, real `/v1/infer`).

**Testability:** p5 Spot + **P5 quota defaults to 0** (entitlement lead time, days);
the one real run timed out, so a smoke test may too. Realistic GPU-validation cost ~$30–80,
gated on quota — not the optimistic in-repo figures.

**Next action:** confirm what "v3" should mean with the requester; do the Phase 0
contradiction fixes (no GPU); defer runtime work until intent + quota are sorted.

---

## Recommended sequencing

1. **Tier 0 bugfix sweep** — small PRs, no hardware, unblocks everything.
2. **Feature 4** (Isaac Sim demo) — cleanest win; needs one short GPU discovery session.
3. **Feature 1** (robot deploy) — no-hardware scaffolding + mocks; defer Jetson tail.
4. **Feature 2** (standalone RL) — ship Anymal standalone; UR3 is a separate unvalidated effort.
5. **Feature 3** (Cosmos v3) — Phase 0 fixes now; runtime deferred pending intent + p5 quota.

## Cross-cutting

- `foundation-stack.ts` is touched by Features 1 & 3; `PLAN.md` by all four (reconcile once).
- `training/envs/` is shared by Features 2 & 4 (UR3 registration + the `omniverse://localhost`
  asset problem — fix once, benefits both).
- G-family-GPU-only constraint must appear in every new lab/launcher.
- Keep CDK changes in separate PRs with `cdk synth` snapshot assertions (incl. a test that
  the inference recipe renders a **real** ECR URI — catches the literal-string regression).
- Any "container builds" claim (cosmos3, the inference image with `ros-humble-ur-robot-driver`
  on a TensorRT/L4T base) should be confirmed by an actual CodeBuild run before merging
  dependent docs. (cosmos3 + cosmos-transfer + inference x86/jetson were confirmed building
  this session.)
