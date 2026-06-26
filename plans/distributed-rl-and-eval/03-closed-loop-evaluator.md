# Feature 3: Closed-loop evaluator (ZMQ policy-server ↔ Isaac Lab)

Models the hi-space LeIsaac pattern: a policy server and a sim client talk over a
transport; the sim drives the robot step-by-step using the policy's actions, and we
measure a binary success rate over N rounds. Runs ON THE LAB 2 WORKSTATION
(g6e.4xlarge L40S). Authored as **Lab 4 Step 4**. Ships LABELLED
"unvalidated until run on the Lab 2 workstation GPU" (honesty rule).

## New modules
### `training/scripts/eval_protocol.py` (NEW)
- Message schema + transport abstraction so tests use an in-process fake (no sockets).
- `encode_obs(obs)->bytes`, `decode_obs`, `encode_action(action)->bytes`,
  `decode_action`. v1 body = JSON (note: ~150KB/msg for 12308 floats — fine for
  episodic eval, NOT realtime; swap to msgpack later behind same signatures).
- `Transport` ABC with `.send(obs)->action`. Two impls:
  - `ZmqTransport(endpoint)` (REQ side) — lazy-import pyzmq.
  - `InProcessTransport(policy_fn)` — calls a python policy fn directly (for tests).

### `training/scripts/eval_policy_server.py` (NEW)
- Loads checkpoint: `resolve_checkpoint` (S3→local download, mirror train.py's S3 +
  isaacsim-guard patterns), then `torch.jit.load` (TorchScript ONLY — a raw rsl_rl
  `model_*.pt` fails; surface a LOUD error telling user to run scriptify first).
- ZMQ REP loop: recv obs → `policy(obs)` → send action. v1 returns single-step action
  (`action[0]` if a horizon is produced). N-step horizon is a forward-compat seam,
  explicitly NOT implemented/claimed.
- CLI: `--checkpoint`, `--endpoint` (default tcp://*:5555), `--device`.

### `training/scripts/eval_sim_client.py` (NEW)
- Lazy-import isaacsim/Isaac Lab INSIDE functions (so importing the module on a laptop
  doesn't crash — mirror train.py guard).
- Init env from `training/envs` registration (TASK_ID). For each of `--eval-rounds`:
  reset → loop {obs → transport.send(obs) → action → env.step} until term/trunc →
  record success.
- **Success detection (RISK):** env doesn't write `info['success']`; it's a
  termination term (`_check_success`, obj z>0.15). Use fallback: prefer
  `info['success']`, else `terminated and not truncated`, else object-height probe if
  exposed. Document ambiguity; keep both paths.
- **Obs flattening (RISK):** handle BOTH dict obs (flatten in export.py's order:
  joint_pos, joint_vel, gripper, object_pose, camera → 12308) AND a pre-flattened
  policy tensor (manager-based envs often return this). Mirror evaluate.py:94-112 branch.
- CLI: `--task`, `--endpoint`, `--eval-rounds`, `--output-dir`, `--max-steps`.

### `training/scripts/eval_mock_env.py` (NEW)
- Tiny gym-like stub (no isaacsim/GPU): fixed obs dim, scripted reward/termination so a
  known policy yields a known success rate. For tests.

### `training/scripts/evaluate.py` (modify)
- Make isaacsim import lazy. Add `--closed-loop` flag: orchestrates client (+ optional
  in-thread server convenience mode, LABELLED unvalidated — two-terminal/separate-process
  is the documented safe default).
- Preserve the existing open-loop path + JSON metrics SCHEMA
  (`success_rate_pct`, `num_episodes`/rounds, `avg_reward`, `avg_cycle_time_sec`,
  `failure_modes{timeout,drop,collision}`) for backward compat.

### `training/requirements.txt` (modify)
- Add `pyzmq>=25` (pure pip, CI-safe).

## Tests — `tests/test_eval_closed_loop.py` (NEW)
- Use `InProcessTransport` + `eval_mock_env` + a canned policy fn (no GPU/sockets):
  - ZMQ message round-trip via encode/decode (and optionally a real loopback socket).
  - Eval-loop control flow over mock env.
  - Success-rate math: e.g. 7/10 scripted successes → 70.0.
- Extend `tests/conftest.py` `fake_boto3` with an `s3` branch (for `resolve_checkpoint`
  dry path). Reuse existing `run_script`/import-fresh patterns.

## Docs — `workshop/lab-4-rl-refinement.md` (modify Step 4)
- Rewrite Step 4 as the closed-loop evaluator that RUNS ON THE LAB 2 WORKSTATION:
  start the Lab 2 box, scriptify the checkpoint, run policy server + sim client (two
  terminals), read success rate. Fix the current fictional JSON example to match the
  real schema. Add the "unvalidated until run on the Lab 2 GPU" label.

## Rebuild/deploy matrix
| Item | Container rebuild? | cdk deploy? |
|---|---|---|
| All eval_*.py, evaluate.py, tests, requirements, docs | No | No |

## Open risks (also in overview)
1. Whole Isaac path GPU-only & currently unproven (legacy `omni.isaac.lab.*` namespace,
   USD paths). Label unvalidated.
2. `info['success']` not guaranteed → fallback success detection.
3. Obs order/scale & dict-vs-flattened tensor — GPU-only verifiable.
4. TorchScript-only checkpoint (raw .pt fails) — loud error + doc.
5. Action horizon out of scope for v1 (single-step only).
6. JSON transport size/latency (not the edge transport) — noted.
7. In-thread server mode vs Isaac GIL — separate-process is the safe default.
