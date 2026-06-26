# Phase 2 — Minimal script refactors (make functions cleanly importable)

Goal: let the CLI call core logic as functions WITHOUT going through argparse.
Keep every `main()` working so existing docs and `run-path-a.sh` are unaffected.
Independent of Phase 1.

## 2.1 `training/scripts/evaluate.py` — extract `run_open_loop(...)`
- Extract the open-loop body (≈ lines 112–254) into:
  `run_open_loop(checkpoint, num_episodes, render_video, output_dir, env, seed) -> dict`
  returning the metrics dict (the existing JSON schema:
  `success_rate_pct, num_episodes, avg_reward, avg_cycle_time_sec,
  failure_modes{timeout,drop,collision}`).
- `main()` parses argv then calls `run_open_loop(...)` — identical behavior to today.
- Keep the lazy isaacsim import INSIDE `run_open_loop` (do not hoist to module top).
- The closed-loop branch already delegates to `eval_sim_client.run_eval` — leave it.
- Effort: MEDIUM (~10 lines of restructuring). Add/keep a test that `main()` still
  produces the same JSON.

## 2.2 `training/scripts/eval_policy_server.py` — extract `serve(...)` (optional but recommended)
- Extract `serve(checkpoint, endpoint, device)` from `main()` (≈ lines 96–143) so the
  CLI can call it directly instead of relying on argv + `sys.exit`.
- `main()` calls `serve(...)`. Preserve the localhost-only default endpoint
  (`tcp://127.0.0.1:5555`) and the non-localhost bind warning, and the TorchScript
  loud-error + torch.jit.load trust warning.
- Effort: LOW. If skipped, the CLI's `pai eval serve` calls `main()` and tolerates its
  exit behavior — but extracting is cleaner for error handling.

## 2.3 `launch_rl_batch.py` — NO code change
- `launch()` does not default `job_queue`/`job_definition` (None → caller must
  resolve). The CLI replicates `main()`'s defaulting by calling the module's
  `_default_queue()` / `_default_job_def()` when the user omits the flags. Document
  this in the subcommand (Phase 3), no edit here.

## 2.4 `scriptify_policy.py` — extract `scriptify(...)` (optional, only if `pai scriptify` ships in v1)
- Extract `scriptify(...)` from `main()`. LOW effort. If deferred, `pai scriptify`
  calls `main()` via argv shim, or drop the subcommand from v1.

## Do NOT touch
- `launch_rl.py`, `export.py`, `eval_sim_client.py`, `eval_protocol.py`,
  `cosmos_setup.py`, `cosmos3_generate.py` — already cleanly importable/callable.
- Do NOT centralize `_account/_role_arn/_isaac_lab_image/_bucket` in this phase
  (known duplication; out of scope — the CLI calls `launch()` which resolves them).
- Do NOT add `__init__.py` to `training/` / `training/scripts/` — we use the
  `sys.path`-prepend shim (Phase 1.5) to match existing test conventions. (Adding
  package init files would change every sibling import and the test harness.)

## Acceptance
- `evaluate.main()` output JSON unchanged (regression test).
- `from evaluate import run_open_loop` works with `training/scripts` on sys.path and
  does NOT import isaacsim at import time.
- All existing tests still pass (`pytest tests/ -q` → same 28 passed / 2 skipped).
