# Feature 1: SageMaker multi-instance RL unlock

**Why quick:** the in-container `containers/isaac-lab/sm-train-entrypoint.sh` ALREADY
parses `/opt/ml/input/config/resourceconfig.json` and launches torchrun with
`--nnodes/--node_rank/--rdzv_endpoint` (NCCL auto). The ONLY blocker is that
`training/scripts/launch_rl.py` hardcodes `ResourceConfig.InstanceCount: 1`.

## Changes
### `training/scripts/launch_rl.py` (modify)
- Add CLI flag `--instance-count` (int, default 1).
- Pass it into the job request: `ResourceConfig.InstanceCount = args.instance_count`.
- When `instance_count > 1`, print a one-line note that this is multi-node NCCL and
  is **unvalidated on hardware** (honesty rule).
- Keep `--dry-run` making ZERO AWS calls (it just prints the request). Verify the
  printed request shows the new InstanceCount.
- No change needed to the entrypoint (it already handles N nodes). Add a code comment
  pointing at sm-train-entrypoint.sh so the linkage is discoverable.

### `tests/` (add)
- Extend the existing launch_rl dry-run test (or add one) asserting:
  - `--instance-count 2 --dry-run` makes zero AWS calls (reuse the fake boto3 stub).
  - The printed/returned request contains `InstanceCount == 2`.

## Rebuild/deploy matrix
| Item | Container rebuild? | cdk deploy? |
|---|---|---|
| launch_rl.py flag | No | No |
| test | No | No |

## Notes / gotchas
- SageMaker sets up inter-node networking automatically for multi-instance training
  jobs (same security context); no SG work needed (unlike Batch).
- MaxRuntimeInSeconds already set; leave as-is.
- This is the **validated-path-adjacent** option: structurally proven container,
  only the launcher count changes. Convergence across nodes still unvalidated — label it.
