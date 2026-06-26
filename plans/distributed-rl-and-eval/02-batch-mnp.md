# Feature 2: AWS Batch Multi-Node Parallel (MNP) RL

Mirror the hi-space workshop's Batch MNP design using THIS repo's conventions.
Reuses the SAME `physical-ai/isaac-lab` image already built by CodeBuild.

## B1. `containers/isaac-lab/batch-train-entrypoint.sh` (NEW) — needs container rebuild
Translates Batch MNP env vars → torchrun, parallel to the SM entrypoint (do NOT
modify sm-train-entrypoint.sh — it's the validated path).
- Read `AWS_BATCH_JOB_NUM_NODES`, `AWS_BATCH_JOB_NODE_INDEX`,
  `AWS_BATCH_JOB_MAIN_NODE_INDEX`, `AWS_BATCH_JOB_MAIN_NODE_PRIVATE_IPV4_ADDRESS`.
- Compute `--nnodes=$NUM_NODES --node_rank=$NODE_INDEX --nproc_per_node=$PROC_PER_NODE`
  `--rdzv_endpoint=$MAIN_IP:29500 --rdzv_backend=c10d`.
- `export NCCL_SOCKET_IFNAME=eth0` (RISK: some AMIs use ens5 — consider leaving unset
  to let NCCL autodetect, or detect at runtime). Document the choice.
- Hyperparameters from ENV (TASK, NUM_ENVS, MAX_ITERATIONS, PROC_PER_NODE) — NOT from
  resourceconfig.json (that's SM-only).
- Symlink skrl/rsl_rl output dir → `/efs/models/<job>` so checkpoints + tensorboard
  persist to EFS.
- Set shm via container (see job def); reference `/dev/shm` for physics data exchange.

## B2. `containers/isaac-lab/Dockerfile` (modify) — needs container rebuild
- `COPY containers/isaac-lab/batch-train-entrypoint.sh /opt/ml/code/batch-train`
- `chmod +x`. Do NOT change the default CMD (keep SM `train` default); Batch job def
  overrides command to call `batch-train`.
- NOTE: adding a file to the container changes the CodeBuild S3 source-asset hash, so
  deploying Foundation retriggers the isaac-lab build automatically.

## B3. `cdk/lib/constructs/batch-rl.ts` (NEW construct)
Props: `{ vpc, projectName, environment, instanceType (default g6.12xlarge),
isaacLabRepo, efsFileSystem, maxvCpus }`.
- **Compute environment** (MANAGED, EC2): instance class g6.12xlarge (4× L4), private
  subnet, launch template with ECS-optimized **GPU** AMI + larger EBS (250GB gp3),
  instance profile (S3 / ECS / EFS / SSM).
- **Security group:** self-referencing inbound ALL (NCCL inter-node), plus NFS 2049
  for EFS.
- **Job queue** → the compute env. Name `${project}-${env}-rl-queue`.
- **MNP job definition** name `${project}-${env}-rl-mnp`:
  - `numNodes` (default 2), single node-range group, all nodes same container.
  - Container: the isaac-lab ECR image; request all 4 GPUs + near-all vCPUs/mem per
    node so Batch places ONE container per instance (don't let it pack 2 ranks/box).
  - `command: ["batch-train"]`; env: TASK, NUM_ENVS, MAX_ITERATIONS, PROC_PER_NODE=4,
    ACCEPT_EULA=Y, NCCL_SOCKET_IFNAME.
  - EFS volume + mount point `/efs` (PREFER Batch-native EfsVolume on the job def;
    drop UserData mount to avoid double-mount). Shared-memory size via linuxParameters.
  - timeout 3600s.
- **RISK:** verify `aws-batch` L2 props (`EfsVolume`, `LinuxParameters.sharedMemorySize`)
  exist at the installed cdk version; fall back to `CfnJobDefinition.volumes/mountPoints`
  + raw `linuxParameters` if missing. Check `cdk/node_modules/aws-cdk-lib/aws-batch/lib/`.

## B4. EFS filesystem
- Create in the Batch stack (or construct): encrypted, in the same VPC, SG allows 2049
  from the compute SG. Mount target per AZ used.

## B5. `cdk/lib/batch-stack.ts` (NEW, opt-in)
- Mirror `workstation-stack.ts`: default-VPC lookup (`Vpc.fromLookup isDefault`),
  reads instanceType/numNodes/maxvCpus from config.json with `--context` overrides.
- Instantiates EFS + BatchRl construct. Pulls `isaacLabRepo` from Foundation (pass via
  props or `Repository.fromRepositoryName`).
- Outputs: job queue name, job definition name, EFS id, a ready-to-run
  `launch_rl_batch.py` example.
- **RISK:** uses default VPC (like workstation) for opt-in independence; if account has
  no default VPC, deploy fails. Documented tradeoff.

## B6. `cdk/bin/app.ts` (modify)
- Add `includeBatch = tryGetContext('batch') === 'true'` opt-in, mirroring
  `includeWorkstation`. Instantiate `BatchStack` with env/config. Deploy:
  `cdk deploy PhysicalAi-dev-Batch --context batch=true`.

## B7. `config.json` (modify)
- Add `"batch": { "instanceType": "g6.12xlarge", "numNodes": 2, "maxvCpus": 96 }`.

## B8. `training/scripts/launch_rl_batch.py` (NEW, laptop submitter)
- Mirror `launch_rl.py` style: argparse (`--task`, `--num-envs`, `--max-iterations`,
  `--num-nodes`, `--job-queue`, `--job-definition`, `--dry-run`), region/account
  resolution identical to launch_rl.py.
- Build `batch.submit_job` request (nodeOverrides for env vars). `--dry-run` prints the
  request and makes ZERO AWS calls.
- Print monitor command (`aws batch describe-jobs --jobs <id>`). Label multi-node NCCL
  convergence unvalidated.

## B9. `tests/` (add)
- `launch_rl_batch.py --dry-run` makes zero AWS calls and prints a submit_job request
  containing the right queue/def/nodes (reuse fake boto3 stub).

## Sequencing
- Parallel: {B1+B2}, {B8+B9}, B7. Serialize: B3→B4→B5→B6 (B5 needs B3/B4; B6 needs B5).
- Deploy order (user): deploy Foundation (rebuilds image from B1/B2) → wait CodeBuild →
  `cdk deploy PhysicalAi-dev-Batch --context batch=true`.

## Open risks (also in overview)
1. Multi-node NCCL convergence unverified on hardware (biggest unknown).
2. EFS mount mechanism: pick ONE (Batch EfsVolume preferred), avoid double-mount;
   `amazon-efs-utils` presence on ECS GPU AMI unverified → nfs4 fallback.
3. aws-batch L2 prop availability at installed cdk version (EfsVolume, shm).
4. g6.12xlarge sizing so exactly one container/instance.
5. Default VPC requirement.
6. NCCL_SOCKET_IFNAME eth0 vs ens5.
