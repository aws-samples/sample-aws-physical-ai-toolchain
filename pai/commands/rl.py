"""pai rl — Launch and monitor Isaac Lab RL training jobs."""

import os
import sys
from pathlib import Path

import click

from pai import config, helpers


def _ensure_repo_on_path():
    """Add repo root to sys.path so training.scripts imports work."""
    # Walk up from pai/commands to repo root (has config.json)
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "config.json").exists():
            repo_path = str(current)
            if repo_path not in sys.path:
                sys.path.insert(0, repo_path)
            return
        current = current.parent


@click.group()
def rl():
    """Launch and monitor Isaac Lab RL training jobs."""
    pass


@rl.command()
@click.option("--task", default="Isaac-Velocity-Flat-Anymal-D-v0", help="Isaac Lab task ID")
@click.option("--num-envs", type=int, default=4096, help="Number of parallel environments")
@click.option("--max-iterations", type=int, default=50, help="Maximum training iterations")
@click.option("--framework", default="rsl_rl", type=click.Choice(["rsl_rl", "skrl", "rl_games"]), help="RL framework")
@click.option("--engine", default="sagemaker", type=click.Choice(["sagemaker", "batch"]), help="Compute engine (SageMaker or AWS Batch)")
@click.option("--instance-type", default="ml.g5.xlarge", help="SageMaker instance type (SageMaker engine only)")
@click.option("--instance-count", type=int, default=1, help="Number of SageMaker instances (SageMaker engine only)")
@click.option("--runtime-min", type=int, default=60, help="Max runtime in minutes (SageMaker engine only)")
@click.option("--num-nodes", type=int, default=2, help="Number of Batch compute nodes (Batch engine only)")
@click.option("--job-queue", default=None, help="Batch job queue name (Batch engine only)")
@click.option("--job-definition", default=None, help="Batch job definition name (Batch engine only)")
@click.option("--dry-run", is_flag=True, help="Preview the job without making AWS calls")
def launch(task, num_envs, max_iterations, framework, engine, instance_type, instance_count, runtime_min, num_nodes, job_queue, job_definition, dry_run):
    """Launch an Isaac Lab RL training job."""
    # Set region before importing (scripts read from env)
    os.environ["AWS_DEFAULT_REGION"] = config.resolve_region()

    # Ensure repo root is on sys.path for training.scripts imports
    _ensure_repo_on_path()

    try:
        if engine == "sagemaker":
            # Lazy-import launch_rl
            from training.scripts import launch_rl

            launch_rl.launch(
                task=task,
                num_envs=num_envs,
                max_iterations=max_iterations,
                framework=framework,
                instance_type=instance_type,
                runtime_min=runtime_min,
                instance_count=instance_count,
                dry_run=dry_run,
            )
        else:  # batch
            # Lazy-import launch_rl_batch
            from training.scripts import launch_rl_batch

            # Use defaults from script if not provided
            if job_queue is None:
                job_queue = launch_rl_batch._default_queue()
            if job_definition is None:
                job_definition = launch_rl_batch._default_job_def()

            launch_rl_batch.launch(
                task=task,
                num_envs=num_envs,
                max_iterations=max_iterations,
                framework=framework,
                num_nodes=num_nodes,
                job_queue=job_queue,
                job_definition=job_definition,
                dry_run=dry_run,
            )
    except Exception as e:
        helpers.friendly_boto_error(e)


@rl.command()
@click.argument("job_name")
@click.option("--engine", default="sagemaker", type=click.Choice(["sagemaker", "batch"]), help="Compute engine")
def status(job_name, engine):
    """Check the status of a training job."""
    import boto3

    region = config.resolve_region()

    try:
        if engine == "sagemaker":
            sm = boto3.client("sagemaker", region_name=region)
            resp = sm.describe_training_job(TrainingJobName=job_name)

            helpers.heading(f"SageMaker Training Job: {job_name}")
            helpers.info(f"Status:          {resp['TrainingJobStatus']}")
            helpers.info(f"Secondary:       {resp.get('SecondaryStatus', 'N/A')}")
            helpers.info(f"Created:         {resp['CreationTime']}")
            if "TrainingStartTime" in resp:
                helpers.info(f"Started:         {resp['TrainingStartTime']}")
            if "TrainingEndTime" in resp:
                helpers.info(f"Ended:           {resp['TrainingEndTime']}")
            helpers.info(f"Instance Type:   {resp['ResourceConfig']['InstanceType']}")
            helpers.info(f"Instance Count:  {resp['ResourceConfig']['InstanceCount']}")
            helpers.info(f"Output Path:     {resp['OutputDataConfig']['S3OutputPath']}")

            if resp['TrainingJobStatus'] in ('Completed', 'Stopped'):
                helpers.success(f"\nJob finished: {resp['TrainingJobStatus']}")
            elif resp['TrainingJobStatus'] == 'Failed':
                helpers.error(f"\nJob failed: {resp.get('FailureReason', 'Unknown')}")
            else:
                helpers.info(f"\nJob in progress: {resp['SecondaryStatus']}")
        else:  # batch
            batch = boto3.client("batch", region_name=region)
            resp = batch.describe_jobs(jobs=[job_name])

            if not resp["jobs"]:
                helpers.error(f"Job not found: {job_name}")
                return

            job = resp["jobs"][0]
            helpers.heading(f"AWS Batch Job: {job_name}")
            helpers.info(f"Status:      {job['status']}")
            helpers.info(f"Job Queue:   {job['jobQueue']}")
            helpers.info(f"Job Def:     {job['jobDefinition']}")
            helpers.info(f"Created:     {job.get('createdAt', 'N/A')}")
            if "startedAt" in job:
                helpers.info(f"Started:     {job['startedAt']}")
            if "stoppedAt" in job:
                helpers.info(f"Stopped:     {job['stoppedAt']}")

            if "nodeProperties" in job:
                helpers.info(f"Nodes:       {job['nodeProperties'].get('numNodes', 'N/A')}")

            if job['status'] == 'SUCCEEDED':
                helpers.success(f"\nJob succeeded")
            elif job['status'] == 'FAILED':
                helpers.error(f"\nJob failed: {job.get('statusReason', 'Unknown')}")
            else:
                helpers.info(f"\nJob in progress")
    except Exception as e:
        helpers.friendly_boto_error(e)


def register(cli: click.Group):
    """Register rl commands with the CLI."""
    cli.add_command(rl)
